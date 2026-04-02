#include "pegasus/pegasus_runtime.h"
#include "pegasus/timer.h"
#include "pegasus/types.h"
#include <seal/seal.h>
#include <seal/util/polyarithsmallmod.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unistd.h>

using namespace seal;

// thread_local: each thread gets its own copy of these timing/memory
// diagnostic variables.  This eliminates data races when multiple
// std::threads call dense(), update_model_dense(), etc. concurrently.
thread_local double tmp_time = 0.0;
thread_local double total_save_model_time = 0.0;
thread_local double total_load_model_time = 0.0;
thread_local double total_offline_time = 0.0;
thread_local double total_online_time = 0.0;
thread_local double save_model_time = 0.0;
thread_local double load_model_time = 0.0;
thread_local double offline_time = 0.0;
thread_local double online_time = 0.0;
thread_local double vm, vm2, rss, rss2;

thread_local bool not_first_epoch;
std::string save_model_loc;
std::string model_name;

void process_mem_usage(double &vm_usage, double &resident_set)
{
    sleep(5);

    vm_usage = 0.0;
    resident_set = 0.0;

    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;

    std::cout << "CHECK RAM USAGE ";
    std::cout << "VM: " << vm_usage << "; RSS: " << resident_set << std::endl;
}

namespace gemini
{
    PegasusRunTime::PegasusRunTime(Parms parms, size_t num_threads)
        : parms_(parms), num_threads_(std::max<size_t>(1, num_threads))
    {
        size_t nlevels = parms.nlevels;

        if (parms.enable_repacking)
        {
            nlevels += (BetterSine::Depth());
        }

        std::string JSON = level2Params(nlevels);

        runtime_ = gemini::RunTime::Create(JSON);
        runtime_->ShowContext(std::cout);
        setUpRuntime();
        setUpSecretKeys();
        setUpPublicKeys();
        setUpFunctors();

        printf("n_{lwe} = %d, n_{lut} = %d, n_{ckks} = %d\n", parms_.lvl0_lattice_dim,
               parms_.lvl1_lattice_dim, parms_.lvl2_lattice_dim);
        printf("KS_base = 2^%d, sk.hamming = %d\n", KS_DC_BASE, SECRET_KEY_HW);
        printf("|msg| < %f, scale = 2^%f, extra_scale = 2^%f, nslots = %d",
               MsgRange(), std::log2(parms_.scale), std::log2(ExtraScaling()),
               parms_.nslots);
        printf(", #thread = %zd\n", num_threads_);
    }

    // ─────────────────────────────────────────────────────────────────
    //  Clone constructor: shared SEAL runtime, own num_threads_
    // ─────────────────────────────────────────────────────────────────
    //
    //  Why sharing runtime_ is safe during training:
    //
    //  1. Evaluator (Add, Sub, Mul, Relin, Rotate) — fully stateless,
    //     all const methods.  Thread-safe by design.
    //
    //  2. Encryptor — has internal PRNG with a mutex.  BUT Encrypt()
    //     is only called during setup (main thread, before pipeline).
    //     During training, threads operate on EXISTING ciphertexts,
    //     never encrypt fresh ones.  No contention.
    //
    //  3. MemoryManager::GetMMProf — already patched to thread_local.
    //     Each std::thread gets its own memory pool.  No contention.
    //
    //  4. ThreadPool — created LOCAL to each DEFINE_MT call using
    //     num_threads_ from THIS PegasusRunTime object.  Two threads
    //     calling act_batch() simultaneously create two independent
    //     ThreadPool(num_threads_) pools.  No sharing.
    //
    //  5. Galois keys, relin keys — read-only after construction.
    //     Shared safely.  (SaveGaloisKey/LoadGaloisKey are unimplemented
    //     in this SEAL fork, so we cannot create a separate runtime
    //     with transferred Galois keys anyway.)
    //
    //  What the clone provides:
    //    - Its own num_threads_ (controls ThreadPool sizing per call)
    //    - Conceptual separation of Module A / Module B runtimes
    //    - Same crypto keys → ciphertexts fully interoperable
    //
    PegasusRunTime::PegasusRunTime(const PegasusRunTime &source, size_t num_threads)
        : parms_(source.parms_), num_threads_(std::max<size_t>(1, num_threads))
    {
        printf("\n=== Cloning PegasusRunTime ===\n"
               "  Source #threads = %zd, Clone #threads = %zd\n",
               source.num_threads_, num_threads_);

        // ── 1. Share the SEAL runtime (shared_ptr, thread-safe) ────────
        //    Evaluator is stateless.  Encryptor not used during training.
        //    Galois/relin/rotation keys: read-only, no SaveGaloisKey API.
        runtime_ = source.runtime_;
        printf("  Shared SEAL runtime (Evaluator, keys, context)\n");

        // ── 2. Share sub-level SEAL contexts (read-only) ───────────────
        lvl0_runtime_ = source.lvl0_runtime_;
        lvl1_runtime_ = source.lvl1_runtime_;

        // ── 3. Copy secret keys (value types, read-only after setup) ───
        lvl0_sk_ntt_ = source.lvl0_sk_ntt_;
        lvl0_sk_non_ntt_ = source.lvl0_sk_non_ntt_;
        lvl1_sk_ntt_ = source.lvl1_sk_ntt_;
        lvl1_sk_non_ntt_ = source.lvl1_sk_non_ntt_;
        lvl2_sk_non_ntt_ = source.lvl2_sk_non_ntt_;

        // ── 4. Copy evaluation keys (deep copy, read-only after setup) ─
        //    These are large (~hundreds of MB) but immutable.
        //    Deep copy avoids any theoretical aliasing issues.
        lutEvalKey_ = source.lutEvalKey_;
        repackKey_ = source.repackKey_;
        lvl2Tolvl0_[0] = source.lvl2Tolvl0_[0];
        lvl1Tolvl0_[0] = source.lvl1Tolvl0_[0];

        // ── 5. Share functors via shared_ptr (const methods only) ──────
        sinFunctor_ = source.sinFunctor_;
        linearTransformer_ = source.linearTransformer_;
        lutFunctor_ = source.lutFunctor_;
        output_bounded_lutFunctor_ = source.output_bounded_lutFunctor_;
        extract_indices_ = source.extract_indices_;

        printf("  Clone complete: shared runtime, %zd threads per DEFINE_MT\n"
               "=== Clone ready ===\n\n",
               num_threads_);
    }

    Status PegasusRunTime::SlotsToCoeffs(Ctx *ct, int nct) const
    {
        ThreadPool pool(num_threads_);
        const size_t work_load = (nct + num_threads_ - 1) / num_threads_;
        for (int w = 0; w < num_threads_; ++w)
        {
            size_t start = w * work_load;
            size_t end = std::min<size_t>(start + work_load, nct);
            if (end > start)
            {
                pool.enqueue(
                    [&](size_t s, size_t e)
                    {
                        for (size_t i = s; i < e; ++i)
                        {
                            CHECK_AND_ABORT(SlotsToCoeffs(ct[i]));
                        }
                    },
                    start, end);
            }
        }
        return Status::Ok();
    }

    Status PegasusRunTime::SlotsToCoeffs(Ctx &out) const
    {
        size_t level = GetNModuli(out) - 1;
        size_t depth = linearTransformer_->depth();
        if (level < depth)
        {
            return Status::NotReady("SlotsToCoeffs require more levels");
        }
        // We keep only one moduli after S2C
        runtime_->DropModuli(&out, level - linearTransformer_->s2c_lvl_start());
        Ctx s2c;
        CHECK_STATUS(linearTransformer_->SlotsToCoeffs(out, &s2c));
        out = s2c;
        return Status::Ok();
    }

    Status PegasusRunTime::RotateLeft(Ctx &out, size_t offset) const
    {
        CHECK_STATUS(runtime_->RotateLeft(&out, offset));
        return Status::Ok();
    }

    Status PegasusRunTime::RotateRight(Ctx &out, size_t offset) const
    {
        CHECK_STATUS(runtime_->RotateRight(&out, offset));
        return Status::Ok();
    }

    Status PegasusRunTime::Add(Ctx &a, Ctx const &b) const
    {
        CHECK_STATUS(runtime_->Add(&a, b));
        return Status::Ok();
    }

    Status PegasusRunTime::Sub(Ctx &a, Ctx const &b) const
    {
        if (&a == &b)
        {
            return Status::NotReady("Sub itself is not supported");
        }
        CHECK_STATUS(runtime_->Sub(&a, b));
        return Status::Ok();
    }

    Status PegasusRunTime::Square(Ctx &a) const
    {
        CHECK_STATUS(runtime_->Mul(&a, a));
        return Status::Ok();
    }

    Status PegasusRunTime::RelinThenRescale(Ctx &a) const
    {
        CHECK_STATUS(runtime_->Relin(&a));
        F64 scale_up = parms_.scale * runtime_->GetModulusPrime(GetNModuli(a) - 1);
        scale_up = std::round(scale_up / a.scale());
        if (scale_up >= 1.)
        {
            CHECK_STATUS(runtime_->MulScalar(&a, 1., scale_up));
        }
        CHECK_STATUS(runtime_->RescaleNext(&a));
        return Status::Ok();
    }

    Status PegasusRunTime::ExtraAllCoefficients(const Ctx &in,
                                                std::vector<lwe::Ctx_st> &lwe_ct)
    {
        std::vector<rlwe::RLWE2LWECt_st> lwe_N_ct(parms_.nslots);
        if (!in.is_ntt_form())
        {
            rlwe::SampleExtract(lwe_N_ct.data(), in, extract_indices_,
                                runtime_->SEALRunTime());
        }
        else
        {
            auto copy{in};
            rlwe::SwitchNTTForm(copy, runtime_->SEALRunTime());
            rlwe::SampleExtract(lwe_N_ct.data(), copy, extract_indices_,
                                runtime_->SEALRunTime());
        }
        lwe_ct.resize(parms_.nslots);

        ThreadPool pool(num_threads_);
        const size_t work_load = (parms_.nslots + num_threads_ - 1) / num_threads_;
        for (int w = 0; w < num_threads_; ++w)
        {
            size_t start = w * work_load;
            size_t end = std::min<size_t>(start + work_load, parms_.nslots);

            if (end > start)
            {
                pool.enqueue(
                    [&](size_t s, size_t e)
                    {
                        for (size_t i = s; i < e; ++i)
                        {
                            rlwe::LWEKeySwitch(&lwe_ct[i], &lwe_N_ct[i], lvl2Tolvl0_,
                                               lvl0_runtime_);
                            lwe_ct[i].scale = in.scale();
                        }
                    },
                    start, end);
            }
        }
        return Status::Ok();
    }

    void PegasusRunTime::MulScalarLWECt(lwe::Ctx_st &out, const lwe::Ctx_st &a,
                                        uint64_t scalar) const
    {
        const auto &q0 =
            lvl0_runtime_->first_context_data()->parms().coeff_modulus()[0];
        using namespace seal::util;
        multiply_poly_scalar_coeffmod(CtData(&a), lwe::params::n() + 1, scalar, q0,
                                      CtData(&out));
        out.scale = a.scale;
    }

    void PegasusRunTime::AddLWECt(lwe::Ctx_st &out, lwe::Ctx_st const &a,
                                  lwe::Ctx_st const &b) const
    {
        using namespace lwe;
        using namespace seal::util;
        const auto &q0 =
            lvl0_runtime_->first_context_data()->parms().coeff_modulus()[0];
        if (!seal::util::are_close(a.scale, b.scale))
        {
            throw std::invalid_argument("AddLWECt scale mismatch");
        }
        add_poly_coeffmod(CtData(&a), CtData(&b), lwe::params::n() + 1, q0,
                          CtData(&out));
        out.scale = a.scale;
    }

    void PegasusRunTime::SubLWECt(lwe::Ctx_st &out, lwe::Ctx_st const &a,
                                  lwe::Ctx_st const &b) const
    {
        using namespace lwe;
        using namespace seal::util;
        if (&a == &b)
        {
            throw std::runtime_error("SubLWECt self-substraction");
        }
        if (!seal::util::are_close(a.scale, b.scale))
        {
            throw std::invalid_argument("SubLWECt scale mismatch");
        }
        const auto &q0 =
            lvl0_runtime_->first_context_data()->parms().coeff_modulus()[0];
        sub_poly_coeffmod(CtData(&a), CtData(&b), lwe::params::n() + 1, q0,
                          CtData(&out));
        out.scale = a.scale;
    }

    void PegasusRunTime::MulConstant(lwe::Ctx_t lvl0_ct, double v) const
    {
        F64 out_scale = lvl0_ct->scale * lutFunctor_->GetPostMultiplier();
        rlwe::RLWE2LWECt_t lvl1_ct;
        lutFunctor_->MulConstant(lvl1_ct, lvl0_ct, v);
        rlwe::LWEKeySwitch(lvl0_ct, lvl1_ct, lvl1Tolvl0_, lvl0_runtime_);
        lvl0_ct->scale = out_scale;
    }

    double PegasusRunTime::DecryptLWE(lwe::Ctx_st const &lwe_ct) const
    {
        return lwe::SymDec(&lwe_ct, lvl0_sk_non_ntt_, lvl0_runtime_);
    }

    std::string PegasusRunTime::level2Params(int level_left) const
    {
        std::stringstream ss;
        ss << "{\"log2PolyDegree\":" << (int)std::log2(parms_.lvl2_lattice_dim)
           << ",";
        ss << "\"nSpecialPrimes\":1,\"seed\":0,";
        ss << "\"moduliArray\":[" << std::to_string(numBitsP0()) << ",";

        level_left = std::max(1, level_left);
        std::string norm_sze = std::to_string(numBitsP0());
        for (int i = 1; i < level_left; ++i)
        {
            ss << norm_sze << ",";
        }

        // Final is the special modulus
        ss << 59 << "]";
        ss << "}";
        return ss.str();
    }

    void PegasusRunTime::setUpRuntime()
    {
        using namespace seal;
        auto lvl2_runtime = runtime_->SEALRunTime();
        const auto &modulus =
            lvl2_runtime->key_context_data()->parms().coeff_modulus();
        EncryptionParameters parms(seal::scheme_type::CKKS);

        // Level-1 RGSW works with 2 moduli
        std::vector<Modulus> lvl1_modulus{modulus.front(), modulus.back()};

        parms.set_coeff_modulus(lvl1_modulus);
        parms.set_poly_modulus_degree(parms_.lvl1_lattice_dim);
        parms.set_galois_generator(5);
        lvl1_runtime_ = SEALContext::Create(parms, true, sec_level_type::none);

        // Level-0 LWE works with 1 modulus
        std::vector<Modulus> lvl0_modulus{modulus.front()};
        parms.set_poly_modulus_degree(parms_.lvl0_lattice_dim);
        parms.set_coeff_modulus(lvl0_modulus);
        lvl0_runtime_ = SEALContext::Create(parms, true, sec_level_type::none);
    }

    void PegasusRunTime::setUpSecretKeys()
    {
        // non-ntt form of the CKKS secret key
        auto lvl2_runtime = runtime_->SEALRunTime();
        auto const &lvl2_sk = runtime_->SEALSecretKey();
        lvl2_sk_non_ntt_.data().resize(parms_.lvl2_lattice_dim);
        std::copy_n((const uint64_t *)lvl2_sk.data().data(), parms_.lvl2_lattice_dim,
                    lvl2_sk_non_ntt_.data().data());
        rlwe::SwitchNTTForm(lvl2_sk_non_ntt_.data().data(), NTTDir::FromNTT, 1,
                            lvl2_runtime);
        // Generate sk_{lut}
        lwe::GenerateHammingSecretKey(lvl1_sk_ntt_, SECRET_KEY_HW, /*is_ntt*/ true,
                                      lvl1_runtime_);
        lvl1_sk_non_ntt_.data().resize(parms_.lvl1_lattice_dim);
        std::copy_n((const uint64_t *)lvl1_sk_ntt_.data().data(),
                    parms_.lvl1_lattice_dim, lvl1_sk_non_ntt_.data().data());
        rlwe::SwitchNTTForm(lvl1_sk_non_ntt_.data().data(), NTTDir::FromNTT, 1,
                            lvl1_runtime_);
        // Generate sk_{lwe}
        lwe::SKInit(lvl0_sk_ntt_, lvl0_sk_non_ntt_, SECRET_KEY_HW, lvl0_runtime_);
    }

    void PegasusRunTime::setUpPublicKeys()
    {
        if (parms_.enable_repacking)
        {
            gemini::RpKeyInit(repackKey_, std::pow(2., numBitsP0()),
                              runtime_->SEALSecretKey(), lvl0_sk_non_ntt_, runtime_);
        }

        rlwe::BKInit(lutEvalKey_, lvl0_sk_non_ntt_, lvl1_sk_ntt_, lvl1_runtime_);
        rlwe::LWEKSKeyInit(lvl2Tolvl0_, KS_DC_BASE, lvl2_sk_non_ntt_.data(),
                           lvl0_sk_ntt_, lvl0_runtime_, runtime_->SEALRunTime());
        rlwe::LWEKSKeyInit(lvl1Tolvl0_, KS_DC_BASE, lvl1_sk_non_ntt_.data(),
                           lvl0_sk_ntt_, lvl0_runtime_, lvl1_runtime_);
    }

    void PegasusRunTime::setUpFunctors()
    {
        using namespace gemini;
        using namespace seal;
        LinearTransformer::Parms ltParams;
        ltParams.nslots = parms_.nslots;
        ltParams.s2c_lvl_start = 2;
        ltParams.c2s_lvl_start = 0; // no C2S is needed
        ltParams.s2cMultiplier = parms_.s2c_multiplier;
        ltParams.c2sMultiplier = 1.;

        if (parms_.enable_repacking)
        {
            sinFunctor_.reset(new BetterSine(runtime_));
        }
        linearTransformer_.reset(new LinearTransformer(ltParams, runtime_));

        lutFunctor_.reset(
            new rlwe::LWEGateBooter(parms_.scale, lutEvalKey_, lvl1_runtime_, 1.));

        output_bounded_lutFunctor_.reset(new rlwe::LWEGateBooter(
            parms_.scale, lutEvalKey_, lvl1_runtime_, ExtraScaling()));

        extract_indices_.resize(parms_.nslots);
        const size_t log2N = (size_t)std::log2(parms_.lvl2_lattice_dim);
        for (size_t i = 0; i < parms_.nslots; ++i)
        {
            extract_indices_[i] = seal::util::reverse_bits(i, log2N - 1);
        }
    }

    Status PegasusRunTime::Repack(Ctx &out,
                                  std::vector<lwe::Ctx_st> const &wires) const
    {
        if (!parms_.enable_repacking)
        {
            return Status::NotReady("Repacking is not ready");
        }

        size_t n = wires.size();
        if (n > parms_.lvl2_lattice_dim / 2)
        {
            return Status::ArgumentError("Repacking too many lwe ciphers to repack");
        }
        if (!IsTwoPower(n))
        {
            return Status::ArgumentError("Repacking invalid number of lwe ciphers");
        }

        CHECK_STATUS(::gemini::Repack(out, sinFunctor_->interval_bound(), wires,
                                      repackKey_, runtime_));
        out.scale() = parms_.scale;
        CHECK_STATUS(sinFunctor_->Apply(out, parms_.scale));
        return Status::Ok();
    }

    void PegasusRunTime::dense(
        Parms parms,
        std::vector<Ctx> &A_cipher,       // D^{i-1}
        std::vector<std::vector<F64>> &W, // W^{i}
        std::vector<F64> &B,              // b^{i}
        std::vector<Ctx> &out_cipher,     // F^{i}
        int layer,
        int epoch)
    {
        int nslots = parms.nslots;
        double scale = parms.scale;
        int moduli = 4;

        tmp_time = 0.0;
        load_model_time = 0.0;
        offline_time = 0.0;
        online_time = 0.0;

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            int w_row = W.size();
            int w_col = W[0].size();
            int len = num_threads_;
            int rounds = std::ceil(1.0 * w_row / num_threads_);

            std::vector<std::vector<Ctx>> W_cipher(num_threads_, std::vector<Ctx>(w_col));
            std::vector<Ctx> B_cipher(w_col);

            tmp_time = 0.0;
            AutoTimer offline_timer(&tmp_time);

            // Initialize output ciphertexts
#pragma omp parallel for
            for (int i = 0; i < w_col; i++)
            {
                runtime_->EncryptZero(&out_cipher[i]);
                runtime_->DropModuli(&out_cipher[i], GetNModuli(out_cipher[i]) - moduli + 1);
                out_cipher[i].scale() = scale;
            }

            offline_timer.stop();
            offline_time += tmp_time;
            total_offline_time += tmp_time;

            process_mem_usage(vm, rss);

            tmp_time = 0.0;
            AutoTimer load_model_timer(&tmp_time);

            // Prepare model bias
            if (not_first_epoch)
            {
                std::filebuf fbin;
                fbin.open(save_model_loc + model_name + "_B" + std::to_string(layer) + "_" + std::to_string(epoch) + ".out", std::ios::in);
                std::istream osin(&fbin);

                for (int i = 0; i < w_col; i++)
                {
                    CHECK_AND_ABORT(runtime_->LoadCtx(&B_cipher[i], osin));
                }

                fbin.close();
            }
            else
            {
#pragma omp parallel for
                for (int i = 0; i < w_col; i++)
                {
                    std::vector<F64> tmp(nslots);
                    for (int j = 0; j < nslots; j++)
                    {
                        tmp[j] = B[i];
                    }
                    CHECK_AND_ABORT(EncodeThenEncrypt(tmp, B_cipher[i]));
                }
            }

#pragma omp parallel for
            for (int i = 0; i < w_col; i++)
            {
                runtime_->DropModuli(&B_cipher[i], GetNModuli(B_cipher[i]) - moduli + 1);
            }

            load_model_timer.stop();
            load_model_time += tmp_time;
            total_load_model_time += tmp_time;

            process_mem_usage(vm2, rss2);
            std::cout << "B RAM USAGE " << (vm2 - vm) / (1024 * 1024) << " GB" << std::endl;

            std::filebuf fbin;
            fbin.open(save_model_loc + model_name + "_W" + std::to_string(layer) + "_" + std::to_string(epoch) + ".out", std::ios::in);
            std::istream osin(&fbin);

            std::cout << "A total of " << rounds << " rounds\n";
            for (int ii = 0; ii < rounds; ii++)
            {
                std::cout << "round " << ii << std::endl;

                if (ii == (rounds - 1) && (ii + 1) * num_threads_ > w_row)
                {
                    len = w_row - ii * num_threads_;
                }

                tmp_time = 0.0;
                AutoTimer load_model_timer(&tmp_time);

                if (ii == 0)
                {
                    process_mem_usage(vm, rss);
                }

                // Prepare model weight
                if (not_first_epoch)
                {
                    for (int i = 0; i < len; i++)
                    {
                        for (int j = 0; j < w_col; j++)
                        {
                            int pos = ii * num_threads_ + i;
                            if (pos < w_row)
                            {
                                CHECK_AND_ABORT(runtime_->LoadCtx(&W_cipher[i][j], osin));
                            }
                        }
                    }
                }
                else
                {
#pragma omp parallel for collapse(2)
                    for (int i = 0; i < len; i++)
                    {
                        for (int j = 0; j < w_col; j++)
                        {
                            int pos = ii * num_threads_ + i;
                            if (pos < w_row)
                            {
                                std::vector<F64> tmp(nslots);
                                for (int k = 0; k < nslots; k++)
                                {
                                    tmp[k] = W[ii * num_threads_ + i][j];
                                }
                                CHECK_AND_ABORT(EncodeThenEncrypt(tmp, W_cipher[i][j]));
                            }
                        }
                    }
                }

#pragma omp parallel for collapse(2)
                for (int i = 0; i < len; i++)
                {
                    for (int j = 0; j < w_col; j++)
                    {
                        runtime_->DropModuli(&W_cipher[i][j], GetNModuli(W_cipher[i][j]) - moduli);
                    }
                }

                if (ii == 0)
                {
                    process_mem_usage(vm2, rss2);
                    std::cout << "W batch RAM USAGE " << (vm2 - vm) / (1024 * 1024) << " GB" << std::endl;
                }

                load_model_timer.stop();
                load_model_time += tmp_time;
                total_load_model_time += tmp_time;

                tmp_time = 0.0;
                AutoTimer online_timer(&tmp_time);

                // Dense function (Wx)
#pragma omp parallel for collapse(2)
                for (int i = 0; i < len; i++)
                {
                    for (int j = 0; j < w_col; j++)
                    {
                        Ctx tmp_cipher = A_cipher[ii * num_threads_ + i];
                        runtime_->DropModuli(&tmp_cipher, GetNModuli(tmp_cipher) - moduli);
                        runtime_->MulRelinRescale(&W_cipher[i][j], tmp_cipher);
                    }
                }

#pragma omp parallel for
                for (int j = 0; j < w_col; j++)
                {
                    for (int i = 0; i < len; i++)
                    {
                        runtime_->Add(&out_cipher[j], W_cipher[i][j]);
                    }
                }

                online_timer.stop();
                online_time += tmp_time;
                total_online_time += tmp_time;
            }

            fbin.close();

            tmp_time = 0.0;
            AutoTimer online_timer(&tmp_time);

            // Add bias (Wx+b)
#pragma omp parallel for
            for (int i = 0; i < w_col; i++)
            {
                runtime_->Add(&out_cipher[i], B_cipher[i]);
            }

            online_timer.stop();
            online_time += tmp_time;
            total_online_time += tmp_time;

            std::cout << "DENSE LOAD MODEL TIME " << load_model_time << std::endl;
            std::cout << "DENSE OFFLINE TIME " << offline_time << std::endl;
            std::cout << "DENSE ONLINE TIME " << online_time << std::endl;

            release_vector_2D(W_cipher);
            release_vector_1D(B_cipher);
            W_cipher.clear();
            B_cipher.clear();
            MemoryManager::SwitchProfile(std::move(old_prof));
        }
    }

    void PegasusRunTime::conv(
        Parms parms,
        std::vector<std::vector<std::vector<Ctx>>> &A_cipher,
        std::vector<std::vector<std::vector<std::vector<F64>>>> &W,
        std::vector<F64> &B,
        std::vector<std::vector<std::vector<Ctx>>> &out_cipher,
        int stride,
        int layer,
        int epoch)
    {
        int nslots = parms.nslots;
        double scale = parms.scale;
        int moduli = 4;

        tmp_time = 0.0;
        load_model_time = 0.0;
        offline_time = 0.0;
        online_time = 0.0;

        int kernel_kc = W.size();
        int kernel_ic = W[0].size();
        int kernel_size = W[0][0].size();
        int out_size = out_cipher.size();

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            std::vector<std::vector<std::vector<std::vector<Ctx>>>> W_cipher(kernel_kc, std::vector<std::vector<std::vector<Ctx>>>(kernel_ic, std::vector<std::vector<Ctx>>(kernel_size, std::vector<Ctx>(kernel_size))));
            std::vector<Ctx> B_cipher(kernel_kc);

            tmp_time = 0.0;
            AutoTimer offline_timer(&tmp_time);

            // Initialize output ciphertexts
#pragma omp parallel for collapse(3)
            for (int i = 0; i < out_size; i++)
            {
                for (int j = 0; j < out_size; j++)
                {
                    for (int kc = 0; kc < kernel_kc; kc++)
                    {
                        runtime_->EncryptZero(&out_cipher[i][j][kc]);
                        runtime_->DropModuli(&out_cipher[i][j][kc], GetNModuli(out_cipher[i][j][kc]) - moduli + 1);
                        out_cipher[i][j][kc].scale() = scale;
                    }
                }
            }

            offline_timer.stop();
            offline_time += tmp_time;
            total_offline_time += tmp_time;

            tmp_time = 0.0;
            AutoTimer load_model_timer(&tmp_time);

            process_mem_usage(vm, rss);

            // Prepare model weight
            if (not_first_epoch)
            {
                std::filebuf fbin;
                fbin.open(save_model_loc + model_name + "_W" + std::to_string(layer) + "_" + std::to_string(epoch) + ".out", std::ios::in);
                std::istream osin(&fbin);

                for (int kc = 0; kc < kernel_kc; kc++)
                {
                    for (int ic = 0; ic < kernel_ic; ic++)
                    {
                        for (int i = 0; i < kernel_size; i++)
                        {
                            for (int j = 0; j < kernel_size; j++)
                            {
                                CHECK_AND_ABORT(runtime_->LoadCtx(&W_cipher[kc][ic][i][j], osin));
                            }
                        }
                    }
                }

                fbin.close();
            }
            else
            {
#pragma omp parallel for collapse(4)
                for (int kc = 0; kc < kernel_kc; kc++)
                {
                    for (int ic = 0; ic < kernel_ic; ic++)
                    {
                        for (int i = 0; i < kernel_size; i++)
                        {
                            for (int j = 0; j < kernel_size; j++)
                            {
                                std::vector<F64> tmp(nslots);
                                for (int ii = 0; ii < nslots; ii++)
                                {
                                    tmp[ii] = W[kc][ic][i][j];
                                }
                                CHECK_AND_ABORT(EncodeThenEncrypt(tmp, W_cipher[kc][ic][i][j]));
                            }
                        }
                    }
                }
            }

#pragma omp parallel for collapse(4)
            for (int kc = 0; kc < kernel_kc; kc++)
            {
                for (int ic = 0; ic < kernel_ic; ic++)
                {
                    for (int i = 0; i < kernel_size; i++)
                    {
                        for (int j = 0; j < kernel_size; j++)
                        {
                            runtime_->DropModuli(&W_cipher[kc][ic][i][j], GetNModuli(W_cipher[kc][ic][i][j]) - moduli);
                        }
                    }
                }
            }

            process_mem_usage(vm2, rss2);
            std::cout << "W RAM USAGE " << (vm2 - vm) / (1024 * 1024) << " GB" << std::endl;

            process_mem_usage(vm, rss);

            // Prepare model bias
            if (not_first_epoch)
            {
                std::filebuf fbin;
                fbin.open(save_model_loc + model_name + "_B" + std::to_string(layer) + "_" + std::to_string(epoch) + ".out", std::ios::in);
                std::istream osin(&fbin);

                for (int kc = 0; kc < kernel_kc; kc++)
                {
                    CHECK_AND_ABORT(runtime_->LoadCtx(&B_cipher[kc], osin));
                }

                fbin.close();
            }
            else
            {
#pragma omp parallel for
                for (int i = 0; i < kernel_kc; i++)
                {
                    std::vector<F64> tmp(nslots);
                    for (int j = 0; j < nslots; j++)
                    {
                        tmp[j] = B[i];
                    }
                    CHECK_AND_ABORT(EncodeThenEncrypt(tmp, B_cipher[i]));
                }
            }

#pragma omp parallel for
            for (int i = 0; i < kernel_kc; i++)
            {
                runtime_->DropModuli(&B_cipher[i], GetNModuli(B_cipher[i]) - moduli + 1);
            }

            process_mem_usage(vm2, rss2);
            std::cout << "B RAM USAGE " << (vm2 - vm) / (1024 * 1024) << " GB" << std::endl;

            load_model_timer.stop();
            load_model_time += tmp_time;
            total_load_model_time += tmp_time;

            tmp_time = 0.0;
            AutoTimer online_timer(&tmp_time);

            // Convolution (Wx)
            for (int kc = 0; kc < kernel_kc; kc++)
            {
                std::cout << "Output channel " << kc << std::endl;
                for (int ic = 0; ic < kernel_ic; ic++)
                {
#pragma omp parallel for collapse(4)
                    for (int i = 0; i < out_size; i++)
                    {
                        for (int j = 0; j < out_size; j++)
                        {
                            for (int ii = 0; ii < kernel_size; ii++)
                            {
                                for (int jj = 0; jj < kernel_size; jj++)
                                {
                                    Ctx tmp_cipher = A_cipher[i * stride + ii][j * stride + jj][ic];
                                    runtime_->DropModuli(&tmp_cipher, GetNModuli(tmp_cipher) - moduli);
                                    runtime_->MulRelinRescale(&tmp_cipher, W_cipher[kc][ic][ii][jj]);
                                    runtime_->Add(&out_cipher[i][j][kc], tmp_cipher);
                                }
                            }
                        }
                    }
                }

                // Add bias
#pragma omp parallel for collapse(2)
                for (int i = 0; i < out_size; i++)
                {
                    for (int j = 0; j < out_size; j++)
                    {
                        runtime_->Add(&out_cipher[i][j][kc], B_cipher[kc]);
                    }
                }
            }

            online_timer.stop();
            online_time += tmp_time;
            total_online_time += tmp_time;

            std::cout << "CONV LOAD MODEL TIME " << load_model_time << std::endl;
            std::cout << "CONV OFFLINE TIME " << offline_time << std::endl;
            std::cout << "CONV ONLINE TIME " << online_time << std::endl;

            release_vector_4D(W_cipher);
            release_vector_1D(B_cipher);
            W_cipher.clear();
            B_cipher.clear();

            MemoryManager::SwitchProfile(std::move(old_prof));
        }
    }

    void PegasusRunTime::maxpool(
        Parms parms,
        std::vector<std::vector<std::vector<Ctx>>> &U_cipher,
        std::vector<std::vector<std::vector<Ctx>>> &MI_cipher,
        std::vector<std::vector<std::vector<Ctx>>> &D_cipher,
        std::vector<std::vector<lwe::Ctx_st>> &U_lwe_cipher,
        std::vector<std::vector<lwe::Ctx_st>> &D_lwe_cipher,
        int mp_len)
    {
        int nslots = parms.nslots;
        double scale = parms.scale;

        tmp_time = 0.0;
        load_model_time = 0.0;
        offline_time = 0.0;
        online_time = 0.0;

        int conv_size = U_cipher.size();
        int mp_size = D_cipher.size();
        int kernel_kc = U_cipher[0][0].size();

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            std::vector<Ctx> tmp_flatten_cipher(conv_size * conv_size);

            for (int kc = 0; kc < kernel_kc; kc++)
            {
                std::cout << "Output channel " << kc << std::endl;

                tmp_time = 0.0;
                AutoTimer online_timer(&tmp_time);

#pragma omp parallel for collapse(4)
                for (int i = 0; i < mp_size; i++)
                {
                    for (int j = 0; j < mp_size; j++)
                    {
                        for (int ii = 0; ii < mp_len; ii++)
                        {
                            for (int jj = 0; jj < mp_len; jj++)
                            {
                                int idx = (i * mp_size + j) * mp_len * mp_len;
                                tmp_flatten_cipher[idx + ii * mp_len + jj] = U_cipher[i * mp_len + ii][j * mp_len + jj][kc];
                            }
                        }
                    }
                }

                online_timer.stop();
                online_time += tmp_time;
                total_online_time += tmp_time;

                // S2C & ReLU
                std::cout << "======= S2C =======\n";
                s2c_and_extract(tmp_flatten_cipher, U_lwe_cipher, conv_size * conv_size, nslots);
                std::cout << "====== RELU =======\n";
                relu_lwe(U_lwe_cipher, D_lwe_cipher, conv_size * conv_size, nslots);
                std::cout << "====== DRELU + REPACK =======\n";
                drelu_lwe(U_lwe_cipher, conv_size * conv_size, nslots);
                repack(U_lwe_cipher, tmp_flatten_cipher, conv_size * conv_size, nslots);

                tmp_time = 0.0;
                AutoTimer online_timer_2(&tmp_time);

#pragma omp parallel for collapse(4)
                for (int i = 0; i < mp_size; i++)
                {
                    for (int j = 0; j < mp_size; j++)
                    {
                        for (int ii = 0; ii < mp_len; ii++)
                        {
                            for (int jj = 0; jj < mp_len; jj++)
                            {
                                int idx = (i * mp_size + j) * mp_len * mp_len;
                                U_cipher[i * mp_len + ii][j * mp_len + jj][kc] = tmp_flatten_cipher[idx + ii * mp_len + jj];
                            }
                        }
                    }
                }

#pragma omp parallel for
                for (int i = 0; i < conv_size * conv_size; i++)
                {
                    U_lwe_cipher[i] = D_lwe_cipher[i];
                }

                // Maxpooling
                std::cout << "====== MAXPOOL =======\n";
                // #pragma omp parallel for collapse(2)
                for (int i = 0; i < mp_size; i++)
                {
                    for (int j = 0; j < mp_size; j++)
                    {
                        int idx = (i * mp_size + j) * mp_len * mp_len;
                        std::vector<lwe::Ctx_st> m0_ct(nslots);
                        std::vector<lwe::Ctx_st> m1_ct(nslots);
                        std::vector<lwe::Ctx_st> max_ct(nslots);

#pragma omp parallel for
                        for (int k = 0; k < nslots; k++)
                        {
                            max_ct[k] = U_lwe_cipher[idx][k];
                        }

                        // Maxpool
                        for (int ii = 1; ii < mp_len * mp_len; ii++)
                        {
#pragma omp parallel for
                            for (int k = 0; k < nslots; k++)
                            {
                                // (m0 + m1)
                                AddLWECt(m0_ct[k], max_ct[k], U_lwe_cipher[idx + ii][k]);
                                // (m0 - m1)
                                SubLWECt(m1_ct[k], max_ct[k], U_lwe_cipher[idx + ii][k]);
                            }
                            Half(m0_ct.data(), m0_ct.size());    // 0.5*x
                            HalfAbs(m1_ct.data(), m1_ct.size()); // 0.5*|x|
#pragma omp parallel for
                            for (int k = 0; k < nslots; k++)
                            {
                                // 0.5*(m0 + m1) + 0.5*|m0 - m1|
                                AddLWECt(max_ct[k], m0_ct[k], m1_ct[k]);
                            }
                        }

#pragma omp parallel for
                        for (int k = 0; k < nslots; k++)
                        {
                            D_lwe_cipher[i * mp_size + j][k] = max_ct[k];
                        }

                        // Find position of the max value
#pragma omp parallel for collapse(2)
                        for (int ii = 0; ii < mp_len * mp_len; ii++)
                        {
                            for (int k = 0; k < nslots; k++)
                            {
                                SubLWECt(U_lwe_cipher[idx + ii][k], U_lwe_cipher[idx + ii][k], max_ct[k]);
                            }
                        }

                        for (int ii = 0; ii < mp_len * mp_len; ii++)
                        {
                            IsNotNegative(U_lwe_cipher[idx + ii].data(), U_lwe_cipher[idx + ii].size());
                        }
                    }
                }

                online_timer_2.stop();
                online_time += tmp_time;
                total_online_time += tmp_time;

                std::cout << "====== REPACK MAXPOOL VALUES =======\n";
                // save max value
                repack(D_lwe_cipher, tmp_flatten_cipher, mp_size * mp_size, nslots);

#pragma omp parallel for collapse(2)
                for (int i = 0; i < mp_size; i++)
                {
                    for (int j = 0; j < mp_size; j++)
                    {
                        D_cipher[i][j][kc] = tmp_flatten_cipher[i * mp_size + j];
                    }
                }

                std::cout << "====== REPACK POSITIONS OF MAX VALUES =======\n";
                // save max value posotion
                repack(U_lwe_cipher, tmp_flatten_cipher, conv_size * conv_size, nslots);

#pragma omp parallel for collapse(4)
                for (int i = 0; i < mp_size; i++)
                {
                    for (int j = 0; j < mp_size; j++)
                    {
                        for (int ii = 0; ii < mp_len; ii++)
                        {
                            for (int jj = 0; jj < mp_len; jj++)
                            {
                                int idx = (i * mp_size + j) * mp_len * mp_len;
                                MI_cipher[i * mp_len + ii][j * mp_len + jj][kc] = tmp_flatten_cipher[idx + ii * mp_len + jj];
                            }
                        }
                    }
                }

                online_timer.stop();
                online_time += tmp_time;
                total_online_time += tmp_time;

                std::cout << std::endl;
            }

            std::cout << "INSIDE MAXPOOL FUNCTION ONLINE TIME " << online_time << std::endl;

            release_vector_1D(tmp_flatten_cipher);
            tmp_flatten_cipher.clear();

            MemoryManager::SwitchProfile(std::move(old_prof));
            process_mem_usage(vm, rss);
            std::cout << "LAYER END CONV RAM USAGE " << (vm) / (1024 * 1024) << " GB" << std::endl;
        }
    }

    void PegasusRunTime::delta_softmax(
        Parms parms,
        std::vector<Ctx> &A_cipher,
        std::vector<Ctx> &B_cipher,
        std::vector<Ctx> &out_cipher)
    {
        tmp_time = 0.0;
        online_time = 0.0;

        int length = A_cipher.size();

        AutoTimer online_timer(&tmp_time);

#pragma omp parallel for
        for (int i = 0; i < length; i++)
        {
            Ctx tmp_cipher = B_cipher[i];
            runtime_->DropModuli(&tmp_cipher, GetNModuli(tmp_cipher) - GetNModuli(A_cipher[0]));
            out_cipher[i] = A_cipher[i];
            runtime_->Sub(&out_cipher[i], tmp_cipher);
        }

        online_timer.stop();
        online_time += tmp_time;
        total_online_time += tmp_time;
        std::cout << "Delta Softmax ONLINE TIME " << online_time << std::endl;
    }

    void PegasusRunTime::delta_1D_dense_1D(
        Parms parms,
        std::vector<std::vector<F64>> &W, // W^{i+1}
        std::vector<Ctx> &B_cipher,       // DE^{i+1}
        std::vector<Ctx> &C_cipher,       // U^{i}
        std::vector<Ctx> &out_cipher,     // DE^{i}
        int layer,
        int epoch)
    {
        int nslots = parms.nslots;
        double scale = parms.scale;
        int moduli = 5;

        tmp_time = 0.0;
        load_model_time = 0.0;
        offline_time = 0.0;
        online_time = 0.0;

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            int w_row = W.size();
            int w_col = W[0].size();
            int len = num_threads_;
            int rounds = std::ceil(1.0 * w_row / num_threads_);

            std::vector<std::vector<Ctx>> W_cipher(num_threads_, std::vector<Ctx>(w_col));

            tmp_time = 0.0;
            AutoTimer offline_timer(&tmp_time);

            // Initialize output ciphertexts
#pragma omp parallel for
            for (int i = 0; i < w_row; i++)
            {
                runtime_->EncryptZero(&out_cipher[i]);
                runtime_->DropModuli(&out_cipher[i], GetNModuli(out_cipher[i]) - moduli + 1);
                out_cipher[i].scale() = scale;
            }

            offline_timer.stop();
            offline_time += tmp_time;
            total_offline_time += tmp_time;

            std::filebuf fbin;
            fbin.open(save_model_loc + model_name + "_W" + std::to_string(layer) + "_" + std::to_string(epoch) + ".out", std::ios::in);
            std::istream osin(&fbin);

            std::cout << "A total of " << rounds << " rounds" << std::endl;
            for (int ii = 0; ii < rounds; ii++)
            {
                std::cout << "round " << ii << std::endl;
                if (ii == (rounds - 1) && (ii + 1) * num_threads_ > w_row)
                {
                    len = w_row - ii * num_threads_;
                }

                // Prepare model weight
                tmp_time = 0.0;
                AutoTimer load_model_timer(&tmp_time);

                if (not_first_epoch)
                {
                    for (int i = 0; i < len; i++)
                    {
                        for (int j = 0; j < w_col; j++)
                        {
                            int pos = ii * num_threads_ + i;
                            if (pos < w_row)
                            {
                                CHECK_AND_ABORT(runtime_->LoadCtx(&W_cipher[i][j], osin));
                            }
                        }
                    }
                }
                else
                {
#pragma omp parallel for collapse(2)
                    for (int i = 0; i < len; i++)
                    {
                        for (int j = 0; j < w_col; j++)
                        {
                            int pos = ii * num_threads_ + i;
                            if (pos < w_row)
                            {
                                std::vector<F64> tmp(nslots);
                                for (int k = 0; k < nslots; k++)
                                {
                                    tmp[k] = W[ii * num_threads_ + i][j];
                                }
                                CHECK_AND_ABORT(EncodeThenEncrypt(tmp, W_cipher[i][j]));
                            }
                        }
                    }
                }

#pragma omp parallel for collapse(2)
                for (int i = 0; i < len; i++)
                {
                    for (int j = 0; j < w_col; j++)
                    {
                        runtime_->DropModuli(&W_cipher[i][j], GetNModuli(W_cipher[i][j]) - moduli);
                    }
                }

                load_model_timer.stop();
                load_model_time += tmp_time;
                total_load_model_time += tmp_time;

                tmp_time = 0.0;
                AutoTimer online_timer(&tmp_time);

                // Calculate delta
#pragma omp parallel for collapse(2)
                for (int i = 0; i < len; i++)
                {
                    for (int j = 0; j < w_col; j++)
                    {
                        Ctx tmp_cipher = B_cipher[j];
                        runtime_->DropModuli(&tmp_cipher, GetNModuli(tmp_cipher) - moduli);
                        runtime_->MulRelinRescale(&W_cipher[i][j], tmp_cipher);
                    }
                }

#pragma omp parallel for
                for (int i = 0; i < len; i++)
                {
                    for (int j = 0; j < w_col; j++)
                    {
                        runtime_->Add(&out_cipher[ii * num_threads_ + i], W_cipher[i][j]);
                    }
                }

                online_timer.stop();
                online_time += tmp_time;
                total_online_time += tmp_time;
            }

            fbin.close();

            tmp_time = 0.0;
            AutoTimer online_timer(&tmp_time);

#pragma omp parallel for
            for (int i = 0; i < w_row; i++)
            {
                Ctx tmp_cipher = C_cipher[i];
                runtime_->DropModuli(&tmp_cipher, GetNModuli(tmp_cipher) - moduli + 1);
                runtime_->MulRelinRescale(&out_cipher[i], tmp_cipher);
            }

            online_timer.stop();
            online_time += tmp_time;
            total_online_time += tmp_time;

            std::cout << "DELTA LOAD MODEL TIME " << load_model_time << std::endl;
            std::cout << "DELTA OFFLINE TIME " << offline_time << std::endl;
            std::cout << "DELTA ONLINE TIME " << online_time << std::endl;

            release_vector_2D(W_cipher);
            W_cipher.clear();

            MemoryManager::SwitchProfile(std::move(old_prof));
        }
    }

    void PegasusRunTime::delta_3D_conv_3D(
        Parms parms,
        std::vector<std::vector<std::vector<std::vector<F64>>>> &W,
        std::vector<std::vector<std::vector<Ctx>>> &B_cipher,
        std::vector<std::vector<std::vector<Ctx>>> &C_cipher,
        std::vector<std::vector<std::vector<Ctx>>> &out_cipher,
        int layer,
        int epoch)
    {
        int nslots = parms.nslots;
        double scale = parms.scale;
        int moduli = 5;

        tmp_time = 0.0;
        load_model_time = 0.0;
        offline_time = 0.0;
        online_time = 0.0;

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            int w_size = W[0][0].size();
            int w_ic = W[0].size();
            int w_kc = W.size();
            int b_size = B_cipher.size();
            int p_size = w_size * 2 + b_size - 2;
            int out_size = out_cipher.size();

            std::vector<std::vector<Ctx>> W_cipher(w_size, std::vector<Ctx>(w_size));
            std::vector<std::vector<Ctx>> P_cipher(p_size, std::vector<Ctx>(p_size));

            tmp_time = 0.0;
            AutoTimer offline_timer(&tmp_time);

            // Initialize output ciphertexts
#pragma omp parallel for collapse(3)
            for (int i = 0; i < out_size; i++)
            {
                for (int j = 0; j < out_size; j++)
                {
                    for (int k = 0; k < w_ic; k++)
                    {
                        runtime_->EncryptZero(&out_cipher[i][j][k]);
                        runtime_->DropModuli(&out_cipher[i][j][k], GetNModuli(out_cipher[i][j][k]) - moduli + 1);
                        out_cipher[i][j][k].scale() = scale;
                    }
                }
            }

            offline_timer.stop();
            offline_time += tmp_time;
            total_offline_time += tmp_time;

            std::filebuf fbin;
            fbin.open(save_model_loc + model_name + "_W" + std::to_string(layer) + "_" + std::to_string(epoch) + ".out", std::ios::in);
            std::istream osin(&fbin);

            for (int kc = 0; kc < w_kc; kc++)
            {
                std::cout << "Output channel " << kc << std::endl;

                tmp_time = 0.0;
                AutoTimer offline_timer(&tmp_time);

#pragma omp parallel for collapse(2)
                for (int i = 0; i < p_size; i++)
                {
                    for (int j = 0; j < p_size; j++)
                    {
                        runtime_->EncryptZero(&P_cipher[i][j]);
                    }
                }

#pragma omp parallel for collapse(2)
                for (int i = 0; i < b_size; i++)
                {
                    for (int j = 0; j < b_size; j++)
                    {
                        P_cipher[i + w_size - 1][j + w_size - 1] = B_cipher[i][j][kc];
                    }
                }

#pragma omp parallel for collapse(2)
                for (int i = 0; i < p_size; i++)
                {
                    for (int j = 0; j < p_size; j++)
                    {
                        runtime_->DropModuli(&P_cipher[i][j], GetNModuli(P_cipher[i][j]) - moduli);
                    }
                }

                offline_timer.stop();
                offline_time += tmp_time;
                total_offline_time += tmp_time;

                for (int ic = 0; ic < w_ic; ic++)
                {
                    tmp_time = 0.0;
                    AutoTimer load_model_timer(&tmp_time);

                    if (not_first_epoch)
                    {
                        for (int i = 0; i < w_size; i++)
                        {
                            for (int j = 0; j < w_size; j++)
                            {
                                CHECK_AND_ABORT(runtime_->LoadCtx(&W_cipher[i][j], osin));
                            }
                        }

                        for (int i = 0; i < w_size; i++)
                        {
                            for (int j = 0; j < w_size; j++)
                            {
                                Ctx tmp_cipher = W_cipher[i][j];
                                W_cipher[i][j] = W_cipher[w_size - 1 - i][w_size - 1 - j];
                                W_cipher[w_size - 1 - i][w_size - 1 - j] = tmp_cipher;
                            }
                        }
                    }
                    else
                    {
#pragma omp parallel for collapse(2)
                        for (int i = 0; i < w_size; i++)
                        {
                            for (int j = 0; j < w_size; j++)
                            {
                                std::vector<F64> tmp(nslots);
                                for (int k = 0; k < nslots; k++)
                                {
                                    tmp[k] = W[kc][ic][w_size - 1 - i][w_size - 1 - j];
                                }
                                CHECK_AND_ABORT(EncodeThenEncrypt(tmp, W_cipher[i][j])); // rotate 180 degree
                            }
                        }
                    }

#pragma omp parallel for collapse(2)
                    for (int i = 0; i < w_size; i++)
                    {
                        for (int j = 0; j < w_size; j++)
                        {
                            runtime_->DropModuli(&W_cipher[i][j], GetNModuli(W_cipher[i][j]) - moduli);
                        }
                    }

                    load_model_timer.stop();
                    load_model_time += tmp_time;
                    total_load_model_time += tmp_time;

                    tmp_time = 0.0;
                    AutoTimer online_timer(&tmp_time);

                    // Calculate delta
#pragma omp parallel for collapse(2)
                    for (int i = 0; i < out_size; i++)
                    {
                        for (int j = 0; j < out_size; j++)
                        {
                            Ctx tmp_cipher;
                            runtime_->EncryptZero(&tmp_cipher);
                            runtime_->DropModuli(&tmp_cipher, GetNModuli(tmp_cipher) - moduli + 1);
                            tmp_cipher.scale() = scale;
                            for (int ii = 0; ii < w_size; ii++)
                            {
                                for (int jj = 0; jj < w_size; jj++)
                                {
                                    Ctx tmp_cipher_2 = P_cipher[i + ii][j + jj];
                                    runtime_->MulRelinRescale(&tmp_cipher_2, W_cipher[ii][jj]);
                                    runtime_->Add(&tmp_cipher, tmp_cipher_2);
                                }
                            }
                            runtime_->Add(&out_cipher[i][j][ic], tmp_cipher);
                        }
                    }

                    online_timer.stop();
                    online_time += tmp_time;
                    total_online_time += tmp_time;
                }

#pragma omp parallel for collapse(2)
                for (int i = 0; i < out_size; i++)
                {
                    for (int j = 0; j < out_size; j++)
                    {
                        Ctx tmp_cipher = C_cipher[i][j][kc];
                        runtime_->DropModuli(&tmp_cipher, GetNModuli(tmp_cipher) - moduli + 1);
                        runtime_->MulRelinRescale(&out_cipher[i][j][kc], tmp_cipher);
                    }
                }
            }

            fbin.close();

            std::cout << "DELTA LOAD MODEL TIME " << load_model_time << std::endl;
            std::cout << "DELTA OFFLINE TIME " << offline_time << std::endl;
            std::cout << "DELTA ONLINE TIME " << online_time << std::endl;

            release_vector_2D(W_cipher);
            release_vector_2D(P_cipher);
            W_cipher.clear();
            P_cipher.clear();

            MemoryManager::SwitchProfile(std::move(old_prof));
        }
    }

    void PegasusRunTime::prepare_delta_maxpooling_from_mat(
        Parms parms,
        std::vector<std::vector<F64>> &W,                         // W^{i+1}
        std::vector<Ctx> &D_cipher,                               // DE^{i+1}
        std::vector<std::vector<std::vector<Ctx>>> &DE_mp_cipher, // DE_mp^{i}
        int layer,
        int epoch)
    {
        int nslots = parms.nslots;
        double scale = parms.scale;
        int moduli = 5;

        tmp_time = 0.0;
        load_model_time = 0.0;
        offline_time = 0.0;
        online_time = 0.0;

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            int d_size = D_cipher.size();
            int w_row = W.size();
            int w_col = W[0].size();
            int de_size = DE_mp_cipher.size();
            int de_kc = DE_mp_cipher[0][0].size();
            int len = num_threads_;
            int rounds = std::ceil(1.0 * w_row / num_threads_);

            std::vector<std::vector<Ctx>> W_cipher(num_threads_, std::vector<Ctx>(w_col));
            std::vector<Ctx> DE_mp_cipher_flatten(w_row);

            tmp_time = 0.0;
            AutoTimer offline_timer(&tmp_time);

            // Initialize output ciphertexts
#pragma omp parallel for
            for (int i = 0; i < w_row; i++)
            {
                runtime_->EncryptZero(&DE_mp_cipher_flatten[i]);
                runtime_->DropModuli(&DE_mp_cipher_flatten[i], GetNModuli(DE_mp_cipher_flatten[i]) - moduli + 1);
                DE_mp_cipher_flatten[i].scale() = scale;
            }

            offline_timer.stop();
            offline_time += tmp_time;
            total_offline_time += tmp_time;

            std::filebuf fbin;
            fbin.open(save_model_loc + model_name + "_W" + std::to_string(layer) + "_" + std::to_string(epoch) + ".out", std::ios::in);
            std::istream osin(&fbin);

            std::cout << "A total of " << rounds << "rounds\n";
            for (int ii = 0; ii < rounds; ii++)
            {
                std::cout << "round " << ii << std::endl;
                if (ii == (rounds - 1) && (ii + 1) * num_threads_ > w_row)
                {
                    len = w_row - ii * num_threads_;
                }

                // Prepare model weight
                tmp_time = 0.0;
                AutoTimer load_model_timer(&tmp_time);

                if (not_first_epoch)
                {
                    for (int i = 0; i < len; i++)
                    {
                        for (int j = 0; j < w_col; j++)
                        {
                            int pos = ii * num_threads_ + i;
                            if (pos < w_row)
                            {
                                CHECK_AND_ABORT(runtime_->LoadCtx(&W_cipher[i][j], osin));
                            }
                        }
                    }
                }
                else
                {
#pragma omp parallel for collapse(2)
                    for (int i = 0; i < len; i++)
                    {
                        for (int j = 0; j < w_col; j++)
                        {
                            int pos = ii * num_threads_ + i;
                            if (pos < w_row)
                            {
                                std::vector<F64> tmp(nslots);
                                for (int k = 0; k < nslots; k++)
                                {
                                    tmp[k] = W[ii * num_threads_ + i][j];
                                }
                                CHECK_AND_ABORT(EncodeThenEncrypt(tmp, W_cipher[i][j]));
                            }
                        }
                    }
                }

#pragma omp parallel for collapse(2)
                for (int i = 0; i < len; i++)
                {
                    for (int j = 0; j < w_col; j++)
                    {
                        runtime_->DropModuli(&W_cipher[i][j], GetNModuli(W_cipher[i][j]) - moduli);
                    }
                }

                load_model_timer.stop();
                load_model_time += tmp_time;
                total_load_model_time += tmp_time;

                tmp_time = 0.0;
                AutoTimer online_timer(&tmp_time);

                // Calculate delta
#pragma omp parallel for collapse(2)
                for (int i = 0; i < len; i++)
                {
                    for (int j = 0; j < w_col; j++)
                    {
                        Ctx tmp_cipher = D_cipher[j];
                        runtime_->DropModuli(&tmp_cipher, GetNModuli(tmp_cipher) - moduli);
                        runtime_->MulRelinRescale(&W_cipher[i][j], tmp_cipher);
                    }
                }

#pragma omp parallel for
                for (int i = 0; i < len; i++)
                {
                    for (int j = 0; j < w_col; j++)
                    {
                        runtime_->Add(&DE_mp_cipher_flatten[ii * num_threads_ + i], W_cipher[i][j]);
                    }
                }

                online_timer.stop();
                online_time += tmp_time;
                total_online_time += tmp_time;
            }

            fbin.close();

            // Reshape to 3D
            tmp_time = 0.0;
            AutoTimer online_timer(&tmp_time);

#pragma omp parallel for collapse(3)
            for (int i = 0; i < de_size; i++)
            {
                for (int j = 0; j < de_size; j++)
                {
                    for (int kc = 0; kc < de_kc; kc++)
                    {
                        DE_mp_cipher[i][j][kc] = DE_mp_cipher_flatten[(i * de_size + j) * de_kc + kc];
                    }
                }
            }

            online_timer.stop();
            online_time += tmp_time;
            total_online_time += tmp_time;

            std::cout << "DELTA MP LOAD MODEL TIME " << load_model_time << std::endl;
            std::cout << "DELTA MP OFFLINE TIME " << offline_time << std::endl;
            std::cout << "DELTA MP ONLINE TIME " << online_time << std::endl;

            release_vector_2D(W_cipher);
            release_vector_1D(DE_mp_cipher_flatten);
            W_cipher.clear();
            DE_mp_cipher_flatten.clear();

            MemoryManager::SwitchProfile(std::move(old_prof));
        }
    }

    void PegasusRunTime::prepare_delta_maxpooling_from_conv(
        Parms parms,
        std::vector<std::vector<std::vector<std::vector<F64>>>> &W,
        std::vector<std::vector<std::vector<Ctx>>> &D_cipher,
        std::vector<std::vector<std::vector<Ctx>>> &DE_mp_cipher,
        int layer,
        int epoch)
    {
        int nslots = parms.nslots;
        double scale = parms.scale;
        int moduli = 5;

        tmp_time = 0.0;
        load_model_time = 0.0;
        offline_time = 0.0;
        online_time = 0.0;

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            int w_size = W[0][0].size();
            int w_ic = W[0].size();
            int w_kc = W.size();
            int d_size = D_cipher.size();
            int p_size = w_size * 2 + d_size - 2;
            int de_mp_size = DE_mp_cipher.size();

            std::vector<std::vector<Ctx>> W_cipher(w_size, std::vector<Ctx>(w_size));
            std::vector<std::vector<Ctx>> P_cipher(p_size, std::vector<Ctx>(p_size));

            tmp_time = 0.0;
            AutoTimer offline_timer(&tmp_time);

            // Initialize output ciphertexts
#pragma omp parallel for collapse(3)
            for (int i = 0; i < de_mp_size; i++)
            {
                for (int j = 0; j < de_mp_size; j++)
                {
                    for (int k = 0; k < w_ic; k++)
                    {
                        runtime_->EncryptZero(&DE_mp_cipher[i][j][k]);
                        runtime_->DropModuli(&DE_mp_cipher[i][j][k], GetNModuli(DE_mp_cipher[i][j][k]) - moduli + 1);
                        DE_mp_cipher[i][j][k].scale() = scale;
                    }
                }
            }

            offline_timer.stop();
            offline_time += tmp_time;
            total_offline_time += tmp_time;

            std::filebuf fbin;
            fbin.open(save_model_loc + model_name + "_W" + std::to_string(layer) + "_" + std::to_string(epoch) + ".out", std::ios::in);
            std::istream osin(&fbin);

            for (int kc = 0; kc < w_kc; kc++)
            {
                std::cout << "Output channel " << kc << std::endl;

                tmp_time = 0.0;
                AutoTimer offline_timer(&tmp_time);

#pragma omp parallel for collapse(2)
                for (int i = 0; i < p_size; i++)
                {
                    for (int j = 0; j < p_size; j++)
                    {
                        runtime_->EncryptZero(&P_cipher[i][j]);
                    }
                }

#pragma omp parallel for collapse(2)
                for (int i = 0; i < d_size; i++)
                {
                    for (int j = 0; j < d_size; j++)
                    {
                        P_cipher[i + w_size - 1][j + w_size - 1] = D_cipher[i][j][kc];
                    }
                }

#pragma omp parallel for collapse(2)
                for (int i = 0; i < p_size; i++)
                {
                    for (int j = 0; j < p_size; j++)
                    {
                        runtime_->DropModuli(&P_cipher[i][j], GetNModuli(P_cipher[i][j]) - moduli);
                    }
                }

                offline_timer.stop();
                offline_time += tmp_time;
                total_offline_time += tmp_time;

                for (int ic = 0; ic < w_ic; ic++)
                {
                    tmp_time = 0.0;
                    AutoTimer load_model_timer(&tmp_time);

                    if (not_first_epoch)
                    {
                        for (int i = 0; i < w_size; i++)
                        {
                            for (int j = 0; j < w_size; j++)
                            {
                                CHECK_AND_ABORT(runtime_->LoadCtx(&W_cipher[i][j], osin));
                            }
                        }

                        for (int i = 0; i < w_size; i++)
                        {
                            for (int j = 0; j < w_size; j++)
                            {
                                Ctx tmp_cipher = W_cipher[i][j];
                                W_cipher[i][j] = W_cipher[w_size - 1 - i][w_size - 1 - j];
                                W_cipher[w_size - 1 - i][w_size - 1 - j] = tmp_cipher;
                            }
                        }
                    }
                    else
                    {
#pragma omp parallel for collapse(2)
                        for (int i = 0; i < w_size; i++)
                        {
                            for (int j = 0; j < w_size; j++)
                            {
                                std::vector<F64> tmp(nslots);
                                for (int k = 0; k < nslots; k++)
                                {
                                    tmp[k] = W[kc][ic][w_size - 1 - i][w_size - 1 - j];
                                }
                                CHECK_AND_ABORT(EncodeThenEncrypt(tmp, W_cipher[i][j])); // rotate 180 degree
                            }
                        }
                    }

#pragma omp parallel for collapse(2)
                    for (int i = 0; i < w_size; i++)
                    {
                        for (int j = 0; j < w_size; j++)
                        {
                            runtime_->DropModuli(&W_cipher[i][j], GetNModuli(W_cipher[i][j]) - moduli);
                        }
                    }

                    load_model_timer.stop();
                    load_model_time += tmp_time;
                    total_load_model_time += tmp_time;

                    tmp_time = 0.0;
                    AutoTimer online_timer(&tmp_time);

                    // Calculate delta
#pragma omp parallel for collapse(2)
                    for (int i = 0; i < de_mp_size; i++)
                    {
                        for (int j = 0; j < de_mp_size; j++)
                        {
                            Ctx tmp_cipher;
                            runtime_->EncryptZero(&tmp_cipher);
                            runtime_->DropModuli(&tmp_cipher, GetNModuli(tmp_cipher) - moduli + 1);
                            tmp_cipher.scale() = scale;
                            for (int ii = 0; ii < w_size; ii++)
                            {
                                for (int jj = 0; jj < w_size; jj++)
                                {
                                    Ctx tmp_cipher_2 = P_cipher[i + ii][j + jj];
                                    runtime_->MulRelinRescale(&tmp_cipher_2, W_cipher[ii][jj]);
                                    runtime_->Add(&tmp_cipher, tmp_cipher_2);
                                }
                            }
                            runtime_->Add(&DE_mp_cipher[i][j][ic], tmp_cipher);
                        }
                    }

                    online_timer.stop();
                    online_time += tmp_time;
                    total_online_time += tmp_time;
                }
            }

            fbin.close();

            std::cout << "DELTA MP LOAD MODEL TIME " << load_model_time << std::endl;
            std::cout << "DELTA MP OFFLINE TIME " << offline_time << std::endl;
            std::cout << "DELTA MP ONLINE TIME " << online_time << std::endl;

            release_vector_2D(W_cipher);
            release_vector_2D(P_cipher);
            W_cipher.clear();
            P_cipher.clear();

            MemoryManager::SwitchProfile(std::move(old_prof));
        }
    }

    void PegasusRunTime::delta_maxpooling(
        Parms parms,
        std::vector<std::vector<std::vector<Ctx>>> &D_cipher,
        std::vector<std::vector<std::vector<Ctx>>> &U_cipher,
        std::vector<std::vector<std::vector<Ctx>>> &MI_cipher,
        std::vector<std::vector<lwe::Ctx_st>> &lwe_cipher,
        std::vector<std::vector<std::vector<Ctx>>> &DE_cipher)
    {
        int nslots = parms.nslots;
        double scale = parms.scale;

        tmp_time = 0.0;
        offline_time = 0.0;
        online_time = 0.0;

        int moduli = GetNModuli(D_cipher[0][0][0]); // level = 4

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            int conv_size = U_cipher.size();
            int conv_kc = U_cipher[0][0].size();

            tmp_time = 0.0;
            AutoTimer online_timer(&tmp_time);

            for (int kc = 0; kc < conv_kc; kc++)
            {
                std::cout << "kc " << kc << std::endl;
#pragma omp parallel for collapse(2)
                for (int i = 0; i < conv_size; i++)
                {
                    for (int j = 0; j < conv_size; j++)
                    {
                        runtime_->DropModuli(&MI_cipher[i][j][kc], GetNModuli(MI_cipher[i][j][kc]) - moduli - 1);
                        runtime_->DropModuli(&U_cipher[i][j][kc], GetNModuli(U_cipher[i][j][kc]) - moduli - 1);
                    }
                }

#pragma omp parallel for collapse(2)
                for (int i = 0; i < conv_size; i++)
                {
                    for (int j = 0; j < conv_size; j++)
                    {
                        runtime_->MulRelinRescale(&U_cipher[i][j][kc], MI_cipher[i][j][kc]);
                    }
                }

#pragma omp parallel for collapse(2)
                for (int i = 0; i < conv_size; i++)
                {
                    for (int j = 0; j < conv_size; j++)
                    {
                        runtime_->MulRelinRescale(&U_cipher[i][j][kc], D_cipher[int(i / 2)][int(j / 2)][kc]);
                        DE_cipher[i][j][kc] = U_cipher[i][j][kc];
                    }
                }
            }

            online_timer.stop();
            online_time += tmp_time;
            total_online_time += tmp_time;

            std::cout << "DELTA MAXPOOLING TIME " << online_time << std::endl;

            MemoryManager::SwitchProfile(std::move(old_prof));
        }
    }

    void PegasusRunTime::update_model_dense(
        Parms parms,
        std::vector<Ctx> &D_cipher,       // D^{i}
        std::vector<Ctx> &DE_cipher,      // DE^{i+1}
        std::vector<std::vector<F64>> &W, // W^{i+1}
        std::vector<F64> &B,              // B^{i+1}
        int moduli,
        int layer,
        int epoch)
    {
        int nslots = parms.nslots;
        double scale = parms.scale;

        tmp_time = 0.0;
        save_model_time = 0.0;
        load_model_time = 0.0;
        offline_time = 0.0;
        online_time = 0.0;

        std::vector<F64> T(nslots);

        int w_row = W.size();
        int w_col = W[0].size();
        int len = num_threads_;
        int rounds = std::ceil(1.0 * w_row / num_threads_);

        Ctx lr_cipher;
        std::vector<F64> lr(nslots, 0);
        lr[0] = 1.0 / nslots; // lr/num_threads_
        CHECK_AND_ABORT(EncodeThenEncrypt(lr, lr_cipher));
        runtime_->DropModuli(&lr_cipher, GetNModuli(lr_cipher) - moduli - 1);

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            std::vector<std::vector<Ctx>> W_cipher(num_threads_, std::vector<Ctx>(w_col));
            std::vector<Ctx> B_cipher(w_col);

            tmp_time = 0.0;
            AutoTimer offline_timer(&tmp_time);

#pragma omp parallel for
            for (int i = 0; i < w_col; i++)
            {
                runtime_->DropModuli(&DE_cipher[i], GetNModuli(DE_cipher[i]) - moduli - 2);
            }

            offline_timer.stop();
            offline_time += tmp_time;
            total_offline_time += tmp_time;

            std::filebuf fwin;
            fwin.open(save_model_loc + model_name + "_W" + std::to_string(layer) + "_" + std::to_string(epoch) + ".out", std::ios::in);
            std::istream oswin(&fwin);

            std::filebuf fwout;
            fwout.open(save_model_loc + model_name + "_W" + std::to_string(layer) + "_" + std::to_string(epoch + 1) + ".out", std::ios::out);
            std::ostream oswout(&fwout);

            std::cout << "A total of " << rounds << " rounds" << std::endl;
            for (int ii = 0; ii < rounds; ii++)
            {
                std::cout << "round " << ii << std::endl;

                if (ii == (rounds - 1) && (ii + 1) * num_threads_ > w_row)
                {
                    len = w_row - ii * num_threads_;
                }

                // Prepare model weight
                tmp_time = 0.0;
                AutoTimer load_model_timer(&tmp_time);

                if (not_first_epoch)
                {
                    for (int i = 0; i < len; i++)
                    {
                        for (int j = 0; j < w_col; j++)
                        {
                            int pos = ii * num_threads_ + i;
                            if (pos < w_row)
                            {
                                CHECK_AND_ABORT(runtime_->LoadCtx(&W_cipher[i][j], oswin));
                            }
                        }
                    }
                }
                else
                {
#pragma omp parallel for collapse(2)
                    for (int i = 0; i < len; i++)
                    {
                        for (int j = 0; j < w_col; j++)
                        {
                            int pos = ii * num_threads_ + i;
                            if (pos < w_row)
                            {
                                std::vector<F64> tmp(nslots);
                                for (int k = 0; k < nslots; k++)
                                {
                                    tmp[k] = W[ii * num_threads_ + i][j];
                                }
                                CHECK_AND_ABORT(EncodeThenEncrypt(tmp, W_cipher[i][j]));
                            }
                        }
                    }
                }

#pragma omp parallel for collapse(2)
                for (int i = 0; i < len; i++)
                {
                    for (int j = 0; j < w_col; j++)
                    {
                        runtime_->DropModuli(&W_cipher[i][j], GetNModuli(W_cipher[i][j]) - moduli);
                    }
                }

                load_model_timer.stop();
                load_model_time += tmp_time;
                total_load_model_time += tmp_time;

                tmp_time = 0.0;
                AutoTimer online_timer(&tmp_time);

                // Update model (W)
#pragma omp parallel for collapse(2)
                for (int i = 0; i < len; i++)
                {
                    for (int j = 0; j < w_col; j++)
                    {
                        Ctx tmp_cipher, tmp_cipher_2;
                        tmp_cipher = D_cipher[ii * num_threads_ + i];
                        runtime_->DropModuli(&tmp_cipher, GetNModuli(tmp_cipher) - moduli - 2);
                        runtime_->MulRelinRescale(&tmp_cipher, DE_cipher[j]);

                        // Rotate and Sum
                        for (int k = log2(nslots) - 1; k >= 0; k--)
                        {
                            tmp_cipher_2 = tmp_cipher;
                            runtime_->RotateLeft(&tmp_cipher_2, pow(2, k));
                            runtime_->Add(&tmp_cipher, tmp_cipher_2);
                        }

                        runtime_->MulRelinRescale(&tmp_cipher, lr_cipher);
                        tmp_cipher_2 = tmp_cipher;

                        for (int k = 0; k < log2(nslots); k++)
                        {
                            runtime_->RotateRight(&tmp_cipher_2, pow(2, k));
                            runtime_->Add(&tmp_cipher, tmp_cipher_2);
                            tmp_cipher_2 = tmp_cipher;
                        }
                        runtime_->Sub(&W_cipher[i][j], tmp_cipher);
                    }
                }

                online_timer.stop();
                online_time += tmp_time;
                total_online_time += tmp_time;

                // Save model weight (W)
                tmp_time = 0.0;
                AutoTimer save_model_timer(&tmp_time);

                for (int i = 0; i < len; i++)
                {
                    for (int j = 0; j < w_col; j++)
                    {
                        // write w to file
                        runtime_->SaveCtx(W_cipher[i][j], oswout);

                        // update the plaintext model
                        // used for debug
                        DecryptThenDecode(W_cipher[i][j], T);
                        W[ii * num_threads_ + i][j] = T[0];
                    }
                }

                save_model_timer.stop();
                save_model_time += tmp_time;
                total_save_model_time += tmp_time;
            }

            fwout.close();
            fwin.close();

            // Prepare model bias
            tmp_time = 0.0;
            AutoTimer load_model_timer(&tmp_time);

            if (not_first_epoch)
            {
                std::filebuf fbin;
                fbin.open(save_model_loc + model_name + "_B" + std::to_string(layer) + "_" + std::to_string(epoch) + ".out", std::ios::in);
                std::istream osbin(&fbin);

                for (int i = 0; i < w_col; i++)
                {
                    CHECK_AND_ABORT(runtime_->LoadCtx(&B_cipher[i], osbin));
                }

                fbin.close();
            }
            else
            {
#pragma omp parallel for
                for (int i = 0; i < w_col; i++)
                {
                    std::vector<F64> tmp(nslots);
                    for (int k = 0; k < nslots; k++)
                    {
                        tmp[k] = B[i];
                    }
                    CHECK_AND_ABORT(EncodeThenEncrypt(tmp, B_cipher[i]));
                }
            }

            load_model_timer.stop();
            load_model_time += tmp_time;
            total_load_model_time += tmp_time;

            tmp_time = 0.0;
            AutoTimer online_timer(&tmp_time);

#pragma omp parallel for
            for (int i = 0; i < w_col; i++)
            {
                runtime_->DropModuli(&B_cipher[i], GetNModuli(B_cipher[i]) - moduli);
            }

            // Update model bias (b)
#pragma omp parallel for
            for (int i = 0; i < w_col; i++)
            {
                Ctx tmp_cipher, tmp_cipher_2;
                tmp_cipher = DE_cipher[i];
                runtime_->DropModuli(&tmp_cipher, GetNModuli(tmp_cipher) - moduli - 1);

                // Rotate and Sum
                for (int k = log2(nslots) - 1; k >= 0; k--)
                {
                    tmp_cipher_2 = tmp_cipher;
                    runtime_->RotateLeft(&tmp_cipher_2, pow(2, k));
                    runtime_->Add(&tmp_cipher, tmp_cipher_2);
                }

                runtime_->MulRelinRescale(&tmp_cipher, lr_cipher);
                tmp_cipher_2 = tmp_cipher;

                for (int k = 0; k < log2(nslots); k++)
                {
                    runtime_->RotateRight(&tmp_cipher_2, pow(2, k));
                    runtime_->Add(&tmp_cipher, tmp_cipher_2);
                    tmp_cipher_2 = tmp_cipher;
                }
                runtime_->Sub(&B_cipher[i], tmp_cipher);
            }

            online_timer.stop();
            online_time += tmp_time;
            total_online_time += tmp_time;

            std::filebuf fbout;
            fbout.open(save_model_loc + model_name + "_B" + std::to_string(layer) + "_" + std::to_string(epoch + 1) + ".out", std::ios::out);
            std::ostream osbout(&fbout);

            // Save model bias (b)
            tmp_time = 0.0;
            AutoTimer save_model_timer(&tmp_time);

            for (int i = 0; i < w_col; i++)
            {
                runtime_->SaveCtx(B_cipher[i], osbout);

                // update plaintext model
                // used for debug
                DecryptThenDecode(B_cipher[i], T);
                B[i] = T[0];
            }

            save_model_timer.stop();
            save_model_time += tmp_time;
            total_save_model_time += tmp_time;

            fbout.close();

            std::cout << "UPDATE MODEL DENSE LOAD MODEL TIME " << load_model_time << std::endl;
            std::cout << "UPDATE MODEL DENSE SAVE MODEL TIME " << save_model_time << std::endl;
            std::cout << "UPDATE MODEL DENSE OFFLINE TIME " << offline_time << std::endl;
            std::cout << "UPDATE MODEL DENSE ONLINE TIME " << online_time << std::endl;

            release_vector_2D(W_cipher);
            release_vector_1D(B_cipher);
            W_cipher.clear();
            B_cipher.clear();

            MemoryManager::SwitchProfile(std::move(old_prof));
        }
    }

    void PegasusRunTime::padding(
        std::vector<std::vector<std::vector<Ctx>>> &A_cipher,
        std::vector<std::vector<std::vector<Ctx>>> &B_cipher,
        int pad_x, int pad_y)
    {
        int a_size = A_cipher.size();
        int a_chnl = A_cipher[0][0].size();
        int b_size = B_cipher.size();

        for (int i = 0; i < a_size; i++)
        {
            for (int j = 0; j < a_size; j++)
            {
                for (int k = 0; k < a_chnl; k++)
                {
                    if (i < pad_x || j < pad_y || (i >= b_size + pad_x) || (j >= b_size + pad_y))
                    {
                        runtime_->EncryptZero(&A_cipher[i][j][k]);
                        runtime_->DropModuli(&A_cipher[i][j][k], GetNModuli(A_cipher[i][j][k]) - GetNModuli(B_cipher[0][0][0]));
                    }
                    else
                    {
                        A_cipher[i][j][k] = B_cipher[i - pad_x][j - pad_y][k];
                    }
                }
            }
        }
    }

    void PegasusRunTime::flatten(
        std::vector<std::vector<std::vector<Ctx>>> &A_3D_cipher,
        std::vector<Ctx> &A_1D_cipher)
    {
        int a_size = A_3D_cipher.size();
        int a_chnl = A_3D_cipher[0][0].size();

        for (int kc = 0; kc < a_chnl; kc++)
        {
            for (int i = 0; i < a_size; i++)
            {
                for (int j = 0; j < a_size; j++)
                {
                    A_1D_cipher[(i * a_size + j) * a_chnl + kc] = A_3D_cipher[i][j][kc];
                }
            }
        }
    }

    void PegasusRunTime::delta_1D_to_3D_dilate(
        Parms parms,
        std::vector<Ctx> &DE_cipher,
        std::vector<std::vector<std::vector<Ctx>>> &DE_3D_cipher,
        int conv_size,
        int stride,
        int moduli)
    {
        int nslots = parms.nslots;
        double scale = parms.scale;

        tmp_time = 0.0;
        offline_time = 0.0;

        std::vector<F64> T(nslots);

        int de_size = DE_cipher.size();
        int de_dilate_size = DE_3D_cipher.size();
        int kernel_kc = DE_3D_cipher[0][0].size();

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            tmp_time = 0.0;
            AutoTimer offline_timer(&tmp_time);

#pragma omp parallel for
            for (int i = 0; i < de_size; i++)
            {
                runtime_->DropModuli(&DE_cipher[i], GetNModuli(DE_cipher[i]) - moduli - 2);
            }

#pragma omp parallel for collapse(3)
            for (int kc = 0; kc < kernel_kc; kc++)
            {
                for (int i = 0; i < de_dilate_size; i++)
                {
                    for (int j = 0; j < de_dilate_size; j++)
                    {
                        runtime_->EncryptZero(&DE_3D_cipher[i][j][kc]);
                        runtime_->DropModuli(&DE_3D_cipher[i][j][kc], GetNModuli(DE_3D_cipher[i][j][kc]) - moduli - 2);
                    }
                }
            }

#pragma omp parallel for collapse(3)
            for (int i = 0; i < conv_size; i++)
            {
                for (int j = 0; j < conv_size; j++)
                {
                    for (int kc = 0; kc < kernel_kc; kc++)
                    {
                        DE_3D_cipher[i * stride][j * stride][kc] = DE_cipher[(i * conv_size + j) * kernel_kc + kc];
                    }
                }
            }

            offline_timer.stop();
            offline_time += tmp_time;
            total_offline_time += tmp_time;

            MemoryManager::SwitchProfile(std::move(old_prof));
        }

        std::cout << "PREPARE DILATE OFFLINE TIME " << offline_time << std::endl;
    }

    void PegasusRunTime::delta_3D_to_3D_dilate(
        Parms parms,
        std::vector<std::vector<std::vector<Ctx>>> &DE_cipher,
        std::vector<std::vector<std::vector<Ctx>>> &DE_3D_cipher,
        int conv_size,
        int stride,
        int moduli)
    {
        int nslots = parms.nslots;
        double scale = parms.scale;

        tmp_time = 0.0;
        offline_time = 0.0;

        std::vector<F64> T(nslots);

        int de_size = DE_cipher.size();
        int de_dilate_size = DE_3D_cipher.size();
        int kernel_kc = DE_3D_cipher[0][0].size();

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            tmp_time = 0.0;
            AutoTimer offline_timer(&tmp_time);

#pragma omp parallel for collapse(3)
            for (int kc = 0; kc < kernel_kc; kc++)
            {
                for (int i = 0; i < conv_size; i++)
                {
                    for (int j = 0; j < conv_size; j++)
                    {
                        runtime_->DropModuli(&DE_cipher[i][j][kc], GetNModuli(DE_cipher[i][j][kc]) - moduli - 2);
                    }
                }
            }

#pragma omp parallel for collapse(3)
            for (int kc = 0; kc < kernel_kc; kc++)
            {
                for (int i = 0; i < de_dilate_size; i++)
                {
                    for (int j = 0; j < de_dilate_size; j++)
                    {
                        runtime_->EncryptZero(&DE_3D_cipher[i][j][kc]);
                        runtime_->DropModuli(&DE_3D_cipher[i][j][kc], GetNModuli(DE_3D_cipher[i][j][kc]) - moduli - 2);
                    }
                }
            }

#pragma omp parallel for collapse(3)
            for (int i = 0; i < conv_size; i++)
            {
                for (int j = 0; j < conv_size; j++)
                {
                    for (int kc = 0; kc < kernel_kc; kc++)
                    {
                        DE_3D_cipher[i * stride][j * stride][kc] = DE_cipher[i][j][kc];
                    }
                }
            }

            offline_timer.stop();
            offline_time += tmp_time;
            total_offline_time += tmp_time;

            MemoryManager::SwitchProfile(std::move(old_prof));
        }

        std::cout << "PREPARE DILATE OFFLINE TIME " << offline_time << std::endl;
    }

    void PegasusRunTime::update_model_conv_with_3D_delta(
        Parms parms,
        std::vector<std::vector<std::vector<Ctx>>> &A_cipher,
        std::vector<std::vector<std::vector<Ctx>>> &DE_cipher,
        std::vector<std::vector<std::vector<std::vector<F64>>>> &W,
        std::vector<F64> &B,
        int moduli,
        int layer,
        int epoch)
    {
        int nslots = parms.nslots;
        double scale = parms.scale;

        tmp_time = 0.0;
        save_model_time = 0.0;
        load_model_time = 0.0;
        offline_time = 0.0;
        online_time = 0.0;

        std::vector<F64> T(nslots);

        int a_size = A_cipher.size();
        int de_size = DE_cipher.size();
        int kernel_size = W[0][0].size();
        int kernel_ic = W[0].size();
        int kernel_kc = W.size();

        Ctx lr_cipher;
        std::vector<F64> lr(nslots, 0);
        lr[0] = 0.01 / nslots;
        CHECK_AND_ABORT(EncodeThenEncrypt(lr, lr_cipher));
        runtime_->DropModuli(&lr_cipher, GetNModuli(lr_cipher) - moduli - 1);

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            std::vector<std::vector<Ctx>> W_cipher(kernel_size, std::vector<Ctx>(kernel_size));
            std::vector<Ctx> B_cipher(kernel_kc);

            tmp_time = 0.0;
            AutoTimer offline_timer(&tmp_time);

#pragma omp parallel for collapse(3)
            for (int i = 0; i < A_cipher.size(); i++)
            {
                for (int j = 0; j < A_cipher[0].size(); j++)
                {
                    for (int k = 0; k < A_cipher[0][0].size(); k++)
                    {
                        runtime_->DropModuli(&A_cipher[i][j][k], GetNModuli(A_cipher[i][j][k]) - moduli - 2);
                    }
                }
            }

#pragma omp parallel for collapse(3)
            for (int i = 0; i < DE_cipher.size(); i++)
            {
                for (int j = 0; j < DE_cipher[0].size(); j++)
                {
                    for (int k = 0; k < DE_cipher[0][0].size(); k++)
                    {
                        runtime_->DropModuli(&DE_cipher[i][j][k], GetNModuli(DE_cipher[i][j][k]) - moduli - 2);
                    }
                }
            }

            offline_timer.stop();
            offline_time += tmp_time;
            total_offline_time += tmp_time;

            std::filebuf fwin;
            fwin.open(save_model_loc + model_name + "_W" + std::to_string(layer) + "_" + std::to_string(epoch) + ".out", std::ios::in);
            std::istream oswin(&fwin);

            std::filebuf fwout;
            fwout.open(save_model_loc + model_name + "_W" + std::to_string(layer) + "_" + std::to_string(epoch + 1) + ".out", std::ios::out);
            std::ostream oswout(&fwout);

            for (int kc = 0; kc < kernel_kc; kc++)
            {
                std::cout << kc << std::endl;
                for (int ic = 0; ic < kernel_ic; ic++)
                {
                    // Prepare model weight (W)
                    tmp_time = 0.0;
                    AutoTimer load_model_timer(&tmp_time);

                    if (not_first_epoch)
                    {
                        for (int i = 0; i < kernel_size; i++)
                        {
                            for (int j = 0; j < kernel_size; j++)
                            {
                                CHECK_AND_ABORT(runtime_->LoadCtx(&W_cipher[i][j], oswin));
                            }
                        }
                    }
                    else
                    {
#pragma omp parallel for collapse(2)
                        for (int i = 0; i < kernel_size; i++)
                        {
                            for (int j = 0; j < kernel_size; j++)
                            {
                                std::vector<F64> tmp(nslots);
                                for (int ii = 0; ii < nslots; ii++)
                                {
                                    tmp[ii] = W[kc][ic][i][j];
                                }
                                CHECK_AND_ABORT(EncodeThenEncrypt(tmp, W_cipher[i][j]));
                            }
                        }
                    }

#pragma omp parallel for collapse(2)
                    for (int i = 0; i < kernel_size; i++)
                    {
                        for (int j = 0; j < kernel_size; j++)
                        {
                            runtime_->DropModuli(&W_cipher[i][j], GetNModuli(W_cipher[i][j]) - moduli);
                        }
                    }

                    load_model_timer.stop();
                    load_model_time += tmp_time;
                    total_load_model_time += tmp_time;

                    // Update model weight (W)
                    tmp_time = 0.0;
                    AutoTimer online_timer(&tmp_time);

#pragma omp parallel for collapse(2)
                    for (int i = 0; i < kernel_size; i++)
                    {
                        for (int j = 0; j < kernel_size; j++)
                        {
                            Ctx tmp_cipher, tmp_cipher_2;
                            runtime_->EncryptZero(&tmp_cipher);
                            runtime_->DropModuli(&tmp_cipher, GetNModuli(tmp_cipher) - moduli - 1);
                            tmp_cipher.scale() = scale;

                            for (int ii = 0; ii < de_size; ii++)
                            {
                                for (int jj = 0; jj < de_size; jj++)
                                {
                                    tmp_cipher_2 = DE_cipher[ii][jj][kc];
                                    runtime_->MulRelinRescale(&tmp_cipher_2, A_cipher[i + ii][j + jj][ic]);
                                    runtime_->Add(&tmp_cipher, tmp_cipher_2);
                                }
                            }

                            // Rotate and Sum
                            for (int k = log2(nslots) - 1; k >= 0; k--)
                            {
                                tmp_cipher_2 = tmp_cipher;
                                runtime_->RotateLeft(&tmp_cipher_2, pow(2, k));
                                runtime_->Add(&tmp_cipher, tmp_cipher_2);
                            }

                            runtime_->MulRelinRescale(&tmp_cipher, lr_cipher);
                            tmp_cipher_2 = tmp_cipher;

                            for (int k = 0; k < log2(nslots); k++)
                            {
                                runtime_->RotateRight(&tmp_cipher_2, pow(2, k));
                                runtime_->Add(&tmp_cipher, tmp_cipher_2);
                                tmp_cipher_2 = tmp_cipher;
                            }

                            runtime_->Sub(&W_cipher[i][j], tmp_cipher);
                        }
                    }

                    online_timer.stop();
                    online_time += tmp_time;
                    total_online_time += tmp_time;

                    // Save model weight (W)
                    tmp_time = 0.0;
                    AutoTimer save_model_timer(&tmp_time);

                    for (int i = 0; i < kernel_size; i++)
                    {
                        for (int j = 0; j < kernel_size; j++)
                        {
                            // write w to file
                            runtime_->SaveCtx(W_cipher[i][j], oswout);

                            // update plaintext model
                            // used for debug
                            DecryptThenDecode(W_cipher[j][i], T);
                            W[kc][ic][i][j] = T[0];
                        }
                        std::cout << std::endl;
                    }

                    save_model_timer.stop();
                    save_model_time += tmp_time;
                    total_save_model_time += tmp_time;
                }
            }

            fwout.close();
            fwin.close();

            // Prepare model bias (b)
            tmp_time = 0.0;
            AutoTimer load_model_timer(&tmp_time);

            if (not_first_epoch)
            {
                std::filebuf fbin;
                fbin.open(save_model_loc + model_name + "_B" + std::to_string(layer) + "_" + std::to_string(epoch) + ".out", std::ios::in);
                std::istream osbin(&fbin);

                for (int i = 0; i < kernel_kc; i++)
                {
                    CHECK_AND_ABORT(runtime_->LoadCtx(&B_cipher[i], osbin));
                }

                fbin.close();
            }
            else
            {
#pragma omp parallel for
                for (int i = 0; i < kernel_kc; i++)
                {
                    std::vector<F64> tmp(nslots);
                    for (int k = 0; k < nslots; k++)
                    {
                        tmp[k] = B[i];
                    }
                    CHECK_AND_ABORT(EncodeThenEncrypt(tmp, B_cipher[i]));
                }
            }

            load_model_timer.stop();
            load_model_time += tmp_time;
            total_load_model_time += tmp_time;

            // Update model bias (b)
            tmp_time = 0.0;
            AutoTimer online_timer(&tmp_time);

#pragma omp parallel for
            for (int i = 0; i < kernel_kc; i++)
            {
                runtime_->DropModuli(&B_cipher[i], GetNModuli(B_cipher[i]) - moduli);
            }

#pragma omp parallel for collapse(3)
            for (int i = 0; i < de_size; i++)
            {
                for (int j = 0; j < de_size; j++)
                {
                    for (int k = 0; k < kernel_kc; k++)
                    {
                        runtime_->DropModuli(&DE_cipher[i][j][k], GetNModuli(DE_cipher[i][j][k]) - moduli - 1);
                    }
                }
            }

#pragma omp parallel for
            for (int k = 0; k < kernel_kc; k++)
            {
                Ctx tmp_cipher, tmp_cipher_2;
                runtime_->EncryptZero(&tmp_cipher);
                runtime_->DropModuli(&tmp_cipher, GetNModuli(tmp_cipher) - moduli - 1);
                tmp_cipher.scale() = scale;
                for (int i = 0; i < de_size; i++)
                {
                    for (int j = 0; j < de_size; j++)
                    {
                        runtime_->Add(&tmp_cipher, DE_cipher[i][j][k]);
                    }
                }

                // Rotate and Sum
                for (int k = log2(nslots) - 1; k >= 0; k--)
                {
                    tmp_cipher_2 = tmp_cipher;
                    runtime_->RotateLeft(&tmp_cipher_2, pow(2, k));
                    runtime_->Add(&tmp_cipher, tmp_cipher_2);
                }

                runtime_->MulRelinRescale(&tmp_cipher, lr_cipher);
                tmp_cipher_2 = tmp_cipher;

                for (int k = 0; k < log2(nslots); k++)
                {
                    runtime_->RotateRight(&tmp_cipher_2, pow(2, k));
                    runtime_->Add(&tmp_cipher, tmp_cipher_2);
                    tmp_cipher_2 = tmp_cipher;
                }
                runtime_->Sub(&B_cipher[k], tmp_cipher);
            }

            online_timer.stop();
            online_time += tmp_time;
            total_online_time += tmp_time;

            std::filebuf fbout;
            fbout.open(save_model_loc + model_name + "_B" + std::to_string(layer) + "_" + std::to_string(epoch + 1) + ".out", std::ios::out);
            std::ostream osbout(&fbout);

            // Save model bias (b)
            tmp_time = 0.0;
            AutoTimer save_model_timer(&tmp_time);

            for (int i = 0; i < kernel_kc; i++)
            {
                // write b to file
                runtime_->SaveCtx(B_cipher[i], osbout);

                // update plaintext model
                // used for debug
                DecryptThenDecode(B_cipher[i], T);
                B[i] = T[0];
            }

            save_model_timer.stop();
            save_model_time += tmp_time;
            total_save_model_time += tmp_time;

            fbout.close();

            std::cout << "UPDATE MODEL CONV LOAD MODEL TIME " << load_model_time << std::endl;
            std::cout << "UPDATE MODEL CONV SAVE MODEL TIME " << save_model_time << std::endl;
            std::cout << "UPDATE MODEL CONV OFFLINE TIME " << offline_time << std::endl;
            std::cout << "UPDATE MODEL CONV ONLINE TIME " << online_time << std::endl;

            release_vector_2D(W_cipher);
            release_vector_1D(B_cipher);
            W_cipher.clear();
            B_cipher.clear();

            MemoryManager::SwitchProfile(std::move(old_prof));
        }
    }

    void PegasusRunTime::s2c_and_extract(std::vector<Ctx> &U_cipher, std::vector<std::vector<lwe::Ctx_st>> &U_lwe_cipher, int length, int nslots)
    {
        std::cout << "S2C & EXTRACT\n";
        tmp_time = 0.0;
        online_time = 0.0;
        AutoTimer online_timer(&tmp_time);

        int rounds = std::ceil(1.0 * length / num_threads_);
        std::cout << "A total of " << rounds << " rounds" << std::endl;
        for (int r = 0; r < rounds; r++)
        {
            std::cout << "round " << r << std::endl;
#pragma omp parallel for
            for (int i = 0; i < num_threads_; i++)
            {
                int pos = r * num_threads_ + i;
                if (pos < length)
                {
                    CHECK_AND_ABORT(SlotsToCoeffs(U_cipher[pos]));

                    // use only if nslots = 2**(2n+1)
                    if (GetNModuli(U_cipher[pos]) != 1)
                    {
                        runtime_->DropModuli(&U_cipher[pos], GetNModuli(U_cipher[pos]) - 1);
                    }
                }
            }
        }

        for (int i = 0; i < length; i++)
        {
            CHECK_AND_ABORT(ExtraAllCoefficients(U_cipher[i], U_lwe_cipher[i]));
        }

        online_timer.stop();
        online_time += tmp_time;
        total_online_time += tmp_time;
        std::cout << "S2C & EXTRACT ONLINE TIME " << online_time << std::endl;
    }

    void PegasusRunTime::repack(std::vector<std::vector<lwe::Ctx_st>> &D_lwe_cipher, std::vector<Ctx> &D_cipher, int length, int nslots)
    {
#pragma omp parallel for
        for (int i = 0; i < length; i++)
        {
            CHECK_AND_ABORT(Repack(D_cipher[i], D_lwe_cipher[i]));
        }
    }

    void PegasusRunTime::repack_with_time(std::vector<std::vector<lwe::Ctx_st>> &D_lwe_cipher, std::vector<Ctx> &D_cipher, int length, int nslots)
    {
        std::cout << "REPACK \n";
        tmp_time = 0.0;
        online_time = 0.0;
        AutoTimer online_timer(&tmp_time);

#pragma omp parallel for
        for (int i = 0; i < length; i++)
        {
            CHECK_AND_ABORT(Repack(D_cipher[i], D_lwe_cipher[i]));
        }

        online_timer.stop();
        online_time += tmp_time;
        total_online_time += tmp_time;
        std::cout << "REPACK ONLINE TIME " << online_time << std::endl;
    }

    void PegasusRunTime::s2c_repack_1D(std::vector<Ctx> &U_cipher,
                                       std::vector<std::vector<lwe::Ctx_st>> &lwe_cipher,
                                       int nslots)
    {
        std::cout << "S2C Repack 1D\n";
        tmp_time = 0.0;
        online_time = 0.0;
        AutoTimer online_timer(&tmp_time);

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            s2c_and_extract(U_cipher, lwe_cipher, U_cipher.size(), nslots);
            repack(lwe_cipher, U_cipher, U_cipher.size(), nslots);

            MemoryManager::SwitchProfile(std::move(old_prof));
        }

        online_timer.stop();
        online_time += tmp_time;
        total_online_time += tmp_time;
        std::cout << "S2C REPACK 1D ONLINE TIME " << online_time << std::endl;
    }

    void PegasusRunTime::s2c_repack_3D(std::vector<std::vector<std::vector<Ctx>>> &U_cipher,
                                       std::vector<std::vector<lwe::Ctx_st>> &lwe_cipher,
                                       int nslots)
    {
        std::cout << "S2C Repack 3D\n";
        tmp_time = 0.0;
        online_time = 0.0;
        AutoTimer online_timer(&tmp_time);

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            for (int i = 0; i < U_cipher.size(); i++)
            {
                for (int j = 0; j < U_cipher[0].size(); j++)
                {
                    s2c_and_extract(U_cipher[i][j], lwe_cipher, U_cipher[0][0].size(), nslots);
                    repack(lwe_cipher, U_cipher[i][j], U_cipher[0][0].size(), nslots);
                }
            }

            MemoryManager::SwitchProfile(std::move(old_prof));
        }

        online_timer.stop();
        online_time += tmp_time;
        total_online_time += tmp_time;
        std::cout << "S2C REPACK 3D ONLINE TIME " << online_time << std::endl;
    }

    void PegasusRunTime::act(std::vector<std::vector<lwe::Ctx_st>> &U_lwe_cipher, int length, int nslots, std::string act_str)
    {
        for (int i = 0; i < length; i++)
        {
            if (act_str == "ReLU")
            {
                ReLU(U_lwe_cipher[i].data(), U_lwe_cipher[i].size());
            }
            else if (act_str == "DReLU")
            {
                DReLU(U_lwe_cipher[i].data(), U_lwe_cipher[i].size());
            }
            else if (act_str == "Sigmoid")
            {
                Sigmoid(U_lwe_cipher[i].data(), U_lwe_cipher[i].size());
            }
            else if (act_str == "DSigmoid")
            {
                DSigmoid(U_lwe_cipher[i].data(), U_lwe_cipher[i].size());
            }
        }
    }

    void PegasusRunTime::act_batch(std::vector<std::vector<lwe::Ctx_st>> &U_lwe_cipher,
                                   std::vector<std::vector<lwe::Ctx_st>> &D_lwe_cipher,
                                   std::vector<Ctx> &D_cipher,
                                   int length, int nslots, std::string act_str)
    {
        std::cout << "ACTIVATIONS & REPACK \n";
        tmp_time = 0.0;
        online_time = 0.0;
        AutoTimer online_timer(&tmp_time);

        int batch_size = num_threads_;
        int len = batch_size;
        int rounds = std::ceil(1.0 * length / batch_size);
        std::cout << "A total of " << rounds << " rounds" << std::endl;

        std::vector<Ctx> rlwe_cipher(batch_size);

        for (int r = 0; r < rounds; r++)
        {
            std::cout << "round " << r << std::endl;

#pragma omp parallel for
            for (int i = 0; i < batch_size; i++)
            {
                int pos = r * batch_size + i;
                if (pos < length)
                {
                    for (int j = 0; j < nslots; j++)
                    {
                        D_lwe_cipher[i][j] = U_lwe_cipher[pos][j];
                    }
                }
            }

            if (r == (rounds - 1) && (r + 1) * batch_size > length)
            {
                len = length - r * batch_size;
            }

            if (act_str == "ReLU")
            {
                act(D_lwe_cipher, len, nslots, "ReLU");
            }
            else if (act_str == "DReLU")
            {
                act(D_lwe_cipher, len, nslots, "DReLU");
            }
            else if (act_str == "Sigmoid")
            {
                act(D_lwe_cipher, len, nslots, "Sigmoid");
            }
            else if (act_str == "DSigmoid")
            {
                act(D_lwe_cipher, len, nslots, "DSigmoid");
            }

            repack(D_lwe_cipher, rlwe_cipher, len, nslots);

#pragma omp parallel for
            for (int i = 0; i < batch_size; i++)
            {
                int pos = r * batch_size + i;
                if (pos < length)
                {
                    D_cipher[pos] = rlwe_cipher[i];
                }
            }
        }

        release_vector_1D(rlwe_cipher);
        rlwe_cipher.clear();

        online_timer.stop();
        online_time += tmp_time;
        total_online_time += tmp_time;
        std::cout << "ACTIVATIONS & REPACK ONLINE TIME " << online_time << std::endl;
    }

    void PegasusRunTime::softmax(std::vector<std::vector<lwe::Ctx_st>> &U_lwe_cipher, std::vector<std::vector<lwe::Ctx_st>> &D_lwe_cipher, std::vector<Ctx> &D_cipher, int length, int nslots)
    {
        std::cout << "SOFTMAX\n";
        tmp_time = 0.0;
        online_time = 0.0;
        AutoTimer online_timer(&tmp_time);

        F64Vec log8(nslots);
        Ctx log8_ct;
        std::vector<lwe::Ctx_st> log8_lwe_ct;
        for (int i = 0; i < nslots; i++)
        {
            // log8[i] = 4.15888; // 1/64
            log8[i] = 3.46574; // 1/32
        }
        CHECK_AND_ABORT(EncodeThenEncrypt(log8, log8_ct));
        CHECK_AND_ABORT(SlotsToCoeffs(log8_ct));
        // use only if nslots = 2**7
        if (GetNModuli(log8_ct) != 1)
        {
            runtime_->DropModuli(&log8_ct, GetNModuli(log8_ct) - 1);
        }
        CHECK_AND_ABORT(ExtraAllCoefficients(log8_ct, log8_lwe_ct));

        std::vector<lwe::Ctx_st> SM_FLAT(length * nslots);
        std::vector<lwe::Ctx_st> SM_SUM(nslots);

        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < nslots; j++)
            {
                SM_FLAT[i * nslots + j] = U_lwe_cipher[i][j];
            }
        }

        // std::cout << "STEP 1: EXP\n";
        Exponent(SM_FLAT.data(), SM_FLAT.size());

        // std::cout << "STEP 2: Scale Exp\n";
#pragma omp parallel for
        for (int i = 0; i < length * nslots; i++)
        {
            std::vector<lwe::Ctx_st> lwe_ct(1);
            lwe_ct[0] = SM_FLAT[i];
            MulConstant(lwe_ct.data(), 0.03125); // 1/32
            SM_FLAT[i] = lwe_ct[0];
        }

        // std::cout << "STEP 3: EXP SUM\n";
#pragma omp parallel for
        for (int j = 0; j < nslots; j++)
        {
            lwe::Ctx_st tmp_lwe = SM_FLAT[j];
            for (int i = 1; i < length; i++)
            {
                AddLWECt(tmp_lwe, tmp_lwe, SM_FLAT[i * nslots + j]);
            }
            SM_SUM[j] = tmp_lwe;
        }

        // std::cout << "STEP 4: LOG\n";
        Log(SM_SUM.data(), SM_SUM.size());

        // Scale EXP_SUM
#pragma omp parallel for
        for (int i = 0; i < nslots; i++)
        {
            AddLWECt(SM_SUM[i], SM_SUM[i], log8_lwe_ct[i]);
        }

        // std::cout << "STEP 5: SUB\n";
#pragma omp parallel for collapse(2)
        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < nslots; j++)
            {
                SubLWECt(SM_FLAT[i * nslots + j], U_lwe_cipher[i][j], SM_SUM[j]);
            }
        }

        // std::cout << "STEP 6: EXP\n";
        Exponent(SM_FLAT.data(), SM_FLAT.size());

#pragma omp parallel for collapse(2)
        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < nslots; j++)
            {
                D_lwe_cipher[i][j] = SM_FLAT[i * nslots + j];
            }
        }

        repack(D_lwe_cipher, D_cipher, length, nslots);

        online_timer.stop();
        online_time += tmp_time;
        total_online_time += tmp_time;

        std::cout << "SOFTMAX & REPACK ONLINE TIME " << online_time << std::endl;
    }

    void PegasusRunTime::relu_lwe(std::vector<std::vector<lwe::Ctx_st>> &U_lwe_cipher, std::vector<std::vector<lwe::Ctx_st>> &D_lwe_cipher, int length, int nslots)
    {
        std::cout << "RELU RETURN LWEs\n";
        tmp_time = 0.0;
        online_time = 0.0;
        AutoTimer online_timer(&tmp_time);

        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < nslots; j++)
            {
                D_lwe_cipher[i][j] = U_lwe_cipher[i][j];
            }

            ReLU(D_lwe_cipher[i].data(), D_lwe_cipher[i].size());
        }
        std::cout << std::endl;

        online_timer.stop();
        online_time += tmp_time;
        total_online_time += tmp_time;
        std::cout << "ReLU RETURN LWEs ONLINE TIME " << online_time << std::endl;
    }

    void PegasusRunTime::drelu_lwe(std::vector<std::vector<lwe::Ctx_st>> &U_lwe_cipher, int length, int nslots)
    {
        std::cout << "DRELU RETURN LWEs\n";
        tmp_time = 0.0;
        online_time = 0.0;
        AutoTimer online_timer(&tmp_time);

        for (int i = 0; i < length; i++)
        {
            DReLU(U_lwe_cipher[i].data(), U_lwe_cipher[i].size());
        }
        std::cout << std::endl;

        online_timer.stop();
        online_time += tmp_time;
        total_online_time += tmp_time;
        std::cout << "DReLU RETURN LWEs ONLINE TIME " << online_time << std::endl;
    }

    void PegasusRunTime::Relu_3D(std::vector<std::vector<std::vector<Ctx>>> U_cipher,
                                 std::vector<std::vector<std::vector<Ctx>>> D_cipher,
                                 std::vector<std::vector<lwe::Ctx_st>> &U_lwe_cipher,
                                 int nslots)
    {
        std::cout << "RELU 3D\n";
        tmp_time = 0.0;
        online_time = 0.0;
        AutoTimer online_timer(&tmp_time);

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            int length = U_cipher.size() * U_cipher[0].size();

            std::vector<Ctx> tmp_cipher(length);

            for (int kc = 0; kc < U_cipher[0][0].size(); kc++)
            {
                for (int i = 0; i < U_cipher.size(); i++)
                {
                    for (int j = 0; i < U_cipher[0].size(); j++)
                    {
                        tmp_cipher[i * U_cipher.size() + j] = U_cipher[i][j][kc];
                    }
                }
                s2c_and_extract(tmp_cipher, U_lwe_cipher, length, nslots);

                for (int i = 0; i < length; i++)
                {
                    ReLU(U_lwe_cipher[i].data(), U_lwe_cipher[i].size());
                }

                repack(U_lwe_cipher, tmp_cipher, length, nslots);

                for (int i = 0; i < U_cipher.size(); i++)
                {
                    for (int j = 0; i < U_cipher[0].size(); j++)
                    {
                        U_cipher[i][j][kc] = tmp_cipher[i * U_cipher.size() + j];
                    }
                }
            }

            release_vector_1D(tmp_cipher);
            tmp_cipher.clear();
            MemoryManager::SwitchProfile(std::move(old_prof));
        }

        online_timer.stop();
        online_time += tmp_time;
        total_online_time += tmp_time;

        std::cout << "ReLU 3D ONLINE TIME " << online_time << std::endl;
    }

    void PegasusRunTime::Drelu_3D(std::vector<std::vector<std::vector<Ctx>>> U_cipher,
                                  std::vector<std::vector<lwe::Ctx_st>> &U_lwe_cipher,
                                  int nslots)
    {
        std::cout << "DRELU 3D\n";
        tmp_time = 0.0;
        online_time = 0.0;
        AutoTimer online_timer(&tmp_time);

        {
            MemoryPoolHandle my_pool = MemoryPoolHandle::New();
            auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

            int length = U_cipher.size() * U_cipher[0].size();

            std::vector<Ctx> tmp_cipher(length);

            for (int kc = 0; kc < U_cipher[0][0].size(); kc++)
            {
                for (int i = 0; i < U_cipher.size(); i++)
                {
                    for (int j = 0; i < U_cipher[0].size(); j++)
                    {
                        tmp_cipher[i * U_cipher.size() + j] = U_cipher[i][j][kc];
                    }
                }
                s2c_and_extract(tmp_cipher, U_lwe_cipher, length, nslots);

                for (int i = 0; i < length; i++)
                {
                    DReLU(U_lwe_cipher[i].data(), U_lwe_cipher[i].size());
                }

                repack(U_lwe_cipher, tmp_cipher, length, nslots);

                for (int i = 0; i < U_cipher.size(); i++)
                {
                    for (int j = 0; i < U_cipher[0].size(); j++)
                    {
                        U_cipher[i][j][kc] = tmp_cipher[i * U_cipher.size() + j];
                    }
                }
            }

            release_vector_1D(tmp_cipher);
            tmp_cipher.clear();

            MemoryManager::SwitchProfile(std::move(old_prof));
        }

        online_timer.stop();
        online_time += tmp_time;
        total_online_time += tmp_time;

        std::cout << "DReLU ONLINE TIME " << online_time << std::endl;
    }

} // namespace gemini
