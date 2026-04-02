// Isolated Softmax Test for HE-SecureNet (ground truth)
//
// Tests the full softmax pipeline step-by-step with LWE decryption
// at every intermediate step to establish ground truth accuracy:
//
//   1. Encode+Encrypt → CKKS → S2C → Extract → LWE
//   2. PBS Exponent on all LWE
//   3. PBS MulConstant(0.03125) on all LWE
//   4. LWE Sum across classes
//   5. PBS Log on sum
//   6. LWE Add log8
//   7. LWE Sub (original - sum)
//   8. PBS Exponent on result
//   9. Repack to CKKS
//
// Reports avg/max error at each step.

#include "pegasus/pegasus_runtime.h"
#include "pegasus/timer.h"
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <vector>

using namespace gemini;

struct ErrorStats
{
    double avg_err = 0.0, max_err = 0.0;
    int count = 0;
    void compute(const double *expected, const double *actual, int n)
    {
        count = n;
        double sum = 0;
        max_err = 0;
        for (int i = 0; i < n; i++)
        {
            double e = std::abs(expected[i] - actual[i]);
            sum += e;
            if (e > max_err)
                max_err = e;
        }
        avg_err = sum / n;
    }
    void print(const char *label) const
    {
        printf("  [%-30s] n=%d avg=%.6e (2^%.1f) max=%.6e (2^%.1f)\n",
               label, count, avg_err, std::log2(avg_err + 1e-30),
               max_err, std::log2(max_err + 1e-30));
    }
};

int main()
{
    setbuf(stdout, NULL); // Disable buffering
    printf("================================================================\n");
    printf("  HE-SecureNet Softmax Ground Truth Test\n");
    printf("================================================================\n\n");

    // Same params as model_b
    const int NSLOTS = 256;
    const int LENGTH = 2;        // 2 classes, minimum for softmax
    const double LOG8 = 3.46574; // ln(32)
    const int SHOW = 8;

    PegasusRunTime::Parms pp;
    pp.lvl0_lattice_dim = lwe::params::n(); // 1024
    pp.lvl1_lattice_dim = 1 << 12;          // 4096
    pp.lvl2_lattice_dim = 1 << 16;          // 65536
    pp.nlevels = 4;
    pp.scale = std::pow(2., 40);
    pp.nslots = NSLOTS;
    pp.enable_repacking = true;

    PegasusRunTime pg_rt(pp, /*num_threads*/ 4);

    printf("Parameters:\n");
    printf("  n_lwe=%d, N_boot=%d, N_ckks=%d\n",
           pp.lvl0_lattice_dim, pp.lvl1_lattice_dim, pp.lvl2_lattice_dim);
    printf("  scale=2^%.0f, nslots=%d\n", std::log2(pp.scale), pp.nslots);
    printf("  MsgRange=%.2f, ExtraScaling=%.6f\n",
           pg_rt.MsgRange(), pg_rt.ExtraScaling());
    printf("\n");

    // Generate test values (same seed as our GPU test, range [-0.5, 0.5])
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    std::vector<std::vector<double>> input_vals(LENGTH, std::vector<double>(NSLOTS));
    for (int i = 0; i < LENGTH; i++)
        for (int k = 0; k < NSLOTS; k++)
            input_vals[i][k] = dist(rng);

    // Compute reference softmax
    std::vector<std::vector<double>> ref_softmax(LENGTH, std::vector<double>(NSLOTS));
    std::vector<double> ref_exp(LENGTH * NSLOTS);
    std::vector<double> ref_exp_scaled(LENGTH * NSLOTS);
    std::vector<double> ref_sum(NSLOTS);
    std::vector<double> ref_log_sum(NSLOTS);
    std::vector<double> ref_log_sum_plus_log8(NSLOTS);
    std::vector<double> ref_sub(LENGTH * NSLOTS);

    for (int j = 0; j < NSLOTS; j++)
    {
        double sum = 0;
        for (int i = 0; i < LENGTH; i++)
        {
            double ex = std::exp(input_vals[i][j]);
            ref_exp[i * NSLOTS + j] = ex;
            ref_exp_scaled[i * NSLOTS + j] = ex * 0.03125;
            sum += ex * 0.03125;
        }
        ref_sum[j] = sum;
        ref_log_sum[j] = std::log(sum);
        ref_log_sum_plus_log8[j] = std::log(sum) + LOG8;
        for (int i = 0; i < LENGTH; i++)
        {
            ref_sub[i * NSLOTS + j] = input_vals[i][j] - ref_log_sum_plus_log8[j];
            ref_softmax[i][j] = std::exp(ref_sub[i * NSLOTS + j]);
        }
    }

    printf("Reference softmax (first %d slots, class 0):\n  ", SHOW);
    for (int k = 0; k < SHOW; k++)
        printf("%.4f ", ref_softmax[0][k]);
    printf("\nReference softmax (first %d slots, class 1):\n  ", SHOW);
    for (int k = 0; k < SHOW; k++)
        printf("%.4f ", ref_softmax[1][k]);
    printf("\n\n");

    // ================================================================
    // Step 0: Encode+Encrypt+S2C+Extract
    // ================================================================
    printf("=== Step 0: Encode + Encrypt + S2C + Extract ===\n");

    // Encrypt log8
    F64Vec log8_vec(NSLOTS, LOG8);
    Ctx log8_ct;
    CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(log8_vec, log8_ct));
    CHECK_AND_ABORT(pg_rt.SlotsToCoeffs(log8_ct));
    if (GetNModuli(log8_ct) != 1)
        pg_rt.runtime_->DropModuli(&log8_ct, GetNModuli(log8_ct) - 1);
    std::vector<lwe::Ctx_st> log8_lwe;
    CHECK_AND_ABORT(pg_rt.ExtraAllCoefficients(log8_ct, log8_lwe));

    // Encrypt inputs
    std::vector<std::vector<lwe::Ctx_st>> U_lwe(LENGTH);
    for (int i = 0; i < LENGTH; i++)
    {
        F64Vec vals(input_vals[i].begin(), input_vals[i].end());
        Ctx ct;
        CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(vals, ct));
        CHECK_AND_ABORT(pg_rt.SlotsToCoeffs(ct));
        if (GetNModuli(ct) != 1)
            pg_rt.runtime_->DropModuli(&ct, GetNModuli(ct) - 1);
        CHECK_AND_ABORT(pg_rt.ExtraAllCoefficients(ct, U_lwe[i]));
    }

    // Decrypt LWE to verify S2C quality
    printf("  LWE decryption after S2C (first %d slots, class 0):\n    ", SHOW);
    {
        ErrorStats err;
        std::vector<double> dec(NSLOTS), exp(NSLOTS);
        for (int j = 0; j < NSLOTS; j++)
        {
            dec[j] = pg_rt.DecryptLWE(U_lwe[0][j]);
            exp[j] = input_vals[0][j];
        }
        for (int k = 0; k < SHOW; k++)
            printf("%.4f(%.4f) ", dec[k], exp[k]);
        printf("\n");
        err.compute(exp.data(), dec.data(), NSLOTS);
        err.print("S2C+Extract class 0");
    }
    {
        ErrorStats err;
        std::vector<double> dec(NSLOTS), exp(NSLOTS);
        for (int j = 0; j < NSLOTS; j++)
        {
            dec[j] = pg_rt.DecryptLWE(U_lwe[1][j]);
            exp[j] = input_vals[1][j];
        }
        err.compute(exp.data(), dec.data(), NSLOTS);
        err.print("S2C+Extract class 1");
    }

    // ================================================================
    // Flatten (matching softmax code exactly)
    // ================================================================
    std::vector<lwe::Ctx_st> SM_FLAT(LENGTH * NSLOTS);
    for (int i = 0; i < LENGTH; i++)
        for (int j = 0; j < NSLOTS; j++)
            SM_FLAT[i * NSLOTS + j] = U_lwe[i][j];

    // Save originals for step 6
    std::vector<lwe::Ctx_st> U_lwe_orig = SM_FLAT;

    // ================================================================
    // Step 1: Exponent PBS
    // ================================================================
    printf("\n=== Step 1: Exponent PBS ===\n");
    pg_rt.Exponent(SM_FLAT.data(), SM_FLAT.size());

    {
        printf("  After Exp (first %d, class 0):\n    ", SHOW);
        ErrorStats err0, err1;
        std::vector<double> dec0(NSLOTS), dec1(NSLOTS);
        for (int j = 0; j < NSLOTS; j++)
        {
            dec0[j] = pg_rt.DecryptLWE(SM_FLAT[j]);
            dec1[j] = pg_rt.DecryptLWE(SM_FLAT[NSLOTS + j]);
        }
        for (int k = 0; k < SHOW; k++)
            printf("%.4f(%.4f) ", dec0[k], ref_exp[k]);
        printf("\n");
        err0.compute(ref_exp.data(), dec0.data(), NSLOTS);
        err1.compute(ref_exp.data() + NSLOTS, dec1.data(), NSLOTS);
        err0.print("Exponent class 0");
        err1.print("Exponent class 1");
    }

    // ================================================================
    // Step 2: MulConstant(0.03125) PBS
    // ================================================================
    printf("\n=== Step 2: MulConstant(0.03125) PBS ===\n");
    for (int i = 0; i < LENGTH * NSLOTS; i++)
    {
        std::vector<lwe::Ctx_st> tmp(1);
        tmp[0] = SM_FLAT[i];
        pg_rt.MulConstant(tmp.data(), 0.03125);
        SM_FLAT[i] = tmp[0];
    }

    {
        printf("  After MulConst (first %d, class 0):\n    ", SHOW);
        ErrorStats err0, err1;
        std::vector<double> dec0(NSLOTS), dec1(NSLOTS);
        for (int j = 0; j < NSLOTS; j++)
        {
            dec0[j] = pg_rt.DecryptLWE(SM_FLAT[j]);
            dec1[j] = pg_rt.DecryptLWE(SM_FLAT[NSLOTS + j]);
        }
        for (int k = 0; k < SHOW; k++)
            printf("%.6f(%.6f) ", dec0[k], ref_exp_scaled[k]);
        printf("\n");
        err0.compute(ref_exp_scaled.data(), dec0.data(), NSLOTS);
        err1.compute(ref_exp_scaled.data() + NSLOTS, dec1.data(), NSLOTS);
        err0.print("MulConst class 0");
        err1.print("MulConst class 1");
    }

    // ================================================================
    // Step 3: Sum across classes
    // ================================================================
    printf("\n=== Step 3: LWE Sum ===\n");
    std::vector<lwe::Ctx_st> SM_SUM(NSLOTS);
    for (int j = 0; j < NSLOTS; j++)
    {
        SM_SUM[j] = SM_FLAT[j];
        for (int i = 1; i < LENGTH; i++)
        {
            lwe::Ctx_st tmp;
            pg_rt.AddLWECt(tmp, SM_SUM[j], SM_FLAT[i * NSLOTS + j]);
            SM_SUM[j] = tmp;
        }
    }

    {
        printf("  Sum (first %d):\n    ", SHOW);
        ErrorStats err;
        std::vector<double> dec(NSLOTS);
        for (int j = 0; j < NSLOTS; j++)
            dec[j] = pg_rt.DecryptLWE(SM_SUM[j]);
        for (int k = 0; k < SHOW; k++)
            printf("%.6f(%.6f) ", dec[k], ref_sum[k]);
        printf("\n");
        err.compute(ref_sum.data(), dec.data(), NSLOTS);
        err.print("Sum");
    }

    // ================================================================
    // Step 4: Log PBS
    // ================================================================
    printf("\n=== Step 4: Log PBS ===\n");
    pg_rt.Log(SM_SUM.data(), SM_SUM.size());

    {
        printf("  Log(sum) (first %d):\n    ", SHOW);
        ErrorStats err;
        std::vector<double> dec(NSLOTS);
        for (int j = 0; j < NSLOTS; j++)
            dec[j] = pg_rt.DecryptLWE(SM_SUM[j]);
        for (int k = 0; k < SHOW; k++)
            printf("%.4f(%.4f) ", dec[k], ref_log_sum[k]);
        printf("\n");
        err.compute(ref_log_sum.data(), dec.data(), NSLOTS);
        err.print("Log(sum)");
    }

    // ================================================================
    // Step 5: Add log8
    // ================================================================
    printf("\n=== Step 5: Add log8 ===\n");
    for (int j = 0; j < NSLOTS; j++)
    {
        lwe::Ctx_st tmp;
        pg_rt.AddLWECt(tmp, SM_SUM[j], log8_lwe[j]);
        SM_SUM[j] = tmp;
    }

    {
        printf("  Log(sum)+log8 (first %d):\n    ", SHOW);
        ErrorStats err;
        std::vector<double> dec(NSLOTS);
        for (int j = 0; j < NSLOTS; j++)
            dec[j] = pg_rt.DecryptLWE(SM_SUM[j]);
        for (int k = 0; k < SHOW; k++)
            printf("%.4f(%.4f) ", dec[k], ref_log_sum_plus_log8[k]);
        printf("\n");
        err.compute(ref_log_sum_plus_log8.data(), dec.data(), NSLOTS);
        err.print("Log(sum)+log8");
    }

    // ================================================================
    // Step 6: Subtract — SM_FLAT[i][j] = U_orig[i][j] - SM_SUM[j]
    // ================================================================
    printf("\n=== Step 6: Subtract ===\n");
    for (int i = 0; i < LENGTH; i++)
        for (int j = 0; j < NSLOTS; j++)
            pg_rt.SubLWECt(SM_FLAT[i * NSLOTS + j],
                           U_lwe_orig[i * NSLOTS + j], SM_SUM[j]);

    {
        printf("  Sub (first %d, class 0):\n    ", SHOW);
        ErrorStats err0, err1;
        std::vector<double> dec0(NSLOTS), dec1(NSLOTS);
        for (int j = 0; j < NSLOTS; j++)
        {
            dec0[j] = pg_rt.DecryptLWE(SM_FLAT[j]);
            dec1[j] = pg_rt.DecryptLWE(SM_FLAT[NSLOTS + j]);
        }
        for (int k = 0; k < SHOW; k++)
            printf("%.4f(%.4f) ", dec0[k], ref_sub[k]);
        printf("\n");
        err0.compute(ref_sub.data(), dec0.data(), NSLOTS);
        err1.compute(ref_sub.data() + NSLOTS, dec1.data(), NSLOTS);
        err0.print("Sub class 0");
        err1.print("Sub class 1");
    }

    // ================================================================
    // Step 7: Final Exponent PBS
    // ================================================================
    printf("\n=== Step 7: Exponent PBS (final) ===\n");
    pg_rt.Exponent(SM_FLAT.data(), SM_FLAT.size());

    {
        printf("  Final softmax (first %d, class 0):\n    ", SHOW);
        ErrorStats err0, err1;
        std::vector<double> dec0(NSLOTS), dec1(NSLOTS);
        for (int j = 0; j < NSLOTS; j++)
        {
            dec0[j] = pg_rt.DecryptLWE(SM_FLAT[j]);
            dec1[j] = pg_rt.DecryptLWE(SM_FLAT[NSLOTS + j]);
        }
        for (int k = 0; k < SHOW; k++)
            printf("%.4f(%.4f) ", dec0[k], ref_softmax[0][k]);
        printf("\n");
        err0.compute(ref_softmax[0].data(), dec0.data(), NSLOTS);
        err1.compute(ref_softmax[1].data(), dec1.data(), NSLOTS);
        err0.print("Softmax class 0 (LWE)");
        err1.print("Softmax class 1 (LWE)");
    }

    // ================================================================
    // Step 8: Repack to CKKS
    // ================================================================
    printf("\n=== Step 8: Repack ===\n");
    std::vector<std::vector<lwe::Ctx_st>> D_lwe(LENGTH, std::vector<lwe::Ctx_st>(NSLOTS));
    for (int i = 0; i < LENGTH; i++)
        for (int j = 0; j < NSLOTS; j++)
            D_lwe[i][j] = SM_FLAT[i * NSLOTS + j];

    std::vector<Ctx> D_cipher(LENGTH);
    pg_rt.repack(D_lwe, D_cipher, LENGTH, NSLOTS);

    {
        for (int i = 0; i < LENGTH; i++)
        {
            F64Vec dec_vals;
            CHECK_AND_ABORT(pg_rt.DecryptThenDecode(D_cipher[i], dec_vals));

            ErrorStats err;
            err.compute(ref_softmax[i].data(), dec_vals.data(), NSLOTS);
            char label[64];
            snprintf(label, sizeof(label), "Softmax class %d (CKKS)", i);
            err.print(label);

            printf("    first %d: ", SHOW);
            for (int k = 0; k < SHOW && k < (int)dec_vals.size(); k++)
                printf("%.4f(%.4f) ", dec_vals[k], ref_softmax[i][k]);
            printf("\n");
        }
    }

    printf("\n================================================================\n");
    printf("  HE-SecureNet Softmax Ground Truth Test Complete\n");
    printf("================================================================\n");
    return 0;
}
