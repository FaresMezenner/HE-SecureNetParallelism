#include "pegasus/repack.h"

#include <Eigen/Dense>
#include <cmath>
namespace gemini
{

  // Debug helper: decrypt & decode a ciphertext, print first few slots
  static void debug_decrypt(const Ctx &ct, const std::shared_ptr<RunTime> runtime,
                            const char *label, size_t nprint = 8, size_t nslots = 1024)
  {
    Ptx ptx;
    auto st = runtime->Decrypt(ct, &ptx);
    if (!st.IsOk())
    {
      printf("  [DBG] %s: decrypt FAILED\n", label);
      return;
    }
    F64Vec decoded;
    runtime->Decode(ptx, nslots, &decoded);
    printf("  [DBG] %s (scale=%.3e, size=%zu, parms=%zu): first %zu slots:\n",
           label, ct.scale(), ct.size(), ct.coeff_modulus_size(), nprint);
    for (size_t i = 0; i < std::min(nprint, decoded.size()); i++)
    {
      printf("    slot[%zu] = %+.8e\n", i, decoded[i]);
    }
    // Also print a few "interesting" slots
    size_t interesting[] = {42, 43, 44, 512, 555, 1023};
    for (size_t idx : interesting)
    {
      if (idx < decoded.size())
      {
        printf("    slot[%zu] = %+.8e\n", idx, decoded[idx]);
      }
    }
  }
  Status RpKeyInit(RpKey &rpk, const F64 scale, const seal::SecretKey &rlwe_sk,
                   const seal::SecretKey &rgsw_sk_non_ntt,
                   const std::shared_ptr<RunTime> runtime)
  {
    auto rlwe_rt = runtime->SEALRunTime();
    const size_t nslots = runtime->MaximumNSlots();
    const size_t NN = rgsw_sk_non_ntt.data().coeff_count();
    if (nslots < NN)
    {
      throw std::invalid_argument("RpKeyInit: require N >= 2*N'");
    }
    auto cast_to_double = [](U64 u) -> F64
    {
      return u > 1 ? -1. : static_cast<F64>(u);
    };

    if (!seal::is_metadata_valid_for(rlwe_sk, rlwe_rt))
    {
      throw std::invalid_argument("RpKeyInit: invalid rlwe_sk");
    }

    F64Vec slots(NN, 0.);
    for (size_t i = 0; i < NN; ++i)
    {
      slots[i] = cast_to_double(rgsw_sk_non_ntt.data()[i]);
    }

    Ptx ptx;
    ptx.parms_id() = rlwe_rt->first_parms_id();
    runtime->Encode(slots, scale, &ptx);
    runtime->Encrypt(ptx, &rpk.key());
    rpk.scale() = scale;

    const size_t g = CeilSqrt(NN);
    rpk.rotated_keys_.resize(g - 1);
    for (size_t j = 1; j < g; ++j)
    {
      std::rotate(slots.begin(), slots.begin() + 1, slots.end());
      CHECK_STATUS(runtime->Encode(slots, scale, &ptx));
      CHECK_STATUS(runtime->Encrypt(ptx, &rpk.rotated_keys_.at(j - 1)));
    }

    auto Montgomerize = [&ptx, runtime](Ctx &ctx)
    {
      if (ptx.coeff_count() !=
          ctx.poly_modulus_degree() * ctx.coeff_modulus_size())
      {
        throw std::invalid_argument("ptx length mismatch");
      }

      for (size_t i = 0; i < ctx.size(); ++i)
      {
        std::copy_n(ctx.data(i), ptx.coeff_count(), ptx.data());
        runtime->Montgomerize(&ptx);
        std::copy_n(ptx.data(), ptx.coeff_count(), ctx.data(i));
      }
    };

    Montgomerize(rpk.key_);
    for (auto &ctx : rpk.rotated_keys_)
    {
      Montgomerize(ctx);
    }

    seal::util::seal_memzero(
        slots.data(),
        slots.size() * sizeof(slots[0])); // clean up secret material
    return Status::Ok();
  }

  /// Wrap an LWE array to a matrix like object

  LWECtArrayWrapper::LWECtArrayWrapper(const std::vector<lwe::Ctx_st> &array,
                                       uint64_t p0, double multiplier)
      : lwe_n_ct_array_(array),
        p0_(p0),
        p0half_(p0 >> 1),
        multiplier_(multiplier)
  {
    if (lwe_n_ct_array_.empty())
    {
      throw std::invalid_argument("LWECtArrayWrapper: Empty array");
    }
  }

  void LWECtArrayWrapper::GetLastColumn(F64Vec &column) const
  {
    const size_t ncols = cols();
    const size_t nrows = rows();
    column.resize(ncols);
    for (size_t i = 0; i < nrows; ++i)
    {
      column[i] = Get(i, ncols);
    }

    for (size_t i = nrows; i < ncols; i += nrows)
    {
      for (size_t k = 0; k < nrows; ++k)
        column[i + k] = column[k];
    }
  }

  /**
   * Input
   *     [a0, b0, c0, ... | a1, b1, c1, ... | .... ]
   *      |<-- stride -->
   *      |<---------------  nslots -------------->
   *
   * Output
   *     [\sum ai, \sum bi, \sum ci, ...]
   *
   * Requirement
   *     nslots/stride is a 2-exponent value.
   */
  Status SumStridedVectors(Ctx &enc_vec, size_t stride, size_t nslots,
                           const std::shared_ptr<RunTime> runtime)
  {
    const size_t vecLength = nslots / stride;
    if (vecLength == 0 || !IsTwoPower(vecLength))
    {
      return Status::ArgumentError("SumStridedVectors: Invalid stride");
    }

    const size_t nsteps = static_cast<size_t>(Log2(vecLength));

    for (size_t i = 0; i < nsteps; ++i)
    {
      auto copy{enc_vec};
      CHECK_STATUS_INTERNAL(runtime->RotateLeft(&copy, (1U << i) * stride),
                            "SumStridedVectors: RotateLeft failed");
      CHECK_STATUS_INTERNAL(runtime->Add(&enc_vec, copy),
                            "SumStridedVectors: Add failed");
    }
    return Status::Ok();
  }

  Status Repack(Ctx &out, double bound,
                const std::vector<lwe::Ctx_st> &lwe_n_ct_array, const RpKey &rpk,
                const std::shared_ptr<RunTime> runtime)
  {
    auto seal_rt = runtime->SEALRunTime();
    if (lwe_n_ct_array.empty())
    {
      Status::ArgumentError("Repack: empty LWE cipher array");
    }

    const uint64_t p0 = runtime->GetModulusPrime(0);
    LWECtArrayWrapper lweMatrix(lwe_n_ct_array, p0, 1. / bound);

    const size_t Nhalf = runtime->MaximumNSlots();
    if (lweMatrix.cols() > Nhalf)
    {
      return Status::ArgumentError("Repack: too many LWE ciphers to pack");
    }

    Ptx ptx;
    ptx.parms_id() = rpk.key().parms_id();

    const size_t nrows = lweMatrix.rows();
    const size_t ncols = lweMatrix.cols();
    const size_t min_n = std::min(nrows, ncols);
    const size_t max_n = std::max(nrows, ncols);

    const size_t g = CeilSqrt(min_n);
    const size_t h = CeilDiv(min_n, g);

    printf("\n===== HE-SecureNet REPACK DEBUG =====\n");
    printf("  nrows=%zu, ncols=%zu, min_n=%zu, max_n=%zu, g=%zu, h=%zu\n",
           nrows, ncols, min_n, max_n, g, h);
    printf("  p0=0x%lX, bound=%.1f, rpk_scale=%.1f\n", p0, bound, rpk.scale());
    printf("  rpk.key parms_id hash=%zu, coeff_mod_size=%zu\n",
           std::hash<seal::parms_id_type>()(rpk.key().parms_id()),
           rpk.key().coeff_modulus_size());

    // Print first few diagonals' first values
    printf("  First 3 LWE matrix entries: M(0,0)=%.8e, M(0,1)=%.8e, M(1,0)=%.8e\n",
           (double)lweMatrix(0, 0), (double)lweMatrix(0, 1), (double)lweMatrix(1, 0));

    // Decrypt rpk.key(0) to see the secret key encoding
    debug_decrypt(rpk.key(), runtime, "rpk.key(0)", 8, max_n);

    std::vector<double> diag(lweMatrix.cols(), 0.);

    // Baby-Steps-Giant-Steps
    for (size_t k = 0; k < h && g * k < min_n; ++k)
    {
      Ctx inner;
      for (size_t j = 0, diag_idx = g * k; j < g && diag_idx < min_n;
           ++j, ++diag_idx)
      {
        // Obtain the diagonal from LWE matrix
        CHECK_STATUS_INTERNAL(
            GetTilingDiagonal(diag, diag_idx, lweMatrix, runtime),
            "MultiplyPlainMatrixCipherVector");

        // Print diag values for first few k,j
        if (k < 2 || (k == 22))
        {
          if (j < 2 || diag_idx == min_n - 1)
          {
            printf("  diag[%zu] (k=%zu,j=%zu) BEFORE rotate: [0]=%.8e [1]=%.8e [43]=%.8e [512]=%.8e [555]=%.8e\n",
                   diag_idx, k, j, diag[0], diag[1],
                   diag.size() > 43 ? diag[43] : 0.0,
                   diag.size() > 512 ? diag[512] : 0.0,
                   diag.size() > 555 ? diag[555] : 0.0);
          }
        }

        // RHS rotated by g * k
        std::rotate(diag.rbegin(), diag.rbegin() + g * k, diag.rend());

        if (k < 2 || (k == 22))
        {
          if (j < 2 || diag_idx == min_n - 1)
          {
            printf("  diag[%zu] (k=%zu,j=%zu) AFTER rotate by %zu: [0]=%.8e [1]=%.8e [43]=%.8e [512]=%.8e [555]=%.8e\n",
                   diag_idx, k, j, g * k, diag[0], diag[1],
                   diag.size() > 43 ? diag[43] : 0.0,
                   diag.size() > 512 ? diag[512] : 0.0,
                   diag.size() > 555 ? diag[555] : 0.0);
          }
        }

        CHECK_STATUS_INTERNAL(runtime->Encode(diag, 1., &ptx), "Encode");

        if (j > 0)
        {
          CHECK_STATUS_INTERNAL(
              runtime->FMAMontgomery(&inner, rpk.rotated_key(j), ptx), "FMA");
        }
        else
        {
          inner = rpk.rotated_key(0);
          CHECK_STATUS_INTERNAL(runtime->MulPlainMontgomery(&inner, ptx),
                                "MulPlain");
        }
      }

      // Debug: decrypt inner BEFORE giant-step rotation
      if (k < 3 || k == 22)
      {
        char buf[128];
        snprintf(buf, sizeof(buf), "inner[k=%zu] BEFORE rotate", k);
        debug_decrypt(inner, runtime, buf, 4, max_n);
      }

      if (k > 0)
      {
        CHECK_STATUS_INTERNAL(runtime->RotateLeft(&inner, g * k), "RotateLeft");

        // Debug: decrypt inner AFTER rotation
        if (k < 3 || k == 22)
        {
          char buf[128];
          snprintf(buf, sizeof(buf), "inner[k=%zu] AFTER rotate by %zu", k, g * k);
          debug_decrypt(inner, runtime, buf, 4, max_n);
        }

        CHECK_STATUS_INTERNAL(runtime->Add(&out, inner), "Add");
      }
      else
      {
        out = inner;
      }

      // Debug: decrypt accumulated out after each giant step
      if (k < 3 || k == 5 || k == 10 || k == 15 || k == 20 || k == h - 1)
      {
        char buf[128];
        snprintf(buf, sizeof(buf), "out after k=%zu", k);
        debug_decrypt(out, runtime, buf, 4, max_n);
      }
    }

    printf("  BSGS loop done. out.scale=%.3e, coeff_mod_size=%zu\n",
           out.scale(), out.coeff_modulus_size());

    // Debug: decrypt out BEFORE SumStridedVectors
    debug_decrypt(out, runtime, "out BEFORE SumStridedVectors", 4, max_n);

    if (nrows < ncols)
    {
      // Sum-Columns
      printf("  Calling SumStridedVectors(stride=%zu, nslots=%zu)\n", nrows, ncols);
      CHECK_STATUS_INTERNAL(SumStridedVectors(out, nrows, ncols, runtime),
                            "SumStridedVectors");
    }

    // Debug: decrypt out AFTER SumStridedVectors, BEFORE rescale
    debug_decrypt(out, runtime, "out AFTER SumStridedVectors, BEFORE rescale", 4, max_n);

    runtime->RescaleNext(&out);
    out.scale() = 1.;

    // Debug: after rescale
    debug_decrypt(out, runtime, "out AFTER rescale (scale=1)", 4, max_n);

    lweMatrix.GetLastColumn(diag);

    printf("  LastColumn: [0]=%.8e [1]=%.8e [43]=%.8e [512]=%.8e\n",
           diag[0], diag[1], diag.size() > 43 ? diag[43] : 0.0, diag.size() > 512 ? diag[512] : 0.0);

    ptx.parms_id() = out.parms_id();
    CHECK_STATUS_INTERNAL(runtime->Encode(diag, 1., &ptx), "Encode Last Column");
    CHECK_STATUS_INTERNAL(runtime->AddPlain(&out, ptx), "AddPlain");

    // Debug: final output
    debug_decrypt(out, runtime, "out FINAL (after AddPlain b-column)", 8, max_n);
    printf("===== END HE-SecureNet REPACK DEBUG =====\n\n");

    return Status::Ok();
  }
} // namespace gemini
