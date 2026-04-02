// Isolated PBS (Programmable Bootstrapping) Test
// Tests the LUT evaluation step in isolation, matching repacking.cc parameters.
//
// Flow:
//   1. Set up Pegasus (same params as repacking.cc)
//   2. Encrypt known values → CKKS → L2R → LWE(n=1024, q0)
//   3. Decrypt LWE to verify L2R quality (pre-PBS baseline)
//   4. Apply PBS (AbsSqrt, Sigmoid) on each LWE ciphertext
//   5. Decrypt LWE after PBS to measure PBS-only error
//
// This isolates the PBS error from repacking error.

#include "pegasus/pegasus_runtime.h"
#include "pegasus/timer.h"
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>

struct ErrorStats
{
    double avg_err = 0.0;
    double max_err = 0.0;
    int count = 0;
    int bad_vals = 0;

    void print(const char *label) const
    {
        printf("[%s] count=%d, avg_err=%.6e (~2^%.1f), max_err=%.6e (~2^%.1f), bad=%d\n",
               label, count, avg_err, std::log2(avg_err + 1e-30),
               max_err, std::log2(max_err + 1e-30), bad_vals);
    }
};

ErrorStats compute_error(const std::vector<double> &expected,
                         const std::vector<double> &actual,
                         double bad_threshold = 1.0)
{
    ErrorStats stats;
    stats.count = std::min(expected.size(), actual.size());
    double sum = 0;
    for (int i = 0; i < stats.count; i++)
    {
        double err = std::abs(expected[i] - actual[i]);
        sum += err;
        if (err > stats.max_err)
            stats.max_err = err;
        if (err > bad_threshold)
            stats.bad_vals++;
    }
    stats.avg_err = sum / stats.count;
    return stats;
}

int pbs_test(int nslots)
{
    using namespace gemini;
    printf("===== PBS Isolated Test (nslots=%d) =====\n\n", nslots);

    // -------------------------------------------------------------------
    // Setup Pegasus — EXACTLY matching repacking.cc
    // -------------------------------------------------------------------
    PegasusRunTime::Parms pp;
    pp.lvl0_lattice_dim = lwe::params::n(); // 1024
    pp.lvl1_lattice_dim = 1 << 12;          // 4096 (PBS ring)
    pp.lvl2_lattice_dim = 1 << 16;          // 65536 (CKKS ring)
    pp.nlevels = 4;
    pp.scale = std::pow(2., 40);
    pp.nslots = nslots;
    pp.enable_repacking = false; // No repacking needed for this test

    PegasusRunTime pg_rt(pp, /*num_threads*/ 4);

    printf("Parameters:\n");
    printf("  n_lwe (lvl0) = %d\n", pp.lvl0_lattice_dim);
    printf("  N_boot (lvl1) = %d\n", pp.lvl1_lattice_dim);
    printf("  N_ckks (lvl2) = %d\n", pp.lvl2_lattice_dim);
    printf("  scale = 2^%.0f\n", std::log2(pp.scale));
    printf("  nslots = %d\n", pp.nslots);
    printf("  MsgRange = %.2f\n", pg_rt.MsgRange());
    printf("\n");

    // -------------------------------------------------------------------
    // Generate random input values (matching repacking.cc range)
    // -------------------------------------------------------------------
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> uniform(-8., 8.);

    F64Vec sqrt_inputs(pp.nslots);
    F64Vec sigmoid_inputs(pp.nslots);
    std::generate_n(sqrt_inputs.begin(), pp.nslots, [&]()
                    { return uniform(rng); });
    std::generate_n(sigmoid_inputs.begin(), pp.nslots, [&]()
                    { return uniform(rng); });

    // Compute ground truth
    std::vector<double> sqrt_gt(pp.nslots), sigmoid_gt(pp.nslots);
    for (int i = 0; i < pp.nslots; i++)
    {
        sqrt_gt[i] = std::sqrt(std::abs(sqrt_inputs[i]));
        sigmoid_gt[i] = 1.0 / (1.0 + std::exp(-sigmoid_inputs[i]));
    }

    // -------------------------------------------------------------------
    // Step 1: Encode → Encrypt → S2C → Extract (L2R)
    // -------------------------------------------------------------------
    Ctx sqrt_ckks, sigmoid_ckks;
    CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(sqrt_inputs, sqrt_ckks));
    CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(sigmoid_inputs, sigmoid_ckks));

    printf("--- L2R Step ---\n");
    F64 s2c_time = 0;
    {
        AutoTimer timer(&s2c_time);
        CHECK_AND_ABORT(pg_rt.SlotsToCoeffs(sqrt_ckks));
        CHECK_AND_ABORT(pg_rt.SlotsToCoeffs(sigmoid_ckks));
    }
    printf("  S2C time: %.2f ms (per ciphertext: %.2f ms)\n",
           s2c_time, s2c_time / 2.0);

    std::vector<lwe::Ctx_st> lwe_sqrt, lwe_sigmoid;
    F64 extract_time = 0;
    {
        AutoTimer timer(&extract_time);
        CHECK_AND_ABORT(pg_rt.ExtraAllCoefficients(sqrt_ckks, lwe_sqrt));
        CHECK_AND_ABORT(pg_rt.ExtraAllCoefficients(sigmoid_ckks, lwe_sigmoid));
    }
    printf("  Extract+KS time: %.2f ms (per ciphertext: %.2f ms)\n",
           extract_time, extract_time / 2.0);

    // -------------------------------------------------------------------
    // Step 2: Decrypt LWE BEFORE PBS (verify L2R quality)
    // -------------------------------------------------------------------
    printf("\n--- Pre-PBS (L2R output) ---\n");
    {
        std::vector<double> sqrt_pre(pp.nslots), sigmoid_pre(pp.nslots);
        for (int i = 0; i < pp.nslots; i++)
        {
            sqrt_pre[i] = pg_rt.DecryptLWE(lwe_sqrt[i]);
            sigmoid_pre[i] = pg_rt.DecryptLWE(lwe_sigmoid[i]);
        }

        // L2R error = decrypted value vs original input
        auto l2r_sqrt_err = compute_error(
            std::vector<double>(sqrt_inputs.begin(), sqrt_inputs.end()),
            sqrt_pre);
        auto l2r_sigmoid_err = compute_error(
            std::vector<double>(sigmoid_inputs.begin(), sigmoid_inputs.end()),
            sigmoid_pre);

        l2r_sqrt_err.print("L2R sqrt_input");
        l2r_sigmoid_err.print("L2R sigmoid_input");

        printf("  First 4 sqrt L2R:  ");
        for (int i = 0; i < std::min(4, pp.nslots); i++)
            printf("%.4f→%.4f  ", sqrt_inputs[i], sqrt_pre[i]);
        printf("\n");

        printf("  First 4 sigmoid L2R:  ");
        for (int i = 0; i < std::min(4, pp.nslots); i++)
            printf("%.4f→%.4f  ", sigmoid_inputs[i], sigmoid_pre[i]);
        printf("\n");
    }

    // -------------------------------------------------------------------
    // Step 3: Apply PBS (LUT evaluation)
    // -------------------------------------------------------------------
    printf("\n--- PBS Step ---\n");
    // Make copies so we can compare
    std::vector<lwe::Ctx_st> lwe_sqrt_pbs = lwe_sqrt;
    std::vector<lwe::Ctx_st> lwe_sigmoid_pbs = lwe_sigmoid;

    F64 pbs_time = 0;
    {
        AutoTimer timer(&pbs_time);
        pg_rt.AbsSqrt(lwe_sqrt_pbs.data(), lwe_sqrt_pbs.size());
        pg_rt.Sigmoid(lwe_sigmoid_pbs.data(), lwe_sigmoid_pbs.size());
    }
    printf("  PBS time: %.2f ms, per LWE: %.4f ms\n",
           pbs_time, pbs_time / (2.0 * pp.nslots));

    // -------------------------------------------------------------------
    // Step 4: Decrypt LWE AFTER PBS (measure PBS quality)
    // -------------------------------------------------------------------
    printf("\n--- Post-PBS Results ---\n");
    std::vector<double> sqrt_post(pp.nslots), sigmoid_post(pp.nslots);
    for (int i = 0; i < pp.nslots; i++)
    {
        sqrt_post[i] = pg_rt.DecryptLWE(lwe_sqrt_pbs[i]);
        sigmoid_post[i] = pg_rt.DecryptLWE(lwe_sigmoid_pbs[i]);
    }

    // PBS error = decrypted value vs ground truth function value
    auto pbs_sqrt_err = compute_error(sqrt_gt, sqrt_post);
    auto pbs_sigmoid_err = compute_error(sigmoid_gt, sigmoid_post);

    pbs_sqrt_err.print("PBS sqrt(|x|)");
    pbs_sigmoid_err.print("PBS sigmoid(x)");

    // Print first few values
    printf("\n  sqrt(|x|) — first 8:\n");
    printf("    input     ground_truth  pbs_result    error\n");
    for (int i = 0; i < std::min(8, pp.nslots); i++)
    {
        double err = std::abs(sqrt_gt[i] - sqrt_post[i]);
        printf("    %+8.4f  %8.4f      %8.4f      %.6f\n",
               sqrt_inputs[i], sqrt_gt[i], sqrt_post[i], err);
    }

    printf("\n  sigmoid(x) — first 8:\n");
    printf("    input     ground_truth  pbs_result    error\n");
    for (int i = 0; i < std::min(8, pp.nslots); i++)
    {
        double err = std::abs(sigmoid_gt[i] - sigmoid_post[i]);
        printf("    %+8.4f  %8.4f      %8.4f      %.6f\n",
               sigmoid_inputs[i], sigmoid_gt[i], sigmoid_post[i], err);
    }

    // -------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------
    printf("\n===== PBS Test Summary (nslots=%d) =====\n", nslots);
    printf("  L2R time:  %.2f ms\n", s2c_time + extract_time);
    printf("  PBS time:  %.2f ms (%.4f ms per LWE)\n",
           pbs_time, pbs_time / (2.0 * pp.nslots));
    printf("  sqrt(|x|) avg_err: %.6e (~2^%.1f)\n",
           pbs_sqrt_err.avg_err, std::log2(pbs_sqrt_err.avg_err + 1e-30));
    printf("  sigmoid(x) avg_err: %.6e (~2^%.1f)\n",
           pbs_sigmoid_err.avg_err, std::log2(pbs_sigmoid_err.avg_err + 1e-30));

    return 0;
}

int main()
{
    pbs_test(/*nslots*/ 64);
    printf("\n\n");
    pbs_test(/*nslots*/ 256);
    return 0;
}
