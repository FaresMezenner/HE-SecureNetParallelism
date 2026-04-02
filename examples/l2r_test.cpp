#include "pegasus/pegasus_runtime.h"
#include "pegasus/timer.h"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <random>

// Test L2R (CKKS -> LWE) error in isolation
// This measures specifically S2C + SampleExtract + LWEKeySwitch error
// WITHOUT any LUT evaluation.
int l2r_test(int nslots)
{
    using namespace gemini;
    PegasusRunTime::Parms pp;

    pp.lvl0_lattice_dim = lwe::params::n(); // 1024
    pp.lvl1_lattice_dim = 1 << 12;          // 4096
    pp.lvl2_lattice_dim = 1 << 16;          // 65536
    pp.nlevels = 4;
    pp.scale = std::pow(2., 40);
    pp.nslots = nslots;
    pp.enable_repacking = false; // We don't need repacking for L2R test

    PegasusRunTime pg_rt(pp, /*num_threads*/ 4);

    // Generate random test values in [-8, 8]
    F64Vec slots(pp.nslots);
    {
        std::mt19937 rng(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<double> dist(-8., 8.);
        std::generate_n(slots.begin(), pp.nslots, [&]()
                        { return dist(rng); });
    }

    // Encrypt
    Ctx ckks_ct;
    CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(slots, ckks_ct));

    // Step 1: S2C
    F64 s2c_time{0.};
    {
        AutoTimer timer(&s2c_time);
        CHECK_AND_ABORT(pg_rt.SlotsToCoeffs(ckks_ct));
    }

    // Step 2: Extract all coefficients (includes key switching)
    std::vector<lwe::Ctx_st> lwe_cts;
    F64 extract_time{0.};
    {
        AutoTimer timer(&extract_time);
        CHECK_AND_ABORT(pg_rt.ExtraAllCoefficients(ckks_ct, lwe_cts));
    }

    printf("=== L2R Test (nslots = %d) ===\n", pp.nslots);
    printf("S2C took %f seconds\n", s2c_time / 1000.);
    printf("Extract+KS took %f seconds\n", extract_time / 1000.);
    printf("Total L2R: %f seconds\n", (s2c_time + extract_time) / 1000.);
    printf("\n");

    // Decrypt each LWE ciphertext and compare with original
    double max_err = 0.;
    double sum_err = 0.;
    double sum_err_sq = 0.;
    int bad_count = 0;

    for (size_t i = 0; i < pp.nslots; ++i)
    {
        double original = slots[i];
        double decrypted = pg_rt.DecryptLWE(lwe_cts[i]);
        double err = std::abs(original - decrypted);

        sum_err += err;
        sum_err_sq += err * err;
        if (err > max_err)
            max_err = err;

        // Count "bad" values with error > 0.1
        if (err > 0.1)
            bad_count++;

        // Print first 8 values
        if (i < 8)
        {
            printf("slot[%3zu]: original=%10.6f, decrypted=%10.6f, error=%e (2^%.1f)\n",
                   i, original, decrypted, err, err > 0 ? std::log2(err) : -999.0);
        }
    }

    double avg_err = sum_err / pp.nslots;
    double rms_err = std::sqrt(sum_err_sq / pp.nslots);

    printf("\n=== L2R Error Summary ===\n");
    printf("Average error: %e ~ 2^%.1f\n", avg_err, std::log2(avg_err));
    printf("RMS error:     %e ~ 2^%.1f\n", rms_err, std::log2(rms_err));
    printf("Max error:     %e ~ 2^%.1f\n", max_err, std::log2(max_err));
    printf("Bad values (err > 0.1): %d / %d (%.1f%%)\n",
           bad_count, pp.nslots, 100.0 * bad_count / pp.nslots);

    // Print LWE ciphertext structure info
    printf("\n=== LWE Ciphertext Info ===\n");
    printf("LWE dimension: %d\n", (int)lwe::params::n());
    printf("LWE scale: %f\n", lwe_cts[0].scale);
    printf("Number of LWE ciphertexts: %zu\n", lwe_cts.size());

    return 0;
}

int main()
{
    printf("=== Test 1: nslots=256 ===\n\n");
    l2r_test(256);

    printf("\n\n=== Test 2: nslots=64 ===\n\n");
    l2r_test(64);

    return 0;
}
