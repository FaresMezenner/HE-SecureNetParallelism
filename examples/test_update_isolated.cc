/*
 * test_update_isolated.cc — PEGASUS Reference for Weight-Update Isolation
 * ========================================================================
 *
 * PURPOSE:
 *   Reference implementation of a single weight update step using PEGASUS.
 *   Encrypt known vectors D and DE, run update_model_dense, decrypt and
 *   log every intermediate value. Compare output with the GPU isolated test.
 *
 * BUILD:
 *   Add to HE-SecureNet/examples/CMakeLists.txt:
 *     add_executable(test_update_iso_exe test_update_isolated.cc)
 *     target_link_libraries(test_update_iso_exe pegasus)
 *
 * RUN (from HE-SecureNet/build/examples/):
 *   ./test_update_iso_exe
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <sys/stat.h>

#include "pegasus/pegasus_runtime.h"
#include "pegasus/timer.h"

using namespace std;
using namespace gemini;

// Globals required by PegasusRunTime
extern std::string save_model_loc;
extern std::string model_name;
extern thread_local bool not_first_epoch;
extern thread_local double total_save_model_time;
extern thread_local double total_load_model_time;
extern thread_local double total_offline_time;
extern thread_local double total_online_time;

static void mkdirs(const std::string& path) {
    std::string partial;
    for (char c : path) {
        partial += c;
        if (c == '/') mkdir(partial.c_str(), 0755);
    }
    if (!partial.empty()) mkdir(partial.c_str(), 0755);
}

int main()
{
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    printf("================================================================\n");
    printf("  PEGASUS REFERENCE: Weight-Update Isolation Test\n");
    printf("================================================================\n\n");

    // ── 1. PEGASUS setup ────────────────────────────────────────────
    PegasusRunTime::Parms pp;
    pp.lvl0_lattice_dim = lwe::params::n();  // 1024
    pp.lvl1_lattice_dim = 1 << 12;           // 4096
    pp.lvl2_lattice_dim = 1 << 16;           // 65536
    pp.nlevels = 8;
    pp.scale = std::pow(2., 40);
    pp.nslots = 512;
    pp.s2c_multiplier = 1.;
    pp.enable_repacking = true;

    printf("--- Initializing PegasusRunTime (N=65536, nslots=512) ---\n");
    PegasusRunTime pg_rt(pp, /*num_threads=*/ 1);
    omp_set_num_threads(1);

    save_model_loc = "/tmp/test_update_iso/";
    model_name = "test_iso";
    mkdirs(save_model_loc);
    not_first_epoch = false;

    int nslots = pp.nslots;    // 512
    double scale = pp.scale;   // 2^40

    printf("  nslots=%d, scale=2^40\n\n", nslots);

    // ── 2. Generate FIXED test data (SAME as GPU test) ──────────────
    // IMPORTANT: Must match the GPU test's data exactly for comparison.
    // GPU test uses NSLOTS=256, PEGASUS uses nslots=512.
    // Fill first 256 with same values, rest with 0.
    printf("--- Generating fixed test data ---\n");
    const int GPU_NSLOTS = 256;  // GPU's slot count
    const double W_INIT_VAL = 0.178;
    const double B_INIT_VAL = -0.002;

    std::vector<double> d_vals(nslots, 0.0);
    for (int i = 0; i < GPU_NSLOTS; i++)
        d_vals[i] = 0.5 + 0.3 * sin(i * 0.1);

    std::vector<double> de_vals(nslots, 0.0);
    for (int i = 0; i < GPU_NSLOTS; i++)
        de_vals[i] = 0.3 * cos(i * 0.07) - 0.1;

    // Expected gradient (sum over 256 active samples, divided by nslots=512)
    // NOTE: PEGASUS divides by nslots=512, GPU divides by NSLOTS=256.
    // For a fair comparison, we compute both.
    double pt_gradient_512 = 0.0;
    for (int i = 0; i < nslots; i++)
        pt_gradient_512 += d_vals[i] * de_vals[i];
    pt_gradient_512 /= nslots;

    double pt_gradient_256 = 0.0;
    for (int i = 0; i < GPU_NSLOTS; i++)
        pt_gradient_256 += d_vals[i] * de_vals[i];
    pt_gradient_256 /= GPU_NSLOTS;

    double pt_w_after_512 = W_INIT_VAL - pt_gradient_512;
    double pt_w_after_256 = W_INIT_VAL - pt_gradient_256;

    printf("  D[0:4]: %.6f, %.6f, %.6f, %.6f\n",
           d_vals[0], d_vals[1], d_vals[2], d_vals[3]);
    printf("  DE[0:4]: %.6f, %.6f, %.6f, %.6f\n",
           de_vals[0], de_vals[1], de_vals[2], de_vals[3]);
    printf("  W_init   = %.6f\n", W_INIT_VAL);
    printf("  PT gradient (nslots=512): %.10e\n", pt_gradient_512);
    printf("  PT gradient (nslots=256): %.10e\n", pt_gradient_256);
    printf("  PT W_after (nslots=512):  %.10e\n", pt_w_after_512);
    printf("  PT W_after (nslots=256):  %.10e\n\n", pt_w_after_256);

    // ── 3. Encrypt D and DE ─────────────────────────────────────────
    printf("--- Encrypting D and DE ---\n");

    // D ciphertext
    Ctx D_cipher;
    pg_rt.EncodeThenEncrypt(d_vals, D_cipher);
    printf("  D: nModuli=%d\n", pg_rt.GetNModuli(D_cipher));

    // Decrypt to verify
    {
        std::vector<double> T(nslots);
        pg_rt.DecryptThenDecode(D_cipher, T);
        printf("  D decrypted[0:4]: %.6f, %.6f, %.6f, %.6f\n",
               T[0], T[1], T[2], T[3]);
    }

    // DE ciphertext
    Ctx DE_cipher;
    pg_rt.EncodeThenEncrypt(de_vals, DE_cipher);
    printf("  DE: nModuli=%d\n", pg_rt.GetNModuli(DE_cipher));

    {
        std::vector<double> T(nslots);
        pg_rt.DecryptThenDecode(DE_cipher, T);
        printf("  DE decrypted[0:4]: %.6f, %.6f, %.6f, %.6f\n",
               T[0], T[1], T[2], T[3]);
    }

    // ── 4. MANUAL weight update (replicate PEGASUS update_model_dense logic) ─
    printf("\n--- Manual PEGASUS weight update (step by step) ---\n");

    int moduli = 5;  // PEGASUS uses moduli=5 for update
    printf("  moduli=%d\n", moduli);

    // lr_cipher: sparse [1/nslots, 0, 0, ...]
    Ctx lr_cipher;
    {
        std::vector<double> lr(nslots, 0.0);
        lr[0] = 1.0 / nslots;
        pg_rt.EncodeThenEncrypt(lr, lr_cipher);
        pg_rt.DropModuliTo(&lr_cipher, moduli + 1);
    }
    printf("  lr_cipher: nModuli=%d, lr[0]=%.10e\n",
           pg_rt.GetNModuli(lr_cipher), 1.0 / nslots);
    {
        std::vector<double> T(nslots);
        pg_rt.DecryptThenDecode(lr_cipher, T);
        printf("  lr decrypted[0:4]: %.10e, %.10e, %.10e, %.10e\n",
               T[0], T[1], T[2], T[3]);
    }

    // W_cipher: all slots = W_INIT_VAL
    Ctx W_cipher;
    {
        std::vector<double> w_vec(nslots, W_INIT_VAL);
        pg_rt.EncodeThenEncrypt(w_vec, W_cipher);
        pg_rt.DropModuliTo(&W_cipher, moduli);
    }
    printf("  W_cipher: nModuli=%d\n", pg_rt.GetNModuli(W_cipher));
    {
        std::vector<double> T(nslots);
        pg_rt.DecryptThenDecode(W_cipher, T);
        printf("  W decrypted[0:4]: %.6f, %.6f, %.6f, %.6f\n",
               T[0], T[1], T[2], T[3]);
    }

    // Step 1: Drop D to moduli+2
    Ctx tmp_cipher = D_cipher;
    pg_rt.DropModuliTo(&tmp_cipher, moduli + 2);
    printf("\n  [Step 1] D dropped to moduli+2=%d: nModuli=%d\n",
           moduli + 2, pg_rt.GetNModuli(tmp_cipher));
    {
        std::vector<double> T(nslots);
        pg_rt.DecryptThenDecode(tmp_cipher, T);
        printf("    D[0:4]: %.8e, %.8e, %.8e, %.8e\n", T[0], T[1], T[2], T[3]);
    }

    // Drop DE to moduli+2
    Ctx de_copy = DE_cipher;
    pg_rt.DropModuliTo(&de_copy, moduli + 2);
    printf("  DE dropped to moduli+2=%d: nModuli=%d\n",
           moduli + 2, pg_rt.GetNModuli(de_copy));
    {
        std::vector<double> T(nslots);
        pg_rt.DecryptThenDecode(de_copy, T);
        printf("    DE[0:4]: %.8e, %.8e, %.8e, %.8e\n", T[0], T[1], T[2], T[3]);
    }

    // Step 2: MulRelinRescale(D, DE)
    printf("\n  [Step 2] MulRelinRescale(D, DE)\n");
    pg_rt.MulRelinRescale(&tmp_cipher, de_copy);
    printf("    After MulRelinRescale: nModuli=%d\n", pg_rt.GetNModuli(tmp_cipher));
    {
        std::vector<double> T(nslots);
        pg_rt.DecryptThenDecode(tmp_cipher, T);
        printf("    product[0:4]: %.8e, %.8e, %.8e, %.8e\n", T[0], T[1], T[2], T[3]);
        // Expected: D[i] * DE[i] for each slot
        printf("    expected[0:4]: %.8e, %.8e, %.8e, %.8e\n",
               d_vals[0]*de_vals[0], d_vals[1]*de_vals[1],
               d_vals[2]*de_vals[2], d_vals[3]*de_vals[3]);
    }

    // Step 3: Rotate-and-Sum (log2(512) = 9 rotations)
    printf("\n  [Step 3] Rotate-and-Sum (9 rotations for nslots=512)\n");
    Ctx tmp_cipher_2;
    for (int k = (int)log2(nslots) - 1; k >= 0; k--) {
        tmp_cipher_2 = tmp_cipher;
        pg_rt.RotateLeft(&tmp_cipher_2, (int)pow(2, k));
        pg_rt.Add(&tmp_cipher, tmp_cipher_2);
    }
    {
        std::vector<double> T(nslots);
        pg_rt.DecryptThenDecode(tmp_cipher, T);
        printf("    After R&S[0:4]: %.8e, %.8e, %.8e, %.8e\n", T[0], T[1], T[2], T[3]);
        double expected_sum = 0.0;
        for (int i = 0; i < nslots; i++) expected_sum += d_vals[i] * de_vals[i];
        printf("    Expected sum = %.8e\n", expected_sum);
    }

    // Step 4: MulRelinRescale(sum, lr_cipher)
    printf("\n  [Step 4] MulRelinRescale(sum, lr_cipher)\n");
    pg_rt.MulRelinRescale(&tmp_cipher, lr_cipher);
    printf("    After ×lr: nModuli=%d\n", pg_rt.GetNModuli(tmp_cipher));
    {
        std::vector<double> T(nslots);
        pg_rt.DecryptThenDecode(tmp_cipher, T);
        printf("    scaled[0:4]: %.8e, %.8e, %.8e, %.8e\n", T[0], T[1], T[2], T[3]);
        printf("    Expected: %.8e (slot 0 only, rest ≈ 0)\n", pt_gradient_512);
    }

    // Step 5: Replicate-Back (9 rotations)
    printf("\n  [Step 5] Replicate-Back (9 right rotations)\n");
    tmp_cipher_2 = tmp_cipher;
    for (int k = 0; k < (int)log2(nslots); k++) {
        pg_rt.RotateRight(&tmp_cipher_2, (int)pow(2, k));
        pg_rt.Add(&tmp_cipher, tmp_cipher_2);
        tmp_cipher_2 = tmp_cipher;
    }
    {
        std::vector<double> T(nslots);
        pg_rt.DecryptThenDecode(tmp_cipher, T);
        printf("    After replicate[0:4]: %.8e, %.8e, %.8e, %.8e\n",
               T[0], T[1], T[2], T[3]);
        printf("    All slots should be ≈ %.8e\n", pt_gradient_512);
    }

    // Step 6: W -= gradient
    printf("\n  [Step 6] W -= gradient\n");
    printf("    W.nModuli=%d, gradient.nModuli=%d\n",
           pg_rt.GetNModuli(W_cipher), pg_rt.GetNModuli(tmp_cipher));
    pg_rt.Sub(&W_cipher, tmp_cipher);
    {
        std::vector<double> T(nslots);
        pg_rt.DecryptThenDecode(W_cipher, T);
        printf("    W_after[0:4]: %.10e, %.10e, %.10e, %.10e\n",
               T[0], T[1], T[2], T[3]);
    }

    // ── 5. Final comparison ─────────────────────────────────────────
    printf("\n=== COMPARISON ===\n");
    {
        std::vector<double> T(nslots);
        pg_rt.DecryptThenDecode(W_cipher, T);
        double he_w = T[0];
        printf("  [PEGASUS] W[0] = %.10e\n", he_w);
        printf("  [PT/512]  W[0] = %.10e  (diff=%.6e)\n",
               pt_w_after_512, he_w - pt_w_after_512);
        printf("  [PT/256]  W[0] = %.10e  (diff=%.6e)\n",
               pt_w_after_256, he_w - pt_w_after_256);
        printf("  STATUS: %s\n",
               (std::abs(he_w - pt_w_after_512) < 1.0) ? "PASS" : "FAIL !!!");
    }

    printf("\n================================================================\n");
    printf("  PEGASUS REFERENCE TEST COMPLETE\n");
    printf("  Compare [PEGASUS] values above with [GPU] test output.\n");
    printf("  The first step where values diverge is the bug location.\n");
    printf("================================================================\n");

    return 0;
}
