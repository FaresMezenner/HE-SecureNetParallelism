/*
 * model_b_bc_4layer.cpp
 * =====================
 * End-to-end HE training of a **4-layer MLP** on the Breast-Cancer dataset
 * using the PEGASUS hybrid CKKS+FHEW framework (HE-SecureNet).
 *
 *   Architecture:  30 → 22 (ReLU) → 22 (ReLU) → 22 (ReLU) → 1 (Sigmoid)
 *   Loss:          Binary cross-entropy  (sigmoid+BCE gradient = D4 − Y)
 *   Batch:         full dataset packed in nslots=512  (455 train, 114 test)
 *   Epochs:        2  (epoch 0 + epoch 1)
 *   Threads:       24  (half-machine experiment)
 *
 * Based on model_b_breast_cancer.cpp (3-layer version).
 */

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/stat.h>

#include "pegasus/pegasus_runtime.h"
#include "pegasus/timer.h"

using namespace std;
using namespace gemini;

// ── Tuneable constants ──────────────────────────────────────────────
static constexpr int num_threads = 24; // half machine (24 cores)

// ── Globals required by PegasusRunTime (defined in pegasus_runtime.cc) ──
extern std::string save_model_loc;
extern std::string model_name;
extern thread_local bool not_first_epoch;
extern thread_local double total_save_model_time;
extern thread_local double total_load_model_time;
extern thread_local double total_offline_time;
extern thread_local double total_online_time;

// ── Dual logging (stdout + file) ────────────────────────────────────
static std::ofstream g_logfile;

static double get_rss_gb()
{
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmRSS:", 0) == 0)
        {
            long kb = 0;
            sscanf(line.c_str(), "VmRSS: %ld kB", &kb);
            return kb / 1048576.0;
        }
    return -1.0;
}

static void log_msg(const std::string &msg)
{
    time_t now = time(nullptr);
    char ts[32];
    strftime(ts, sizeof(ts), "%H:%M:%S", localtime(&now));
    std::string line = std::string("[") + ts + "] " + msg;
    std::cout << line << std::endl;
    if (g_logfile.is_open())
        g_logfile << line << std::endl;
}

static void log_time_mem(const std::string &label, double secs)
{
    char buf[256];
    snprintf(buf, sizeof(buf), "%-42s %10.1f s | RSS %.1f GB",
             label.c_str(), secs, get_rss_gb());
    log_msg(buf);
}

static void mkdirs(const std::string &path)
{
    std::string partial;
    for (char c : path)
    {
        partial += c;
        if (c == '/')
        {
            mkdir(partial.c_str(), 0755);
        }
    }
    if (!partial.empty())
    {
        mkdir(partial.c_str(), 0755);
    }
}

// ── ReLU / Sigmoid in plaintext (for evaluation) ────────────────────
static inline double pt_relu(double x) { return x > 0.0 ? x : 0.0; }
static inline double pt_sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// =====================================================================
//                             MAIN
// =====================================================================
int main(int argc, char *argv[])
{

    // ── Open log file ───────────────────────────────────────────────
    std::string log_dir = "./logs";
    mkdirs(log_dir);
    {
        time_t now = time(nullptr);
        char buf[64];
        strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&now));
        std::string log_path = log_dir + "/model_b_bc_4l_" + std::string(buf) + ".log";
        g_logfile.open(log_path, std::ios::out);
        log_msg("Log file: " + log_path);
    }

    // ── Pegasus parameter setup ─────────────────────────────────────
    PegasusRunTime::Parms pp;
    pp.lvl0_lattice_dim = lwe::params::n(); // 1024
    pp.lvl1_lattice_dim = 1 << 12;          // 4096
    pp.lvl2_lattice_dim = 1 << 16;          // 65536
    pp.nlevels = 8;                         // CKKS multiplicative levels
    pp.scale = std::pow(2., 40);
    pp.nslots = 512; // batch = all 455 train samples
    pp.s2c_multiplier = 1.;
    pp.enable_repacking = true;

    log_msg("Creating PegasusRunTime (N=65536, nslots=512, nlevels=8) ...");
    log_msg("Architecture: 30 -> 22(ReLU) -> 22(ReLU) -> 22(ReLU) -> 1(Sigmoid)");
    log_msg("Threads: " + std::to_string(num_threads) + "  Epochs: 2");
    double setup_time = 0.0;
    {
        AutoTimer t(&setup_time);
        // (constructor prints context info)
    }
    PegasusRunTime pg_rt(pp, /*num_threads*/ num_threads);
    omp_set_num_threads(num_threads);
    log_time_mem("SETUP (keygen + context)", setup_time);

    // ── Model save location ─────────────────────────────────────────
    save_model_loc = "./model/model_b_bc_4l/";
    model_name = "model_b_bc_4l";
    mkdirs(save_model_loc);
    log_msg("Model checkpoint dir: " + save_model_loc);

    // ── Network dimensions ──────────────────────────────────────────
    const int l0 = 30;            // input  (features)
    const int l1 = 22;            // hidden 1
    const int l2 = 22;            // hidden 2
    const int l3 = 22;            // hidden 3  (NEW)
    const int l4 = 1;             // output (binary)
    const int nslots = pp.nslots; // 512
    const int n_train = 455;
    const int n_test = 114;
    const int total_epochs = 2;

    not_first_epoch = false;

    // ── Read Breast-Cancer dataset ──────────────────────────────────
    log_msg("Reading Breast Cancer dataset ...");

    // x_train: 455 rows × 30 features (row-major in file)
    vector<vector<F64>> x_train(l0, vector<F64>(nslots, 0.0));
    vector<vector<F64>> x_test(l0, vector<F64>(nslots, 0.0));
    vector<F64> y_train_raw(nslots, 0.0);
    vector<F64> y_test_raw(nslots, 0.0);

    {
        ifstream fin("../dataset/breast_cancer/x_train.out");
        if (!fin.is_open())
        {
            log_msg("ERROR: cannot open x_train.out");
            return 1;
        }
        for (int j = 0; j < n_train; ++j)
            for (int i = 0; i < l0; ++i)
                fin >> x_train[i][j];
        fin.close();
    }
    {
        ifstream fin("../dataset/breast_cancer/x_test.out");
        if (!fin.is_open())
        {
            log_msg("ERROR: cannot open x_test.out");
            return 1;
        }
        for (int j = 0; j < n_test; ++j)
            for (int i = 0; i < l0; ++i)
                fin >> x_test[i][j];
        fin.close();
    }
    {
        ifstream fin("../dataset/breast_cancer/y_train.out");
        if (!fin.is_open())
        {
            log_msg("ERROR: cannot open y_train.out");
            return 1;
        }
        for (int j = 0; j < n_train; ++j)
            fin >> y_train_raw[j];
        fin.close();
    }
    {
        ifstream fin("../dataset/breast_cancer/y_test.out");
        if (!fin.is_open())
        {
            log_msg("ERROR: cannot open y_test.out");
            return 1;
        }
        for (int j = 0; j < n_test; ++j)
            fin >> y_test_raw[j];
        fin.close();
    }
    log_msg("Dataset loaded: train=" + std::to_string(n_train) + " test=" + std::to_string(n_test) + " features=" + std::to_string(l0));

    // ── Cipher-domain vectors ───────────────────────────────────────
    // Input activations  D0  (30 cts)
    vector<vector<F64>> D0_plain(l0, vector<F64>(nslots, 0.0));
    vector<Ctx> D0_cipher(l0);

    // Layer 1 intermediates  (30 → 22, ReLU)
    vector<Ctx> U1_cipher(l1);
    vector<Ctx> D1_cipher(l1);
    vector<Ctx> DE1_cipher(l1);
    vector<vector<F64>> W1(l0, vector<F64>(l1, 0.0));
    vector<F64> B1(l1, 0.0);

    // Layer 2 intermediates  (22 → 22, ReLU)
    vector<Ctx> U2_cipher(l2);
    vector<Ctx> D2_cipher(l2);
    vector<Ctx> DE2_cipher(l2);
    vector<vector<F64>> W2(l1, vector<F64>(l2, 0.0));
    vector<F64> B2(l2, 0.0);

    // Layer 3 intermediates  (22 → 22, ReLU)  — NEW
    vector<Ctx> U3_cipher(l3);
    vector<Ctx> D3_cipher(l3);
    vector<Ctx> DE3_cipher(l3);
    vector<vector<F64>> W3(l2, vector<F64>(l3, 0.0));
    vector<F64> B3(l3, 0.0);

    // Layer 4 intermediates  (22 → 1, Sigmoid)  — was Layer 3
    vector<Ctx> U4_cipher(l4);
    vector<Ctx> D4_cipher(l4);
    vector<Ctx> DE4_cipher(l4);
    vector<vector<F64>> W4(l3, vector<F64>(l4, 0.0));
    vector<F64> B4(l4, 0.0);

    // LWE buffers – sized for the *largest* layer that needs activation
    // l1=l2=l3=22, l4=1 → max_act_dim = 22
    const int max_act_dim = std::max({l1, l2, l3, l4});
    vector<vector<lwe::Ctx_st>> U_lwe_cipher(max_act_dim, vector<lwe::Ctx_st>(nslots));
    vector<vector<lwe::Ctx_st>> D_lwe_cipher(max_act_dim, vector<lwe::Ctx_st>(nslots));

    // Label ciphertexts (1 output class)
    vector<vector<F64>> Y(l4, vector<F64>(nslots, 0.0));
    vector<Ctx> Y_cipher(l4);

    // ── Initialise weights (Xavier-like, ±0.3) ─────────────────────
    log_msg("Initialising weights ...");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.3, 0.3);

    for (int i = 0; i < l0; ++i)
        for (int j = 0; j < l1; ++j)
            W1[i][j] = dis(gen);
    for (int i = 0; i < l1; ++i)
        B1[i] = dis(gen);
    for (int i = 0; i < l1; ++i)
        for (int j = 0; j < l2; ++j)
            W2[i][j] = dis(gen);
    for (int i = 0; i < l2; ++i)
        B2[i] = dis(gen);
    for (int i = 0; i < l2; ++i)
        for (int j = 0; j < l3; ++j)
            W3[i][j] = dis(gen);
    for (int i = 0; i < l3; ++i)
        B3[i] = dis(gen);
    for (int i = 0; i < l3; ++i)
        for (int j = 0; j < l4; ++j)
            W4[i][j] = dis(gen);
    for (int i = 0; i < l4; ++i)
        B4[i] = dis(gen);

    // ── Encrypt inputs (only once – same data every epoch) ──────────
    log_msg("Encrypting input features (D0) ...");
    double enc_time = 0.0;
    {
        AutoTimer t(&enc_time);
        for (int i = 0; i < l0; ++i)
            D0_plain[i] = x_train[i]; // already padded with zeros at [455..511]

#pragma omp parallel for
        for (int i = 0; i < l0; ++i)
            CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(D0_plain[i], D0_cipher[i]));
    }
    log_time_mem("ENCRYPT D0 (30 cts)", enc_time);

    // ── Encrypt labels (only once) ──────────────────────────────────
    log_msg("Encrypting labels (Y) ...");
    double enc_y_time = 0.0;
    {
        AutoTimer t(&enc_y_time);
        // Binary: Y[0][j] = label ∈ {0,1}
        for (int j = 0; j < nslots; ++j)
            Y[0][j] = y_train_raw[j]; // padded zeros for j>=455

        for (int i = 0; i < l4; ++i)
            CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(Y[i], Y_cipher[i]));

        // S2C → extract → repack to refresh levels
        pg_rt.s2c_and_extract(Y_cipher, U_lwe_cipher, l4, nslots);
        pg_rt.repack(U_lwe_cipher, Y_cipher, l4, nslots);
    }
    log_time_mem("ENCRYPT Y (1 ct + S2C+repack)", enc_y_time);

    // ==================================================================
    //                      TRAINING LOOP
    // ==================================================================
    for (int epoch = 0; epoch < total_epochs; ++epoch)
    {
        log_msg("=========================================");
        log_msg("============= EPOCH " + std::to_string(epoch) + " =============");
        log_msg("=========================================");

        if (epoch != 0)
            not_first_epoch = true;

        double epoch_time = 0.0;
        AutoTimer epoch_timer(&epoch_time);

        // ─────────────────────────────────────────────────────────────
        //  LAYER 1 – FORWARD PROPAGATION  (30 → 22, ReLU)
        // ─────────────────────────────────────────────────────────────
        log_msg("--- Layer 1 FORWARD ---");

        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.dense(pp, D0_cipher, W1, B1, U1_cipher, 1, epoch);
            timer.stop();
            log_time_mem("  L1 dense(30x22)", t);
        }
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.s2c_and_extract(U1_cipher, U_lwe_cipher, l1, nslots);
            timer.stop();
            log_time_mem("  L1 s2c_extract(22)", t);
        }
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, D1_cipher, l1, nslots, "ReLU");
            timer.stop();
            log_time_mem("  L1 ReLU(22)", t);
        }
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, U1_cipher, l1, nslots, "DReLU");
            timer.stop();
            log_time_mem("  L1 DReLU(22)", t);
        }

        // ─────────────────────────────────────────────────────────────
        //  LAYER 2 – FORWARD PROPAGATION  (22 → 22, ReLU)
        // ─────────────────────────────────────────────────────────────
        log_msg("--- Layer 2 FORWARD ---");

        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.dense(pp, D1_cipher, W2, B2, U2_cipher, 2, epoch);
            timer.stop();
            log_time_mem("  L2 dense(22x22)", t);
        }
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.s2c_and_extract(U2_cipher, U_lwe_cipher, l2, nslots);
            timer.stop();
            log_time_mem("  L2 s2c_extract(22)", t);
        }
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, D2_cipher, l2, nslots, "ReLU");
            timer.stop();
            log_time_mem("  L2 ReLU(22)", t);
        }
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, U2_cipher, l2, nslots, "DReLU");
            timer.stop();
            log_time_mem("  L2 DReLU(22)", t);
        }

        // ─────────────────────────────────────────────────────────────
        //  LAYER 3 – FORWARD PROPAGATION  (22 → 22, ReLU)  — NEW
        // ─────────────────────────────────────────────────────────────
        log_msg("--- Layer 3 FORWARD ---");

        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.dense(pp, D2_cipher, W3, B3, U3_cipher, 3, epoch);
            timer.stop();
            log_time_mem("  L3 dense(22x22)", t);
        }
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.s2c_and_extract(U3_cipher, U_lwe_cipher, l3, nslots);
            timer.stop();
            log_time_mem("  L3 s2c_extract(22)", t);
        }
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, D3_cipher, l3, nslots, "ReLU");
            timer.stop();
            log_time_mem("  L3 ReLU(22)", t);
        }
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, U3_cipher, l3, nslots, "DReLU");
            timer.stop();
            log_time_mem("  L3 DReLU(22)", t);
        }

        // ─────────────────────────────────────────────────────────────
        //  LAYER 4 – FORWARD PROPAGATION  (22 → 1, Sigmoid)
        // ─────────────────────────────────────────────────────────────
        log_msg("--- Layer 4 FORWARD ---");

        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.dense(pp, D3_cipher, W4, B4, U4_cipher, 4, epoch);
            timer.stop();
            log_time_mem("  L4 dense(22x1)", t);
        }
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.s2c_and_extract(U4_cipher, U_lwe_cipher, l4, nslots);
            timer.stop();
            log_time_mem("  L4 s2c_extract(1)", t);
        }
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, D4_cipher, l4, nslots, "Sigmoid");
            timer.stop();
            log_time_mem("  L4 Sigmoid(1)", t);
        }

        // ─────────────────────────────────────────────────────────────
        //  LAYER 4 – DELTA (BACKWARD)
        //  δ₄ = D4 − Y   (sigmoid + BCE: gradient simplifies)
        // ─────────────────────────────────────────────────────────────
        log_msg("--- Layer 4 DELTA ---");
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.delta_softmax(pp, D4_cipher, Y_cipher, DE4_cipher);
            timer.stop();
            log_time_mem("  L4 delta_softmax(1)", t);
        }

        // ─────────────────────────────────────────────────────────────
        //  LAYER 3 – DELTA (BACKWARD)
        //  δ₃ = (W4ᵀ · δ₄) ⊙ A'(U3)
        // ─────────────────────────────────────────────────────────────
        log_msg("--- Layer 3 DELTA ---");
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.delta_1D_dense_1D(pp, W4, DE4_cipher, U3_cipher, DE3_cipher, 4, epoch);
            timer.stop();
            log_time_mem("  L3 delta_1D_dense_1D(W4:22x1)", t);
        }
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.s2c_repack_1D(DE3_cipher, U_lwe_cipher, nslots);
            timer.stop();
            log_time_mem("  L3 s2c_repack_1D(22)", t);
        }

        // ─────────────────────────────────────────────────────────────
        //  LAYER 2 – DELTA (BACKWARD)
        //  δ₂ = (W3ᵀ · δ₃) ⊙ A'(U2)
        // ─────────────────────────────────────────────────────────────
        log_msg("--- Layer 2 DELTA ---");
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.delta_1D_dense_1D(pp, W3, DE3_cipher, U2_cipher, DE2_cipher, 3, epoch);
            timer.stop();
            log_time_mem("  L2 delta_1D_dense_1D(W3:22x22)", t);
        }
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.s2c_repack_1D(DE2_cipher, U_lwe_cipher, nslots);
            timer.stop();
            log_time_mem("  L2 s2c_repack_1D(22)", t);
        }

        // ─────────────────────────────────────────────────────────────
        //  LAYER 1 – DELTA (BACKWARD)
        //  δ₁ = (W2ᵀ · δ₂) ⊙ A'(U1)
        // ─────────────────────────────────────────────────────────────
        log_msg("--- Layer 1 DELTA ---");
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.delta_1D_dense_1D(pp, W2, DE2_cipher, U1_cipher, DE1_cipher, 2, epoch);
            timer.stop();
            log_time_mem("  L1 delta_1D_dense_1D(W2:22x22)", t);
        }
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.s2c_repack_1D(DE1_cipher, U_lwe_cipher, nslots);
            timer.stop();
            log_time_mem("  L1 s2c_repack_1D(22)", t);
        }

        // ─────────────────────────────────────────────────────────────
        //  WEIGHT UPDATES
        // ─────────────────────────────────────────────────────────────
        log_msg("--- Layer 1 WEIGHT UPDATE ---");
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.update_model_dense(pp, D0_cipher, DE1_cipher, W1, B1, 5, 1, epoch);
            timer.stop();
            log_time_mem("  L1 update_model_dense(30x22)", t);
        }

        log_msg("--- Layer 2 WEIGHT UPDATE ---");
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.update_model_dense(pp, D1_cipher, DE2_cipher, W2, B2, 5, 2, epoch);
            timer.stop();
            log_time_mem("  L2 update_model_dense(22x22)", t);
        }

        log_msg("--- Layer 3 WEIGHT UPDATE ---");
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.update_model_dense(pp, D2_cipher, DE3_cipher, W3, B3, 5, 3, epoch);
            timer.stop();
            log_time_mem("  L3 update_model_dense(22x22)", t);
        }

        log_msg("--- Layer 4 WEIGHT UPDATE ---");
        {
            double t = 0.0;
            AutoTimer timer(&t);
            pg_rt.update_model_dense(pp, D3_cipher, DE4_cipher, W4, B4, 5, 4, epoch);
            timer.stop();
            log_time_mem("  L4 update_model_dense(22x1)", t);
        }

        epoch_timer.stop();
        log_time_mem("EPOCH " + std::to_string(epoch) + " TOTAL", epoch_time);

        // ─────────────────────────────────────────────────────────────
        //  PLAINTEXT EVALUATION  (using decrypted weights on TEST set)
        // ─────────────────────────────────────────────────────────────
        log_msg("--- EVALUATION (plaintext, test set) ---");
        {
            int correct = 0;
            for (int k = 0; k < n_test; ++k)
            {
                // Layer 1: ReLU(x·W1 + B1)
                vector<double> h1(l1, 0.0);
                for (int j = 0; j < l1; ++j)
                {
                    double sum = B1[j];
                    for (int i = 0; i < l0; ++i)
                        sum += x_test[i][k] * W1[i][j];
                    h1[j] = pt_relu(sum);
                }
                // Layer 2: ReLU(h1·W2 + B2)
                vector<double> h2(l2, 0.0);
                for (int j = 0; j < l2; ++j)
                {
                    double sum = B2[j];
                    for (int i = 0; i < l1; ++i)
                        sum += h1[i] * W2[i][j];
                    h2[j] = pt_relu(sum);
                }
                // Layer 3: ReLU(h2·W3 + B3)  — NEW
                vector<double> h3(l3, 0.0);
                for (int j = 0; j < l3; ++j)
                {
                    double sum = B3[j];
                    for (int i = 0; i < l2; ++i)
                        sum += h2[i] * W3[i][j];
                    h3[j] = pt_relu(sum);
                }
                // Layer 4: Sigmoid(h3·W4 + B4)
                double out = B4[0];
                for (int i = 0; i < l3; ++i)
                    out += h3[i] * W4[i][0];
                double pred = pt_sigmoid(out);

                double label = y_test_raw[k];
                double pred_class = (pred >= 0.5) ? 1.0 : 0.0;
                if (pred_class == label)
                    ++correct;
            }
            double accuracy = 1.0 * correct / n_test;
            log_msg("  Epoch " + std::to_string(epoch) + " | Test Accuracy: " + std::to_string(accuracy) + " (" + std::to_string(correct) + "/" + std::to_string(n_test) + ")");
        }
    }

    // ==================================================================
    //  FINAL SUMMARY
    // ==================================================================
    log_msg("=========================================");
    log_msg("            TRAINING COMPLETE            ");
    log_msg("=========================================");
    log_msg("TOTAL SAVE MODEL TIME  " + std::to_string(total_save_model_time) + " s");
    log_msg("TOTAL LOAD MODEL TIME  " + std::to_string(total_load_model_time) + " s");
    log_msg("TOTAL OFFLINE TIME     " + std::to_string(total_offline_time) + " s");
    log_msg("TOTAL ONLINE TIME      " + std::to_string(total_online_time) + " s");
    log_msg("Final RSS: " + std::to_string(get_rss_gb()) + " GB");

    if (g_logfile.is_open())
        g_logfile.close();
    return 0;
}
