/*
 * model_b_bc_dual_rt.cpp
 * ======================
 * Pipeline-parallel HE training with TWO SEPARATE PegasusRunTime objects.
 *
 * MOTIVATION:
 *   The previous pipeline experiment (model_b_bc_parallel.cpp) shared a
 *   single PegasusRunTime between two std::threads.  Shared-resource
 *   contention caused each operation to run ~2× slower with 12 threads
 *   instead of 24:
 *
 *     1. MemoryManager::switch_mutex_ — static mutex serialises
 *        SwitchProfile() calls  (mitigated: GetMMProf now thread_local).
 *     2. Encryptor PRNG — internal mutex in SEAL's UniformRandomGenerator
 *        serialises concurrent Encrypt() calls.
 *     3. ThreadPool explosion — DEFINE_MT creates ThreadPool(num_threads_)
 *        on every LUT call.  With 12-thread shared runtime, each call
 *        spawned only 12 workers; with 24-thread runtime, each call
 *        spawns 24 workers.  Under pipeline parallelism both threads may
 *        invoke LUT simultaneously, yielding 48 workers — acceptable on a
 *        machine with sufficient cores or as a theoretical upper bound.
 *
 *   This test creates TWO PegasusRunTime objects  (pg_rt_A, pg_rt_B)
 *   sharing the same cryptographic keys but each with:
 *     - its own SEAL Encryptor (fresh PRNG — no contention)
 *     - its own Evaluator / Decryptor
 *     - its own ThreadPool sizing (num_threads_=24)
 *   Ciphertexts remain interoperable because the keys are identical
 *   (transferred via SaveKey/LoadKey during clone construction).
 *
 * THREAD COUNT ANALYSIS (per concurrent invocation):
 *   Each PegasusRunTime has num_threads_=24.
 *   Thread A calls omp_set_num_threads(24) → OMP parallel regions use 24.
 *   Thread B calls omp_set_num_threads(24) → OMP parallel regions use 24.
 *   DEFINE_MT creates ThreadPool(24) per invocation.
 *   Peak thread count when both modules execute LUT simultaneously:
 *     2 std::threads + 2×24 ThreadPool workers + 2×24 OMP workers = 98
 *   On a 24-core machine this is ~4× oversubscribed, but the goal is to
 *   measure the THEORETICAL speedup assuming sufficient compute resources:
 *   what operation speeds does each module achieve with 24 threads?
 *
 * BUFFER PROTOCOL:
 *   D1PipelineBuffer with double-buffer + consumption counter.
 *   Thread B deep-copies D1 from buffer then signals consumed.
 *   Safe for arbitrary TOTAL_STEPS.
 *
 * Architecture:  30 → 22 (ReLU) → 22 (ReLU) → 22 (ReLU) → 1 (Sigmoid)
 *   + Local Classifier A:  22 → 1 (Sigmoid)
 *
 * Module A  =  L1 forward  +  Local Classifier AL  +  backprop + update
 * Module B  =  L2→L3→L4 forward  +  L4→L3→L2 backprop + update
 *
 * CSV + text logging for post-hoc Gantt diagram and analysis.
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
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <iomanip>
#include <sys/stat.h>

#include "pegasus/pegasus_runtime.h"
#include "pegasus/timer.h"

using namespace std;
using namespace gemini;

// ── Tuneable constants ──────────────────────────────────────────────
static constexpr int NUM_THREADS = 8; // FULL machine per module (theoretical)
static constexpr int TOTAL_STEPS = 2; // training steps — arbitrary

// ── Globals required by PegasusRunTime (defined in pegasus_runtime.cc) ──
extern std::string save_model_loc;
extern std::string model_name;
extern thread_local bool not_first_epoch;
extern thread_local double total_save_model_time;
extern thread_local double total_load_model_time;
extern thread_local double total_offline_time;
extern thread_local double total_online_time;

// =====================================================================
//  UTILITIES
// =====================================================================
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

static void mkdirs(const std::string &path)
{
    std::string partial;
    for (char c : path)
    {
        partial += c;
        if (c == '/')
            mkdir(partial.c_str(), 0755);
    }
    if (!partial.empty())
        mkdir(partial.c_str(), 0755);
}

static inline double pt_relu(double x) { return x > 0.0 ? x : 0.0; }
static inline double pt_sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// =====================================================================
//  THREAD-SAFE TEXT LOGGER
// =====================================================================
struct TSLogger
{
    std::mutex mtx;
    std::ofstream logfile;

    void log(const std::string &tag, const std::string &msg)
    {
        time_t now = time(nullptr);
        char ts[32];
        strftime(ts, sizeof(ts), "%H:%M:%S", localtime(&now));
        std::string line = std::string("[") + ts + "][" + tag + "] " + msg;
        std::lock_guard<std::mutex> lk(mtx);
        std::cout << line << std::endl;
        if (logfile.is_open())
            logfile << line << std::endl;
    }

    void log_time_mem(const std::string &tag, const std::string &label,
                      double secs)
    {
        char buf[300];
        snprintf(buf, sizeof(buf), "%-44s %10.1f s | RSS %.1f GB",
                 label.c_str(), secs, get_rss_gb());
        log(tag, buf);
    }
};

// =====================================================================
//  CSV OPERATION RECORDER (thread-safe)
// =====================================================================
using clk = std::chrono::steady_clock;

struct OpRecord
{
    std::string thread;    // "main", "A", "B"
    std::string module;    // "A", "B", "MAIN"
    std::string operation; // e.g. "dense", "ReLU", "DReLU", "update" ...
    int layer;
    int step;
    std::string phase;     // "FF", "BP", "UPDATE", "EVAL", "SETUP"
    std::string par_group; // "L1_A_s0", "PAR_A_s0", "PAR_B_s0", "EVAL_s0"
    double start_s;
    double end_s;
    double duration_s;
    double rss_gb;
};

struct CSVRecorder
{
    std::mutex mtx;
    std::vector<OpRecord> records;
    clk::time_point t0;

    CSVRecorder() : t0(clk::now()) {}

    double now_s() const
    {
        return std::chrono::duration<double>(clk::now() - t0).count();
    }

    void add(const OpRecord &r)
    {
        std::lock_guard<std::mutex> lk(mtx);
        records.push_back(r);
    }

    void write(const std::string &path)
    {
        std::ofstream f(path);
        f << "thread,module,operation,layer,step,phase,parallel_group,"
             "start_s,end_s,duration_s,rss_gb\n";
        for (auto &r : records)
            f << r.thread << "," << r.module << "," << r.operation << ","
              << r.layer << "," << r.step << "," << r.phase << ","
              << r.par_group << ","
              << std::fixed << std::setprecision(1)
              << r.start_s << "," << r.end_s << ","
              << r.duration_s << "," << std::setprecision(2)
              << r.rss_gb << "\n";
    }
};

// ── Helper: execute + time + record an HE operation ─────────────────
static void timed_op(CSVRecorder &csv, TSLogger &tlog,
                     const std::string &thr, const std::string &mod,
                     const std::string &op, int layer, int step,
                     const std::string &phase, const std::string &pg,
                     std::function<void()> fn)
{
    double t_start = csv.now_s();
    tlog.log(thr, ">> " + op + " L" + std::to_string(layer));

    fn(); // ← the actual HE operation

    double t_end = csv.now_s();
    double dur = t_end - t_start;
    double rss = get_rss_gb();

    tlog.log_time_mem(thr, "   " + op + " L" + std::to_string(layer), dur);
    csv.add({thr, mod, op, layer, step, phase, pg,
             t_start, t_end, dur, rss});
}

// =====================================================================
//  FLEXIBLE DOUBLE-BUFFER WITH CONSUMPTION COUNTER
// =====================================================================
//
//  Protocol for arbitrary TOTAL_STEPS:
//
//  buf[0] used for even steps (0, 2, 4, ...)
//  buf[1] used for odd  steps (1, 3, 5, ...)
//
//  Thread A (producer):
//    Before writing buf[step%2]:
//      wait until d1_consumed_step >= step - 2
//      (ensures B finished reading the PREVIOUS use of this same slot)
//      For step 0,1 this is trivially true (d1_consumed_step starts at -1 >= -2, -1)
//    After writing:
//      set d1_ready_step = step, notify B
//
//  Thread B (consumer):
//    Before reading buf[step%2]:
//      wait until d1_ready_step >= step
//    After consuming (deep-copy to local + L2 dense done):
//      set d1_consumed_step = step, notify A
//
struct D1PipelineBuffer
{
    std::mutex mtx;
    std::condition_variable cv;
    int d1_ready_step = -1;    // A sets after writing (B waits on this)
    int d1_consumed_step = -1; // B sets after consuming (A waits on this)

    // Thread A: block until buf[step%2] is safe to overwrite
    void wait_buf_free(int step)
    {
        std::unique_lock<std::mutex> lk(mtx);
        int need_consumed = step - 2;
        cv.wait(lk, [&]
                { return d1_consumed_step >= need_consumed; });
    }

    // Thread A: signal that D1 for `step` is ready in buf[step%2]
    void signal_ready(int step)
    {
        {
            std::lock_guard<std::mutex> lk(mtx);
            d1_ready_step = step;
        }
        cv.notify_all();
    }

    // Thread B: block until D1 for `step` is available
    void wait_ready(int step)
    {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [&]
                { return d1_ready_step >= step; });
    }

    // Thread B: signal that buf[step%2] has been consumed (B finished reading)
    void signal_consumed(int step)
    {
        {
            std::lock_guard<std::mutex> lk(mtx);
            d1_consumed_step = step;
        }
        cv.notify_all();
    }
};

// =====================================================================
//                             MAIN
// =====================================================================
int main(int /*argc*/, char * /*argv*/[])
{
    // ── Open log file ───────────────────────────────────────────────
    std::string log_dir = "./logs";
    mkdirs(log_dir);
    TSLogger tlog;
    {
        time_t now = time(nullptr);
        char buf[64];
        strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&now));
        std::string log_path = log_dir + "/model_b_dual_" + std::string(buf) + ".log";
        tlog.logfile.open(log_path, std::ios::out);
        tlog.log("MAIN", "Log file: " + log_path);
    }

    CSVRecorder csv;
    tlog.log("MAIN", "=== DUAL-RUNTIME pipeline-parallel experiment ===");
    tlog.log("MAIN", "Two separate PegasusRunTime objects (shared keys, independent Encryptor)");
    tlog.log("MAIN", "Threads per module: " + std::to_string(NUM_THREADS) + " (full machine each, intentionally oversubscribed)");
    tlog.log("MAIN", "Total steps: " + std::to_string(TOTAL_STEPS));

    // ── Pegasus parameter setup ─────────────────────────────────────
    PegasusRunTime::Parms pp;
    pp.lvl0_lattice_dim = lwe::params::n(); // 1024
    pp.lvl1_lattice_dim = 1 << 12;          // 4096
    pp.lvl2_lattice_dim = 1 << 16;          // 65536
    pp.nlevels = 8;
    pp.scale = std::pow(2., 40);
    pp.nslots = 512;
    pp.s2c_multiplier = 1.;
    pp.enable_repacking = true;

    tlog.log("MAIN", "Arch: 30->22(ReLU)->22(ReLU)->22(ReLU)->1(Sigmoid) + LocalClf 22->1(Sigmoid)");

    // ── Create PRIMARY runtime (pg_rt_A) ────────────────────────────
    tlog.log("MAIN", "Creating pg_rt_A (primary — full keygen, " + std::to_string(NUM_THREADS) + " threads) ...");
    double setup_start = csv.now_s();
    PegasusRunTime pg_rt_A(pp, NUM_THREADS);
    omp_set_num_threads(NUM_THREADS);
    double setup_a_end = csv.now_s();
    tlog.log_time_mem("MAIN", "SETUP pg_rt_A (keygen + context)", setup_a_end - setup_start);
    csv.add({"main", "MAIN", "setup_A", 0, -1, "SETUP", "SETUP_A",
             setup_start, setup_a_end, setup_a_end - setup_start, get_rss_gb()});

    tlog.log("MAIN", "RSS after pg_rt_A: " + to_string(get_rss_gb()) + " GB");

    // ── Clone SECONDARY runtime (pg_rt_B) ───────────────────────────
    //   Shares crypto keys with pg_rt_A but gets a fresh SEAL runtime:
    //     - Own Encryptor (fresh PRNG — no contention)
    //     - Own Evaluator / Decryptor
    //     - Own num_threads_ = 24
    //   Evaluation keys (lutEvalKey, repackKey, KS keys) are deep-copied.
    //   Functors (BetterSine, LinearTransformer, LUTFunctor) are shared
    //   via shared_ptr — their methods are const/read-only.
    tlog.log("MAIN", "Cloning pg_rt_B (shared keys, fresh Encryptor, " + std::to_string(NUM_THREADS) + " threads) ...");
    double clone_start = csv.now_s();
    PegasusRunTime pg_rt_B(pg_rt_A, NUM_THREADS);
    double clone_end = csv.now_s();
    tlog.log_time_mem("MAIN", "CLONE pg_rt_B", clone_end - clone_start);
    csv.add({"main", "MAIN", "clone_B", 0, -1, "SETUP", "SETUP_B",
             clone_start, clone_end, clone_end - clone_start, get_rss_gb()});

    tlog.log("MAIN", "RSS after pg_rt_B clone: " + to_string(get_rss_gb()) + " GB");
    double setup_end = csv.now_s();
    csv.add({"main", "MAIN", "setup_total", 0, -1, "SETUP", "SETUP",
             setup_start, setup_end, setup_end - setup_start, get_rss_gb()});

    // ── Model save location ─────────────────────────────────────────
    save_model_loc = "./model/model_b_dual/";
    model_name = "dual";
    mkdirs(save_model_loc);
    tlog.log("MAIN", "Checkpoint dir: " + save_model_loc);

    // ── Network dimensions ──────────────────────────────────────────
    const int l0 = 30; // input
    const int l1 = 22; // hidden 1
    const int l2 = 22; // hidden 2
    const int l3 = 22; // hidden 3
    const int l4 = 1;  // output
    const int lAL = 1; // local classifier output
    const int nslots = pp.nslots;
    const int n_train = 455;
    const int n_test = 114;

    not_first_epoch = false;

    // ── Read Breast-Cancer dataset ──────────────────────────────────
    tlog.log("MAIN", "Reading Breast Cancer dataset ...");
    vector<vector<F64>> x_train(l0, vector<F64>(nslots, 0.0));
    vector<vector<F64>> x_test(l0, vector<F64>(nslots, 0.0));
    vector<F64> y_train_raw(nslots, 0.0);
    vector<F64> y_test_raw(nslots, 0.0);

    {
        ifstream fin("../dataset/breast_cancer/x_train.out");
        if (!fin.is_open())
        {
            tlog.log("MAIN", "ERROR: cannot open x_train.out");
            return 1;
        }
        for (int j = 0; j < n_train; ++j)
            for (int i = 0; i < l0; ++i)
                fin >> x_train[i][j];
    }
    {
        ifstream fin("../dataset/breast_cancer/x_test.out");
        if (!fin.is_open())
        {
            tlog.log("MAIN", "ERROR: cannot open x_test.out");
            return 1;
        }
        for (int j = 0; j < n_test; ++j)
            for (int i = 0; i < l0; ++i)
                fin >> x_test[i][j];
    }
    {
        ifstream fin("../dataset/breast_cancer/y_train.out");
        if (!fin.is_open())
        {
            tlog.log("MAIN", "ERROR: cannot open y_train.out");
            return 1;
        }
        for (int j = 0; j < n_train; ++j)
            fin >> y_train_raw[j];
    }
    {
        ifstream fin("../dataset/breast_cancer/y_test.out");
        if (!fin.is_open())
        {
            tlog.log("MAIN", "ERROR: cannot open y_test.out");
            return 1;
        }
        for (int j = 0; j < n_test; ++j)
            fin >> y_test_raw[j];
    }
    tlog.log("MAIN", "Dataset loaded: train=" + to_string(n_train) + " test=" + to_string(n_test) + " features=" + to_string(l0));

    // ── Cipher-domain vectors ───────────────────────────────────────
    // Module A intermediates (Thread A only — uses pg_rt_A)
    vector<Ctx> D0_cipher(l0);
    vector<Ctx> U1_cipher(l1);
    vector<Ctx> D1_cipher(l1); // Thread A's own L1 output

    // Double buffer: D1 handoff from A → B
    vector<Ctx> D1_buffer_0(l1); // even steps
    vector<Ctx> D1_buffer_1(l1); // odd steps

    // Thread B local copy of D1 (deep-copied from buffer, safe to use while A writes next)
    vector<Ctx> D1_local_B(l1);

    // Labels (encrypted with pg_rt_A — compatible with both runtimes)
    vector<vector<F64>> Y(l4, vector<F64>(nslots, 0.0));
    vector<Ctx> Y_cipher(l4);

    // ── Weights ─────────────────────────────────────────────────────
    vector<vector<F64>> W1(l0, vector<F64>(l1, 0.0));
    vector<F64> B1(l1, 0.0);

    // Local Classifier AL (22→1)
    vector<vector<F64>> W_AL(l1, vector<F64>(lAL, 0.0));
    vector<F64> B_AL(lAL, 0.0);

    // Module B weights
    vector<vector<F64>> W2(l1, vector<F64>(l2, 0.0));
    vector<F64> B2(l2, 0.0);
    vector<vector<F64>> W3(l2, vector<F64>(l3, 0.0));
    vector<F64> B3(l3, 0.0);
    vector<vector<F64>> W4(l3, vector<F64>(l4, 0.0));
    vector<F64> B4(l4, 0.0);

    // ── Module A intermediates ──────────────────────────────────────
    vector<Ctx> U_AL_cipher(lAL);
    vector<Ctx> D_AL_cipher(lAL);
    vector<Ctx> DE_AL_cipher(lAL);
    vector<Ctx> U1_deriv_cipher(l1); // DReLU(L1) output
    vector<Ctx> DE1_cipher(l1);

    // ── Module B intermediates ──────────────────────────────────────
    vector<Ctx> U2_cipher(l2), D2_cipher(l2), DE2_cipher(l2);
    vector<Ctx> U3_cipher(l3), D3_cipher(l3), DE3_cipher(l3);
    vector<Ctx> U4_cipher(l4), D4_cipher(l4), DE4_cipher(l4);
    vector<Ctx> U2_deriv_cipher(l2); // DReLU(L2) output
    vector<Ctx> U3_deriv_cipher(l3); // DReLU(L3) output

    // ── LWE buffers (SEPARATE per module to avoid data races) ───────
    const int max_dim = std::max({l1, l2, l3, l4, lAL});

    // Module A scratch
    vector<vector<lwe::Ctx_st>> U_lwe_a(max_dim, vector<lwe::Ctx_st>(nslots));
    vector<vector<lwe::Ctx_st>> D_lwe_a(max_dim, vector<lwe::Ctx_st>(nslots));
    // Saved S2C output for deferred L1 DReLU
    vector<vector<lwe::Ctx_st>> L1_lwe_saved(l1, vector<lwe::Ctx_st>(nslots));

    // Module B scratch
    vector<vector<lwe::Ctx_st>> U_lwe_b(max_dim, vector<lwe::Ctx_st>(nslots));
    vector<vector<lwe::Ctx_st>> D_lwe_b(max_dim, vector<lwe::Ctx_st>(nslots));
    // Saved S2C output for deferred L2/L3 DReLU
    vector<vector<lwe::Ctx_st>> L2_lwe_saved(l2, vector<lwe::Ctx_st>(nslots));
    vector<vector<lwe::Ctx_st>> L3_lwe_saved(l3, vector<lwe::Ctx_st>(nslots));

    // ── Initialise weights (Xavier-like, ±0.3) ─────────────────────
    tlog.log("MAIN", "Initialising weights ...");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.3, 0.3);

    for (int i = 0; i < l0; ++i)
        for (int j = 0; j < l1; ++j)
            W1[i][j] = dis(gen);
    for (int j = 0; j < l1; ++j)
        B1[j] = dis(gen);

    for (int i = 0; i < l1; ++i)
        for (int j = 0; j < lAL; ++j)
            W_AL[i][j] = dis(gen);
    for (int j = 0; j < lAL; ++j)
        B_AL[j] = dis(gen);

    for (int i = 0; i < l1; ++i)
        for (int j = 0; j < l2; ++j)
            W2[i][j] = dis(gen);
    for (int j = 0; j < l2; ++j)
        B2[j] = dis(gen);
    for (int i = 0; i < l2; ++i)
        for (int j = 0; j < l3; ++j)
            W3[i][j] = dis(gen);
    for (int j = 0; j < l3; ++j)
        B3[j] = dis(gen);
    for (int i = 0; i < l3; ++i)
        for (int j = 0; j < l4; ++j)
            W4[i][j] = dis(gen);
    for (int j = 0; j < l4; ++j)
        B4[j] = dis(gen);

    // ── Encrypt inputs (once, using pg_rt_A) ────────────────────────
    //   Both runtimes share the same keys → ciphertexts are interoperable.
    //   Thread B uses pg_rt_B's Evaluator on these ciphertexts — works
    //   because both SEAL contexts have identical parameters and keys.
    tlog.log("MAIN", "Encrypting D0 (30 cts) via pg_rt_A ...");
    {
        vector<vector<F64>> D0_plain(l0, vector<F64>(nslots, 0.0));
        double t = 0.0;
        AutoTimer timer(&t);
        for (int i = 0; i < l0; ++i)
            D0_plain[i] = x_train[i];
#pragma omp parallel for
        for (int i = 0; i < l0; ++i)
            CHECK_AND_ABORT(pg_rt_A.EncodeThenEncrypt(D0_plain[i], D0_cipher[i]));
        timer.stop();
        tlog.log_time_mem("MAIN", "ENCRYPT D0", t);
    }

    // ── Encrypt labels (once, via pg_rt_A, with S2C+repack refresh) ─
    tlog.log("MAIN", "Encrypting Y (1 ct) via pg_rt_A ...");
    {
        double t = 0.0;
        AutoTimer timer(&t);
        for (int j = 0; j < nslots; ++j)
            Y[0][j] = y_train_raw[j];
        for (int i = 0; i < l4; ++i)
            CHECK_AND_ABORT(pg_rt_A.EncodeThenEncrypt(Y[i], Y_cipher[i]));

        // Refresh levels via S2C → extract → repack (using Module A's LWE scratch)
        pg_rt_A.s2c_and_extract(Y_cipher, U_lwe_a, l4, nslots);
        pg_rt_A.repack(U_lwe_a, Y_cipher, l4, nslots);
        timer.stop();
        tlog.log_time_mem("MAIN", "ENCRYPT Y + refresh", t);
    }

    tlog.log("MAIN", "Peak RSS after full setup: " + to_string(get_rss_gb()) + " GB");

    // ==================================================================
    //                  TRAINING LOOP  (pipeline-parallel)
    // ==================================================================
    //  Flexible buffer via D1PipelineBuffer (consumption counter).
    D1PipelineBuffer d1_pipe;

    tlog.log("MAIN", "Launching pipeline — Thread A (pg_rt_A) & Thread B (pg_rt_B)");
    tlog.log("MAIN", "Buffer protocol: double-buffer + consumption counter (arbitrary steps)");
    tlog.log("MAIN", "Total steps: " + std::to_string(TOTAL_STEPS));
    double pipeline_start = csv.now_s();

    // ══════════════════════════════════════════════════════════════════
    //  THREAD A: uses pg_rt_A exclusively
    //  L1 forward + Module A (local clf + L1 backprop)
    //  Runs its OWN step loop.  Only halts at copy step if buf not consumed.
    // ══════════════════════════════════════════════════════════════════
    std::thread thread_a([&]()
                         {
        omp_set_num_threads(NUM_THREADS);
        not_first_epoch = false; // thread_local: Thread A's own copy

        for (int step = 0; step < TOTAL_STEPS; ++step)
        {
            int epoch = step;
            if (step > 0) not_first_epoch = true;

            std::string pg_l1 = "L1_A_s" + to_string(step);
            std::string pg_a  = "PAR_A_s" + to_string(step);

            tlog.log("A", "=========== STEP " + to_string(step) + " ===========");
            double step_start = csv.now_s();

            // ─── L1 Forward ──────────────────────────────────────────
            timed_op(csv, tlog, "A", "A", "dense", 1, step, "FF", pg_l1, [&]
                     { pg_rt_A.dense(pp, D0_cipher, W1, B1, U1_cipher, 1, epoch); });

            timed_op(csv, tlog, "A", "A", "s2c_extract", 1, step, "FF", pg_l1, [&]
                     { pg_rt_A.s2c_and_extract(U1_cipher, U_lwe_a, l1, nslots); });

            timed_op(csv, tlog, "A", "A", "save_lwe", 1, step, "FF", pg_l1, [&]
                     {
                         for (int i = 0; i < l1; ++i)
                             L1_lwe_saved[i] = U_lwe_a[i];
                     });

            timed_op(csv, tlog, "A", "A", "ReLU", 1, step, "FF", pg_l1, [&]
                     { pg_rt_A.act_batch(U_lwe_a, D_lwe_a, D1_cipher, l1, nslots, "ReLU"); });

            // ─── Wait for buf[step%2] to be free ─────────────────────
            timed_op(csv, tlog, "A", "A", "wait_buf_free", 1, step, "SYNC", pg_l1, [&]
                     { d1_pipe.wait_buf_free(step); });

            // ─── Copy D1 → double buffer + SIGNAL Thread B ──────────
            auto& D1_buf = (step % 2 == 0) ? D1_buffer_0 : D1_buffer_1;
            timed_op(csv, tlog, "A", "A", "copy_D1_buffer", 1, step, "FF", pg_l1, [&]
                     {
                         for (int i = 0; i < l1; ++i)
                             D1_buf[i] = D1_cipher[i];
                     });
            d1_pipe.signal_ready(step);
            tlog.log("A", "D1 buffer ready -> signaled step " + to_string(step));

            // ─── AL Forward: dense(22→1) → S2C → Sigmoid ────────────
            timed_op(csv, tlog, "A", "A", "dense", 10, step, "FF", pg_a, [&]
                     { pg_rt_A.dense(pp, D1_cipher, W_AL, B_AL, U_AL_cipher, 10, epoch); });

            timed_op(csv, tlog, "A", "A", "s2c_extract", 10, step, "FF", pg_a, [&]
                     { pg_rt_A.s2c_and_extract(U_AL_cipher, U_lwe_a, lAL, nslots); });

            timed_op(csv, tlog, "A", "A", "Sigmoid", 10, step, "FF", pg_a, [&]
                     { pg_rt_A.act_batch(U_lwe_a, D_lwe_a, D_AL_cipher, lAL, nslots, "Sigmoid"); });

            // ─── AL Delta:  DE_AL = D_AL − Y ────────────────────────
            timed_op(csv, tlog, "A", "A", "delta_softmax", 10, step, "BP", pg_a, [&]
                     { pg_rt_A.delta_softmax(pp, D_AL_cipher, Y_cipher, DE_AL_cipher); });

            // ─── L1 DReLU (deferred, from saved LWE) ────────────────
            timed_op(csv, tlog, "A", "A", "DReLU", 1, step, "BP", pg_a, [&]
                     { pg_rt_A.act_batch(L1_lwe_saved, D_lwe_a, U1_deriv_cipher, l1, nslots, "DReLU"); });

            // ─── L1 Delta:  DE1 = (W_AL^T · DE_AL) ⊙ DReLU(U1) ─────
            timed_op(csv, tlog, "A", "A", "delta_1D_dense_1D", 1, step, "BP", pg_a, [&]
                     { pg_rt_A.delta_1D_dense_1D(pp, W_AL, DE_AL_cipher, U1_deriv_cipher,
                                               DE1_cipher, 10, epoch); });

            // ─── L1 S2C+Repack (refresh levels for weight update) ────
            timed_op(csv, tlog, "A", "A", "s2c_repack_1D", 1, step, "BP", pg_a, [&]
                     { pg_rt_A.s2c_repack_1D(DE1_cipher, U_lwe_a, nslots); });

            // ─── Weight Updates ──────────────────────────────────────
            timed_op(csv, tlog, "A", "A", "update_W_AL", 10, step, "UPDATE", pg_a, [&]
                     { pg_rt_A.update_model_dense(pp, D1_cipher, DE_AL_cipher,
                                                W_AL, B_AL, 5, 10, epoch); });

            timed_op(csv, tlog, "A", "A", "update_W1", 1, step, "UPDATE", pg_a, [&]
                     { pg_rt_A.update_model_dense(pp, D0_cipher, DE1_cipher,
                                                W1, B1, 5, 1, epoch); });

            double step_end = csv.now_s();
            double step_dur = step_end - step_start;
            tlog.log_time_mem("A", "Thread A STEP " + to_string(step) + " TOTAL", step_dur);
            csv.add({"A", "A", "step_total_A", 0, step, "STEP", "STEP_A_s" + to_string(step),
                     step_start, step_end, step_dur, get_rss_gb()});

            // ─── Local Classifier eval (plaintext, Thread A exclusive) ─
            {
                int correct = 0;
                for (int k = 0; k < n_test; ++k)
                {
                    vector<double> h1(l1, 0.0);
                    for (int j = 0; j < l1; ++j)
                    {
                        double sum = B1[j];
                        for (int i = 0; i < l0; ++i)
                            sum += x_test[i][k] * W1[i][j];
                        h1[j] = pt_relu(sum);
                    }
                    double out_al = B_AL[0];
                    for (int i = 0; i < l1; ++i)
                        out_al += h1[i] * W_AL[i][0];
                    double pred_al = pt_sigmoid(out_al);
                    if (((pred_al >= 0.5) ? 1.0 : 0.0) == y_test_raw[k])
                        ++correct;
                }
                double acc = 1.0 * correct / n_test;
                tlog.log("A", "[LOCAL CLF A] Step " + to_string(step) + " Accuracy: "
                         + to_string(acc) + " (" + to_string(correct) + "/" + to_string(n_test) + ")");
                csv.add({"A", "A", "eval_local", 0, step, "EVAL", "EVAL_A_s" + to_string(step),
                         csv.now_s(), csv.now_s(), 0.0, get_rss_gb()});
            }
        }
        tlog.log("A", "Thread A FINISHED all steps | RSS " + to_string(get_rss_gb()) + " GB"); }); // end thread_a

    // ══════════════════════════════════════════════════════════════════
    //  THREAD B: uses pg_rt_B exclusively
    //  L2→L3→L4 forward + backward + updates
    //  Waits for D1, deep-copies to local, signals consumed.
    // ══════════════════════════════════════════════════════════════════
    std::thread thread_b([&]()
                         {
        omp_set_num_threads(NUM_THREADS);
        not_first_epoch = false; // thread_local: Thread B's own copy

        for (int step = 0; step < TOTAL_STEPS; ++step)
        {
            int epoch = step;
            if (step > 0) not_first_epoch = true;

            std::string pg_b = "PAR_B_s" + to_string(step);

            // ─── Wait for Thread A to produce D1 for this step ───────
            timed_op(csv, tlog, "B", "B", "wait_D1_ready", 1, step, "SYNC", pg_b, [&]
                     { d1_pipe.wait_ready(step); });

            auto& D1_buf = (step % 2 == 0) ? D1_buffer_0 : D1_buffer_1;

            tlog.log("B", "=========== STEP " + to_string(step) + " (D1 received) ===========");
            double step_start = csv.now_s();

            // ─── Deep-copy D1 from buffer → local + signal consumed ──
            timed_op(csv, tlog, "B", "B", "copy_D1_local", 1, step, "FF", pg_b, [&]
                     {
                         for (int i = 0; i < l1; ++i)
                             D1_local_B[i] = D1_buf[i];
                     });
            d1_pipe.signal_consumed(step);
            tlog.log("B", "D1 buffer consumed -> signaled step " + to_string(step));

            // ─── L2 Forward ──────────────────────────────────────────
            timed_op(csv, tlog, "B", "B", "dense", 2, step, "FF", pg_b, [&]
                     { pg_rt_B.dense(pp, D1_local_B, W2, B2, U2_cipher, 2, epoch); });

            timed_op(csv, tlog, "B", "B", "s2c_extract", 2, step, "FF", pg_b, [&]
                     { pg_rt_B.s2c_and_extract(U2_cipher, U_lwe_b, l2, nslots); });

            timed_op(csv, tlog, "B", "B", "save_lwe", 2, step, "FF", pg_b, [&]
                     {
                         for (int i = 0; i < l2; ++i)
                             L2_lwe_saved[i] = U_lwe_b[i];
                     });

            timed_op(csv, tlog, "B", "B", "ReLU", 2, step, "FF", pg_b, [&]
                     { pg_rt_B.act_batch(U_lwe_b, D_lwe_b, D2_cipher, l2, nslots, "ReLU"); });

            // ─── L3 Forward ──────────────────────────────────────────
            timed_op(csv, tlog, "B", "B", "dense", 3, step, "FF", pg_b, [&]
                     { pg_rt_B.dense(pp, D2_cipher, W3, B3, U3_cipher, 3, epoch); });

            timed_op(csv, tlog, "B", "B", "s2c_extract", 3, step, "FF", pg_b, [&]
                     { pg_rt_B.s2c_and_extract(U3_cipher, U_lwe_b, l3, nslots); });

            timed_op(csv, tlog, "B", "B", "save_lwe", 3, step, "FF", pg_b, [&]
                     {
                         for (int i = 0; i < l3; ++i)
                             L3_lwe_saved[i] = U_lwe_b[i];
                     });

            timed_op(csv, tlog, "B", "B", "ReLU", 3, step, "FF", pg_b, [&]
                     { pg_rt_B.act_batch(U_lwe_b, D_lwe_b, D3_cipher, l3, nslots, "ReLU"); });

            // ─── L4 Forward ──────────────────────────────────────────
            timed_op(csv, tlog, "B", "B", "dense", 4, step, "FF", pg_b, [&]
                     { pg_rt_B.dense(pp, D3_cipher, W4, B4, U4_cipher, 4, epoch); });

            timed_op(csv, tlog, "B", "B", "s2c_extract", 4, step, "FF", pg_b, [&]
                     { pg_rt_B.s2c_and_extract(U4_cipher, U_lwe_b, l4, nslots); });

            timed_op(csv, tlog, "B", "B", "Sigmoid", 4, step, "FF", pg_b, [&]
                     { pg_rt_B.act_batch(U_lwe_b, D_lwe_b, D4_cipher, l4, nslots, "Sigmoid"); });

            // ─── L4 Delta:  DE4 = D4 − Y ────────────────────────────
            timed_op(csv, tlog, "B", "B", "delta_softmax", 4, step, "BP", pg_b, [&]
                     { pg_rt_B.delta_softmax(pp, D4_cipher, Y_cipher, DE4_cipher); });

            // ─── L3 DReLU (deferred, from saved L3 LWE) ─────────────
            timed_op(csv, tlog, "B", "B", "DReLU", 3, step, "BP", pg_b, [&]
                     { pg_rt_B.act_batch(L3_lwe_saved, D_lwe_b, U3_deriv_cipher, l3, nslots, "DReLU"); });

            // ─── L3 Delta ────────────────────────────────────────────
            timed_op(csv, tlog, "B", "B", "delta_1D_dense_1D", 3, step, "BP", pg_b, [&]
                     { pg_rt_B.delta_1D_dense_1D(pp, W4, DE4_cipher, U3_deriv_cipher,
                                               DE3_cipher, 4, epoch); });

            timed_op(csv, tlog, "B", "B", "s2c_repack_1D", 3, step, "BP", pg_b, [&]
                     { pg_rt_B.s2c_repack_1D(DE3_cipher, U_lwe_b, nslots); });

            // ─── L2 DReLU (deferred, from saved L2 LWE) ─────────────
            timed_op(csv, tlog, "B", "B", "DReLU", 2, step, "BP", pg_b, [&]
                     { pg_rt_B.act_batch(L2_lwe_saved, D_lwe_b, U2_deriv_cipher, l2, nslots, "DReLU"); });

            // ─── L2 Delta ────────────────────────────────────────────
            timed_op(csv, tlog, "B", "B", "delta_1D_dense_1D", 2, step, "BP", pg_b, [&]
                     { pg_rt_B.delta_1D_dense_1D(pp, W3, DE3_cipher, U2_deriv_cipher,
                                               DE2_cipher, 3, epoch); });

            timed_op(csv, tlog, "B", "B", "s2c_repack_1D", 2, step, "BP", pg_b, [&]
                     { pg_rt_B.s2c_repack_1D(DE2_cipher, U_lwe_b, nslots); });

            // ─── Weight Updates (L4, L3, L2) ─────────────────────────
            timed_op(csv, tlog, "B", "B", "update_W4", 4, step, "UPDATE", pg_b, [&]
                     { pg_rt_B.update_model_dense(pp, D3_cipher, DE4_cipher,
                                                W4, B4, 5, 4, epoch); });

            timed_op(csv, tlog, "B", "B", "update_W3", 3, step, "UPDATE", pg_b, [&]
                     { pg_rt_B.update_model_dense(pp, D2_cipher, DE3_cipher,
                                                W3, B3, 5, 3, epoch); });

            timed_op(csv, tlog, "B", "B", "update_W2", 2, step, "UPDATE", pg_b, [&]
                     { pg_rt_B.update_model_dense(pp, D1_local_B, DE2_cipher,
                                                W2, B2, 5, 2, epoch); });

            double step_end = csv.now_s();
            double step_dur = step_end - step_start;
            tlog.log_time_mem("B", "Thread B STEP " + to_string(step) + " TOTAL", step_dur);
            csv.add({"B", "B", "step_total_B", 0, step, "STEP", "STEP_B_s" + to_string(step),
                     step_start, step_end, step_dur, get_rss_gb()});
        }
        tlog.log("B", "Thread B FINISHED all steps | RSS " + to_string(get_rss_gb()) + " GB"); }); // end thread_b

    // ── Wait for BOTH threads to complete ALL steps ─────────────────
    tlog.log("MAIN", "Pipeline running — waiting for both threads ...");
    thread_a.join();
    tlog.log("MAIN", "Thread A joined");
    thread_b.join();
    tlog.log("MAIN", "Thread B joined");

    double pipeline_end = csv.now_s();
    double pipeline_dur = pipeline_end - pipeline_start;
    tlog.log_time_mem("MAIN", "PIPELINE TOTAL (all steps)", pipeline_dur);
    csv.add({"main", "MAIN", "pipeline_total", 0, TOTAL_STEPS - 1, "STEP", "PIPELINE",
             pipeline_start, pipeline_end, pipeline_dur, get_rss_gb()});

    // ==================================================================
    //  EVALUATION (plaintext, test set) — after ALL steps complete
    // ==================================================================
    tlog.log("MAIN", "--- FINAL EVALUATION (plaintext, test set) ---");

    // Full model accuracy:  W1 → W2 → W3 → W4
    {
        int correct = 0;
        for (int k = 0; k < n_test; ++k)
        {
            vector<double> h1(l1, 0.0);
            for (int j = 0; j < l1; ++j)
            {
                double sum = B1[j];
                for (int i = 0; i < l0; ++i)
                    sum += x_test[i][k] * W1[i][j];
                h1[j] = pt_relu(sum);
            }
            vector<double> h2(l2, 0.0);
            for (int j = 0; j < l2; ++j)
            {
                double sum = B2[j];
                for (int i = 0; i < l1; ++i)
                    sum += h1[i] * W2[i][j];
                h2[j] = pt_relu(sum);
            }
            vector<double> h3(l3, 0.0);
            for (int j = 0; j < l3; ++j)
            {
                double sum = B3[j];
                for (int i = 0; i < l2; ++i)
                    sum += h2[i] * W3[i][j];
                h3[j] = pt_relu(sum);
            }
            double out = B4[0];
            for (int i = 0; i < l3; ++i)
                out += h3[i] * W4[i][0];
            double pred = pt_sigmoid(out);
            if (((pred >= 0.5) ? 1.0 : 0.0) == y_test_raw[k])
                ++correct;
        }
        double acc = 1.0 * correct / n_test;
        tlog.log("MAIN", "[FULL MODEL] Final Accuracy: " + to_string(acc) + " (" + to_string(correct) + "/" + to_string(n_test) + ")");
        csv.add({"main", "MAIN", "eval_full", 0, TOTAL_STEPS - 1, "EVAL", "EVAL_final",
                 csv.now_s(), csv.now_s(), 0.0, get_rss_gb()});
    }

    // Local classifier accuracy:  W1 → W_AL
    {
        int correct = 0;
        for (int k = 0; k < n_test; ++k)
        {
            vector<double> h1(l1, 0.0);
            for (int j = 0; j < l1; ++j)
            {
                double sum = B1[j];
                for (int i = 0; i < l0; ++i)
                    sum += x_test[i][k] * W1[i][j];
                h1[j] = pt_relu(sum);
            }
            double out_al = B_AL[0];
            for (int i = 0; i < l1; ++i)
                out_al += h1[i] * W_AL[i][0];
            double pred_al = pt_sigmoid(out_al);
            if (((pred_al >= 0.5) ? 1.0 : 0.0) == y_test_raw[k])
                ++correct;
        }
        double acc = 1.0 * correct / n_test;
        tlog.log("MAIN", "[LOCAL CLF A] Final Accuracy: " + to_string(acc) + " (" + to_string(correct) + "/" + to_string(n_test) + ")");
        csv.add({"main", "A", "eval_local", 0, TOTAL_STEPS - 1, "EVAL", "EVAL_final",
                 csv.now_s(), csv.now_s(), 0.0, get_rss_gb()});
    }

    // ==================================================================
    //  FINAL SUMMARY
    // ==================================================================
    tlog.log("MAIN", "=========================================");
    tlog.log("MAIN", "         TRAINING COMPLETE               ");
    tlog.log("MAIN", "=========================================");
    tlog.log("MAIN", "EXPERIMENT: DUAL-RUNTIME pipeline-parallel");
    tlog.log("MAIN", "  Threads per module: " + to_string(NUM_THREADS));
    tlog.log("MAIN", "  Total steps:        " + to_string(TOTAL_STEPS));
    tlog.log("MAIN", "  Separate runtimes:  pg_rt_A, pg_rt_B (shared keys)");
    tlog.log("MAIN", "NOTE: Timing globals are thread_local — per-thread copies.");
    tlog.log("MAIN", "Main-thread timing (setup only):");
    tlog.log("MAIN", "  TOTAL SAVE MODEL TIME  " + to_string(total_save_model_time) + " s");
    tlog.log("MAIN", "  TOTAL LOAD MODEL TIME  " + to_string(total_load_model_time) + " s");
    tlog.log("MAIN", "  TOTAL OFFLINE TIME     " + to_string(total_offline_time) + " s");
    tlog.log("MAIN", "  TOTAL ONLINE TIME      " + to_string(total_online_time) + " s");
    tlog.log("MAIN", "Final RSS: " + to_string(get_rss_gb()) + " GB");

    // Write CSV
    std::string csv_path = "./logs/dual_rt_ops.csv";
    csv.write(csv_path);
    tlog.log("MAIN", "CSV written: " + csv_path);

    if (tlog.logfile.is_open())
        tlog.logfile.close();
    return 0;
}
