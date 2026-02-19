#include "pegasus/pegasus_runtime.h"
#include "pegasus/timer.h"
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <cmath>
using namespace gemini;
using namespace seal;
using namespace std;

extern thread_local double tmp_time;
extern thread_local double total_save_model_time;
extern thread_local double total_load_model_time;
extern thread_local double total_offline_time;
extern thread_local double total_online_time;
extern thread_local double save_model_time;
extern thread_local double load_model_time;
extern thread_local double offline_time;
extern thread_local double online_time;
extern int num_threads;

void s2c_and_extract(PegasusRunTime &pg_rt, vector<Ctx> &U_cipher, vector<vector<lwe::Ctx_st>> &U_lwe_cipher, int length, int nslots)
{
    std::cout << "S2C & EXTRACT\n";
    tmp_time = 0.0;
    online_time = 0.0;
    AutoTimer online_timer(&tmp_time);

    int rounds = std::ceil(1.0 * length / num_threads);
    std::cout << "s2c rounds " << rounds << std::endl;
    for (int r = 0; r < rounds; r++)
    {
#pragma omp parallel for
        for (int i = 0; i < num_threads; i++)
        {
            int pos = r * num_threads + i;
            if (pos < length)
            {
                CHECK_AND_ABORT(pg_rt.SlotsToCoeffs(U_cipher[pos]));

                // use only if nslots = 2**(2n+1)
                if (GetNModuli(U_cipher[pos]) != 1)
                {
                    pg_rt.runtime_->DropModuli(&U_cipher[pos], GetNModuli(U_cipher[pos]) - 1);
                }
            }
        }
    }

    for (int i = 0; i < length; i++)
    {
        CHECK_AND_ABORT(pg_rt.ExtraAllCoefficients(U_cipher[i], U_lwe_cipher[i]));
    }

    online_timer.stop();
    online_time += tmp_time;
    total_online_time += tmp_time;
    std::cout << "S2C & EXTRACT TIME " << online_time << std::endl;
}

void repack(PegasusRunTime &pg_rt, vector<vector<lwe::Ctx_st>> &D_lwe_cipher, vector<Ctx> &D_cipher, int length, int nslots)
{
    std::cout << "REPACK\n";

#pragma omp parallel for
    for (int i = 0; i < length; i++)
    {
        CHECK_AND_ABORT(pg_rt.Repack(D_cipher[i], D_lwe_cipher[i]));
    }
    cout << endl;
}

void relu(PegasusRunTime &pg_rt, vector<vector<lwe::Ctx_st>> &U_lwe_cipher, vector<vector<lwe::Ctx_st>> &D_lwe_cipher, int length, int nslots)
{
    std::cout << "RELU\n";

    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < nslots; j++)
        {
            D_lwe_cipher[i][j] = U_lwe_cipher[i][j];
        }

        pg_rt.ReLU(D_lwe_cipher[i].data(), D_lwe_cipher[i].size());
    }
    cout << endl;
}

void drelu(PegasusRunTime &pg_rt, vector<vector<lwe::Ctx_st>> &U_lwe_cipher, int length, int nslots)
{
    std::cout << "DRELU\n";

    for (int i = 0; i < length; i++)
    {
        pg_rt.DReLU(U_lwe_cipher[i].data(), U_lwe_cipher[i].size());
    }
    cout << endl;
}

void act(PegasusRunTime &pg_rt, vector<vector<lwe::Ctx_st>> &U_lwe_cipher, int length, int nslots, string act_str)
{
    tmp_time = 0.0;
    online_time = 0.0;
    AutoTimer online_timer(&tmp_time);

    for (int i = 0; i < length; i++)
    {
        if (act_str == "ReLU")
        {
            pg_rt.ReLU(U_lwe_cipher[i].data(), U_lwe_cipher[i].size());
        }
        else if (act_str == "DReLU")
        {
            pg_rt.DReLU(U_lwe_cipher[i].data(), U_lwe_cipher[i].size());
        }
    }

    online_timer.stop();
    online_time += tmp_time;
    total_online_time += tmp_time;
    std::cout << "ACT TIME " << online_time << std::endl;
}

void act_batch(PegasusRunTime &pg_rt,
               vector<vector<lwe::Ctx_st>> &U_lwe_cipher,
               vector<vector<lwe::Ctx_st>> &D_lwe_cipher,
               vector<Ctx> &D_cipher,
               int length,
               int nslots,
               string act_str)
{
    int batch_size = num_threads;
    int len = batch_size;
    int rounds = std::ceil(1.0 * length / batch_size);
    std::cout << "act rounds " << rounds << std::endl;

    tmp_time = 0.0;
    online_time = 0.0;
    AutoTimer online_timer(&tmp_time);

    vector<Ctx> rlwe_cipher(batch_size);

    for (int r = 0; r < rounds; r++)
    {
        std::cout << r << std::endl;

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
            act(pg_rt, D_lwe_cipher, len, nslots, "ReLU");
        }
        else if (act_str == "DReLU")
        {
            act(pg_rt, D_lwe_cipher, len, nslots, "DReLU");
        }

        repack(pg_rt, D_lwe_cipher, rlwe_cipher, len, nslots);

        F64Vec T(nslots);
        for (int i = 0; i < batch_size; i++)
        {
            double cmp = pg_rt.DecryptLWE(D_lwe_cipher[i][0]);
            pg_rt.DecryptThenDecode(rlwe_cipher[i], T);
            std::cout << cmp << T[0] << std::endl;
        }

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
    std::cout << "ACT & REPACK TIME " << online_time << std::endl;
}

void Relu(PegasusRunTime &pg_rt, vector<vector<lwe::Ctx_st>> &U_lwe_cipher, vector<vector<lwe::Ctx_st>> &D_lwe_cipher, vector<Ctx> &D_cipher, int length, int nslots)
{
    tmp_time = 0.0;
    online_time = 0.0;
    AutoTimer online_timer(&tmp_time);

    relu(pg_rt, U_lwe_cipher, D_lwe_cipher, length, nslots);
    repack(pg_rt, D_lwe_cipher, D_cipher, length, nslots);

    online_timer.stop();
    online_time += tmp_time;
    total_online_time += tmp_time;

    std::cout << "RELU & REPACK TIME " << online_time << std::endl;
}

void Drelu(PegasusRunTime &pg_rt, vector<vector<lwe::Ctx_st>> &U_lwe_cipher, vector<Ctx> &U_cipher, int length, int nslots)
{
    tmp_time = 0.0;
    online_time = 0.0;
    AutoTimer online_timer(&tmp_time);

    drelu(pg_rt, U_lwe_cipher, length, nslots);
    repack(pg_rt, U_lwe_cipher, U_cipher, length, nslots);

    online_timer.stop();
    online_time += tmp_time;
    total_online_time += tmp_time;

    std::cout << "DRELU & REPACK TIME " << online_time << std::endl;
}

void Softmax(PegasusRunTime &pg_rt, vector<vector<lwe::Ctx_st>> &U_lwe_cipher, vector<vector<lwe::Ctx_st>> &D_lwe_cipher, vector<Ctx> &D_cipher, int length, int nslots)
{
    tmp_time = 0.0;
    online_time = 0.0;
    AutoTimer online_timer(&tmp_time);

    std::cout << "SOFTMAX\n";

    F64Vec log8(nslots);
    Ctx log8_ct;
    vector<lwe::Ctx_st> log8_lwe_ct;
    for (int i = 0; i < nslots; i++)
    {
        // log8[i] = 4.15888; // 1/64
        log8[i] = 3.46574; // 1/32
    }
    CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(log8, log8_ct));
    CHECK_AND_ABORT(pg_rt.SlotsToCoeffs(log8_ct));
    // use only if nslots = 2**7
    if (GetNModuli(log8_ct) != 1)
    {
        pg_rt.runtime_->DropModuli(&log8_ct, GetNModuli(log8_ct) - 1);
    }
    CHECK_AND_ABORT(pg_rt.ExtraAllCoefficients(log8_ct, log8_lwe_ct));

    vector<lwe::Ctx_st> SM_FLAT(length * nslots);
    vector<lwe::Ctx_st> SM_SUM(nslots);

    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < nslots; j++)
        {
            SM_FLAT[i * nslots + j] = U_lwe_cipher[i][j];
        }
    }

    // std::cout << "STEP 1: EXP\n";
    pg_rt.Exponent(SM_FLAT.data(), SM_FLAT.size());

    // std::cout << "STEP 2: Scale Exp\n";
#pragma omp parallel for
    for (int i = 0; i < length * nslots; i++)
    {
        vector<lwe::Ctx_st> lwe_ct(1);
        lwe_ct[0] = SM_FLAT[i];
        // pg_rt.MulConstant(lwe_ct.data(), 0.015625); // 1/64
        pg_rt.MulConstant(lwe_ct.data(), 0.03125); // 1/32
        SM_FLAT[i] = lwe_ct[0];
    }

    // std::cout << "STEP 3: EXP SUM\n";
#pragma omp parallel for
    for (int j = 0; j < nslots; j++)
    {
        lwe::Ctx_st tmp_lwe = SM_FLAT[j];
        for (int i = 1; i < length; i++)
        {
            pg_rt.AddLWECt(tmp_lwe, tmp_lwe, SM_FLAT[i * nslots + j]);
        }
        SM_SUM[j] = tmp_lwe;
    }

    // std::cout << "STEP 4: LOG\n";
    pg_rt.Log(SM_SUM.data(), SM_SUM.size());

    // Scale EXP_SUM
#pragma omp parallel for
    for (int i = 0; i < nslots; i++)
    {
        pg_rt.AddLWECt(SM_SUM[i], SM_SUM[i], log8_lwe_ct[i]);
    }

    // std::cout << "STEP 5: SUB\n";
#pragma omp parallel for collapse(2)
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < nslots; j++)
        {
            pg_rt.SubLWECt(SM_FLAT[i * nslots + j], U_lwe_cipher[i][j], SM_SUM[j]);
        }
    }

    // std::cout << "STEP 6: EXP\n";
    pg_rt.Exponent(SM_FLAT.data(), SM_FLAT.size());

#pragma omp parallel for collapse(2)
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < nslots; j++)
        {
            D_lwe_cipher[i][j] = SM_FLAT[i * nslots + j];
        }
    }

    repack(pg_rt, D_lwe_cipher, D_cipher, length, nslots);

    online_timer.stop();
    online_time += tmp_time;
    total_online_time += tmp_time;

    std::cout << "SOFTMAX & REPACK TIME " << online_time << std::endl;
}

void Repack(PegasusRunTime &pg_rt, vector<Ctx> &U_cipher, vector<vector<lwe::Ctx_st>> &U_lwe_cipher, int length, int nslots)
{
    tmp_time = 0.0;
    online_time = 0.0;
    AutoTimer online_timer(&tmp_time);

    std::cout << "S2C & EXTRACT\n";

#pragma omp parallel for
    for (int i = 0; i < length; i++)
    {
        CHECK_AND_ABORT(pg_rt.SlotsToCoeffs(U_cipher[i]));

        // use onyl if nslots = 2**7
        if (GetNModuli(U_cipher[i]) != 1)
        {
            pg_rt.runtime_->DropModuli(&U_cipher[i], GetNModuli(U_cipher[i]) - 1);
        }

        CHECK_AND_ABORT(pg_rt.ExtraAllCoefficients(U_cipher[i], U_lwe_cipher[i]));
    }
    cout << endl;

    std::cout << "REPACK\n";

#pragma omp parallel for
    for (int i = 0; i < length; i++)
    {
        CHECK_AND_ABORT(pg_rt.Repack(U_cipher[i], U_lwe_cipher[i]));
    }
    cout << endl;

    online_timer.stop();
    online_time += tmp_time;
    total_online_time += tmp_time;

    std::cout << "REPACK TIME " << online_time << std::endl;
}

void write_w_to_file(vector<vector<F64>> W, std::string model_name, int epoch, int n, int m)
{
    std::ofstream out("model_" + model_name + "_" + to_string(epoch) + "_ct.out", std::ios_base::app);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            out << W[i][j] << " ";
        }
    }
    out << std::endl;

    out.close();
}

void write_w_conv_to_file(vector<vector<vector<vector<F64>>>> W, std::string model_name, int epoch, int kc, int ic, int n, int m)
{
    std::ofstream out("model_" + model_name + "_" + to_string(epoch) + "_ct.out", std::ios_base::app);

    for (int i = 0; i < kc; i++)
    {
        for (int j = 0; j < ic; j++)
        {
            for (int ii = 0; ii < n; ii++)
            {
                for (int jj = 0; jj < m; jj++)
                {
                    out << W[i][j][ii][jj] << " ";
                }
            }
        }
    }
    out << std::endl;

    out.close();
}

void write_b_to_file(vector<F64> B, std::string model_name, int epoch, int length)
{
    std::ofstream out("model_" + model_name + "_" + to_string(epoch) + "_ct.out", std::ios_base::app);

    for (int i = 0; i < length; i++)
    {
        out << B[i] << " ";
    }
    out << std::endl;

    out.close();
}

void read_intermediate_results(PegasusRunTime &pg_rt, string modelname, string filename, vector<vector<F64>> &A, vector<Ctx> &A_cipher, vector<vector<lwe::Ctx_st>> &A_lwe_cipher, int length, int nslots, int moduli)
{
    string inFileName = "../model/model_" + modelname + "/model_" + modelname + "_" + filename + ".out";
    cout << inFileName << endl;
    ifstream inFile;
    inFile.open(inFileName.c_str());

    if (inFile.is_open())
    {
        for (int i = 0; i < nslots; i++)
        {
            for (int j = 0; j < length; j++)
            {
                inFile >> A[j][i];
            }
        }
    }
    inFile.close();

#pragma omp parallel for
    for (int i = 0; i < length; i++)
    {
        CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(A[i], A_cipher[i]));
        pg_rt.runtime_->DropModuli(&A_cipher[i], GetNModuli(A_cipher[i]) - moduli);
    }

    size_t found = filename.find('U');
    if (found != std::string::npos)
    {
        s2c_and_extract(pg_rt, A_cipher, A_lwe_cipher, length, nslots);
    }
}

void read_intermediate_conv_results_plain(string modelname, string filename, vector<vector<vector<vector<F64>>>> &A, int length, int channel, int nslots, int moduli)
{
    string inFileName = "../model/model_" + modelname + "/model_" + modelname + "_" + filename + ".out";
    cout << inFileName << endl;
    ifstream inFile;
    inFile.open(inFileName.c_str());

    if (inFile.is_open())
    {
        for (int ns = 0; ns < nslots; ns++)
        {
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    for (int k = 0; k < channel; k++)
                    {
                        inFile >> A[i][j][k][ns];
                    }
                }
            }
        }
    }
    inFile.close();

    /*
  #pragma omp parallel for collapse(3)
    for(int i = 0; i < length; i++) {
        for(int j = 0; j < length; j++) {
            for(int k = 0; k < channel; k++) {
                CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(A[i][j][k], A_cipher[i][j][k]));
                pg_rt.runtime_->DropModuli(&A_cipher[i][j][k], GetNModuli(A_cipher[i][j][k]) - moduli);
            }
        }
    }
    */
}

/*
void sigmoid(PegasusRunTime &pg_rt, vector<vector<lwe::Ctx_st>> &U_lwe_cipher, vector<vector<lwe::Ctx_st>> &D_lwe_cipher, int length, int nslots) {
    std::cout << "Sigmoid\n";

    for(int i = 0; i < length; i++) {
        for(int j = 0; j < nslots; j++) {
            D_lwe_cipher[i][j]= U_lwe_cipher[i][j];
        }

        pg_rt.Sigmoid(D_lwe_cipher[i].data(), D_lwe_cipher[i].size());
    }
    cout << endl;
}
*/

void Relu_3D(PegasusRunTime &pg_rt,
             std::vector<std::vector<std::vector<Ctx>>> U_cipher,
             std::vector<std::vector<std::vector<Ctx>>> D_cipher,
             std::vector<std::vector<lwe::Ctx_st>> &U_lwe_cipher,
             int nslots)
{
    std::cout << "RELU 3D\n";

    {
        MemoryPoolHandle my_pool = MemoryPoolHandle::New();
        auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

        int length = U_cipher.size() * U_cipher[0].size();

        vector<Ctx> tmp_cipher(length);

        for (int kc = 0; kc < U_cipher[0][0].size(); kc++)
        {
            for (int i = 0; i < U_cipher.size(); i++)
            {
                for (int j = 0; i < U_cipher[0].size(); j++)
                {
                    tmp_cipher[i * U_cipher.size() + j] = U_cipher[i][j][kc];
                }
            }
            s2c_and_extract(pg_rt, tmp_cipher, U_lwe_cipher, length, nslots);

            for (int i = 0; i < length; i++)
            {
                pg_rt.ReLU(U_lwe_cipher[i].data(), U_lwe_cipher[i].size());
            }

            repack(pg_rt, U_lwe_cipher, tmp_cipher, length, nslots);

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
}

void Drelu_3D(PegasusRunTime &pg_rt,
              std::vector<std::vector<std::vector<Ctx>>> U_cipher,
              std::vector<std::vector<lwe::Ctx_st>> &U_lwe_cipher,
              int nslots)
{
    std::cout << "dRELU 3D\n";

    {
        MemoryPoolHandle my_pool = MemoryPoolHandle::New();
        auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

        int length = U_cipher.size() * U_cipher[0].size();

        vector<Ctx> tmp_cipher(length);

        for (int kc = 0; kc < U_cipher[0][0].size(); kc++)
        {
            for (int i = 0; i < U_cipher.size(); i++)
            {
                for (int j = 0; i < U_cipher[0].size(); j++)
                {
                    tmp_cipher[i * U_cipher.size() + j] = U_cipher[i][j][kc];
                }
            }
            s2c_and_extract(pg_rt, tmp_cipher, U_lwe_cipher, length, nslots);

            for (int i = 0; i < length; i++)
            {
                pg_rt.DReLU(U_lwe_cipher[i].data(), U_lwe_cipher[i].size());
            }

            repack(pg_rt, U_lwe_cipher, tmp_cipher, length, nslots);

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
}

void Repack_3D(PegasusRunTime &pg_rt,
               std::vector<std::vector<std::vector<Ctx>>> U_cipher,
               std::vector<std::vector<lwe::Ctx_st>> &lwe_cipher,
               int nslots)
{
    std::cout << "Repack 3D\n";

    {
        MemoryPoolHandle my_pool = MemoryPoolHandle::New();
        auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));

        for (int i = 0; i < U_cipher.size(); i++)
        {
            for (int j = 0; j < U_cipher[0].size(); j++)
            {
                pg_rt.s2c_and_extract_2(U_cipher[i][j], lwe_cipher, U_cipher[0][0].size(), nslots);
                pg_rt.repack_2(lwe_cipher, U_cipher[i][j], U_cipher[0][0].size(), nslots);
            }
        }

        MemoryManager::SwitchProfile(std::move(old_prof));
    }
}
