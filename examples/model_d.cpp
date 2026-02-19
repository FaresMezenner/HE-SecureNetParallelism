#include "pegasus/pegasus_runtime.h"
#include "pegasus/timer.h"
#include "mnist/mnist_reader.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>

using namespace gemini;
using namespace seal;
using namespace std;

int num_threads = 64;
extern thread_local double total_save_model_time;
extern thread_local double total_load_model_time;
extern thread_local double total_offline_time;
extern thread_local double total_online_time;
extern thread_local bool not_first_epoch;
extern std::string save_model_loc;
extern std::string model_name;

int main(int argc, char *argv[])
{
    // Pegasus setup
    PegasusRunTime::Parms pp;

    pp.lvl0_lattice_dim = lwe::params::n(); // 1024
    pp.lvl1_lattice_dim = 1 << 12;          // 4096
    pp.lvl2_lattice_dim = 1 << 16;          // 65536
    pp.nlevels = 8;                         // CKKS levels
    pp.scale = std::pow(2., 40);
    pp.nslots = 128; // The number of batch size
    pp.s2c_multiplier = 1.;
    pp.enable_repacking = true;

    PegasusRunTime pg_rt(pp, /*num_threads*/ num_threads);
    omp_set_num_threads(num_threads);

    int batch_start_point = 0; // The starting index of the training samples
    int nslots = pp.nslots;    // The number of slots in a ciphertext is equal to the batch size
    int total_epoches = 10;    // The total number of iterations for model training
    not_first_epoch = false;   // A flag to determine whether it's in the first epoch or not

    // Read MNIST dataset
    mnist::MNIST_dataset<vector, vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<vector, vector, uint8_t, uint8_t>("../dataset/mnist", 1, 10000);
    save_model_loc = "/app/model/model_d/";
    model_name = "model_d";

    // Variables for input data
    int l0_conv = 28;
    int l0_kc = 1;
    vector<vector<vector<vector<F64>>>> D0(l0_conv, vector<vector<vector<F64>>>(l0_conv, vector<vector<F64>>(l0_kc, vector<F64>(nslots, 0.0))));
    vector<vector<vector<Ctx>>> D0_cipher(l0_conv, vector<vector<Ctx>>(l0_conv, vector<Ctx>(l0_kc)));

    // Variables for intermediate layer results
    // Variables for Layer 1
    int l1 = 12;
    int l1_mp = 12;
    int l1_mp_len = 2;
    int l1_conv = 24;
    int l1_size = 5;
    int l1_ic = 1;
    int l1_kc = 6;
    vector<vector<vector<Ctx>>> U1_cipher(l1_conv, vector<vector<Ctx>>(l1_conv, vector<Ctx>(l1_kc)));
    vector<vector<vector<Ctx>>> D1_cipher(l1_conv, vector<vector<Ctx>>(l1_conv, vector<Ctx>(l1_kc)));
    vector<vector<vector<Ctx>>> DE1_cipher(l1_conv, vector<vector<Ctx>>(l1_conv, vector<Ctx>(l1_kc)));
    vector<vector<vector<Ctx>>> D1_mp_cipher(l1_mp, vector<vector<Ctx>>(l1_mp, vector<Ctx>(l1_kc)));
    vector<vector<vector<Ctx>>> DE1_mp_cipher(l1_mp, vector<vector<Ctx>>(l1_mp, vector<Ctx>(l1_kc)));
    vector<vector<vector<Ctx>>> MI1_cipher(l1_conv, vector<vector<Ctx>>(l1_conv, vector<Ctx>(l1_kc)));
    vector<vector<vector<vector<F64>>>> W1(l1_kc, vector<vector<vector<F64>>>(l1_ic, vector<vector<F64>>(l1_size, vector<F64>(l1_size, 0.0))));
    vector<F64> B1(l1_kc, 0.0);

    // Variables for Layer 2
    int l2 = 256;
    int l2_mp = 4;
    int l2_mp_len = 2;
    int l2_conv = 8;
    int l2_size = 5;
    int l2_ic = 6;
    int l2_kc = 16;
    vector<vector<vector<Ctx>>> U2_cipher(l2, vector<vector<Ctx>>(l2, vector<Ctx>(l2_kc)));
    vector<Ctx> D2_cipher(l2);
    vector<vector<vector<Ctx>>> DE2_cipher(l2_conv, vector<vector<Ctx>>(l2_conv, vector<Ctx>(l2_kc)));
    vector<vector<vector<Ctx>>> D2_mp_cipher(l2_mp, vector<vector<Ctx>>(l2_mp, vector<Ctx>(l2_kc)));
    vector<vector<vector<Ctx>>> DE2_mp_cipher(l2_mp, vector<vector<Ctx>>(l2_mp, vector<Ctx>(l2_kc)));
    vector<vector<vector<Ctx>>> MI2_cipher(l2_conv, vector<vector<Ctx>>(l2_conv, vector<Ctx>(l2_kc)));
    vector<vector<vector<vector<F64>>>> W2(l2_kc, vector<vector<vector<F64>>>(l2_ic, vector<vector<F64>>(l2_size, vector<F64>(l2_size, 0.0))));
    vector<F64> B2(l2_kc, 0.0);

    // Variables for Layer 3
    int l3 = 100;
    vector<Ctx> U3_cipher(l3);
    vector<Ctx> D3_cipher(l3);
    vector<Ctx> DE3_cipher(l3);
    vector<vector<F64>> W3(l2, vector<F64>(l3, 0.0));
    vector<F64> B3(l3, 0.0);

    // Variables for Layer 4
    int l4 = 10;
    vector<Ctx> U4_cipher(l4);
    vector<Ctx> D4_cipher(l4);
    vector<Ctx> DE4_cipher(l4);
    vector<vector<F64>> W4(l3, vector<F64>(l4, 0.0));
    vector<F64> B4(l4, 0.0);

    // Variables for LWE ciphertexts
    vector<vector<lwe::Ctx_st>> U_lwe_cipher(l0_conv * l0_conv, vector<lwe::Ctx_st>(nslots));
    vector<vector<lwe::Ctx_st>> D_lwe_cipher(l0_conv * l0_conv, vector<lwe::Ctx_st>(nslots));

    // Variables for input labels
    vector<vector<F64>> Y(l4, vector<F64>(nslots, 0.0));
    vector<Ctx> Y_cipher(l4);

    // Init model weight and bias
    // Random number generator
    std::random_device rd;  // Non-deterministic random seed
    std::mt19937 gen(rd()); // Mersenne twister engine seeded with random device
    std::uniform_real_distribution<double> dis(-0.3, 0.3);

    // W1
    for (int kc = 0; kc < l1_kc; kc++)
    {
        for (int ic = 0; ic < l1_ic; ic++)
        {
            for (int i = 0; i < l1_size; i++)
            {
                for (int j = 0; j < l1_size; j++)
                {
                    W1[kc][ic][i][j] = dis(gen);
                }
            }
        }
    }

    // B1
    for (int i = 0; i < l1_kc; i++)
    {
        B1[i] = dis(gen);
    }

    // W2
    for (int kc = 0; kc < l2_kc; kc++)
    {
        for (int ic = 0; ic < l2_ic; ic++)
        {
            for (int i = 0; i < l2_size; i++)
            {
                for (int j = 0; j < l2_size; j++)
                {
                    W2[kc][ic][i][j] = dis(gen);
                }
            }
        }
    }

    // B2
    for (int i = 0; i < l2_kc; i++)
    {
        B2[i] = dis(gen);
    }

    // W3
    for (int i = 0; i < l2; i++)
    {
        for (int j = 0; j < l3; j++)
        {
            W3[i][j] = dis(gen);
        }
    }

    // B3
    for (int i = 0; i < l3; i++)
    {
        B3[i] = dis(gen);
    }

    // W4
    for (int i = 0; i < l3; i++)
    {
        for (int j = 0; j < l4; j++)
        {
            W4[i][j] = dis(gen);
        }
    }

    // B4
    for (int i = 0; i < l4; i++)
    {
        B4[i] = dis(gen);
    }

    // The Training Loop
    for (int epoch = 0; epoch < total_epoches; epoch++)
    {
        std::cout << "=========" << std::endl;
        std::cout << "EPOCH " << epoch << std::endl;
        std::cout << "=========" << std::endl;

        if (epoch != 0)
            not_first_epoch = true;
        batch_start_point = nslots * epoch;

        std::cout << "Read the input data" << std::endl;
        for (int kc = 0; kc < l0_kc; kc++)
        {
            for (int i = 0; i < l0_conv; i++)
            {
                for (int j = 0; j < l0_conv; j++)
                {
                    for (int k = 0; k < nslots; k++)
                    {
                        D0[i][j][kc][k] = int(dataset.test_images[batch_start_point + k][kc * l0_conv * l0_conv + i * l0_conv + j]) / 255.0;
                    }
                }
            }
        }
        for (int kc = 0; kc < l0_kc; kc++)
        {
            for (int i = 0; i < l0_conv; i++)
            {
                for (int j = 0; j < l0_conv; j++)
                {
                    pg_rt.EncodeThenEncrypt(D0[i][j][kc], D0_cipher[i][j][kc]);
                }
            }
        }

        std::cout << "Read the input labels" << std::endl;
        for (int i = 0; i < nslots; i++)
        {
            Y[int(dataset.test_labels[batch_start_point + i])][i] = 1.0;
        }
        for (int i = 0; i < l4; i++)
        {
            pg_rt.EncodeThenEncrypt(Y[i], Y_cipher[i]);
        }
        pg_rt.s2c_and_extract(Y_cipher, U_lwe_cipher, l4, nslots);
        pg_rt.repack(U_lwe_cipher, Y_cipher, l4, nslots);

        std::cout << "===========================" << std::endl;
        std::cout << "Layer 1 FORWARD PROPAGATION" << std::endl;
        std::cout << "===========================" << std::endl;

        // The linear function of the Layer 1

        pg_rt.conv(pp, D0_cipher, W1, B1, U1_cipher, 1, 1, epoch);
        pg_rt.maxpool(pp, U1_cipher, MI1_cipher, D1_mp_cipher, U_lwe_cipher, D_lwe_cipher, l1_mp_len);

        std::cout << "===========================" << std::endl;
        std::cout << "Layer 2 FORWARD PROPAGATION" << std::endl;
        std::cout << "===========================" << std::endl;

        // The linear function of the Layer 2
        {

            pg_rt.conv(pp, D1_mp_cipher, W2, B2, U2_cipher, 1, 2, epoch);
            // Create a tmp_cipher for catching output values
            vector<vector<vector<Ctx>>> tmp_output_cipher(l2_mp, vector<vector<Ctx>>(l2_mp, vector<Ctx>(l2_kc)));
            pg_rt.maxpool(pp, U2_cipher, MI2_cipher, tmp_output_cipher, U_lwe_cipher, D_lwe_cipher, l2_mp_len);

            // Flatten
            pg_rt.flatten(tmp_output_cipher, D2_cipher);
            release_vector_3D(tmp_output_cipher);
            tmp_output_cipher.clear();
        }

        std::cout << "===========================" << std::endl;
        std::cout << "Layer 3 FORWARD PROPAGATION" << std::endl;
        std::cout << "===========================" << std::endl;

        // The linear function of Layer 3
        pg_rt.dense(pp, D2_cipher, W3, B3, U3_cipher, 3, epoch);

        // The activation function of the Layer 3
        pg_rt.s2c_and_extract(U3_cipher, U_lwe_cipher, l3, nslots);
        pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, D3_cipher, l3, nslots, "ReLU");

        // The derivative of the activation function of the Layer 3
        pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, U3_cipher, l3, nslots, "DReLU");

        std::cout << "===========================" << std::endl;
        std::cout << "Layer 4 FORWARD PROPAGATION" << std::endl;
        std::cout << "===========================" << std::endl;

        // The linear function of Layer 4
        pg_rt.dense(pp, D3_cipher, W4, B4, U4_cipher, 4, epoch);

        // The activation function of the Layer 4
        pg_rt.s2c_and_extract(U4_cipher, U_lwe_cipher, l4, nslots);
        pg_rt.softmax(U_lwe_cipher, D_lwe_cipher, D4_cipher, l4, nslots);

        std::cout << "==========================" << std::endl;
        std::cout << "Layer 4 DELTA CALCULATIONS" << std::endl;
        std::cout << "==========================" << std::endl;

        // Calculate delta of Layer 4
        pg_rt.delta_softmax(pp, D4_cipher, Y_cipher, DE4_cipher);

        std::cout << "==========================" << std::endl;
        std::cout << "Layer 3 DELTA CALCULATIONS" << std::endl;
        std::cout << "==========================" << std::endl;

        // Calculate delta of Layer 3
        pg_rt.delta_1D_dense_1D(pp, W4, DE4_cipher, U3_cipher, DE3_cipher, 4, epoch);
        // Repack delta to a higher level
        pg_rt.s2c_repack_1D(DE3_cipher, U_lwe_cipher, nslots);

        std::cout << "==========================" << std::endl;
        std::cout << "Layer 2 DELTA CALCULATIONS" << std::endl;
        std::cout << "==========================" << std::endl;

        // Calculate delta of Layer 2
        pg_rt.prepare_delta_maxpooling_from_mat(pp, W3, DE3_cipher, DE2_mp_cipher, 2, epoch);
        pg_rt.delta_maxpooling(pp, DE2_mp_cipher, U2_cipher, MI2_cipher, U_lwe_cipher, DE2_cipher);
        // Repack delta to a higher level
        pg_rt.s2c_repack_3D(DE2_cipher, U_lwe_cipher, nslots);

        std::cout << "==========================" << std::endl;
        std::cout << "Layer 1 DELTA CALCULATIONS" << std::endl;
        std::cout << "==========================" << std::endl;

        // Calculate delta of Layer 1
        pg_rt.prepare_delta_maxpooling_from_conv(pp, W2, DE2_cipher, DE1_mp_cipher, 1, epoch);
        pg_rt.delta_maxpooling(pp, DE1_mp_cipher, U1_cipher, MI1_cipher, U_lwe_cipher, DE1_cipher);
        // Repack delta to a higher level
        pg_rt.s2c_repack_3D(DE1_cipher, U_lwe_cipher, nslots);

        std::cout << "======================" << std::endl;
        std::cout << "Layer 1 WEIGHT UPDATES" << std::endl;
        std::cout << "======================" << std::endl;

        // Update model weights of Layer 1
        pg_rt.update_model_conv_with_3D_delta(pp, D0_cipher, DE1_cipher, W1, B1, 5, 1, epoch);

        std::cout << "======================" << std::endl;
        std::cout << "Layer 2 WEIGHT UPDATES" << std::endl;
        std::cout << "======================" << std::endl;

        // Update model weights of Layer 2
        pg_rt.update_model_conv_with_3D_delta(pp, D1_mp_cipher, DE2_cipher, W2, B2, 5, 2, epoch);

        std::cout << "======================" << std::endl;
        std::cout << "Layer 3 WEIGHT UPDATES" << std::endl;
        std::cout << "======================" << std::endl;

        // Update model weights of Layer 3
        pg_rt.update_model_dense(pp, D2_cipher, DE3_cipher, W3, B3, 5, 3, epoch);

        std::cout << "======================" << std::endl;
        std::cout << "Layer 4 WEIGHT UPDATES" << std::endl;
        std::cout << "======================" << std::endl;

        // Update model weights of Layer 4
        pg_rt.update_model_dense(pp, D3_cipher, DE4_cipher, W4, B4, 5, 4, epoch);
    }

    // The Training Time Information
    std::cout << "TOTAL SAVE MODEL TIME " << total_save_model_time << std::endl;
    std::cout << "TOTAL LOAD MODEL TIME " << total_load_model_time << std::endl;
    std::cout << "TOTAL OFFLINE TIME " << total_offline_time << std::endl;
    std::cout << "TOTAL ONLINE MODEL TIME " << total_online_time << std::endl;

    return 0;
}
