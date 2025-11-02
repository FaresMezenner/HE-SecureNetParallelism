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
extern double total_save_model_time;
extern double total_load_model_time;
extern double total_offline_time;
extern double total_online_time;
extern bool not_first_epoch;
extern std::string save_model_loc;
extern std::string model_name;

int main(int argc, char *argv[]) {
    // Pegasus setup
    PegasusRunTime::Parms pp;

    pp.lvl0_lattice_dim = lwe::params::n(); // 1024
    pp.lvl1_lattice_dim = 1 << 12; // 4096
    pp.lvl2_lattice_dim = 1 << 16; // 65536
    pp.nlevels = 8; // CKKS levels
    pp.scale = std::pow(2., 40);
    pp.nslots = 128; // The number of batch size 
    pp.s2c_multiplier = 1.;
    pp.enable_repacking = true;

    PegasusRunTime pg_rt(pp, /*num_threads*/ num_threads);
    omp_set_num_threads(num_threads);

    int batch_start_point = 0; // The starting index of the training samples
    int nslots = pp.nslots; // The number of slots in a ciphertext is equal to the batch size
    int total_epoches = 10; // The total number of iterations for model training 
    not_first_epoch = false; // A flag to determine whether it's in the first epoch or not

    // Read MNIST dataset
    mnist::MNIST_dataset<vector, vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<vector, vector, uint8_t, uint8_t>("../dataset/mnist", 1, 10000);
    save_model_loc = "/app/model/model_b/";
    model_name = "model_b";

    // Variables for input data
    int l0 = 784;
    vector<vector<F64>> D0(l0, vector<F64>(nslots, 0.0));
    vector<Ctx> D0_cipher(l0);

    // Variables for intermediate layer results
    // Variables for Layer 1
    int l1 = 128;
    vector<Ctx> U1_cipher(l1);
    vector<Ctx> D1_cipher(l1);
    vector<Ctx> DE1_cipher(l1);
    vector<vector<F64>> W1(l0, vector<F64>(l1, 0.0));
    vector<F64> B1(l1, 0.0);

    // Variables for Layer 2
    int l2 = 128;
    vector<Ctx> U2_cipher(l2);
    vector<Ctx> D2_cipher(l2);
    vector<Ctx> DE2_cipher(l2);
    vector<vector<F64>> W2(l1, vector<F64>(l2, 0.0));
    vector<F64> B2(l2, 0.0);

    // Variables for Layer 3
    int l3 = 10;
    vector<Ctx> U3_cipher(l3);
    vector<Ctx> D3_cipher(l3);
    vector<Ctx> DE3_cipher(l3);
    vector<vector<F64>> W3(l2, vector<F64>(l3, 0.0));
    vector<F64> B3(l3, 0.0);

    // Variables for LWE ciphertexts
    vector<vector<lwe::Ctx_st>> U_lwe_cipher(l1, vector<lwe::Ctx_st>(nslots));
    vector<vector<lwe::Ctx_st>> D_lwe_cipher(l1, vector<lwe::Ctx_st>(nslots));

    // Variables for input labels
    vector<vector<F64>> Y(l3, vector<F64>(nslots, 0.0));
    vector<Ctx> Y_cipher(l3);

    // Init model weight and bias
    // Random number generator
    std::random_device rd; // Non-deterministic random seed
    std::mt19937 gen(rd()); // Mersenne twister engine seeded with random device
    std::uniform_real_distribution<double> dis(-0.3, 0.3);

    // W1
    for(int i = 0; i < l0; i++){
        for(int j = 0; j < l1; j++){
            W1[i][j] = dis(gen);
        }
    }

    // B1
    for(int i = 0; i < l1; i++){
        B1[i] = dis(gen);
    }

    // W2
    for(int i = 0; i < l1; i++){
        for(int j = 0; j < l2; j++){
            W2[i][j] = dis(gen);
        }
    }

    // B2
    for(int i = 0; i < l2; i++){
        B2[i] = dis(gen);
    }

    // W3
    for(int i = 0; i < l2; i++){
        for(int j = 0; j < l3; j++){
            W3[i][j] = dis(gen);
        }
    }

    // B3
    for(int i = 0; i < l3; i++){
        B3[i] = dis(gen);
    }

    // The Training Loop
    for(int epoch = 0; epoch < total_epoches; epoch++){
        std::cout << "=========" << std::endl;
        std::cout << "EPOCH " << epoch << std::endl;
        std::cout << "=========" << std::endl;

        if(epoch != 0) not_first_epoch = true;
        batch_start_point = nslots * epoch;

        std::cout << "Read the input data" << std::endl;
        for(int i = 0; i < l0; i++){
            for(int j = 0; j < nslots; j++){
                D0[i][j] = int(dataset.test_images[batch_start_point + j][i]) / 255.0;
            }
            pg_rt.EncodeThenEncrypt(D0[i], D0_cipher[i]);
        }

        std::cout << "Read the input labels" << std::endl;
        for(int i = 0; i < nslots; i++){
            Y[int(dataset.test_labels[batch_start_point + i])][i] = 1.0;
        }
        for(int i = 0; i < l3; i++){
            pg_rt.EncodeThenEncrypt(Y[i], Y_cipher[i]);
        }
        pg_rt.s2c_and_extract(Y_cipher, U_lwe_cipher, l3, nslots);
        pg_rt.repack(U_lwe_cipher, Y_cipher, l3, nslots);


        std::cout << "===========================" << std::endl;
        std::cout << "Layer 1 FORWARD PROPAGATION" << std::endl;
        std::cout << "===========================" << std::endl;

        // The linear function of Layer 1
        pg_rt.dense(pp, D0_cipher, W1, B1, U1_cipher, 1, epoch);

        // The activation function of the Layer 1
        pg_rt.s2c_and_extract(U1_cipher, U_lwe_cipher, l1, nslots);
        pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, D1_cipher, l1, nslots, "ReLU");

        // The derivative of the activation function of the Layer 1
        pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, U1_cipher, l1, nslots, "DReLU");

        std::cout << "===========================" << std::endl;
        std::cout << "Layer 2 FORWARD PROPAGATION" << std::endl;
        std::cout << "===========================" << std::endl;

        // The linear function of Layer 2
        pg_rt.dense(pp, D1_cipher, W2, B2, U2_cipher, 2, epoch);

        // The activation function of the Layer 2
        pg_rt.s2c_and_extract(U2_cipher, U_lwe_cipher, l2, nslots);
        pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, D2_cipher, l2, nslots, "ReLU");

        // The derivative of the activation function of the Layer 2
        pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, U2_cipher, l2, nslots, "DReLU");

        std::cout << "===========================" << std::endl;
        std::cout << "Layer 3 FORWARD PROPAGATION" << std::endl;
        std::cout << "===========================" << std::endl;

        // The linear function of Layer 3
        pg_rt.dense(pp, D2_cipher, W3, B3, U3_cipher, 3, epoch);

        // The activation function of the Layer 3
        pg_rt.s2c_and_extract(U3_cipher, U_lwe_cipher, l3, nslots);
        pg_rt.softmax(U_lwe_cipher, D_lwe_cipher, D3_cipher, l3, nslots);

        std::cout << "==========================" << std::endl;
        std::cout << "Layer 3 DELTA CALCULATIONS" << std::endl;
        std::cout << "==========================" << std::endl;

        // Calculate delta of Layer 3
        pg_rt.delta_softmax(pp, D3_cipher, Y_cipher, DE3_cipher);

        std::cout << "==========================" << std::endl;
        std::cout << "Layer 2 DELTA CALCULATIONS" << std::endl;
        std::cout << "==========================" << std::endl;

        // Calculate delta of Layer 2
        pg_rt.delta_1D_dense_1D(pp, W3, DE3_cipher, U2_cipher, DE2_cipher, 3, epoch);
        // Repack delta to a higher level
        pg_rt.s2c_repack_1D(DE2_cipher, U_lwe_cipher, nslots);

        std::cout << "==========================" << std::endl;
        std::cout << "Layer 1 DELTA CALCULATIONS" << std::endl;
        std::cout << "==========================" << std::endl;

        // Calculate delta of Layer 1
        pg_rt.delta_1D_dense_1D(pp, W2, DE2_cipher, U1_cipher, DE1_cipher, 2, epoch);
        // Repack delta to a higher level
        pg_rt.s2c_repack_1D(DE1_cipher, U_lwe_cipher, nslots);

        std::cout << "======================" << std::endl;
        std::cout << "Layer 1 WEIGHT UPDATES" << std::endl;
        std::cout << "======================" << std::endl;

        // Update model weights of Layer 1
        pg_rt.update_model_dense(pp, D0_cipher, DE1_cipher, W1, B1, 5, 1, epoch);

        std::cout << "======================" << std::endl;
        std::cout << "Layer 2 WEIGHT UPDATES" << std::endl;
        std::cout << "======================" << std::endl;

        // Update model weights of Layer 2
        pg_rt.update_model_dense(pp, D1_cipher, DE2_cipher, W2, B2, 5, 2, epoch);

        std::cout << "======================" << std::endl;
        std::cout << "Layer 3 WEIGHT UPDATES" << std::endl;
        std::cout << "======================" << std::endl;

        // Update model weights of Layer 3
        pg_rt.update_model_dense(pp, D2_cipher, DE3_cipher, W3, B3, 5, 3, epoch);
    }

    // The Training Time Information
    std::cout << "TOTAL SAVE MODEL TIME " << total_save_model_time << std::endl;
    std::cout << "TOTAL LOAD MODEL TIME " << total_load_model_time << std::endl;
    std::cout << "TOTAL OFFLINE TIME " << total_offline_time << std::endl;
    std::cout << "TOTAL ONLINE MODEL TIME " << total_online_time << std::endl;

    return 0;
}
