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
extern bool not_first_epoch;
extern std::string save_model_loc;
extern std::string model_name;

int main(int argc, char *argv[]) {
  double vm, vm2, rss, rss2;

  PegasusRunTime::Parms pp;

  pp.lvl0_lattice_dim = lwe::params::n(); //1024
  pp.lvl1_lattice_dim = 1 << 12; //4096
  pp.lvl2_lattice_dim = 1 << 16; //65536
  pp.nlevels = 7; // CKKS levels
  pp.scale = std::pow(2., 40);
  pp.nslots = 512;
  pp.s2c_multiplier = 1.;
  pp.enable_repacking = true;

  process_mem_usage(vm, rss);

  PegasusRunTime pg_rt(pp, /*num_threads*/ num_threads);
  omp_set_num_threads(num_threads);

  int batch_start_point = 0;
  int check_point = 0;
  int nslots = pp.nslots;
  int l0, l1;
  double total_time{0.};
  int total_epoches = 15;

  not_first_epoch = false;
  save_model_loc = "/app/model/model_a/";
  model_name = "model_a";

  // READ BREAST CANCER DATASET
  l0 = 30; 
  l1 = 1;

  vector<vector<F64>> x_train(l0, vector<F64>(nslots, 0.0));
  vector<vector<F64>> x_test(l0, vector<F64>(nslots, 0.0));
  vector<vector<F64>> y_train(l1, vector<F64>(nslots, 0.0));
  vector<vector<F64>> y_test(l1, vector<F64>(nslots, 0.0));

  vector<vector<F64>> W1(l0, vector<F64>(l1, 0.0));
  vector<F64> B1(l1, 0.0);
  vector<vector<F64>> W1_2(l0, vector<F64>(l1, 0.0)); // backup
  vector<F64> B1_2(l1, 0.0); // backup

  vector<vector<F64>> A(l0, vector<F64>(nslots, 0.0));
  vector<Ctx> A_cipher(l0);
  vector<vector<F64>> Y(l1, vector<F64>(nslots, 0.0));
  vector<Ctx> Y_cipher(l1);

  vector<vector<F64>> U1(l1, vector<F64>(nslots, 0.0));
  vector<Ctx> U_cipher(l1);
  vector<vector<lwe::Ctx_st>> U_lwe_cipher(l1, vector<lwe::Ctx_st>(nslots));

  vector<vector<F64>> D1(l1, vector<F64>(nslots, 0.0));
  vector<Ctx> D_cipher(l1);
  vector<vector<lwe::Ctx_st>> D_lwe_cipher(l1, vector<lwe::Ctx_st>(nslots));

  vector<vector<F64>> DE1(l1, vector<F64>(nslots, 0.0));
  vector<Ctx> DE1_cipher(l1);

  vector<lwe::Ctx_st> lwe_ct;
  F64Vec T(nslots);

  {
      total_time = 0.0;
      AutoTimer total_timer(&total_time);

      string inFileName = "../dataset/breast_cancer/x_train.out";
      ifstream inFile;
      inFile.open(inFileName.c_str());
      if(inFile.is_open()) {
          for(int j = 0; j < nslots; j++) {
              for(int i = 0; i < l0; i++) {
                  if(j < 455) {
                  inFile >> x_train[i][j];
                  A[i][j] = x_train[i][j];
                  }else{
                      x_train[i][j] = 0;
                      A[i][j] = 0;
                  }
              }
          }
      }

      string inFileName_2 = "../dataset/breast_cancer/x_test.out";
      ifstream inFile_2;
      inFile_2.open(inFileName_2.c_str());
      if(inFile_2.is_open()) {
          for(int j = 0; j < nslots; j++) {
              for(int i = 0; i < l0; i++) {
                  if(j < 114) {
                  inFile_2 >> x_test[i][j];
                  }else{
                      x_test[i][j] = 0;
                  }
              }
          }
      }

      string inFileName_3 = "../dataset/breast_cancer/y_train.out";
      ifstream inFile_3;
      inFile_3.open(inFileName_3.c_str());
      if(inFile_3.is_open()) {
          for(int j = 0; j < nslots; j++) {
              if(j < 455) {
                  inFile_3 >> y_train[0][j];
                  Y[0][j] = y_train[0][j];
              }else{
                  y_train[0][j] = 0;
                  Y[0][j] = 0;
              }
          }
      }

      string inFileName_4 = "../dataset/breast_cancer/y_test.out";
      ifstream inFile_4;
      inFile_4.open(inFileName_4.c_str());
      if(inFile_4.is_open()) {
          for(int j = 0; j < nslots; j++) {
              if(j < 114) {
                  inFile_4 >> y_test[0][j];
              }else{
                  y_test[0][j] = 0;
              }
          }
      }

      // A
#pragma omp parallel for
      for(int i = 0; i < l0; i++) {
          CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(A[i], A_cipher[i]));
      }
      // Y
#pragma omp parallel for
      for(int i = 0; i < l1; i++) {
          CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(Y[i], Y_cipher[i]));
      }

      pg_rt.s2c_and_extract(Y_cipher, U_lwe_cipher, l1, nslots);
      pg_rt.repack(U_lwe_cipher, Y_cipher, l1, nslots);

      total_timer.stop();
      cout << "PREPARE A & Y TIME " << total_time << endl;

      std::random_device rd; // Non-deterministic random seed
      std::mt19937 gen(rd()); // Mersenne Twister engine seeded with random device
      std::uniform_real_distribution<double> dis(-1.0, 1.0);

      // W1
      for(int i = 0; i < l0; i++) {
          for(int j = 0; j < l1; j++) {
              W1[i][j] = dis(gen);
              W1_2[i][j] = W1[i][j];
          }
      }

      // B1
      for(int i = 0; i < l1; i++) {
          B1[i] = dis(gen);
          B1_2[i] = B1[i];
      }
  }

  // Plaintext domain training
  for(int epoch = 0; epoch < total_epoches; epoch++) {
      // FORWARD
      for(int k = 0; k < 455; k++) {
          for(int i = 0; i < l1; i++) {
              F64 tmp = 0.0;
              for(int j = 0; j < l0; j++) {
                  tmp += x_train[j][k] * W1[j][i];
              }
              tmp += B1[i];
              U1[i][k] = tmp;
              D1[i][k] = 1.0 / (1.0 + exp(-1.0 * U1[i][k])); // sigmoid
          }
      }

      // BACKWARD
      for(int k = 0; k < 455; k++) {
          for(int i = 0; i < l1; i++) {
              DE1[i][k] = y_train[i][k] - D1[i][k];
          }
      }

      // UPDATE
      for(int i = 0; i < l0; i++) {
          for(int j = 0; j < l1; j++) {
              F64 tmp = 0.0;
              for(int k = 0; k < 455; k++) {
                  tmp += x_train[i][k] * DE1[j][k];
              }
              W1[i][j] += tmp * 1.0 / 455;
              // std::cout << tmp * 1.0 / 455 << std::endl;
          }
          // std::cout << W1[i][0] << std::endl;
      }

      // EVALUATE
      int correct = 0;
      for(int k = 0; k < 114; k++) {
          F64 tmp = 0.0;
          for(int i = 0; i < l1; i++) {
              for(int j = 0; j < l0; j++) {
                  tmp += x_test[j][k] * W1[j][i];
              }
              tmp += B1[i];
          }
          if(tmp >= 0.5) {
              tmp = 1.0;
          }else{
              tmp = 0.0;
          }
          if(tmp == y_test[0][k]) {
              correct++;
          }
      }
      std::cout << "Epoch: " << epoch << " Testing Accuracy: " << 1.0 * correct / 114 << std::endl;
  }

  // Restore
  // W1
  for(int i = 0; i < l0; i++) {
      for(int j = 0; j < l1; j++) {
          W1[i][j] = W1_2[i][j];
      }
  }

  // B1
  for(int i = 0; i < l1; i++) {
      B1[i] = B1_2[i];
  }

  // Ciphertext domain training
  for(int epoch = 0; epoch < total_epoches; epoch++) {
      std::cout << std::endl;
      std::cout << "=====================================" << std::endl;
      std::cout << "============== EPOCH " << epoch << " ==============\n";
      std::cout << "=====================================" << std::endl;

      if(epoch != 0) { not_first_epoch = false; }

      {
          std::cout << std::endl;
          std::cout << "========== FORWARD ==========\n";
          std::cout << std::endl;

          pg_rt.dense(pp, A_cipher, W1, B1, U_cipher, 1, epoch);
      }

      {
          std::cout << std::endl;
          std::cout << "========== SIGMOID ==========\n";
          std::cout << std::endl;

          pg_rt.s2c_and_extract(U_cipher, U_lwe_cipher, l1, nslots);
          pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, D_cipher, l1, nslots, "Sigmoid");
      }

      {
          std::cout << std::endl;
          std::cout << "========== BACKWARD ==========\n";
          std::cout << std::endl;

          pg_rt.delta_softmax(pp, D_cipher, Y_cipher, DE1_cipher); 
      }

      {
          std::cout << std::endl;
          std::cout << "========== UPDATE MODEL ==========\n";
          std::cout << std::endl;

          vector<Ctx> A_cipher_2(l0);
#pragma omp parallel for 
          for(int i = 0; i < l0; i++) {
              A_cipher_2[i] = A_cipher[i];
          }

          pg_rt.update_model_dense(pp, A_cipher_2, DE1_cipher, W1, B1, 4, 1, epoch);
          /*
          // Check Answer
          for(int i = 0; i < 30; i++) {
              std::cout << W1[i][0] << std::endl;
          }
          */

          release_vector_1D(A_cipher_2);
          A_cipher_2.clear();
      }

      // EVALUATE in the pt domain
      std::cout << "EVALUATE\n";
      int correct = 0;
      for(int k = 0; k < 114; k++) {
          F64 tmp = 0.0;
          for(int i = 0; i < l1; i++) {
              for(int j = 0; j < l0; j++) {
                  tmp += x_test[j][k] * W1[j][i];
              }
              tmp += B1[i];
          }
          std::cout << tmp << " ";
          if(tmp >= 0.5) {
              tmp = 1.0;
              std::cout << "1 ";
          }else{
              tmp = 0.0;
              std::cout << "0 ";
          }
          std::cout << y_test[0][k] << std::endl;
          if(tmp == y_test[0][k]) {
              correct++;
          }
      }
      std::cout << "Accuracy: " << 1.0 * correct / 114 << std::endl;
  }

  std::cout << "END END\n";

  return 0;
}
