#=========================================
# This parser consists of four parts:
# 1) Parse the ONNX model structure, which is stored in a dictionary called `model`.
# 2) Print the C++ training code before the training loop.
# 3) Print the C++ training code for the training loop (i.e., forward propagation, backpropagation, and weight update).
# 4) Print the C++ training code after the training loop, showing the training time.
#=========================================

import pdb
import sys
import onnx
import numpy as np
from collections import defaultdict

print("#=========================================")
print("#=== Part 1: Parse the Model Structure ===")
print("#=========================================")
# Load the ONNX graph file
model_name = sys.argv[1]
onnx_model = onnx.load("../onnx/%s.onnx" % model_name)
inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
model_graph = inferred_model.graph

num_layer = 0
is_maxpool = False
model = defaultdict(dict)

# Understand the input layer
model[num_layer]["name"] =  "input"
model[num_layer]["output"] =  []
for d in model_graph.input[0].type.tensor_type.shape.dim:
    if (d.HasField("dim_value")):
        # print (d.dim_value, end=", ")  
        model[num_layer]["output"].append(d.dim_value)
    '''
    elif (d.HasField("dim_param")):
        # print (d.dim_param, end=", ") 
    else:
        # print ("?", end=", ")
    '''

num_layer = num_layer + 1

# Understand intermediate layers
for i, x in enumerate(model_graph.node):
    node = x
    value = ""
    if i < len(model_graph.value_info):
        value = model_graph.value_info[i]
    # Parse the information of conv layers
    if 'conv2d' in node.name or 'Conv' in node.name:
        if node.op_type == 'Reshape' or node.op_type == 'Transpose':
            continue
        model[num_layer]["name"] = "conv2d"
        if 'output_flatten' in model[num_layer - 1].keys():
            model[num_layer]["input"] = model[num_layer - 1]['output_flatten']
        elif 'output_maxpool' in model[num_layer - 1].keys():
            model[num_layer]["input"] = model[num_layer - 1]['output_maxpool']
        else:
            model[num_layer]["input"] = model[num_layer - 1]['output']
        if node.op_type == 'Conv':
            model[num_layer]["conv"] = {}
            for d in node.attribute:
                model[num_layer]["conv"][d.name] = d.ints
            model[num_layer]["output"] = []
            for d in value.type.tensor_type.shape.dim:
                if (d.HasField("dim_value")):
                    model[num_layer]["output"].append(d.dim_value)
            model[num_layer]["output"].reverse()
        elif node.op_type == 'Relu':
            model[num_layer]["act"] = 'relu'
            num_layer = num_layer + 1
    # Parse the information of dense layers
    elif 'dense' in node.name:
        model[num_layer]["name"] = "dense"
        if 'output_flatten' in model[num_layer - 1].keys():
            model[num_layer]["input"] = model[num_layer - 1]['output_flatten']
        elif 'output_maxpool' in model[num_layer - 1].keys():
            model[num_layer]["input"] = model[num_layer - 1]['output_maxpool']
        else:
            model[num_layer]["input"] = model[num_layer - 1]['output']
        if node.op_type == 'MatMul':
            model[num_layer]["output"] = []
            for d in value.type.tensor_type.shape.dim:
                if (d.HasField("dim_value")):
                    model[num_layer]["output"].append(d.dim_value)
        elif node.op_type == 'Relu':
            model[num_layer]["act"] = 'relu'
            num_layer = num_layer + 1
        elif node.op_type == 'Softmax':
            model[num_layer]["act"] = 'softmax'
            num_layer = num_layer + 1
    # Parse the information of maxpooling layers
    elif 'max_pooling2d' in node.name:
        is_maxpool = True
        if node.op_type == 'MaxPool':
            model[num_layer - 1]["maxpool"] = {}
            for d in node.attribute:
                model[num_layer - 1]["maxpool"][d.name] = d.ints
            model[num_layer - 1]["output_maxpool"] = []
            for d in value.type.tensor_type.shape.dim:
                if (d.HasField("dim_value")):
                    model[num_layer - 1]["output_maxpool"].append(d.dim_value)
    elif 'flatten' in node.name:
        if 'maxpool' in model[num_layer - 1].keys():
            model[num_layer - 1]["output_flatten"] = [np.prod(model[num_layer - 1]["output_maxpool"])]
        else:
            model[num_layer - 1]["output_flatten"] = [np.prod(model[num_layer - 1]["output"])]

# Understand the output layer
model[num_layer]["name"] =  "output"
model[num_layer]["output"] =  []
for d in model_graph.output[0].type.tensor_type.shape.dim:
    if (d.HasField("dim_value")):
        # print (d.dim_value, end=", ")
        model[num_layer]["output"].append(d.dim_value)
    '''
    elif (d.HasField("dim_param")):
        print (d.dim_param, end=", ")
    else:
        print ("?", end=", ")
    '''

# Print the model structure for debug
print("The model structure is: ")
for x in model.values():
    print(x)

print("# =========================================================")
print("# ========= Part 2. Generate the Private C++ Code  ========")
print("# =========================================================")

# Declare the saving location for the private C++ training code
f = open('../examples/%s.cpp' % model_name, 'w')

# Determine the level requirements for the CKKS scheme
if len(model) == 3:
    nlevels = 7 # without maxpooling
else:
    nlevels = 8 # with maxpooling
    
# Declare directory for saving encrypted model weight
save_model_loc = '/app/model/' + model_name + '/'

# Print header files
f.write('#include "pegasus/pegasus_runtime.h"\n')
f.write('#include "pegasus/timer.h"\n')
f.write('#include "mnist/mnist_reader.hpp"\n')
f.write('#include <iostream>\n')
f.write('#include <fstream>\n')
f.write('#include <string>\n')
f.write('#include <ctime>\n')
f.write('\n')
f.write('using namespace gemini;\n')
f.write('using namespace seal;\n')
f.write('using namespace std;\n')
f.write('\n')
# Determine global variables 
f.write('int num_threads = 64;\n')
f.write('extern thread_local double total_save_model_time;\n')
f.write('extern thread_local double total_load_model_time;\n')
f.write('extern thread_local double total_offline_time;\n')
f.write('extern thread_local double total_online_time;\n')
f.write('extern thread_local bool not_first_epoch;\n')
f.write('extern std::string save_model_loc;\n')
f.write('extern std::string model_name;\n')
f.write('\n')
# The main function begins
f.write('int main(int argc, char *argv[]) {\n')
# f.write('double vm, vm2, rss, rss2;\n')
f.write('// Pegasus setup\n')
f.write('PegasusRunTime::Parms pp;\n')
f.write('\n')
f.write('pp.lvl0_lattice_dim = lwe::params::n(); // 1024\n')
f.write('pp.lvl1_lattice_dim = 1 << 12; // 4096\n')
f.write('pp.lvl2_lattice_dim = 1 << 16; // 65536\n')
f.write('pp.nlevels = %d; // CKKS levels\n' % nlevels)
f.write('pp.scale = std::pow(2., 40);\n')
f.write('pp.nslots = 128; // The number of batch size \n')
f.write('pp.s2c_multiplier = 1.;\n')
f.write('pp.enable_repacking = true;\n')
# f.write('process_mem_usage(vm, rss);\n')
f.write('\n')
f.write('PegasusRunTime pg_rt(pp, /*num_threads*/ num_threads);\n')
f.write('omp_set_num_threads(num_threads);\n')
f.write('\n')
f.write('int batch_start_point = 0; // The starting index of the training samples\n')
f.write('int nslots = pp.nslots; // The number of slots in a ciphertext is equal to the batch size\n')
f.write('int total_epoches = 10; // The total number of iterations for model training \n')
f.write('not_first_epoch = false; // A flag to determine whether it\'s in the first epoch or not\n')
f.write('\n')
f.write('// Read MNIST dataset\n')
f.write('mnist::MNIST_dataset<vector, vector<uint8_t>, uint8_t> dataset =\n')
f.write('mnist::read_dataset<vector, vector, uint8_t, uint8_t>("../dataset/mnist", 1, 10000);\n')
f.write('save_model_loc = "%s";\n' % save_model_loc)
f.write('model_name = "%s";\n' % model_name)
f.write('\n')

# ============================================
# === Part 2.1 Variables of the Input Data ===
# ============================================
# Deal with the input data. 
f.write("// Variables for input data\n")
if 'output_flatten' in model[0].keys():
    # Reshape input data as a vector
    f.write('int l0 = %d;\n' % (model[0]['output_flatten'][0]))
    f.write('vector<vector<F64>> D0(l0, vector<F64>(nslots, 0.0));\n')
    f.write('vector<Ctx> D0_cipher(l0);\n')
    f.write('\n')
else:
    # Reshape input data as a 3D tensor
    f.write('int l0_conv = %d;\n' % (model[0]['output'][0]))
    f.write('int l0_kc = %d;\n' % (model[0]['output'][-1]))
    f.write('vector<vector<vector<vector<F64>>>> D0(l0_conv, vector<vector<vector<F64>>>(l0_conv, vector<vector<F64>>(l0_kc, vector<F64>(nslots, 0.0))));\n')
    f.write('vector<vector<vector<Ctx>>> D0_cipher(l0_conv, vector<vector<Ctx>>(l0_conv, vector<Ctx>(l0_kc)));\n')
    f.write('\n')

# ===============================================================
# === Part 2.2 Variables Used for Saving Intermediate Results ===
# ===============================================================
f.write("// Variables for intermediate layer results\n")
print("Creating Variables for Model Training")
for idx, value in model.items():
    if idx == 0 or idx == num_layer: continue
    print(idx, value)

    f.write('// Variables for Layer %d\n' % idx)
    if value['name'] == 'dense':
        # Variables used for dense layers
        f.write('int l%d = %d;\n' % (idx, value['output'][0]))
        # f.write('vector<vector<F64>> U%d(l%d, vector<F64>(nslots, 0.0));\n' %(idx, idx))
        f.write('vector<Ctx> U%d_cipher(l%d);\n' %(idx, idx))
        # f.write('vector<vector<F64>> D%d(l%d, vector<F64>(nslots, 0.0));\n' %(idx, idx))
        f.write('vector<Ctx> D%d_cipher(l%d);\n' %(idx, idx))
        # f.write('vector<vector<F64>> DE%d(l%d, vector<F64>(nslots, 0.0));\n' %(idx, idx))
        f.write('vector<Ctx> DE%d_cipher(l%d);\n' %(idx, idx))
        # Model weight and bias for dense layers
        f.write('vector<vector<F64>> W%d(l%d, vector<F64>(l%d, 0.0));\n' %(idx, idx - 1, idx))
        f.write('vector<F64> B%d(l%d, 0.0);\n\n' %(idx, idx))
    elif value['name'] == 'conv2d':
        # Variables used for conv2d (and maxpooling) layers
        if "output_flatten" in value.keys():
            f.write('int l%d = %d;\n' % (idx, value['output_flatten'][0]))
            if "output_maxpool" in value.keys():
                f.write('int l%d_mp = %d;\n' % (idx, value['output_maxpool'][-1]))
                f.write('int l%d_mp_len = %d;\n' % (idx, value['maxpool']['kernel_shape'][0]))
        else:
            if "output_maxpool" in value.keys():
                f.write('int l%d = %d;\n' % (idx, value['output_maxpool'][-1]))
                f.write('int l%d_mp = %d;\n' % (idx, value['output_maxpool'][-1]))
                f.write('int l%d_mp_len = %d;\n' % (idx, value['maxpool']['kernel_shape'][0]))
            else:
                f.write('int l%d = %d;\n' % (idx, value['output'][-1]))
        f.write('int l%d_conv = %d;\n' % (idx, value['output'][0]))
        f.write('int l%d_size = %d;\n' % (idx, value['conv']['kernel_shape'][0]))
        if 'maxpool' in model[idx - 1].keys():
            f.write('int l%d_ic = %d;\n' % (idx, value['input'][0]))
        else:
            f.write('int l%d_ic = %d;\n' % (idx, value['input'][-1]))
        f.write('int l%d_kc = %d;\n' % (idx, value['output'][-1]))

        if "output_flatten" in value.keys():
            if 'maxpool' not in value.keys():
                # f.write('vector<vector<F64>> U%d(l%d, vector<F64>(nslots, 0.0));\n' %(idx, idx))
                f.write('vector<Ctx> U%d_cipher(l%d);\n' %(idx, idx))
            else:
                # f.write('vector<vector<vector<vector<F64>>>> U%d(l%d, vector<vector<vector<F64>>>(l%d, vector<vector<F64>>(l%d_kc, vector<F64>(nslots, 0.0))));\n' %(idx, idx, idx, idx))
                f.write('vector<vector<vector<Ctx>>> U%d_cipher(l%d, vector<vector<Ctx>>(l%d, vector<Ctx>(l%d_kc)));\n' %(idx, idx, idx, idx))
            # f.write('vector<vector<F64>> D%d(l%d, vector<F64>(nslots, 0.0));\n' %(idx, idx))
            f.write('vector<Ctx> D%d_cipher(l%d);\n' %(idx, idx))
            if model[idx + 1]['name'] == 'dense' and 'maxpool' not in value.keys():
                # f.write('vector<vector<F64>> DE%d(l%d, vector<F64>(nslots, 0.0));\n' %(idx, idx))
                f.write('vector<Ctx> DE%d_cipher(l%d);\n' %(idx, idx))
            else:
                # f.write('vector<vector<vector<vector<F64>>>> DE%d(l%d_conv, vector<vector<vector<F64>>>(l%d_conv, vector<vector<F64>>(l%d_kc, vector<F64>(nslots, 0.0))));\n' %(idx, idx, idx, idx))
                f.write('vector<vector<vector<Ctx>>> DE%d_cipher(l%d_conv, vector<vector<Ctx>>(l%d_conv, vector<Ctx>(l%d_kc)));\n' %(idx, idx, idx, idx))
        else:
            # f.write('vector<vector<vector<vector<F64>>>> U%d(l%d_conv, vector<vector<vector<F64>>>(l%d_conv, vector<vector<F64>>(l%d_kc, vector<F64>(nslots, 0.0))));\n' %(idx, idx, idx, idx))
            f.write('vector<vector<vector<Ctx>>> U%d_cipher(l%d_conv, vector<vector<Ctx>>(l%d_conv, vector<Ctx>(l%d_kc)));\n' %(idx, idx, idx, idx))
            # f.write('vector<vector<vector<vector<F64>>>> D%d(l%d_conv, vector<vector<vector<F64>>>(l%d_conv, vector<vector<F64>>(l%d_kc, vector<F64>(nslots, 0.0))));\n' %(idx, idx, idx, idx))
            f.write('vector<vector<vector<Ctx>>> D%d_cipher(l%d_conv, vector<vector<Ctx>>(l%d_conv, vector<Ctx>(l%d_kc)));\n' %(idx, idx, idx, idx))
            # f.write('vector<vector<vector<vector<F64>>>> DE%d(l%d_conv, vector<vector<vector<F64>>>(l%d_conv, vector<vector<F64>>(l%d_kc, vector<F64>(nslots, 0.0))));\n' %(idx, idx, idx, idx))
            f.write('vector<vector<vector<Ctx>>> DE%d_cipher(l%d_conv, vector<vector<Ctx>>(l%d_conv, vector<Ctx>(l%d_kc)));\n' %(idx, idx, idx, idx))

        if 'maxpool' in value.keys():
            # f.write('vector<vector<vector<vector<F64>>>> D%d_mp(l%d_mp, vector<vector<vector<F64>>>(l%d_mp, vector<vector<F64>>(l%d_kc, vector<F64>(nslots, 0.0))));\n' %(idx, idx, idx, idx))
            f.write('vector<vector<vector<Ctx>>> D%d_mp_cipher(l%d_mp, vector<vector<Ctx>>(l%d_mp, vector<Ctx>(l%d_kc)));\n' %(idx, idx, idx, idx))
            # f.write('vector<vector<vector<vector<F64>>>> DE%d_mp(l%d_mp, vector<vector<vector<F64>>>(l%d_mp, vector<vector<F64>>(l%d_kc, vector<F64>(nslots, 0.0))));\n' %(idx, idx, idx, idx))
            f.write('vector<vector<vector<Ctx>>> DE%d_mp_cipher(l%d_mp, vector<vector<Ctx>>(l%d_mp, vector<Ctx>(l%d_kc)));\n' %(idx, idx, idx, idx))
            # f.write('vector<vector<vector<vector<F64>>>> MI%d(l%d_conv, vector<vector<vector<F64>>>(l%d_conv, vector<vector<F64>>(l%d_kc, vector<F64>(nslots, 0.0))));\n' %(idx, idx, idx, idx))
            f.write('vector<vector<vector<Ctx>>> MI%d_cipher(l%d_conv, vector<vector<Ctx>>(l%d_conv, vector<Ctx>(l%d_kc)));\n' %(idx, idx, idx, idx))

        # Model weight and bias for dense layers
        f.write('vector<vector<vector<vector<F64>>>> W%d(l%d_kc, vector<vector<vector<F64>>>(l%d_ic, vector<vector<F64>>(l%d_size, vector<F64>(l%d_size, 0.0))));\n' %(idx, idx, idx, idx, idx))
        f.write('vector<F64> B%d(l%d_kc, 0.0);\n\n' %(idx, idx))

# Print variables used for LWE ciphertexts
f.write("// Variables for LWE ciphertexts\n")
if 'output_maxpool' in model[1].keys():
    f.write('vector<vector<lwe::Ctx_st>> U_lwe_cipher(l0_conv * l0_conv, vector<lwe::Ctx_st>(nslots));\n')
    f.write('vector<vector<lwe::Ctx_st>> D_lwe_cipher(l0_conv * l0_conv, vector<lwe::Ctx_st>(nslots));\n\n')
else:
    f.write('vector<vector<lwe::Ctx_st>> U_lwe_cipher(l1, vector<lwe::Ctx_st>(nslots));\n')
    f.write('vector<vector<lwe::Ctx_st>> D_lwe_cipher(l1, vector<lwe::Ctx_st>(nslots));\n\n')
# ==========================================
# === Part 2.3 Variables of Input Labels ===
# ==========================================
f.write("// Variables for input labels\n")
f.write('vector<vector<F64>> Y(l%d, vector<F64>(nslots, 0.0));\n' % (num_layer - 1))
f.write('vector<Ctx> Y_cipher(l%d);\n\n' % (num_layer - 1))

# =========================================
# === Part 2.4 Initialize Model Weights ===
# =========================================
f.write('// Init model weight and bias\n')
f.write('// Random number generator\n')
f.write('std::random_device rd; // Non-deterministic random seed\n')
f.write('std::mt19937 gen(rd()); // Mersenne twister engine seeded with random device\n')
f.write('std::uniform_real_distribution<double> dis(-0.3, 0.3);\n')
f.write('\n')
for idx, value in model.items():
    if idx == 0 or idx == num_layer: continue

    if value['name'] == 'dense':
        f.write('// W%d\n' % (idx)) 
        f.write('for(int i = 0; i < l%d; i++){\n' % (idx - 1)) 
        f.write('for(int j = 0; j < l%d; j++){\n' % (idx))
        f.write('W%d[i][j] = dis(gen);\n' % (idx))
        f.write('}\n')
        f.write('}\n\n')
        f.write('// B%d\n' % (idx)) 
        f.write('for(int i = 0; i < l%d; i++){\n' % (idx)) 
        f.write('B%d[i] = dis(gen);\n' % (idx))
        f.write('}\n\n')
    elif value['name'] == 'conv2d':
        f.write('// W%d\n' % (idx)) 
        f.write('for(int kc = 0; kc < l%d_kc; kc++){\n' % idx) 
        f.write('for(int ic = 0; ic < l%d_ic; ic++){\n' % idx) 
        f.write('for(int i = 0; i < l%d_size; i++){\n' % idx) 
        f.write('for(int j = 0; j < l%d_size; j++){\n' % (idx))
        f.write('W%d[kc][ic][i][j] = dis(gen);\n' % (idx))
        f.write('}\n')
        f.write('}\n')
        f.write('}\n')
        f.write('}\n\n')

        f.write('// B%d\n' % (idx)) 
        f.write('for(int i = 0; i < l%d_kc; i++){\n' % (idx)) 
        f.write('B%d[i] = dis(gen);\n' % (idx))
        f.write('}\n\n')

print("# =======================================")
print("# === Part 3. Print the Training Loop ===")
print("# =======================================")
f.write('// The Training Loop\n')
f.write('for(int epoch = 0; epoch < total_epoches; epoch++){\n')
f.write('std::cout << "=========" << std::endl;\n')
f.write('std::cout << "EPOCH " << epoch << std::endl;\n')
f.write('std::cout << "=========" << std::endl;\n')
f.write('\n')
f.write('if(epoch != 0) not_first_epoch = true;\n')
f.write('batch_start_point = nslots * epoch;\n')
f.write('\n')

# =======================================================================
# === Part 3.1. Read Inputs and Lables in the Beginning of each Epoch ===
# =======================================================================
f.write('std::cout << "Read the input data" << std::endl;\n')
if 'output_flatten' in model[0].keys():
    # Encrypt input as a 1D vector
    f.write('for(int i = 0; i < l0; i++){\n') 
    f.write('for(int j = 0; j < nslots; j++){\n')
    f.write('D0[i][j] = int(dataset.test_images[batch_start_point + j][i]) / 255.0;\n')
    f.write('}\n')
    f.write('pg_rt.EncodeThenEncrypt(D0[i], D0_cipher[i]);\n')
    f.write('}\n\n')
else:
    # Encrypt input as a 3D tensor
    f.write('for(int kc = 0; kc < l0_kc; kc++) {\n')
    f.write('for(int i = 0; i < l0_conv; i++) {\n')
    f.write('for(int j = 0; j < l0_conv; j++) {\n')
    f.write('for(int k = 0; k < nslots; k++) {\n')
    f.write('D0[i][j][kc][k] = int(dataset.test_images[batch_start_point + k][kc * l0_conv * l0_conv + i * l0_conv + j]) / 255.0;\n')
    f.write('}\n')
    f.write('}\n')
    f.write('}\n')
    f.write('}\n')

    f.write('for(int kc = 0; kc < l0_kc; kc++) {\n')
    f.write('for(int i = 0; i < l0_conv; i++) {\n')
    f.write('for(int j = 0; j < l0_conv; j++) {\n')
    f.write('pg_rt.EncodeThenEncrypt(D0[i][j][kc], D0_cipher[i][j][kc]);\n')
    f.write('}\n')
    f.write('}\n')
    f.write('}\n\n')

# Encrypt Labels
f.write('std::cout << "Read the input labels" << std::endl;\n')
f.write('for(int i = 0; i < nslots; i++){\n') 
f.write('Y[int(dataset.test_labels[batch_start_point + i])][i] = 1.0;\n')
f.write('}\n')

f.write('for(int i = 0; i < l%d; i++){\n' % (num_layer - 1)) 
f.write('pg_rt.EncodeThenEncrypt(Y[i], Y_cipher[i]);\n')
f.write('}\n')

f.write('pg_rt.s2c_and_extract(Y_cipher, U_lwe_cipher, l%d, nslots);\n' % (num_layer - 1))
f.write('pg_rt.repack(U_lwe_cipher, Y_cipher, l%d, nslots);\n\n' % (num_layer - 1))

# =====================================
# === Part 3.2. Forward Propagation ===
# =====================================
# Determine the ciphertext level required for weight updates
if len(model) == 3:
    level = 4
else:
    level = 5

# Parse the layer sequentially 
print(" >>> Generating Code for Forward Propagation <<< ")
for idx, value in model.items():
    if idx == 0 or idx == num_layer: continue # skip input and output
    print(idx, value)

    f.write('\n')
    f.write('std::cout << "===========================" << std::endl;\n')
    f.write('std::cout << "Layer %d FORWARD PROPAGATION" << std::endl;\n' % idx)
    f.write('std::cout << "===========================" << std::endl;\n')

    # Dense functions
    if value['name'] == 'dense':
        f.write('\n// The linear function of Layer %d\n' % idx)
        # Print the dense function
        f.write('pg_rt.dense(pp, D%d_cipher, W%d, B%d, U%d_cipher, %d, epoch);\n' % 
                (idx - 1, idx, idx, idx, idx))
        # Print the activation function
        f.write('\n// The activation function of the Layer %d\n' % idx)
        f.write('pg_rt.s2c_and_extract(U%d_cipher, U_lwe_cipher, l%d, nslots);\n' % (idx, idx))
        if 'relu' in value['act']:
            f.write('pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, D%d_cipher, l%d, nslots, "ReLU");\n' % (idx, idx))
        elif 'softmax' in value['act']:
            f.write('pg_rt.softmax(U_lwe_cipher, D_lwe_cipher, D%d_cipher, l%d, nslots);\n' % (idx, idx))

        # Print the derivative of the activation function
        if idx != num_layer - 1:
            f.write('\n// The derivative of the activation function of the Layer %d\n' % idx)
            f.write('pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, U%d_cipher, l%d, nslots, "DReLU");\n' % (idx, idx))
    # Conv functions
    elif value['name'] == 'conv2d':
        f.write('\n// The linear function of the Layer %d\n' % idx)
        is_bracket = False
        # f.write('MemoryPoolHandle my_pool = MemoryPoolHandle::New();\n')
        # f.write('auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));\n')

        the_input = 'D%d_cipher' % (idx - 1)
        # If padding is required, generate a temp 3-D variable
        if 'pads' in value['conv'].keys():
            if value['conv']['pads'] != 0:
                f.write('{\n')
                is_bracket = True
                the_input = 'tmp_input_cipher'
                f.write('// Create a tmp_cipher for padding\n')
                f.write('int input_padding = %d;\n' % (value['input'][0] + value['conv']['pads'][0] + value['conv']['pads'][-1]))
                f.write('vector<vector<vector<Ctx>>> tmp_input_cipher(input_padding, vector<vector<Ctx>>(input_padding, vector<Ctx>(l%d_ic)));\n' % (idx))
                if idx != 1 and 'output_maxpool' in model[idx - 1].keys():
                    f.write('pg_rt.padding(tmp_input_cipher, D%d_mp_cipher, %d, %d);\n' % (idx - 1, value['conv']['pads'][0], value['conv']['pads'][1]))
                else:
                    f.write('pg_rt.padding(tmp_input_cipher, D%d_cipher, %d, %d);\n' % (idx - 1, value['conv']['pads'][0], value['conv']['pads'][1]))
        else:
            if idx != 1 and 'output_maxpool' in model[idx - 1].keys():
                the_input = 'D%d_mp_cipher' % (idx - 1)

        # Print conv (and maxpooling) functions
        if 'maxpool' not in value.keys():
            the_output = 'U%d_cipher' % (idx)
            # If the following layer is a dense layer, generate a temp 3-D variable to catch the return of the conv layer
            if 'output_flatten' in value.keys():
                if is_bracket == False:
                    f.write('{\n')
                    is_bracket = True
                f.write('// Create a tmp_cipher for catching output values\n')
                the_output = 'tmp_output_cipher'
                f.write('vector<vector<vector<Ctx>>> tmp_output_cipher(l%d_conv, vector<vector<Ctx>>(l%d_conv, vector<Ctx>(l%d_kc)));\n' % (idx, idx, idx))

            f.write('\n')
            f.write('pg_rt.conv(pp, %s, W%d, B%d, %s, %d, %d, epoch);\n' 
                    % (the_input, idx, idx, the_output, value['conv']['strides'][0], idx))
            f.write('\n')
        else:
            if 'output_flatten' in value.keys():
                if is_bracket == False:
                    f.write('{\n')
                    is_bracket = True
            f.write('\n')
            f.write('pg_rt.conv(pp, %s, W%d, B%d, U%d_cipher, %d, %d, epoch);\n' 
                    % (the_input, idx, idx, idx, value['conv']['strides'][0], idx))

            the_output = 'D%d_mp_cipher' % (idx)
            if 'output_flatten' in value.keys():
                f.write('// Create a tmp_cipher for catching output values\n')
                the_output = 'tmp_output_cipher'
                f.write('vector<vector<vector<Ctx>>> tmp_output_cipher(l%d_mp, vector<vector<Ctx>>(l%d_mp, vector<Ctx>(l%d_kc)));\n' % (idx, idx, idx))

            f.write('pg_rt.maxpool(pp, U%d_cipher, MI%d_cipher, %s, U_lwe_cipher, D_lwe_cipher, l%d_mp_len);\n' % (idx, idx, the_output, idx))
            f.write('\n')

        # If the following layer is a dense layer, we need to flatten the output
        if 'output_flatten' in value.keys():
            f.write('// Flatten\n')
            if 'maxpool' in value.keys():
                f.write('pg_rt.flatten(tmp_output_cipher, D%d_cipher);\n' % idx)
            else:
                f.write('pg_rt.flatten(tmp_output_cipher, U%d_cipher);\n' % idx)

            if 'pads' in value['conv'].keys():
                f.write('\n')
                f.write('release_vector_3D(tmp_input_cipher);\n')
                f.write('tmp_input_cipher.clear();\n')

            f.write('release_vector_3D(tmp_output_cipher);\n')
            f.write('tmp_output_cipher.clear();\n')

        # f.write('MemoryManager::SwitchProfile(std::move(old_prof));\n')
        if is_bracket == True:
            f.write('}\n')

        # Print activation functions
        if 'maxpool' not in value.keys():
            if 'output_flatten' not in value.keys():
                if 'relu' in value['act']:
                    f.write('\n// The activation function of the Layer %d\n' % idx)
                    f.write('pg_rt.Relu_3D(U%d_cipher, D%d_cipher, U_lwe_cipher, nslots);\n' % (idx, idx))
            else:
                f.write('\n// The activation function of the Layer %d\n' % idx)
                f.write('pg_rt.s2c_and_extract(U%d_cipher, U_lwe_cipher, l%d, nslots);\n' % (idx, idx))
                if 'relu' in value['act']:
                    f.write('pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, D%d_cipher, l%d, nslots, "ReLU");\n' % (idx, idx))

        # Print the derivative of the activation functions
        if 'output_flatten' in value.keys() and 'maxpool' not in value.keys():
            f.write('\n// The derivative of the activation function of the Layer %d\n' % idx)
            f.write('pg_rt.act_batch(U_lwe_cipher, D_lwe_cipher, U%d_cipher, l%d, nslots, "DReLU");\n' % (idx, idx))
        elif 'maxpool' not in value.keys():
            f.write('\n// The derivative of the activation function of the Layer %d\n' % idx)
            f.write('pg_rt.Drelu_3D(U%d_cipher, U_lwe_cipher, nslots);\n' % (idx))

# =================================
# === Part 3.3. Backpropagation ===
# =================================
# Parse the layer in a reverse way
print(" >>> Generating Code for BackPropagation <<< ")
for idx in range(num_layer - 1, 0, -1):
    print(idx, model[idx])
    f.write('\n')
    f.write('std::cout << "==========================" << std::endl;\n')
    f.write('std::cout << "Layer %d DELTA CALCULATIONS" << std::endl;\n' % idx)
    f.write('std::cout << "==========================" << std::endl;\n')

    f.write('\n// Calculate delta of Layer %d\n' % idx)

    if idx == num_layer - 1:
        # Special function to calculate the delta of the last layer
        f.write('pg_rt.delta_softmax(pp, D%d_cipher, Y_cipher, DE%d_cipher);\n'%(idx, idx))
        continue
    if model[idx]['name'] == 'dense':
       if model[idx + 1]['name'] == 'dense':
            # Calculate delta with a 1-D delta and a 1-D data
            f.write('pg_rt.delta_1D_dense_1D(pp, W%d, DE%d_cipher, U%d_cipher, DE%d_cipher, %d, epoch);\n' % 
                    (idx + 1, idx + 1, idx, idx, idx + 1))
    elif model[idx]['name'] == 'conv2d':
        if 'maxpool' in model[idx].keys():
            if model[idx + 1]['name'] == 'dense':
                # Calculate delta for maxpooling layer, where the input delta is a 1-D vector
                f.write('pg_rt.prepare_delta_maxpooling_from_mat(pp, W%d, DE%d_cipher, DE%d_mp_cipher, %d, epoch);\n' % 
                        (idx + 1, idx + 1, idx, idx))
                f.write('pg_rt.delta_maxpooling(pp, DE%d_mp_cipher, U%d_cipher, MI%d_cipher, U_lwe_cipher, DE%d_cipher);\n' % 
                        (idx, idx, idx, idx,))
            elif model[idx + 1]['name'] == 'conv2d':
                # Calculate delta for maxpooling layer, where the input delta is a 3-D tensor
                f.write('pg_rt.prepare_delta_maxpooling_from_conv(pp, W%d, DE%d_cipher, DE%d_mp_cipher, %d, epoch);\n' % 
                        (idx + 1, idx + 1, idx, idx))
                f.write('pg_rt.delta_maxpooling(pp, DE%d_mp_cipher, U%d_cipher, MI%d_cipher, U_lwe_cipher, DE%d_cipher);\n' % 
                        (idx, idx, idx, idx))
        elif model[idx + 1]['name'] == 'dense':
            # Calculate 1D delta with a 1-D delta and a 1-D data
            f.write('pg_rt.delta_1D_dense_1D(pp, W%d, DE%d_cipher, U%d_cipher, DE%d_cipher, %d, epoch);\n' % 
                    (idx + 1, idx + 1, idx, idx, idx + 1))
        elif model[idx + 1]['name'] == 'conv2d' and 'output_flatten' in model[idx + 1].keys():
            # Calculate 3D delta with a 1-D delta.
            # First reshape 1D delta into 3-D, and dilate the 3-D delta according to stride length
            f.write('{\n')
            # f.write('MemoryPoolHandle my_pool = MemoryPoolHandle::New();\n')
            # f.write('auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));\n')
            f.write('// Create a tmp_cipher for data dilation\n')
            f.write('int l%d_dilate = l%d_conv + (l%d_conv - 1) * (%d - 1);\n' % (idx+1, idx+1, idx+1, model[idx+1]['conv']['strides'][1]))
            f.write('vector<vector<vector<Ctx>>> tmp_cipher(l%d_dilate, vector<vector<Ctx>>(l%d_dilate, vector<Ctx>(l%d_kc)));\n' % (idx+1, idx+1, idx+1))
            f.write('pg_rt.delta_1D_to_3D_dilate(pp, DE%d_cipher, tmp_cipher, l%d_conv, %d, 4);\n' % (idx+1, idx+1, model[idx+1]['conv']['strides'][0]))
            f.write('pg_rt.delta_3D_conv_3D(pp, W%d, tmp_cipher, U%d_cipher, DE%d_cipher, %d, epoch);\n' % 
                    (idx + 1, idx, idx, idx + 1))
            f.write('\n')
            f.write('release_vector_3D(tmp_cipher);\n')
            f.write('tmp_cipher.clear();\n')
            # f.write('MemoryManager::SwitchProfile(std::move(old_prof));\n')
            f.write('}\n')
        elif model[idx + 1]['name'] == 'conv2d' and 'output_flatten' not in model[idx + 1].keys():
            # Calculate 3-D delta with a 3-D tensor delta
            f.write('pg_rt.delta_3D_conv_3D(pp, W%d, DE%d_cipher, U%d_cipher, DE%d_cipher, %d, epoch);\n' % 
                    (idx + 1, idx + 1, idx, idx, idx + 1))

    # Repack delta to a higher CKKS level
    if idx != num_layer - 1 and idx != 0:
        f.write('// Repack delta to a higher level\n')
        if model[idx + 1]['name'] == 'dense' and 'maxpool' not in model[idx].keys():
            f.write('pg_rt.s2c_repack_1D(DE%d_cipher, U_lwe_cipher, nslots);\n' % (idx))
        else:
            f.write('pg_rt.s2c_repack_3D(DE%d_cipher, U_lwe_cipher, nslots);\n' % (idx))

# ================================
# === Part 3.4. Weight Updates ===
# ================================
# Parse the structure sequentially again
print(" >>> Generating Code for Weight Updates <<< ")
for idx, value in model.items():
    if idx == 0 or idx == num_layer: continue
    print(idx, value)

    f.write('\n')
    f.write('std::cout << "======================" << std::endl;\n')
    f.write('std::cout << "Layer %d WEIGHT UPDATES" << std::endl;\n' % idx)
    f.write('std::cout << "======================" << std::endl;\n')

    f.write('\n// Update model weights of Layer %d\n' % idx)

    # Update weights of dense layers
    if value['name'] == 'dense':
        f.write('pg_rt.update_model_dense(pp, D%d_cipher, DE%d_cipher, W%d, B%d, %d, %d, epoch);\n' % 
                (idx - 1, idx, idx, idx, level, idx))
    # Update weights of conv2d layers
    elif value['name'] == 'conv2d':
        is_bracket = False;
        # f.write('MemoryPoolHandle my_pool = MemoryPoolHandle::New();\n')
        # f.write('auto old_prof = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool)));\n')

        # If padding is required, generate a temp 3-D variable
        if 'pads' in value['conv'].keys():
            if value['conv']['pads'] != 0:
                f.write('{\n')
                is_bracket = True;
                f.write('// Create a tmp_cipher for padding\n')
                f.write('int input_padding = %d;\n' % (value['input'][0] + value['conv']['pads'][0] + value['conv']['pads'][-1]))
                f.write('vector<vector<vector<Ctx>>> tmp_input_cipher(input_padding, vector<vector<Ctx>>(input_padding, vector<Ctx>(l%d_ic)));\n' % (idx))
                if idx != 1 and 'output_maxpool' in model[idx - 1].keys():
                    f.write('pg_rt.padding(tmp_input_cipher, D%d_mp_cipher, %d, %d);\n' % (idx - 1, value['conv']['pads'][0], value['conv']['pads'][1]))
                else:
                    f.write('pg_rt.padding(tmp_input_cipher, D%d_cipher, %d, %d);\n' % (idx - 1, value['conv']['pads'][0], value['conv']['pads'][1]))
        # If conv with strides, create a temp 3-D variable for dilation
        if 'strides' in value['conv'].keys() and value['conv']['strides'][0] > 1:
            if is_bracket == False:
                f.write('{\n')
                is_bracket = True
            f.write('// Create a tmp_cipher for data dilation\n')
            f.write('int l%d_dilate = l%d_conv + (l%d_conv - 1) * (%d - 1);\n' % (idx, idx, idx, value['conv']['strides'][1]))
            f.write('vector<vector<vector<Ctx>>> tmp_cipher(l%d_dilate, vector<vector<Ctx>>(l%d_dilate, vector<Ctx>(l%d_kc)));\n' % (idx, idx, idx))

        if model[idx + 1]["name"] == 'conv2d':
            if 'pads' in value['conv'].keys():
                the_input = 'tmp_input_cipher'
            elif 'output_maxpool' in model[idx - 1].keys():
                the_input = 'D%d_mp_cipher' % (idx - 1)
            else:
                the_input = 'D%d_cipher' % (idx - 1)
            if 'strides' in value['conv'].keys() and value['conv']['strides'][0] > 1:
                the_output = 'tmp_cipher'
                f.write('pg_rt.delta_3D_to_3D_dilate(pp, DE%d_cipher, tmp_cipher, l%d_conv, %d, 4);\n' % (idx, idx, value['conv']['strides'][0]))
            else:
                the_output = 'DE%d_cipher' % (idx)

            if is_bracket:
                f.write('\n')
            f.write('pg_rt.update_model_conv_with_3D_delta(pp, %s, %s, W%d, B%d, %d, %d, epoch);\n' %
                    (the_input, the_output, idx, idx, level, idx))
        else:
            if 'pads' in value['conv'].keys():
                the_input = 'tmp_input_cipher'
            elif 'output_maxpool' in model[idx - 1].keys():
                the_input = 'D%d_mp_cipher' % (idx - 1)
            else:
                the_input = 'D%d_cipher' % (idx - 1)
            if 'strides' in value['conv'].keys() and value['conv']['strides'][0] > 1:
                the_output = 'tmp_cipher'
                f.write('pg_rt.delta_1D_to_3D_dilate(pp, DE%d_cipher, tmp_cipher, l%d_conv, %d, 4);\n' % (idx, idx, value['conv']['strides'][0]))
            else:
                the_output = 'DE%d_cipher' % (idx)
                # f.write('pg_rt.delta_1D_to_3D_dilate(pp, DE%d_cipher, tmp_cipher, l%d_conv, 1, 4);\n' % (idx, idx))

            if is_bracket:
                f.write('\n')
            f.write('pg_rt.update_model_conv_with_3D_delta(pp, %s, %s, W%d, B%d, %d, %d, epoch);\n' %
                    (the_input, the_output, idx, idx, level, idx))

        if is_bracket:
            f.write('\n');
        # Release memory of temp variables 
        if 'pads' in value['conv'].keys():
            f.write('release_vector_3D(tmp_input_cipher);\n')
            f.write('tmp_input_cipher.clear();\n')
        if 'strides' in value['conv'].keys() and value['conv']['strides'][0] > 1:
            f.write('release_vector_3D(tmp_cipher);\n')
            f.write('tmp_cipher.clear();\n')
        # f.write('MemoryManager::SwitchProfile(std::move(old_prof));\n')
        if is_bracket == True:
            f.write('}\n')
    

f.write('}\n')
print("# ================================================")
print("# === Part 4. Output Training Time Information ===")
print("# ================================================")
f.write('\n')
f.write('// The Training Time Information\n')
f.write('std::cout << "TOTAL SAVE MODEL TIME " << total_save_model_time << std::endl;\n')
f.write('std::cout << "TOTAL LOAD MODEL TIME " << total_load_model_time << std::endl;\n')
f.write('std::cout << "TOTAL OFFLINE TIME " << total_offline_time << std::endl;\n')
f.write('std::cout << "TOTAL ONLINE MODEL TIME " << total_online_time << std::endl;\n')
f.write('\n')
f.write('return 0;\n')
f.write('}\n')

f.close()
