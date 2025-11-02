# HE-SecureNet: Private, Efficient and Usable Homomorphic Encryption-based Model Training

This repository contains the implementation of the paper **HE-SecureNet: Private, Efficient and Usable Homomorphic Encryption-based Model Training** by Thomas Schneider, Huan-Chih Wang, and Hossein Yalame.

The paper was accepted at the Workshop on Privacy in the Electronic Society (WPES), held in conjunction with ACM CCS 2025, and is available [here](https://eprint.iacr.org/2025/1591.pdf).

## Repository Structure

- cmake - CMake helper files.
- dataset - Breast Cancer and MNIST datasets.
- examples - Private C++ training codes.
    - model_a: Logistic Regression on the Breast Cancer Dataset
	- model_b: MLP on the MNIST Dataset
	- model_c: CNN on the MNIST Dataset
	- model_d: LeNet-5 on the MNIST Dataset
- licenses - Licensing information of the utilized libraries.
- onnx - ONNX graph files.
- parser - Our private training code generator.
- pegasus - The Pegasus backend
- python - Python scripts to generate corresponding ONNX model graph file.
- thirdparty - The thirdparty libraries for Pegasus.

## Requirements

- C++ compiler that supports at least C++14 standard
- Cmake >= 3.10
- GMP
- OpenMP
- [MNIST Reader](https://github.com/wichtounet/mnist/tree/master)
- Python Packages
    - numpy 1.26.4
    - onnx 1.16.2
    - onnxruntime 1.18.0
    - scipy 1.14.1
    - tensosflow 2.19.0
    - tf2onnx 1.16.1

## Installation
After cloning or downloading this repository, there are two ways to build HE-SecureNet.

### Install Manually
```
cd HE-SecureNet
# Install Python packages
pip3 install -r requirements.txt
# Download the MNIST dataset
cd dataset
git clone https://github.com/wichtounet/mnist.git
cd ../
# Build HE-SecureNet
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF
make -j8
```

### Installation with docker
In the HE-SecureNet directory, run ```docker build -t he-securenet .``` to build the docker image.

## Usage
The following steps demonstrate how to use HE-SecureNet, including:
1.	Generating the ONNX file using TensorFlow
2.	Creating the private C++ training code with HE-SecureNet’s ONNX parser
3.	Executing private model training

We provide an example of training the MLP model, which denoted as "model_b" in this repo.

### Generate ONNX graph file
1. Navigate to ```python``` folder and run ```python3 model_b.py``` to generate the ONNX graph file.
2. The ONNX file is created in the ```onnx``` folder.

### Generate private C++ training code
1. Navigate to ```parser``` folder and run ```python3 parser.py model_b``` to get the C++ training code.
2. The private training code is stored in the ```examples``` folder.

### Execute HE-SecureNet
1. Navigate to ```build``` folder and run ```make -j8``` to generate executable binary file.
2. Run ```./model_b_exe``` to start training the model in the ciphertext domain.

## Cite HE-SecureNet
```
@inproceedings{HESecureNET,
  title={{HE-SecureNet: An Efficient and Usable Framework for Model Training via Homomorphic Encryption}},
  author={ Schneider, Thomas and Wang, Huan-Chih, and Yalame, Hossein},
  booktitle={WPES},
  year={2025}
}
```
