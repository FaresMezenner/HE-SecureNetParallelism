# Install script for directory: /home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SEAL-3.5/seal" TYPE FILE FILES
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/batchencoder.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/biguint.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/ciphertext.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/ckks.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/modulus.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/context.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/decryptor.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/intencoder.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/encryptionparams.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/encryptor.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/evaluator.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/galoiskeys.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/intarray.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/keygenerator.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/kswitchkeys.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/memorymanager.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/plaintext.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/publickey.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/randomgen.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/randomtostd.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/relinkeys.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/seal.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/secretkey.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/serializable.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/serialization.h"
    "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/Gemini-SEAL/native/src/seal/valcheck.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/build/thirdparty/Gemini-SEAL/native/src/seal/util/cmake_install.cmake")

endif()

