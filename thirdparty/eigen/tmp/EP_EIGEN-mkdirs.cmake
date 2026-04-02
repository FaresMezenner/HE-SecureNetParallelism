# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/eigen/src"
  "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/eigen/build"
  "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/eigen/EP_EIGEN-prefix"
  "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/eigen/tmp"
  "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/eigen/stamp"
  "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/eigen/EP_EIGEN-prefix/src"
  "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/eigen/stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/eigen/stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/eigen/stamp${cfgdir}") # cfgdir has leading slash
endif()
