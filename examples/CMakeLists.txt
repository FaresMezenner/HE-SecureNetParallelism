find_package(MNIST PATHS ../dataset/mnist)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)
include_directories(${MNIST_INCLUDE_DIR})

add_executable(lut_exe lut.cc)
target_link_libraries(lut_exe pegasus)

add_executable(repacking_exe repacking.cc)
target_link_libraries(repacking_exe pegasus)

add_executable(model_a_exe model_a_e2e.cpp)
target_link_libraries(model_a_exe pegasus)

add_executable(model_b_exe model_b.cpp)
target_link_libraries(model_b_exe pegasus)

add_executable(model_c_exe model_c.cpp)
target_link_libraries(model_c_exe pegasus)

add_executable(model_d_exe model_d.cpp)
target_link_libraries(model_d_exe pegasus)
