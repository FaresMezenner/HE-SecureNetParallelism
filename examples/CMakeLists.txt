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

add_executable(model_b_bc_exe model_b_breast_cancer.cpp)
target_link_libraries(model_b_bc_exe pegasus)

add_executable(model_b_bc_4l_exe model_b_bc_4layer.cpp)
target_link_libraries(model_b_bc_4l_exe pegasus)

add_executable(model_b_par_exe model_b_bc_parallel.cpp)
target_link_libraries(model_b_par_exe pegasus)
target_compile_options(model_b_par_exe PRIVATE -pthread)
target_link_options(model_b_par_exe PRIVATE -pthread)

add_executable(model_b_pipe_exe model_b_bc_pipeline.cpp)
target_link_libraries(model_b_pipe_exe pegasus)
target_compile_options(model_b_pipe_exe PRIVATE -pthread)
target_link_options(model_b_pipe_exe PRIVATE -pthread)

add_executable(model_b_dual_rt_exe model_b_bc_dual_rt.cpp)
target_link_libraries(model_b_dual_rt_exe pegasus)
target_compile_options(model_b_dual_rt_exe PRIVATE -pthread)
target_link_options(model_b_dual_rt_exe PRIVATE -pthread)
