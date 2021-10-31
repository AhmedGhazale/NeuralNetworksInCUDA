cmake_minimum_required(VERSION 3.20)
project(NeuralNetworks CUDA)

set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)


add_executable(NeuralNetworks main.cu NeuralNetwork.cpp NeuralNetwork.h NeuralNetwork.cpp MatrixMultiplication.cuh Tensor.h activition_functions.cuh tensor_ops.h scaler_ops.h reduction_ops.h)

target_link_libraries(NeuralNetworks CUDA::curand )

set_target_properties(NeuralNetworks PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
