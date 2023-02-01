cmake_minimum_required(VERSION 3.20)
project(NeuralNetworks CUDA)

set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include_directories(include)

add_executable(NeuralNetworks main.cu )


target_link_libraries(NeuralNetworks CUDA::curand )

set_target_properties(NeuralNetworks PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
