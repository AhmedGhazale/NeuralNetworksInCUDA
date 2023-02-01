//
// Created by ahmed on ٢٣‏/٨‏/٢٠٢١.
//

#ifndef NEURALNETWORKS_ACTIVITION_FUNCTIONS_CUH
#define NEURALNETWORKS_ACTIVITION_FUNCTIONS_CUH
const int SIGMOID_THREADS = 1024;
#include <cassert>     /* assert */


__global__
void sigmoid_kernel(const float* a, float* b, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx>=size)return;
    b[idx] = 1.0f / (1.0f + exp(-a[idx]));
    //b[idx] = a[idx];
}
__global__
void sigmoid_prime_kernel(const float* a, float* b, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=size)return;
    float sigmoid_val =  1.0f / (1.0f + exp(-a[idx]));
    b[idx] = sigmoid_val * (1 - sigmoid_val);
}

Tensor<float> sigmoidTensor(Tensor<float> &a){

    int BLOCKS = ceil(double(a.size())/double(SIGMOID_THREADS));

    //Tensor<float>b(a.shape(),Fill::ZEROS, Device::CUDA);
    Tensor<float>b(a.shape(),Fill::NOFILL, Device::CUDA);
    assert(a.get_device() == Device::CUDA);

    dim3 threads(SIGMOID_THREADS);
    dim3 blocks(BLOCKS);

    sigmoid_kernel<<<blocks,threads>>>(a.get_pointer(),b.get_pointer(),a.size());

    return b;
}


Tensor<float> sigmoidPrimeTensor(Tensor<float> &a){

    assert(a.get_device() == Device::CUDA);

    int BLOCKS = ceil(double(a.size())/double(SIGMOID_THREADS));

    Tensor<float>b(a.shape(),Fill::NOFILL,Device::CUDA);
    dim3 threads(SIGMOID_THREADS);
    dim3 blocks(BLOCKS);

    sigmoid_prime_kernel<<<blocks,threads>>>(a.get_pointer(),b.get_pointer(),a.size());
    cudaDeviceSynchronize();
    return b;
}



#endif //NEURALNETWORKS_ACTIVITION_FUNCTIONS_CUH
