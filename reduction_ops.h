//
// Created by ahmed on ٥‏/٩‏/٢٠٢١.
//

#ifndef NEURALNETWORKS_REDUCTION_OPS_H
#define NEURALNETWORKS_REDUCTION_OPS_H


#include "Tensor.h"

const int REDUCE_THREADS = 1024;

__global__
void reduce_sum_kernel(const float* a, float* b, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx>=size)return;
    atomicAdd(b, a[idx]);
}


Tensor<float> reduceSumTensor(Tensor<float> &a){

    int BLOCKS = ceil(double(a.size())/double(REDUCE_THREADS));

    Tensor<float>b({1},Fill::ZEROS, Device::CUDA);

    assert(a.get_device() == Device::CUDA);

    dim3 threads(REDUCE_THREADS);
    dim3 blocks(BLOCKS);

    reduce_sum_kernel<<<blocks,threads>>>(a.get_pointer(),b.get_pointer(),a.size());

    return b;
}


#endif //NEURALNETWORKS_REDUCTION_OPS_H
