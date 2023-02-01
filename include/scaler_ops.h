//
// Created by ahmed on ٤‏/٩‏/٢٠٢١.
//

#ifndef NEURALNETWORKS_SCALER_OPS_H
#define NEURALNETWORKS_SCALER_OPS_H

const int SCALAR_THREADS = 1024;

__global__
void scalar_multi_kernel(const float* a, float* b,float val, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx>=size)return;
    b[idx] = a[idx] * val;
}

__global__
void scalar_pow2_kernel(const float* a, float* b, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx>=size)return;
    b[idx] = a[idx] * a[idx];
}


Tensor<float> scalarMultiplyTensor(Tensor<float> &a, float v){

    int BLOCKS = ceil(double(a.size())/double(SCALAR_THREADS));

    Tensor<float>b(a.shape(),Fill::NOFILL, Device::CUDA);
    assert(a.get_device() == Device::CUDA);

    dim3 threads(SCALAR_THREADS);
    dim3 blocks(BLOCKS);

    scalar_multi_kernel<<<blocks,threads>>>(a.get_pointer(),b.get_pointer(),v,a.size());

    return b;
}

Tensor<float> scalarSquareTensor(Tensor<float> &a){

    int BLOCKS = ceil(double(a.size())/double(SCALAR_THREADS));

    //Tensor<float>b(a.shape(),Fill::ZEROS, Device::CUDA);
    Tensor<float>b(a.shape(),Fill::NOFILL, Device::CUDA);
    assert(a.get_device() == Device::CUDA);

    dim3 threads(SCALAR_THREADS);
    dim3 blocks(BLOCKS);

    scalar_pow2_kernel<<<blocks,threads>>>(a.get_pointer(),b.get_pointer(),a.size());

    return b;
}

#endif //NEURALNETWORKS_SCALER_OPS_H
