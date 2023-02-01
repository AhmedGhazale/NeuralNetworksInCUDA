//
// Created by ahmed on ٤‏/٩‏/٢٠٢١.
//

#ifndef NEURALNETWORKS_TENSOR_OPS_H
#define NEURALNETWORKS_TENSOR_OPS_H

const int SUBTRACT_THREADS = 1024;

__global__
void subtract_kernel(const float* a,const float* b,  float* c, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx>=size)return;

    c[idx] = a[idx] - b[idx];

}

__global__
void multi_kernel(const float* a,const float* b,  float* c, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx>=size)return;

    c[idx] = a[idx] * b[idx];

}




Tensor<float> subtractTensor(Tensor<float> &a, Tensor<float> &b){

    int BLOCKS = ceil(double(a.size())/double(SUBTRACT_THREADS));



    assert(a.get_device() == Device::CUDA && b.get_device() == Device::CUDA && a.shape() == b.shape() );

    Tensor<float>c(a.shape(),Fill::NOFILL, Device::CUDA);

    dim3 threads(SUBTRACT_THREADS);
    dim3 blocks(BLOCKS);

    subtract_kernel<<<blocks,threads>>>(a.get_pointer(),b.get_pointer(),c.get_pointer(), a.size());

    return c;
}

Tensor<float> multiplyTensor(Tensor<float> &a, Tensor<float> &b){

    int BLOCKS = ceil(double(a.size())/double(SUBTRACT_THREADS));

    assert(a.get_device() == Device::CUDA && b.get_device() == Device::CUDA && a.shape() == b.shape() );

    Tensor<float>c(a.shape(),Fill::NOFILL, Device::CUDA);

    dim3 threads(SUBTRACT_THREADS);
    dim3 blocks(BLOCKS);

    multi_kernel<<<blocks,threads>>>(a.get_pointer(),b.get_pointer(),c.get_pointer(), a.size());

    return c;
}



#endif //NEURALNETWORKS_TENSOR_OPS_H
