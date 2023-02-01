//
// Created by ahmed on ٢٣‏/٨‏/٢٠٢١.
//

#ifndef NEURALNETWORKS_MATRIXMULTIPLICATION_CUH
#define NEURALNETWORKS_MATRIXMULTIPLICATION_CUH
#include "Tensor.h"
#include <cassert>     /* assert */


const int THREADS = 32;


__global__ void matmul_kernel(const float *a, const float *b, float *c, int M, int K, int N, bool transA, bool transB) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col>=N )return;
    // Iterate over row, and down column

    float tmp = 0;
    for (int i = 0; i < K; i++) {
        // Accumulate results for a single element
        float v1;
        float v2;
        if (transA){
            v1 = a[i * M + row];
        }else{
            v1 = a[row * K + i] ;
        }

        if (transB){
            //v2 = a[col * K + i];
            v2 = b[col * K + i];
        }else{
            v2 = b[i * N + col];
        }
        //printf(" * %d %d  - %d %f %f\n",row,col, i,v1,v2);

        //tmp += a[row * K + k] * b[k * N + col];
        tmp += v1 * v2;
    }
    //c[ row * N + col] = tmp ;
    c[row * N + col] = tmp ;
}

void matrixMultiplication(const float *a,const float *b, float *c, int M, int K, int N, bool transA , bool transB){
    // MxN = MxK * KxN
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block

    int BLOCKS_X = ceil(double(N)/double(THREADS));
    int BLOCKS_Y = ceil(double(M)/double(THREADS));

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS_X, BLOCKS_Y);

    matmul_kernel<<<blocks, threads>>>(a, b, c, M, K, N, transA,transB);
    cudaDeviceSynchronize();
}

Tensor<float> mutmulTensor(Tensor<float> &a, Tensor<float> &b, bool transA = false, bool transB = false){

    auto shapeA = a.shape();
    auto shapeB = b.shape();
    if (transA){
        reverse(shapeA.begin(),shapeA.end());
    }if (transB){
        reverse(shapeB.begin(), shapeB.end());
    }
    assert(shapeA.size() == 2 && shapeB.size()==2 &&  shapeA[1] == shapeB[0]);
    assert(a.get_device() == Device::CUDA && b.get_device() == Device::CUDA);

    auto shape = {shapeA[0], shapeB[1]};

    Tensor<float>c(shape,Fill::NOFILL, Device::CUDA);


    matrixMultiplication(a.get_pointer(),b.get_pointer(),c.get_pointer(),shapeA[0],shapeA[1],shapeB[1], transA, transB);

    return c;
}

#endif //NEURALNETWORKS_MATRIXMULTIPLICATION_CUH
