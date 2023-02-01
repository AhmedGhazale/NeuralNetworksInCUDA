//
// Created by ahmed on ٩‏/٨‏/٢٠٢١.
//

#ifndef MATMUL_TENSOR_H
#define MATMUL_TENSOR_H
using namespace std;


#include <cuda.h>
#include <curand.h>
#include <memory>
#include <search.h>

enum Device{
    CUDA = 1,
    CPU = 2
};
enum Fill{
    ZEROS = 0,
    RAND = 1,
    NOFILL = 2,
    ONES = 3
};


std::default_random_engine m_generator;

template<class T>
class Tensor{
private:
    //T* m_data;
    shared_ptr<T>m_data;
    int m_size;
    //T* m_data_dev;
    shared_ptr<T>m_data_dev;
    Device m_device;
    vector<int>m_shape;


public:

    Tensor(){}

    explicit Tensor(vector<int>shape, Fill fill,  Device device){
        int size = 1;
        for (auto i : shape)size *= i;
        m_size = size;

        if (device== Device::CPU){
            tensor_init_cpu(fill, size);
        }else{
            tensor_init_gpu(fill, size);
        }
        m_size = size;
        m_device = device;
        m_shape = shape;
    }

    int size(){
        return m_size;
    }
    vector<int> shape(){
        return m_shape;
    }

    void print(int limit=1e9){
        bool on_gpu = false;
        if (m_device == Device::CUDA){
            //cout<<"Move Tensor to cpu first"<<endl;
            this->cpu();
            on_gpu = true;
            //return;
        }
        cout<<"[";
        for( int i = 0;i<min(m_size,limit);i++){
            cout<<m_data.get()[i]<<", ";
        }
        cout<<"]"<<endl;
        if (on_gpu)this->cuda();
    }

    T* get_pointer(){
        if (m_device == Device::CPU){
            return m_data.get();
        }else{
            return m_data_dev.get();
        }
    }

    void cuda(){
        if (m_device == Device::CUDA)return;
        T* data_dev;
        cudaMalloc(&data_dev, m_size * sizeof(T));
        cudaMemcpy( data_dev, m_data.get(), m_size * sizeof(T), cudaMemcpyHostToDevice);
        m_data.reset();
        m_data_dev.reset();
        m_data_dev = std::shared_ptr<T>(data_dev,[&](T* ptr){ cudaFree(ptr); });
        m_device = Device::CUDA;
    }
    void cpu(){
        if(m_device == Device::CPU)return;
        T* data;
        data = (T*)malloc(m_size * sizeof(T));
        cudaMemcpy(data, m_data_dev.get(), m_size * sizeof(T), cudaMemcpyDeviceToHost);
        m_data_dev.reset();
        m_data.template reset(data);
        m_device=Device::CPU;
    }
    Tensor<T> slice(int s, int e) {
        int other_dims_size = 1;
        for (int i = 1; i < this->shape().size(); i++) {
            other_dims_size *= this->shape()[i];
        }
        auto new_shape = this->shape();
        new_shape[0] = e - s;
        int s_b = s * other_dims_size;
        int e_b = e * other_dims_size;
        int size_b = e_b - s_b;

        Tensor<T> res(new_shape, Fill::NOFILL, m_device);
        if (m_device == CPU) {
            memcpy(res.get_pointer(),this->get_pointer() + s_b,  size_b * sizeof(T));
        } else {
            cudaMemcpy(res.get_pointer(), this->get_pointer() + s_b,  size_b * sizeof(T), cudaMemcpyDeviceToDevice);
        }
        return res;
    }

    Device get_device(){
        return m_device;
    }
private:

    void tensor_init_cpu(Fill fill, int size){
        T* data = (T*)malloc(size * sizeof(T));

        if (fill == Fill::ZEROS) {
            std::generate(data, data + size, []() { return 0; });
        }else if (fill == Fill::ONES){
            std::generate(data,data+size, [](){return 1;});
        }else if (fill== Fill::RAND){
            std::normal_distribution<T> distribution(0,1);
            std::generate(data,data+size, [&distribution](){return distribution(m_generator);});
        }else{
            // perform nothing
        }
        m_data = shared_ptr<T>(data);
    }
    void tensor_init_gpu(Fill fill, int size){
        T* data_dev;
        cudaMalloc(&data_dev, size * sizeof(T));

        if (fill == Fill::RAND) {
            curandGenerator_t prng;
            curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
            curandGenerateNormal(prng, data_dev, size, 0, 1);\
        }else if (fill == Fill::ZEROS){
            cudaMemset(data_dev,0, size * sizeof(T));
        //}else if (fill == Fill::ONES){
        //    cudaMemset(m_data_dev,1, size * sizeof(T));
        }else{
            // perform nothing
        }
        m_data_dev = std::shared_ptr<T>(data_dev,[&](T* ptr){ cudaFree(ptr); });
    }

};



#endif //MATMUL_TENSOR_H
