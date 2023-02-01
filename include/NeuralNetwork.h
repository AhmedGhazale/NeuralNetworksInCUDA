
#ifndef NEURALNETWORKS_NEURALNETWORK_H
#define NEURALNETWORKS_NEURALNETWORK_H
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include "Tensor.h"
#include "activition_functions.cuh"
#include "MatrixMultiplication.cuh"
#include "tensor_ops.h"
#include "scaler_ops.h"
#include "reduction_ops.h"

class NeuralNetwork{

public:

    NeuralNetwork(int input_size):m_distribution(0,1){
        m_input_size = input_size;
        m_last_shape = input_size;
    }


    void addDenseLayer(int neurons){

        auto w = Tensor<float>({neurons, m_last_shape}, Fill::RAND,Device::CPU);
        w.cuda();
        m_weights.push_back(w);
        m_layers.push_back(neurons);
        m_last_shape = neurons;
    }


    void forward(Tensor<float> &x, vector<Tensor<float>> &Z, vector<Tensor<float>> &A){

        A.push_back(x);
        auto a = x;
        for (int i =0;i<m_layers.size();i++){

            Tensor<float> z = mutmulTensor(a, m_weights[i], false, true);
            a = sigmoidTensor(z);
            Z.push_back(z);
            A.push_back(a);
        }

    }
    float backward(Tensor<float> &y, vector<Tensor<float>> &A,  vector<Tensor<float>> &Z,vector<Tensor<float>> &dW){

        Tensor<float> dz = subtractTensor(A[A.size()-1], y);
        auto loss_tensor_pow = scalarSquareTensor(dz);
        auto loss_tensor = reduceSumTensor(loss_tensor_pow);
        loss_tensor.cpu();
        float loss = loss_tensor.get_pointer()[0];

        float m = y.shape()[0];

        for (int i = m_layers.size();i>0;i--){

            auto dw = mutmulTensor(dz, A[i - 1], true, false);
            dw = scalarMultiplyTensor(dw, 1 / m);
            dW.push_back(dw);
            if(i>1){
                auto muti = mutmulTensor(dz, m_weights[i-1]);
                auto sig_prime = sigmoidPrimeTensor(Z[i-2]);
                dz = multiplyTensor(muti, sig_prime);
            }
        }
        reverse(dW.begin(), dW.end());
        return loss;
    }

    float step(Tensor<float> &x_batch, Tensor<float> &y_batch, float lr){
        vector<Tensor<float>>Z;
        vector<Tensor<float>>A;
        vector<Tensor<float>>dW;

        forward(x_batch,Z,A);
        float loss = backward(y_batch,A,Z,dW);
        updateWeights(dW, lr);
        //exit(0);

        return loss;
    }
    void updateWeights(vector<Tensor<float>>& dW, float lr){
        for(int i =0;i<m_weights.size();i++){
            auto dw = scalarMultiplyTensor(dW[i], lr);
            m_weights[i] = subtractTensor(m_weights[i], dw);
        }
    }
    void fit(Tensor<float>& X, Tensor<float> &Y, int batch_size, float lr, int epochs){
        int steps = X.shape()[0]/batch_size;

        for (int i = 0 ; i<epochs ; i++){
            float epoch_loss = 0;
            for (int j=0 ; j<steps ; j++){
                auto batch_x = X.slice(j * batch_size, (j+1) * batch_size);
                auto batch_y = Y.slice(j * batch_size, (j+1) * batch_size);
                float loss = step(batch_x, batch_y,lr);
                //float loss = step(X, Y,lr);
                //cout<<loss<<endl;
                epoch_loss += loss;
            }
            float accuracy = calc_accuracy(X,Y);
            printf("epoch: %d, loss: %f, train accuracy:%f \n",i, epoch_loss / steps,accuracy);
        }
    }
    float calc_accuracy(Tensor<float>& X, Tensor<float> &Y){
        vector<Tensor<float>>Z;
        vector<Tensor<float>>A;

        forward(X,Z,A);

        float correct = 0;
        int total = Y.shape()[0];
        int output_dim = Y.shape()[1];

        auto pred = A[A.size()-1];
        auto pred_org_device = pred.get_device();
        auto true_org_device = Y.get_device();

        pred.cpu();
        Y.cpu();

        for (int i=0;i< total;i++){
            int max_idx_pred = 0;
            int max_idx_true = 0;

            for(int j =0;j<output_dim;j++){
                if(pred.get_pointer()[i * output_dim + j] > pred.get_pointer()[i * output_dim + max_idx_pred]){
                    max_idx_pred = j;
                }
                if(Y.get_pointer()[i * output_dim + j] > Y.get_pointer()[i * output_dim + max_idx_true]){
                    max_idx_true= j;
                }
            }
            if (max_idx_pred == max_idx_true)correct +=1;
        }
        float accuracy = correct / total;
        if (true_org_device == Device::CUDA)Y.cuda();
        if (pred_org_device == Device::CUDA)pred.cuda();

        return accuracy;
    }

    void cuda(){
        for (auto& weights : m_weights){
            weights.cuda();
        }
    }

//private:

    int m_input_size;
    std::vector<int>m_layers;

    //std::vector<std::vector<float>>m_weights;
    std::vector<Tensor<float>>m_weights;
    std::vector<std::vector<int>>m_weights_shapes;

    int m_last_shape;
    std::default_random_engine m_generator;
    std::normal_distribution<float> m_distribution;
};


#endif //NEURALNETWORKS_NEURALNETWORK_H
