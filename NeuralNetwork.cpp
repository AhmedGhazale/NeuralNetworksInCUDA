//
// Created by ahmed on ٢٣‏/٨‏/٢٠٢١.
//

#include "NeuralNetwork.h"

//NeuralNetwork::NeuralNetwork(int input_size):m_distribution(0,1){
//    m_input_size = input_size;
//    m_last_shape = input_size;
//}



//void NeuralNetwork::addDenseLayer(int neurons) {
//
//    //std::vector<float> w(m_last_shape * neurons);
//    //m_weights_shapes.push_back({m_last_shape, neurons});
//    //std::generate(w.begin(),w.end(),[this](){return m_distribution(m_generator);});
//    auto w = Tensor<float>({neurons, m_last_shape}, Fill::RAND);
//    m_weights.push_back(w);
//    m_layers.push_back(neurons);
//    m_last_shape = neurons;
//}
//
