# Neural Networks From Scratch 
___

* This is an implementation of Neural Networks from scratch using CUDA.
* it's a header only library to ease the integration in ur projects
* it provides an API similar to Keras

## Quick Start
u will need CUDA installed to be able to use the library.
``` bash
git clone https://github.com/AhmedGhazale/NeuralNetworksInCUDA.git
cd NeuralNetworksInCUDA
mkdir build
cd build
cmake ..
make 
./NeuralNetworks ../data/mnist_train.csv
```
This will train a Neural Network on Mnist dataset for 100 epochs.  
at the end, it will achive arround **%92** training accuracy.  

___
to build u own model and train on ur own data u can see the example below.  
U need to add the include directory to ur project.
``` c++
#include "NeuralNetwork.h"
#include "Tensor.h"
    
    Tensor<float>X;
    Tensor<float>Y;
    
    // fill X and Y with data, check the Tensor.h for reference
    
    // constructing the model
    int input_size = 784;
    int first_layer_size = 500;
    int second_layer_size = 300;
    int thirds_layer_size = 200;
    int output_layer_size = 10;
    
    NeuralNetwork model(input_size);
    model.addDenseLayer(first_layer_size);
    model.addDenseLayer(second_layer_size);
    model.addDenseLayer(thirds_layer_size);
    model.addDenseLayer(output_layer_size);
    
    // setting training parameters
    int epochs = 100
    int batch_size = 64;
    float lr = .005;
    
    // strart training
    model.fit(X, Y, batch_size, lr, epochs);
```

## Supported Layers

* DenseLayer (fully connected)

## Supported Activation Functions
* Sigmoid

## Supported Operations on Tenosr Datatype

* Reduction operations
    * reduceSumTensor
* Scalar operations
    * scalarMultiplyTensor
    * scalarSquareTensor
* Tensor-Tensor operations
    * multiplyTensors
    * subtractTensors

## Supported Loss Funcitons
* u can only train a regression model using MSE as a loss function.



