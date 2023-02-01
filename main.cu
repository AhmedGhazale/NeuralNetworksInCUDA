#include <iostream>
#include <random>
#include <algorithm>
#include "NeuralNetwork.h"
#include "Tensor.h"
#include "activition_functions.cuh"
#include <chrono>
#include <sstream>
#include "fstream"
#include "reduction_ops.h"

using namespace std;
using namespace std::chrono;

void readMNIST(string csv_path, Tensor<float>& X, Tensor<float>& Y, bool one_hot = true){
    X = Tensor<float>({60000, 784},Fill::NOFILL,Device::CPU);
    if (one_hot) {
        Y = Tensor<float>({60000, 10}, Fill::ZEROS, Device::CPU);
    }else{
        Y = Tensor<float>({60000, 1}, Fill::NOFILL, Device::CPU);
    }

    fstream fin;
    fin.open(csv_path, ios::in);
    string line, word, temp;

    for(int i =0;i<60000;i++){
        fin >> temp;
        stringstream s(temp);

        for (int j =0;j<785;j++){
            getline(s, word, ',');
            if(j == 0){
                if (one_hot){
                    int val = stoi(word);
                    Y.get_pointer()[i * 10 + val] = 1;
                }else {
                    Y.get_pointer()[i] = stof(word);
                }
            }else{
                X.get_pointer()[i * 784 + j] = stof(word);
            }
        }
    }

}


int main(int argc, char** argv) {

    if (argc<2){
        cout<<"to run please provide the path to mnist dataset"<<endl;
        cout<<"for example run: ./NeuralNetworks ../data/mnist_train.csv"<<endl;
        exit(0);
    }

//    string mnist_path = "../data/mnist_train.csv";

    string mnist_path = argv[1];

    Tensor<float>X;
    Tensor<float>Y;
    readMNIST(mnist_path,X, Y);

    X.cuda();
    Y.cuda();

    X = scalarMultiplyTensor(X, 1.0 / 255);

    cout<<"X Tensor shape: (";
    string d = "";
    for(int i : X.shape()){
        cout<<d<<i;
        d = ", ";
    }
    cout<<")"<<endl;


    cout<<"Y Tensor shape: (";
    d = "";
    for(int i : Y.shape()){
        cout<<d<<i;
        d = ", ";
    }
    cout<<")"<<endl;


    NeuralNetwork model(784);
    model.addDenseLayer(500);
    model.addDenseLayer(300);
    model.addDenseLayer(200);
    model.addDenseLayer(10);

    model.fit(X,Y,50,.005,100);







    return 0;
}
