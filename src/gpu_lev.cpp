#include "gpu_lev.hpp"
#include <driver_types.h>
#include <string>
#include <vector>
#include <cuda_runtime.h>

using namespace std;


vector<string> gpu_lev(const string& word1, const string& word2) {
    char* word1_device;
    char* word2_device;

    cudaSetDevice(0);
    cudaMalloc(&word1_device, sizeof(char) * word1.size());
    cudaMalloc(&word2_device, sizeof(char) * word2.size());

    cudaMemcpy(word1_device, word1.data(), word1.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(word2_device, word2.data(), word2.size(), cudaMemcpyHostToDevice);

    int* x_matrix = create_X_matrix(word1_device,word1.size());

    //now we want to create D matrix



    return vector<string>();
}

