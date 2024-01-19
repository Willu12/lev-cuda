#include "kernels.cuh"
#include <cstdio>
#include <cuda_runtime_api.h>

#define ALPHABET_SIZE 95

__global__ void create_x_matrix(int* x_matrix,const char* word,const char* alphabet, int size) {
    int tid = threadIdx.x;
    if(tid >= ALPHABET_SIZE) return;
    
    for(int j =0; j<size + 1; j++) {

        if(j == 0) {
            x_matrix[tid] = 0;
            continue;
        }
        int index = tid * (size + 1) + j;
        x_matrix[index] = word[j - 1] == alphabet[tid] ? j : x_matrix[index - 1];
    }
}

__device__ int calculate_d_value(int* d_matrix, char* word1, char* word2, int* x_matrix, int current_index, int length) {
    int i = current_index / length;
    int j = current_index % length;
    
    if(i == 0) return j;
    if(j == 0) return i;

    if(word1[i - 1] == word2[j - 1]) return d_matrix[current_index - 1 - length];

    // wiemy że litera m = word2[i -1] znajduje się na miejscu w alfabecie będącym jej wartością ASCII 
    int l = word1[i - 1] - 32;

    if(x_matrix[l * length + j] == 0) {
        return 1 + min(d_matrix[current_index - length],
        min(d_matrix[current_index - length - 1],i + j - 1));
    }
 
    return 1 + min(
        d_matrix[current_index - length],
        min(d_matrix[current_index - length - 1],
        d_matrix[(i - 1) * length + x_matrix[l * length + j] - 1] + (j - 1 - x_matrix[l * length + j]))
    );
}

__global__ void create_d_matrix(int* d_matrix, char* word1, char* word2, int* x_matrix, int len, int correct_length,int i) {

    //okej wywolujemy tyle wątkow ile liter ma slowo 1
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // wywalmy tidy ktore są za duże
    if(tid > correct_length) return;
    int index = tid + i * (correct_length + 1); 
     d_matrix[index] = calculate_d_value(d_matrix, word1, word2,x_matrix,index, correct_length + 1);
}



int* create_X_matrix(char* word, int len) {
    
    // we assume that word consists only of ascii characters
    int * x_matrix;
    char* alphabet_device;
    const std::string ALPHABET = std::string(" !\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~");

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc(&x_matrix, sizeof(int) * (1 + len) * ALPHABET.size());
    cudaMalloc(&alphabet_device, sizeof(char) * ALPHABET.size());
    
    cudaStatus = cudaMemcpy(alphabet_device, ALPHABET.data(), ALPHABET.size() * sizeof(char), cudaMemcpyHostToDevice);
    
    create_x_matrix<<<1,256>>>(x_matrix,word,alphabet_device,len);

    return x_matrix;
}

int* create_D_matrix(char* word1,char* word2, int len1, int len2,int *x_matrix) {
    int* d_matrix;

    cudaMalloc(&d_matrix, sizeof(int) * (len1 + 1) * (len2 + 1));
    
    const int threads_per_blocks = 512;

    const int blocks = (len2 + 1) / threads_per_blocks + !!(len2 % threads_per_blocks);

    for(int i =0; i < len1 + 1; i++ ) {
        
        cudaDeviceSynchronize();
        create_d_matrix<<<blocks,threads_per_blocks>>>(d_matrix,word1,word2,x_matrix,len1,len2,i);
    }
    

    return d_matrix;
}