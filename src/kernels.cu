#include "kernels.cuh"

#define ALPHABET_SIZE 95

__global__ void create_x_matrix(int* x_matrix, char* word, char* alphabet, int size) {
    int tid = threadIdx.x;

    for(int j =0; j<size; j++) {

        if(tid == 0 || j == 0) {
            x_matrix[tid] = 0;
            continue;
        }

        int index = tid * size + j;

        x_matrix[index] = word[j - 1] == alphabet[tid] ? j : x_matrix[index - 1];
    }
}

__device__ int calculate_d_value(int* d_matrix, char* word1, char* word2, int* x_matrix, int current_index, int length) {
    
    int i = current_index /length;
    int j = current_index % length;

    if(i == 0) return j;
    if(j == 0) return i;

    if(word1[j - 1] == word2[i - 1]) return d_matrix[current_index - 1 - length];

    // wiemy że litera m = word2[i -1] znajduje się na miejscu w alfabecie będącym jej wartością ASCII 
    int l = word2[i - 1] - 32;

    if(x_matrix[l + j * ALPHABET_SIZE] == 0) {
        return 1 + min(d_matrix[current_index - length],
        min(d_matrix[current_index - length - 1],i + j -1));
    }

    return 1 + min(
        d_matrix[current_index - length],
        min(d_matrix[current_index - length - 1],
        d_matrix[(i - 1) * length + x_matrix[l * ALPHABET_SIZE + j] - 1] + (j - 1 - x_matrix[l * ALPHABET_SIZE + j]))
    );
}

__global__ void create_d_matrix(int* d_matrix, char* word1, char* word2, int* x_matrix, int size, int length) {

    //okej wywolujemy tyle wątkow ile liter ma slowo 1
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i =0; i<size + 1; i++) {

        d_matrix[tid + length * i] = calculate_d_value(d_matrix, word1, word2,x_matrix,tid + length * i,length);
        __syncthreads();
    }
}



int* create_X_matrix(char* word, int len) {
    
    // we assume that word consists only of ascii characters
    int * x_matrix;
    char* alphabet_device;

    cudaMalloc(&x_matrix, sizeof(int) * (1 + len) * ALPHABET.size());
    cudaMalloc(&alphabet_device, sizeof(char) * ALPHABET.size());

    cudaMemcpy(alphabet_device, ALPHABET.data(), ALPHABET.size(), cudaMemcpyHostToDevice);

    create_x_matrix<<<1,256>>>(x_matrix,word,alphabet_device,len);


    return x_matrix;
}

int* create_D_matrix(char* word1,char* word2, int len1, int len2,int *x_matrix) {
    int* d_matrix;

    cudaMalloc(&d_matrix, sizeof(int) * (len1 + 1) * (len2 + 1));
    
    

    return d_matrix;
}