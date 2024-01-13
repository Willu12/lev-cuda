#include "gpu_lev.hpp"
#include "kernels.cuh"
#include <cstdio>


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
    int* d_matrix = create_D_matrix(word1_device, word2_device, word1.size(), word2.size(),x_matrix);


    //copy d_matrix on cpu
    int * d_matrix_cpu = (int*)malloc(sizeof(int) * (word1.size() + 1) * (word2.size() + 1));


    cudaMemcpy(d_matrix_cpu, d_matrix, sizeof(int) * (word1.size() + 1) * (word2.size() + 1), cudaMemcpyDeviceToHost);
   // printf("%d\n",d_matrix_cpu[word1.size() * (55) +  word2.size() -1]);

    return obtain_operation(d_matrix_cpu, word1, word2);
}

vector<string> obtain_operation(const int* verif, const string& str1, const string& str2) {
    vector<string> list = vector<string>();
    unsigned int i = str1.size() + 1;
    unsigned int j = str2.size() + 1;
    const int len = str1.size();

    for(int k =0; k< i * j; k++) {
        //printf("verif[%d] = %d\n",k,verif[k]);
    }

    while(i > 0 || j > 0) {
        printf("%d,%d \n",i,j);
        if (i > 0 && j>=0 && verif[i * len + j] == verif[(i - 1) * len+ j] + 1) {
            list.push_back(string("Delete "+ string(1,str1[i - 1]) + " at position " + to_string(i - 1)));
            i--;
            printf("jeden\n");
        }
        else if(j > 0 && i>=0 && verif[i * len + j] == verif[i * len + j - 1] + 1) {
            list.push_back(string("Insert "+ string(1,str2[i - 1]) + " at position " + to_string(i - 1)));
            j--;
            printf("dwa\n");
        }
        else {
            
            if(i > 0 && j > 0 && verif[i * len + j] == verif[(i - 1) * len + j - 1] + 1) {
                if(str1[i - 1] != str2[j - 1]) {
                    list.push_back(string("Substitute "+ string(1,str1[i - 1]) + " at position " + to_string(i - 1)
                     + " with " + string(1,str2[j - 1])));
                }   
            }
            i--;
            j--;
        }
    }

    return list;
}