#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>


__global__ void create_x_matrix(int* x_matrix, const char* word, const char* alphabet, int size);
__global__ void create_d_matrix(int* d_matrix, char* word1, char* word2, int* x_matrix, int length, int correct_length,int i);
__device__ int calculate_d_value(int* d_matrix, char* word1, char* word2, int* x_matrix, int current_index, int length);
int* create_X_matrix(char* word, int len);
int* create_D_matrix(char* word1,char* word2, int len1, int len2,int *x_matrix);