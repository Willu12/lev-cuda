#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>


__global__ void create_x_matrix(int* x_matrix, const char* word, const char* alphabet, int size);
int* create_X_matrix(char* word, int len);
int* create_D_matrix(char* word1,char* word2, int len1, int len2,int *x_matrix);