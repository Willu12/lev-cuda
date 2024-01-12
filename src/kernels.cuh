#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>

const std::string ALPHABET = " !\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";


__global__ void create_x_matrix(int* x_matrix, char* word, char* alphabet, int size);
int* create_X_matrix(char* word, int len);
int* create_D_matrix(char* word1,char* word2, int len1, int len2);