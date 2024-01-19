#pragma once

#include <vector>
#include <string>
#include <cuda_runtime.h>


using namespace std;


vector<string> gpu_lev(const string& word1, const string& word2);
int* create_X_matrix(char* word, int len);
int* create_D_matrix(char* word1,char* word2, int len1, int len2,int *x_matrix);
vector<string> obtain_operation(const int* d_matrix, const string& word1, const string& word2);