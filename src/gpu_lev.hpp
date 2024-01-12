#pragma once

#include <vector>
#include <string>
#include "kernels.cuh"


using namespace std;


//robimy funkcje ktora zwraca vektor rozwiazan
vector<string> gpu_lev(const string& word1, const string& word2);
int* create_X_matrix(char* word, int len);