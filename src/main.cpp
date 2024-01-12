#include <chrono>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#include "utils.hpp"

using namespace std;

vector<string> cpu_lev(const string& word1, const string& word2);
vector<string> obtain_operations(const vector<vector<int>>& verif, const string& str1, const string& str2);

int main(int argc, char** argv) { 
    
    string jeden = read_file("data/jeden.txt");
    string dwa = read_file("data/dwa.txt");

    vector<string> cpu_edits = cpu_lev(jeden, dwa);
    
    save_edits_to_file(cpu_edits, "cpu_results");
    return 0;
}

vector<string> cpu_lev(const string& word1, const string& word2) {

    auto start = chrono::high_resolution_clock::now();

    int size1 = word1.size();
    int size2 = word2.size();

    vector<vector<int>> verif(size1 + 1, vector<int>(size2 + 1));

    // If one of the words has zero length, the distance is equal to the size of the other word.
    if (size1 == 0 || size2 == 0)
        return vector<string>();

    // Sets the first row and the first column of the verification matrix with the numerical order from 0 to the length of each word.
    for (int i = 0; i <= size1; i++)
        verif[i][0] = i;
    for (int j = 0; j <= size2; j++)
        verif[0][j] = j;

    // Verification step / matrix filling.
    for (int i = 1; i <= size1; i++) {
        for (int j = 1; j <= size2; j++) {
            // Sets the modification cost.
            // 0 means no modification (i.e. equal letters) and 1 means that a modification is needed (i.e. unequal letters).
            int cost = (word2[j - 1] == word1[i - 1]) ? 0 : 1;

            // Sets the current position of the matrix as the minimum value between a (deletion), b (insertion) and c (substitution).
            // a = the upper adjacent value plus 1: verif[i - 1][j] + 1
            // b = the left adjacent value plus 1: verif[i][j - 1] + 1
            // c = the upper left adjacent value plus the modification cost: verif[i - 1][j - 1] + cost
            verif[i][j] = min(
                min(verif[i - 1][j] + 1, verif[i][j - 1] + 1),
                verif[i - 1][j - 1] + cost
            );
        }
    }

    const auto stop = chrono::high_resolution_clock::now();
    const chrono::duration<double, milli> fp_ms = stop - start;

    cout<<"cpu_lev took: " <<fp_ms.count()<<endl;

    vector<string> edits = obtain_operations(verif, word1, word2);

    return edits;
}

vector<string> obtain_operations(const vector<vector<int>>& verif, const string& str1, const string& str2) {
    
    vector<string> list = vector<string>();
    unsigned int i = str1.size();
    unsigned int j = str2.size();

    while(i > 0 || j > 0) {
        if (i > 0 && verif[i][j] == verif[i - 1][j] + 1) {
            list.push_back(string("Delete "+ string(1,str1[i - 1]) + " at position " + to_string(i - 1)));
            i--;
        }
        else if(j > 0 && verif[i][j] == verif[i][j - 1] + 1) {
            list.push_back(string("Insert "+ string(1,str2[i - 1]) + " at position " + to_string(i - 1)));
            j--;
        }
        else {
            if(i > 0 && j > 0 && verif[i][j] == verif[i - 1][j - 1] + 1) {
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