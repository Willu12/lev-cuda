#include <chrono>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>


std::string read_file(const std::string& path);
std::vector<std::string> cpu_lev(const std::string& word1, const std::string& word2);
std::vector<std::string> obtain_operations(const std::vector<std::vector<int>>& verif, const std::string& str1, const std::string& str2);
void save_edits_to_file(const std::vector<std::string>& edits, const std::string& file_path);

int main(int argc, char** argv) { 
    
    std::string jeden = read_file("data/jeden.txt");
    std::string dwa = read_file("data/dwa.txt");

    std::vector<std::string> cpu_edits = cpu_lev(jeden, dwa);
    
    save_edits_to_file(cpu_edits, "cpu_results");
    return 0;
}

std::vector<std::string> cpu_lev(const std::string& word1, const std::string& word2) {

    auto start = std::chrono::high_resolution_clock::now();

    int size1 = word1.size();
    int size2 = word2.size();

    std::vector<std::vector<int>> verif(size1 + 1, std::vector<int>(size2 + 1));

    // If one of the words has zero length, the distance is equal to the size of the other word.
    if (size1 == 0 || size2 == 0)
        return std::vector<std::string>();

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
            verif[i][j] = std::min(
                std::min(verif[i - 1][j] + 1, verif[i][j - 1] + 1),
                verif[i - 1][j - 1] + cost
            );
        }
    }

    const auto stop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> fp_ms = stop - start;

    std::cout<<"cpu_lev took: " <<fp_ms.count()<<std::endl;

    std::vector<std::string> edits = obtain_operations(verif, word1, word2);

    return edits;
}

std::vector<std::string> obtain_operations(const std::vector<std::vector<int>>& verif, const std::string& str1, const std::string& str2) {
    
    std::vector<std::string> list = std::vector<std::string>();
    unsigned int i = str1.size();
    unsigned int j = str2.size();

    while(i > 0 || j > 0) {
        if (i > 0 && verif[i][j] == verif[i - 1][j] + 1) {
            list.push_back(std::string("Delete "+ std::string(1,str1[i - 1]) + " at position " + std::to_string(i - 1)));
            i--;
        }
        else if(j > 0 && verif[i][j] == verif[i][j - 1] + 1) {
            list.push_back(std::string("Insert "+ std::string(1,str2[i - 1]) + " at position " + std::to_string(i - 1)));
            j--;
        }
        else {
            if(i > 0 && j > 0 && verif[i][j] == verif[i - 1][j - 1] + 1) {
                if(str1[i - 1] != str2[j - 1]) {
                    list.push_back(std::string("Substitute "+ std::string(1,str1[i - 1]) + " at position " + std::to_string(i - 1)
                     + " with " + std::string(1,str2[j - 1])));
                }   
            }
            i--;
            j--;
        }
    }
    return list;
}

std::string read_file(const std::string& path) {
    
    std::ifstream input_file(path);
    if (!input_file.is_open()) {
        std::cout<<"Error opening file: " << path << std::endl;
        return "";
    }

    std::string content((std::istreambuf_iterator<char>(input_file)),
                        std::istreambuf_iterator<char>());

    input_file.close();

    return content;
}


void save_edits_to_file(const std::vector<std::string>& edits, const std::string& file_name) {     
    
    std::ofstream file(file_name);

    for(const auto& edit: edits) {
        file << edit << '\n';
    }

    file.close();
}