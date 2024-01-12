#pragma once
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

std::string read_file(const std::string& path);
void save_edits_to_file(const std::vector<std::string>& edits, const std::string& file_path);


