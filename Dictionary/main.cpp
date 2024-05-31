#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

constexpr int VECTOR_SIZE = 3;

// Read the CSV file and return data as a vector of vectors of floats
std::vector<std::vector<float>> read_csv(const std::string &csv_file)
{
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << csv_file << std::endl;
        return {};
    }
    
    std::vector<std::vector<float>> data;
    std::string line;

    while (std::getline(file, line))
    {
        std::vector<float> row;
        size_t pos = 0;
        while ((pos = line.find(',')) != std::string::npos)
        {
            row.push_back(std::stof(line.substr(0, pos)));
            line.erase(0, pos + 1);
        }
        row.push_back(std::stof(line));
        data.push_back(row);
    }

    return data;
}

// Find the index of the element in 'data' that is closest to 'target_data'
int find_closest_array_index(const std::vector<std::vector<float>> &data, const std::vector<float> &target_data)
{
    std::vector<float> distances;
    for (const auto &row : data)
    {
        float distance = 0.0f;
        for (size_t i = 0; i < VECTOR_SIZE; ++i)
        {
            distance += std::pow(row[i] - target_data[i], 2);
        }
        distances.push_back(std::sqrt(distance));
    }

    auto min_it = std::min_element(distances.begin(), distances.end());
    return std::distance(distances.begin(), min_it);
}

int main()
{
    std::vector<std::vector<float>> data = read_csv("data.csv");
    if (data.empty()) {
        return 1;
    }
    
    std::vector<float> target = {1500, 10, 500};

    int closest_index = find_closest_array_index(data, target);

    std::vector<float> closest = data[closest_index];

    return 0;
}
