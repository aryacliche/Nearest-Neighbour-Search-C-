#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <array>
#include <stdexcept>
#include <cstdlib> // for std::rand and std::srand
#include <ctime>   // for std::time
#include <cmath>
#include <algorithm> // For std::copy, std::sort
#include <numeric>
#include <random>   
#include <chrono>
#include <unordered_map>
#include <nlohmann/json.hpp>

#define FEATURE_SIZE 4096

#ifdef DEBUG
    #define DEBUG_PRINT std::cout
#else
    #define DEBUG_PRINT if (false) std::cout
#endif

struct VGGNetFeature {
    std::array<float, FEATURE_SIZE> values; // We know that each of the features is compuslorily 4096 dimensional   
};

std::vector<VGGNetFeature> readVGGNetFeatures(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file");
    }

    int num_points, feature_dim;
    file.read(reinterpret_cast<char*>(&num_points), sizeof(int));
    file.read(reinterpret_cast<char*>(&feature_dim), sizeof(int));

    if (feature_dim != FEATURE_SIZE) {
        throw std::runtime_error("Unexpected feature dimension");
    }

    // Now we will read bin files' data and put them into each of the features. This step has a lot of error handling
    std::vector<VGGNetFeature> features(num_points);
    for (auto& feature : features) {
        file.read(reinterpret_cast<char*>(feature.values.data()), FEATURE_SIZE * sizeof(float));
    }

    return features;
}

double euclideanDistance(const std::array<float, FEATURE_SIZE>& a, const std::array<float, FEATURE_SIZE>& b) {
    std::array<float, FEATURE_SIZE> diff;
    std::transform(a.begin(), a.end(), b.begin(), diff.begin(),
                   [](float x, float y) { return x - y; });
    
    float sum_of_squares = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0f);
    return sum_of_squares;
}

double cosineDistance(const std::array<float, FEATURE_SIZE>& a, const std::array<float, FEATURE_SIZE>& b) {
    float dot_product = std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
    float norm_a = std::sqrt(std::inner_product(a.begin(), a.end(), a.begin(), 0.0f));
    float norm_b = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0f));
    
    // Avoid division by zero
    if (norm_a == 0 || norm_b == 0) {
        return 1.0; // Return maximum distance if either vector is zero
    }
    return dot_product / (norm_a * norm_b);
}

std::vector<uint64_t> readLabelsAsInt(const std::string filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file");
    }

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    
    std::unordered_map<std::string, int> label_to_int;
    int current_label = 0;
    std::vector<u_int64_t> int_labels(labels.size());

    for (const auto& label : labels) {
        if (label_to_int.find(label) == label_to_int.end()) {
            label_to_int[label] = current_label++;
        }
    }

    std::transform(labels.begin(), labels.end(), int_labels.begin(),
                   [&label_to_int](const std::string& label) { return label_to_int[label]; });
    return int_labels;
}