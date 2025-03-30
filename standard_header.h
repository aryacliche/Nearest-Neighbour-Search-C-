#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <array>
#include <stdexcept>
#include <cstdlib>      // for std::rand and std::srand
#include <ctime>        // for std::time
#include <algorithm>    // for std::copy, std::sort
#include <cmath>
#include <numeric>
#include <random>   
#include <chrono>
#include <unordered_map>
#include <nlohmann/json.hpp>

#ifdef DEBUG
    #define DEBUG_PRINT std::cout
#else
    #define DEBUG_PRINT if (false) std::cout
#endif

struct VGGNetFeature {
    std::vector<float> values; 
};

std::vector<VGGNetFeature> readVGGNetFeatures(const std::string& filename, int &feature_dim) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file");
    }

    int num_points;
    file.read(reinterpret_cast<char*>(&num_points), sizeof(int));
    file.read(reinterpret_cast<char*>(&feature_dim), sizeof(int));

    std::cout << "Dealing with features of size " << feature_dim << std::endl;
    std::cout << "Dealing with " << num_points << " points" << std::endl;

    // Now we will read bin files' data and put them into each of the features. This step has a lot of error handling
    std::vector<VGGNetFeature> features(num_points);
    for (auto& feature : features) {
        feature.values.resize(feature_dim); // Ensure the vector is resized
        file.read(reinterpret_cast<char*>(feature.values.data()), feature_dim * sizeof(float));
    }

    return features;
}

double euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> diff;
    std::transform(a.begin(), a.end(), b.begin(), diff.begin(),
                   [](float x, float y) { return x - y; });
    
    float sum_of_squares = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0f);
    return sum_of_squares;
}

double cosineDistance(const std::vector<float>& a, const std::vector<float>& b) {
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

void dataMatrixFromVGGNETFeatures (std::vector<VGGNetFeature> features, double** data) {
    auto size = features.size();
    auto feature_dim = features[0].values.size();
    for (int i = 0; i < size; i++) {
        data[i] = new double[feature_dim];
        for (int j = 0; j < feature_dim; j++) {
            data[i][j] = features[i].values[j];
        }
    }
}

class LabelMaker {  // This linked list is for keeping track of the frequency of labels seen
    struct Label {
        size_t index;
        int count;
        Label* lessPopular;
        Label* morePopular;
        
        Label(size_t index, Label* morePopular) : index(index), count(1), lessPopular(nullptr), morePopular(morePopular ) {}
    };
    
    Label* mostPopular;

    public:
        LabelMaker(){
            mostPopular = nullptr;
        }

        void updateCounters(size_t index) {
            Label* current = mostPopular;
            Label* previous = nullptr;
            while (true) {
                if (current == nullptr) {  // We have reached the end of the list. We need to make a new node
                    Label* newLabel = new Label(index, previous);
                    if (previous != nullptr) {
                        previous->lessPopular = newLabel;
                    }
                    else { // This is the only node in the list thus is the most popular
                        mostPopular = newLabel;
                    }
                    break;  // We don't need to look further
                }
                else {  // we have not yet reached the end of the list
                    if (current->index == index) {  // We have a match
                        current->count++;
                        // Now we will sit down and swap with each of the more popular nodes if applicable
                        while (current->morePopular != nullptr && current->count > current->morePopular->count) {
                            size_t temp_count = current->count;
                            size_t temp_index = current->index;
                            current->count = current->morePopular->count;
                            current->index = current->morePopular->index;
                            current->morePopular->count = temp_count;
                            current->morePopular->index = temp_index;

                            // Now we will go one step up the list
                            current = current->morePopular;
                        }
                        break;
                    }
                    else {  // This one wasn't a match, let's go lower into the list
                        previous = current;
                        current = current->lessPopular;
                    }
                }
            }
        }

        std::vector<size_t> topKLabels(size_t k) {
            Label* current = mostPopular;
            std::vector<size_t> topK;
            while (current != nullptr && topK.size() < k) {
                topK.push_back(current->count);
                current = current->lessPopular;
            }
            return topK;
        }

};