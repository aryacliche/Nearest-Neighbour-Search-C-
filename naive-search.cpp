#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <stdexcept>
#include <cstdlib> // for std::rand and std::srand
#include <ctime>   // for std::time
#include <cmath>
#include <algorithm> // For std::copy
#include <numeric>
#include <chrono>

#define FEATURE_SIZE 4096

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
    return std::sqrt(sum_of_squares);
}

void merge(const std::vector<VGGNetFeature>& features, std::vector<size_t>& indices, 
		size_t left, size_t mid, size_t right, const VGGNetFeature& reference) {
	std::vector<size_t> leftIndices(indices.begin() + left, indices.begin() + mid + 1);
	std::vector<size_t> rightIndices(indices.begin() + mid + 1, indices.begin() + right + 1);

	size_t i = 0, j = 0, k = left;
	while (i < leftIndices.size() && j < rightIndices.size()) {
		if (euclideanDistance(features[leftIndices[i]].values, reference.values) <= euclideanDistance(features[rightIndices[j]].values, reference.values)) {
			indices[k++] = leftIndices[i++];
		} else {
			indices[k++] = rightIndices[j++];
		}
	}

	while (i < leftIndices.size()) indices[k++] = leftIndices[i++];
	while (j < rightIndices.size()) indices[k++] = rightIndices[j++];
}

void indexedMergeSort(const std::vector<VGGNetFeature>& features, std::vector<size_t>& indices, 
		size_t left, size_t right, const VGGNetFeature& reference) {
	if (left < right) {
		size_t mid = left + (right - left) / 2;
		indexedMergeSort(features, indices, left, mid, reference);
		indexedMergeSort(features, indices, mid + 1, right, reference);
		merge(features, indices, left, mid, right, reference);
	}
}

int main(int argc, char const *argv[])
{
	const std::string train_filename = argv[1];//"data/vggnet_imagenet_train_features.bin";
	const std::string val_filename = argv[2];//"data/vggnet_imagenet_val_features.bin";

	// Load the training dataset (probably VGGNET features of IMAGENET/MIRFLICKR) [Store in the heap]
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<VGGNetFeature> training_features = readVGGNetFeatures(train_filename);
	auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time for loading training dataset: " << elapsed_seconds.count() << "s\n";

	// Load the validation dataset (probably VGGNET features of IMAGENET/MIRFLICKR) [Store in the heap]
	start = std::chrono::high_resolution_clock::now();
    std::vector<VGGNetFeature> val_features = readVGGNetFeatures(val_filename);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time for loading validation dataset: " << elapsed_seconds.count() << "s\n";

	// We also want an "index" vector for the training features
	std::vector<size_t> training_indices(training_features.size());
	std::iota(training_indices.begin(), training_indices.end(), 0);    // Fills it with 0, 1, 2, ....

	std::srand(std::time(nullptr)); // use current time as seed for random generator
    
    for (auto i = 0; i < 1; i++) {
        int random_index = std::rand() % val_features.size();
        VGGNetFeature random_query = val_features[0];

        std::vector<size_t> sorted_indices(training_indices); // Makes a proper copy of training_indices
        
        start = std::chrono::high_resolution_clock::now();
        indexedMergeSort(training_features, sorted_indices, 0, sorted_indices.size() - 1, random_query);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Execution time: " << elapsed_seconds.count() << "s\n";
        for (auto j = 0; j < 10; j++) {
            std::cout << sorted_indices[j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Done" << std::endl;
}
