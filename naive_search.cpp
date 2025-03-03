#include "standard_header.h"

bool compareFeatures(int index1, int index2, std::vector<VGGNetFeature>& features, VGGNetFeature& reference) {
    return euclideanDistance(features[index1].values, reference.values) < euclideanDistance(features[index2].values, reference.values);
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
    elapsed_seconds = end - start;
    std::cout << "Time for loading validation dataset: " << elapsed_seconds.count() << "s\n";

	// We also want an "index" vector for the training features
	std::vector<size_t> training_indices(training_features.size());
	std::iota(training_indices.begin(), training_indices.end(), 0);    // Fills it with 0, 1, 2, ....

	std::srand(std::time(nullptr)); // use current time as seed for random generator
    
    auto running_total_time = 0;

    for (auto i = 0; i < 100; i++) {
        int random_index = std::rand() % val_features.size();
        VGGNetFeature random_query = val_features[random_index];

        std::vector<size_t> sorted_indices(training_indices); // Makes a proper copy of training_indices
        
        start = std::chrono::high_resolution_clock::now();
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](size_t a, size_t b) {
            return compareFeatures(a, b, training_features, random_query);
        });
        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        running_total_time += elapsed_seconds.count();
        std::cout << "Execution time: " << elapsed_seconds.count() << "s\n";
    }
    std::cout << "Average time: " << running_total_time / 100.0 << "s\n";
    std::cout << "Done" << std::endl;
}
