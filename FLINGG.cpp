#include "standard_header.h"

int main(int argc, char const *argv[]){
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

    // Parameters of FLINGG operation
    int N = training_features.size();
    int B = 2 * int(sqrt(N));
    int R = 25;
    int m = 10;
    int L = 4;
    N = N - N % B;
    training_features.resize(N);    // We only keep the first N features of the training set

    // We also want an "index" vector for the training features
	std::vector<size_t> training_indices(training_features.size());
	std::iota(training_indices.begin(), training_indices.end(), 0);    // Fills it with 0, 1, 2, ....

    // Creating groups
    std::shuffle(training_indices.begin(), training_indices.end(), std::default_random_engine(std::time(nullptr)));
    std::vector<std::vector<size_t>> groups(B);
    for (int i = 0; i < N; i++) {   // Other methods to make this copy are also O(N) anyways so this is fine
        groups[i % B].push_back(training_indices[i]);
    }
}