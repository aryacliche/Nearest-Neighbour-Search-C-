#include "FLINNG.h"

int main(int argc, char const *argv[]){
    // Taking in inputs
    const std::string train_features_file_name = argv[1];//"data/vggnet_imagenet_train_features.bin";
	const std::string val_features_file_name = argv[2];//"data/vggnet_imagenet_val_features.bin";
    const std::string train_labels_file_name = argv[3];//"data/vggnet_imagenet_train_labels.txt";
	const std::string val_labels_file_name = argv[4];//"data/vggnet_imagenet_val_labels.txt";
    const std::string temp_dir = argv[5];   //"temp";
    std::string group_creation_algorithm = argv[6]; // "random" or "labelled"
    int forced_N = -1;
    if (argc == 8) {
        forced_N = std::stoi(argv[7]);
    }

    int num_threads = 32;
    omp_set_num_threads(num_threads);

    // Verifying that OpenMP is enabled
    #ifdef _OPENMP
        std::cout << "OpenMP is enabled! Version: " << _OPENMP << std::endl;
    #else
        std::cout << "OpenMP is NOT enabled!" << std::endl;
    #endif

	// Estimate the size of the training dataset
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<VGGNetFeature> training_features = readVGGNetFeatures(train_features_file_name);
    std::vector<int> training_labels = readLabelsAsInt(train_labels_file_name);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time for loading training dataset: " << elapsed_seconds.count() << "s\n";

    // Load the validation dataset (probably VGGNET features of IMAGENET/MIRFLICKR) [Store in the heap]
	start = std::chrono::high_resolution_clock::now();
    std::vector<VGGNetFeature> val_features = readVGGNetFeatures(val_features_file_name);
    std::vector<int> val_labels = readLabelsAsInt(val_labels_file_name);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Time for loading validation dataset: " << elapsed_seconds.count() << "s\n";

    int N = (forced_N != -1) ? std::min(forced_N, int(training_features.size())) : training_features.size();
    int B = 2 * int(sqrt(N));
    N = N - N % B;
    training_features.resize(N);    // We only keep the first N features of the training set

    // We also want an "index" vector for the training features
    std::vector<size_t> training_indices(N);
	std::iota(training_indices.begin(), training_indices.end(), 0);    // Fills it with 0, 1, 2, ....

    // Parameters that should be read from a config file
    std::ifstream config_file("config.json");
    if (!config_file) {
        return false;
    }
    nlohmann::json config;
    config_file >> config;
    config_file.close();
    
    int d = 4096;
    int R = config["R"];
    int t = config["t"];
    int m = config["m"];
    int L = config["L"];
    double w = config["w"];
    
    int l = 1 << L; 

    // Initialise R x B groups of size N / B
    std::vector<std::vector<std::vector<size_t>>> groups(R, std::vector<std::vector<size_t>>(B, std::vector<size_t>(N / B))); // TODO : Make this into an array if possible
    
    // Each mask is a vector of m bool-vectors. We need B such masks per iteration. There are R iterations. Thus finally masks will be a R x B x m x 2**L matrix OR vector of vector of vector of vector of bools
    std::vector<std::vector<std::vector<std::vector<bool>>>> masks(R, std::vector<std::vector<std::vector<bool>>>(B, std::vector<std::vector<bool>>(m, std::vector<bool>(l, false))));

    // We will use p-stable LSH functions taken from the paper "Locality-Sensitive Hashing Scheme Based on p-Stable Distributions" by Piotr Indyk and Rajeev Motwani (https://dl.acm.org/doi/pdf/10.1145/997817.997857)
    std::vector<std::vector<float>> vecs(m, std::vector<float>(d));
    std::vector<double> t_vals(m);
    
    bool masks_present = checkMetadata(temp_dir, N, B, R, m, d, l);    

    if (masks_present == false) {
        offlinePrep(training_indices, training_labels, groups, masks, vecs, t_vals, temp_dir, training_features, N, B, R, m, d, l, w, group_creation_algorithm);
    }
    else {
        loadProcessedData(m, d, temp_dir, t_vals, vecs, groups, masks, N, B, R, l);
    }
    updateMetadata(temp_dir, N, B, R, m, d, l, w, t);   

    //// Querying with the validation set
    // Discarding contents of results.csv
    std::ofstream csv_file(temp_dir + "/results.csv", std::ios::trunc);
    if (!csv_file) {
        throw std::runtime_error("Cannot open results CSV file");
    }
    csv_file << "precision" << "," << "recall" << "\n";
    csv_file.close();

    double running_total_time = 0.0;
    int num_queries = 100;
    #pragma omp parallel for reduction(+:running_total_time)
    for (auto i = 0; i < num_queries; i++) {
        int random_index = std::rand() % val_features.size();
        VGGNetFeature random_query = val_features[random_index];
        running_total_time += evaluateQuery(training_features, training_indices, groups, masks, vecs, t_vals, random_query, N, B, R, m, d, l, w, t, temp_dir);
    }
    std::cout << "Average time: " << running_total_time / num_queries << "ms\n";
    std::cout << "Done" << std::endl;
}   
