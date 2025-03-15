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
    int d = 4096;
    int m = 10;
    int t = 6;
    int L = 4;
    int w = 100;
    N = N - N % B;
    training_features.resize(N);    // We only keep the first N features of the training set

    // We also want an "index" vector for the training features
	std::vector<size_t> training_indices(training_features.size());
	std::iota(training_indices.begin(), training_indices.end(), 0);    // Fills it with 0, 1, 2, ....

    //// Creating R x B groups of size N / B randomly
    std::vector<std::vector<std::vector<size_t>>> groups(R, std::vector<std::vector<size_t>>(B, std::vector<size_t>(N / B)));
    for (int r = 0; r < R; r++) {
        std::shuffle(training_indices.begin(), training_indices.end(), std::default_random_engine(std::time(nullptr)));
        for (int i = 0; i < N; i++) {   // Other methods to make this copy are also O(N) anyways so this is fine
            groups[r][i % B].push_back(training_indices[i]);
        }
    }

    //// Generating masks
    // Each mask is a vector of m bool-vectors. We need B such masks per iteration. There are R iterations. Thus finally masks will be a R x B x m x 2**L matrix OR vector of vector of vector of vector of bools
    std::vector<std::vector<std::vector<std::vector<bool>>>> masks(R, std::vector<std::vector<std::vector<bool>>>(B, std::vector<std::vector<bool>>(m, std::vector<bool>(1 << L, false))));

    //// Generating hash functions
    // We will use p-stable LSH functions taken from the paper "Locality-Sensitive Hashing Scheme Based on p-Stable Distributions" by Piotr Indyk and Rajeev Motwani (https://dl.acm.org/doi/pdf/10.1145/997817.997857)
    std::vector<std::vector<float>> vecs(m, std::vector<float>(d));
    std::default_random_engine generator(std::time(nullptr));
    std::normal_distribution<float> distribution(0.0, 1.0);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < d; ++j) {
            vecs[i][j] = distribution(generator);
        }
    }

    // Let's generate the offset t for each hash function
    std::vector<int> t_vals(m);
    std::uniform_int_distribution<int> uniform_dist(0, w);
    for (int i = 0; i < m; ++i) {
        t_vals[i] = uniform_dist(generator);
    }

    std::cout << "LSH functions initialised successfully\n";

    if (false) {
        //// Filling masks
        // We will go per group and make the masks
        for (int r = 0; r < R; r++) {
            for (int b = 0; b < B; b++) {
                for (int i = 0; i < N / B; i++) {
                    for (int j = 0; j < m; j++) {
                        float sum = 0;
                        for (int k = 0; k < d; k++) {
                            sum += vecs[j][k] * training_features[groups[r][b][i]].values[k];
                        }
                        int hash_val = int((sum + t_vals[j]) / w) % (1 << L);
                        masks[r][b][j][hash_val] = true;
                    }
                }
            }
        }

        std::cout << "Masks generated successfully\n";

        //// Saving masks
        // We will now save each of the masks as a binary file
        for (int r = 0; r < R; r++) {
            for (int b = 0; b < B; b++) {
                std::string filename = "temp/mask_" + std::to_string(r) + "_" + std::to_string(b) + ".bin";
                std::ofstream file(filename, std::ios::binary);
                if (!file) {
                    throw std::runtime_error("Cannot open file");
                }

                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < (1 << L); j++) {
                        bool mask_value = masks[r][b][i][j];
                        file.write(reinterpret_cast<char*>(&mask_value), sizeof(bool));
                    }
                }
            }
        }

        std::cout << "Masks saved successfully\n";
    }
    else {
        //// Loading masks
        for (int r = 0; r < R; r++) {
            for (int b = 0; b < B; b++) {
            std::string filename = "temp/mask_" + std::to_string(r) + "_" + std::to_string(b) + ".bin";
            std::ifstream file(filename, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Cannot open file");
            }

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < (1 << L); j++) {
                bool mask_value;
                file.read(reinterpret_cast<char*>(&mask_value), sizeof(bool));
                masks[r][b][i][j] = mask_value;
                }
            }
            }
        }

        std::cout << "Masks loaded successfully\n";
    }

    //// Querying with the validation set
    auto running_total_time = 0;
    for (auto i = 0; i < 100; i++) {
        int random_index = std::rand() % val_features.size();
        VGGNetFeature random_query = val_features[random_index];

        std::vector<size_t> query_hash_values(m);   // This stores the hash values of the query
        std::vector<size_t> reported_neighbours(training_indices); // Makes a proper copy of training_indices. Will finally be the reported neighbours
        
        start = std::chrono::high_resolution_clock::now();
        
        // Calculating hash values of the query
        for (int i = 0; i < m; i++) {
            float sum = 0;
            for (int j = 0; j < d; j++) {
                sum += vecs[i][j] * random_query.values[j];
            }
            int hash_val = int((sum + t_vals[i]) / w) % (1 << L);
            query_hash_values[i] = hash_val;
        }

        // Checking against all the masks
        for (int r = 0; r < R; r++) {
            std::vector<size_t> iteration_neighbours;
            for (int b = 0; b < B; b++) {
                auto sum = 0;
                for (int i = 0; i < m; i++) {
                    if (masks[r][b][i][query_hash_values[i]]) 
                        sum++;
                }
                if (sum >= t) {
                    iteration_neighbours.insert(iteration_neighbours.end(), groups[r][b].begin(), groups[r][b].end());
                }
            }

            std::cout << "Number of neighbours reported in iteration " << r << ": " << iteration_neighbours.size() << std::endl;

            // Take intersection of reported neighbours and the current set of neighbours
            std::sort(iteration_neighbours.begin(), iteration_neighbours.end());
            std::sort(reported_neighbours.begin(), reported_neighbours.end());
            std::vector<size_t> new_reported_neighbours;
            std::set_intersection(iteration_neighbours.begin(), iteration_neighbours.end(), reported_neighbours.begin(), reported_neighbours.end(), std::back_inserter(new_reported_neighbours));
            reported_neighbours = new_reported_neighbours;
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        running_total_time += elapsed_seconds.count();
        std::cout << "Execution time: " << elapsed_seconds.count() << "s\n";
        std::cout << "          Number of neighbours reported finally : " << reported_neighbours.size() << std::endl;
    }
    std::cout << "Average time: " << running_total_time / 100.0 << "s\n";
    std::cout << "Done" << std::endl;


}   
