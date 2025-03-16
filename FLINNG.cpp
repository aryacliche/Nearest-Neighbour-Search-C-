#include "standard_header.h"

int main(int argc, char const *argv[]){
    const std::string train_mask_filename = argv[1];//"data/vggnet_imagenet_train_features.bin";
	const std::string val_mask_filename = argv[2];//"data/vggnet_imagenet_val_features.bin";
    int forced_N = -1;
    if (argc == 4) {
        forced_N = std::stoi(argv[3]);
    }

	// Load the training dataset (probably VGGNET features of IMAGENET/MIRFLICKR) [Store in the heap]
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<VGGNetFeature> training_features = readVGGNetFeatures(train_mask_filename);
	auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time for loading training dataset: " << elapsed_seconds.count() << "s\n";

	// Load the validation dataset (probably VGGNET features of IMAGENET/MIRFLICKR) [Store in the heap]
	start = std::chrono::high_resolution_clock::now();
    std::vector<VGGNetFeature> val_features = readVGGNetFeatures(val_mask_filename);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Time for loading validation dataset: " << elapsed_seconds.count() << "s\n";

    int N = (forced_N != -1) ? std::min(forced_N, static_cast<int>(training_features.size())) : training_features.size();
    int B = 2 * int(sqrt(N));
    int R = 25;
    int d = 4096;
    int m = 10;
    int t = 6;
    int L = 4;
    int l = 1 << L;
    int w = 1;
    N = N - N % B;
    training_features.resize(N);    // We only keep the first N features of the training set

    // We also want an "index" vector for the training features
	std::vector<size_t> training_indices(training_features.size());
	std::iota(training_indices.begin(), training_indices.end(), 0);    // Fills it with 0, 1, 2, ....

    //// Creating R x B groups of size N / B randomly
    std::vector<std::vector<std::vector<size_t>>> groups(R, std::vector<std::vector<size_t>>(B, std::vector<size_t>(N / B))); // TODO : Make this into an array if possible
    for (int r = 0; r < R; r++) {
        std::shuffle(training_indices.begin(), training_indices.end(), std::default_random_engine(std::time(nullptr))); // TODO : Verify that this is properly permuting the thing.
        for (int i = 0; i < N; i++) {   // Other methods to make this copy are also O(N) anyways so this is fine
            groups[r][i % B][i / B] = training_indices[i];
        }
    }

    //// Generating masks
    // Each mask is a vector of m bool-vectors. We need B such masks per iteration. There are R iterations. Thus finally masks will be a R x B x m x 2**L matrix OR vector of vector of vector of vector of bools
    std::vector<std::vector<std::vector<std::vector<bool>>>> masks(R, std::vector<std::vector<std::vector<bool>>>(B, std::vector<std::vector<bool>>(m, std::vector<bool>(l, false))));

    //// Generating hash functions
    // We will use p-stable LSH functions taken from the paper "Locality-Sensitive Hashing Scheme Based on p-Stable Distributions" by Piotr Indyk and Rajeev Motwani (https://dl.acm.org/doi/pdf/10.1145/997817.997857)
    std::vector<std::vector<float>> vecs(m, std::vector<float>(d));
    std::vector<int> t_vals(m);
    
    if (true) {
        // Generating and saving the vecs mask_file
        std::default_random_engine generator(std::time(nullptr));
        std::normal_distribution<float> distribution(0.0, 1.0);

        for (int i = 0; i < m; ++i) {
            std::string vec_mask_filename = "temp/vec_"+std::to_string(i)+".bin";
            std::ofstream vec_mask_file(vec_mask_filename, std::ios::binary);
            if (!vec_mask_file) {
                throw std::runtime_error("Cannot open vec_mask_file");
            }
            for (int j = 0; j < d; ++j) {
                vecs[i][j] = distribution(generator);
                float vec_value = vecs[i][j];
                vec_mask_file.write(reinterpret_cast<char*>(&vec_value), sizeof(float));
            }
            vec_mask_file.close();
        }

        // Let's generate the offset t for each hash function (and save it as well)
        std::uniform_int_distribution<int> uniform_dist(0, w);
        std::string t_mask_filename = "temp/t_values.bin";
        std::ofstream t_mask_file(t_mask_filename, std::ios::binary);
        if (!t_mask_file) {
            throw std::runtime_error("Cannot open t_file");
        }
        for (int i = 0; i < m; ++i) {
            t_vals[i] = uniform_dist(generator);
            int t_value = t_vals[i];
            t_mask_file.write(reinterpret_cast<char*>(&t_value), sizeof(int));
        }
        t_mask_file.close();
        
        std::cout << "LSH functions initialised and saved successfully\n";
    
        //// Filling masks
        // We will go per group and make the masks
        start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < R; r++) {
            for (int b = 0; b < B; b++) {
                for (int i = 0; i < N / B; i++) {
                    for (int j = 0; j < m; j++) {
                        float sum = 0;
                        for (int k = 0; k < d; k++) {
                            sum += vecs[j][k] * training_features[groups[r][b][i]].values[k];
                        }
                        int hash_val = (int((sum + t_vals[j]) / w) % (l) + l) % l;  // I need to do it this way because C++ will output -7 % 5 = -2 instead of 3.
                        masks[r][b][j][hash_val] = true;
                    }
                }
                
                // Saving the mask to disk
                std::string mask_mask_filename = "temp/mask_" + std::to_string(r) + "_" + std::to_string(b) + ".bin";
                std::ofstream mask_mask_file(mask_mask_filename, std::ios::binary);
                if (!mask_mask_file) {
                    throw std::runtime_error("Cannot open mask_mask_file");
                }

                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < (l); j++) {
                        bool mask_value = masks[r][b][i][j];
                        mask_mask_file.write(reinterpret_cast<char*>(&mask_value), sizeof(bool));
                    }
                }
                mask_mask_file.close();
            }
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "Time for making and saving masks: " << elapsed_seconds.count() << "s\n";
    }
    else {
        //// Loading the LSH functions
        for (int i = 0; i < m; i++) {
            std::string vec_filename = "temp/vec_" + std::to_string(i) + ".bin";
            std::ifstream vec_file(vec_filename, std::ios::binary);
            if (!vec_file) {
                throw std::runtime_error("Cannot open vec_file");
            }

            for (int j = 0; j < d; j++) {
                float vec_value;
                vec_file.read(reinterpret_cast<char*>(&vec_value), sizeof(float));
                vecs[i][j] = vec_value;
            }
        }

        std::string t_filename = "temp/t_values.bin";
        std::ifstream t_file(t_filename, std::ios::binary);
        if (!t_file) {
            throw std::runtime_error("Cannot open t_file");
        }
        for (int i = 0 ; i < m ; i ++) {
            int t_value;
            t_file.read(reinterpret_cast<char*>(&t_value), sizeof(int));
            t_vals[i] = t_value;
        }

        std::cout << "LSH functions loaded successfully\n";

        //// Loading masks
        start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < R; r++) {
            for (int b = 0; b < B; b++) {
            std::string mask_filename = "temp/mask_" + std::to_string(r) + "_" + std::to_string(b) + ".bin";
            std::ifstream mask_file(mask_filename, std::ios::binary);
            if (!mask_file) {
                throw std::runtime_error("Cannot open mask_file");
            }

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < (l); j++) {
                bool mask_value;
                mask_file.read(reinterpret_cast<char*>(&mask_value), sizeof(bool));
                masks[r][b][i][j] = mask_value;
                }
            }
            }
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "Time for making and saving masks: " << elapsed_seconds.count() << "s\n";
    }

    //// Querying with the validation set
    float running_total_time = 0.0;
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
            int hash_val = (int((sum + t_vals[i]) / w) % l + l) % l;
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
