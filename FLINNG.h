#include "standard_header.h"
#include "omp.h"
#include "naive_search.h"

void random_group_creation (std::vector<std::vector<std::vector<uint64_t>>> &groups, std::vector<uint64_t> training_indices, std::string temp_dir, int R, int B, int N) {
    //// Creating R x B groups of size N / B randomly
    for (int r = 0; r < R; r++) {
        std::shuffle(training_indices.begin(), training_indices.end(), std::default_random_engine(std::time(nullptr))); // TODO : Verify that this is properly permuting the thing.
        for (int b = 0; b < B; b++) {   // Other methods to make this copy are also O(N) anyways so this is fine
            std::string group_filename = temp_dir+"/group_"+std::to_string(r)+"_"+std::to_string(b)+".bin";
            std::ofstream group_file(group_filename, std::ios::binary);
            if (!group_file) {
                throw std::runtime_error("Cannot open group_file");
            }
            for (auto i = 0; i < N / B; i ++) {
                groups[r][b][i] = training_indices[i + b * (N / B)];
                uint64_t group_value = groups[r][b][i];
                group_file.write(reinterpret_cast<char*>(&group_value), sizeof(uint64_t));
            }
        }
    }
}

void labelled_group_creation (std::vector<std::vector<std::vector<uint64_t>>> &groups, std::vector<uint64_t> training_indices, std::vector<uint64_t> &training_labels, std::string temp_dir, int R, int B, int N) {
    std::vector<uint64_t> unique_clusters(training_labels.begin(), training_labels.end());
    std::sort(unique_clusters.begin(), unique_clusters.end());
    unique_clusters.erase(std::unique(unique_clusters.begin(), unique_clusters.end()), unique_clusters.end());
    std::mt19937 rng(std::time(nullptr));

    for (auto r = 0 ; r < R ; r++) {
        std::vector<std::vector<uint64_t>> iter_groups;
        std::vector<uint64_t> outcasts;

        for (uint64_t cluster : unique_clusters) {
            std::vector<uint64_t> cluster_indices;
            
            // Find indices belonging to this cluster
            for (uint64_t i = 0; i < training_labels.size(); ++i) {
                if (training_labels[i] == cluster && i < N) {
                    cluster_indices.push_back(i);
                }
            }

            // Shuffle the indices
            std::shuffle(cluster_indices.begin(), cluster_indices.end(), rng);

            uint64_t group_size = N / B;
            uint64_t num_pure_groups = cluster_indices.size() / group_size;

            for (uint64_t i = 0; i < num_pure_groups; ++i) {
                iter_groups.push_back(std::vector<uint64_t>(
                    cluster_indices.begin() + i * group_size,
                    cluster_indices.begin() + (i + 1) * group_size
                ));
            }

            // Collect outcast elements
            outcasts.insert(outcasts.end(), 
                            cluster_indices.begin() + num_pure_groups * group_size, 
                            cluster_indices.end());
        }

        // Shuffle outcasts
        std::shuffle(outcasts.begin(), outcasts.end(), rng);

        // Distribute outcasts into additional groups
        uint64_t group_size = N / B;
        uint64_t num_extra_groups = outcasts.size() / group_size;
        
        for (uint64_t i = 0; i < num_extra_groups; ++i) {
            iter_groups.push_back(std::vector<uint64_t>(
                outcasts.begin() + i * group_size,
                outcasts.begin() + (i + 1) * group_size
            ));
        }

        groups[r] = iter_groups;
        
        for (int b = 0; b < B; b++) {
            std::string group_filename = temp_dir+"/group_"+std::to_string(r)+"_"+std::to_string(b)+".bin";
            std::ofstream group_file(group_filename, std::ios::binary);
            if (!group_file) {
                throw std::runtime_error("Cannot open group_file");
            }
            for (auto i = 0; i < N / B; i ++) {
                uint64_t group_value = groups[r][b][i];
                group_file.write(reinterpret_cast<char*>(&group_value), sizeof(uint64_t));
            }
        }
    }
}

void offlinePrep (std::vector<uint64_t> &training_indices, std::vector<uint64_t> &training_labels, std::vector<std::vector<std::vector<uint64_t>>> &groups, std::vector<std::vector<std::vector<std::vector<bool>>>> &masks, std::vector<std::vector<float>> &vecs, std::vector<double> &t_vals, std::string temp_dir, std::vector<VGGNetFeature> &training_features, int N, int B, int R, int m, int d, int l, double w, std::string group_creation_algorithm) {
    
    if (group_creation_algorithm == "random") {
        random_group_creation(groups, training_indices, temp_dir, R, B, N);
    }
    else if (group_creation_algorithm == "labelled") {
        labelled_group_creation(groups, training_indices, training_labels, temp_dir, R, B, N);
    }
    else {
        throw std::runtime_error("Invalid group_creation_algorithm");
    }

    for (auto r = 0; r < R; r++) {
        for (auto b = 0; b < B; b++) {
            std::sort(groups[r][b].begin(), groups[r][b].end());    // We are sorting the group before we save it to save on intersection cost in online setup
            
            #ifdef DEBUG
            for (auto i = 0; i < N / B; i ++) {
                    DEBUG_PRINT << groups[r][b][i] << " ";
            }
            DEBUG_PRINT << std::endl;
            #endif
        }
        DEBUG_PRINT << "------------------------------------" << std::endl;
    }

    // Generating and saving the vecs mask_file
    std::default_random_engine generator(std::time(nullptr));
    std::normal_distribution<float> distribution(0.0, 1.0);

    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        std::string vec_filename = temp_dir+"/vec_"+std::to_string(i)+".bin";
        std::ofstream vec_file(vec_filename, std::ios::binary);
        if (!vec_file) {
            throw std::runtime_error("Cannot open vec_file");
        }
        for (int j = 0; j < d; ++j) {
            vecs[i][j] = distribution(generator);
            float vec_value = vecs[i][j];
            vec_file.write(reinterpret_cast<char*>(&vec_value), sizeof(float));
        }
        vec_file.close();
    }

    // Let's generate the offset t for each hash function (and save it as well)
    std::uniform_real_distribution<double> uniform_dist(0, w);  
    std::string t_mask_filename = temp_dir+"/t_values.bin";
    std::ofstream t_mask_file(t_mask_filename, std::ios::binary);
    if (!t_mask_file) {
        throw std::runtime_error("Cannot open t_file");
    }
    for (int i = 0; i < m; ++i) {
        t_vals[i] = uniform_dist(generator);
        double t_value = t_vals[i];
        t_mask_file.write(reinterpret_cast<char*>(&t_value), sizeof(double));
    }
    t_mask_file.close();
    
    std::cout << "LSH functions initialised and saved successfully\n";

    //// Filling masks
    // We will go per group and make the masks  
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < R; r++) {
        #pragma omp parallel for
        for (int b = 0; b < B; b++) {
            for (int i = 0; i < N / B; i++) {
                for (int j = 0; j < m; j++) {
                    float dot_prod = std::inner_product(
                        vecs[j].begin(), vecs[j].end(),
                        training_features[groups[r][b][i]].values.begin(),
                        0.0f
                    );
                    int hash_val = (int((dot_prod + t_vals[j]) / w) % (l) + l) % l;  // I need to do it this way because C++ will output -7 % 5 = -2 instead of 3.
                    masks[r][b][j][hash_val] = true;
                }
            }
            
            // Saving the mask to disk
            std::string mask_mask_filename = temp_dir+"/mask_" + std::to_string(r) + "_" + std::to_string(b) + ".bin";
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

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time for making and saving masks: " << elapsed_seconds.count() << "ms\n";
}

void loadProcessedData (int m, int d, std::string temp_dir, std::vector<double> &t_vals, std::vector<std::vector<float>> &vecs, std::vector<std::vector<std::vector<uint64_t>>> &groups, std::vector<std::vector<std::vector<std::vector<bool>>>> &masks, int N, int B, int R, int l) {
    //// Loading the LSH functions
    for (int i = 0; i < m; i++) {
        std::string vec_filename = temp_dir+"/vec_" + std::to_string(i) + ".bin";
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

    std::string t_filename = temp_dir+"/t_values.bin";
    std::ifstream t_file(t_filename, std::ios::binary);
    if (!t_file) {
        throw std::runtime_error("Cannot open t_file");
    }
    for (int i = 0 ; i < m ; i ++) {
        double t_value;
        t_file.read(reinterpret_cast<char*>(&t_value), sizeof(double));
        t_vals[i] = t_value;
    }

    std::cout << "LSH functions loaded successfully\n";

    //// Loading groups
    //// Loading the LSH functions
    for (int r = 0; r < R; r++) {
        for (int b = 0; b < B; b++) {
            std::string group_filename = temp_dir+"/group_" + std::to_string(r) + "_" + std::to_string(b) + ".bin";
            std::ifstream group_file(group_filename, std::ios::binary);
            if (!group_file) {
                throw std::runtime_error("Cannot open group_file");
            }

            for (int i = 0; i < N / B; i++) {
                uint64_t group_value;
                group_file.read(reinterpret_cast<char*>(&group_value), sizeof(uint64_t));
                groups[r][b][i] = group_value;
            }
        }
    }

    //// Loading masks
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < R; r++) {
        for (int b = 0; b < B; b++) {
            std::string mask_filename = temp_dir+"/mask_" + std::to_string(r) + "_" + std::to_string(b) + ".bin";
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

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time for loading masks: " << elapsed_seconds.count() << "ms\n";

    std::ofstream yes_file(temp_dir + "/yes.bin", std::ios::binary);
    yes_file.write("yes", 3);
    yes_file.close();
}

double evaluateQuery(std::vector<VGGNetFeature> &training_features, std::vector<uint64_t> &training_indices, std::vector<std::vector<std::vector<uint64_t>>> &groups, std::vector<std::vector<std::vector<std::vector<bool>>> > &masks, std::vector<std::vector<float>> &vecs, std::vector<double> &t_vals, VGGNetFeature &random_query, int N, int B, int R, int m, int d, int l, double w, int t, std::string temp_dir) {
    std::vector<uint64_t> query_hash_values(m);   // This stores the hash values of the query
    std::vector<uint64_t> reported_neighbours(training_indices); // Makes a proper copy of training_indices. Will finally be the reported neighbours
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Calculating hash values of the query
    DEBUG_PRINT << "query_hash_values: \n";
    for (int i = 0; i < m; i++) {
        float sum = 0;
        for (int j = 0; j < d; j++) {
            sum += vecs[i][j] * random_query.values[j];
        }
        int hash_val = (int((sum + t_vals[i]) / w) % l + l) % l;
        query_hash_values[i] = hash_val;
        DEBUG_PRINT << hash_val << " ";
    }
    DEBUG_PRINT << std::endl;

    // Checking against all the masks
    for (int r = 0; r < R; r++) {
        std::vector<uint64_t> iteration_neighbours;
        for (int b = 0; b < B; b++) {
            #ifdef DEBUG
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < l; j++) {
                    if (masks[r][b][i][j] == true) {
                        DEBUG_PRINT << "(" << i << "," << j << ") ";
                    }
                }
                DEBUG_PRINT << std::endl;
            }
            DEBUG_PRINT << "------------\n" << std::endl;
            #endif

            auto sum = 0;
            for (int i = 0; i < m; i++) {
                sum += masks[r][b][i][query_hash_values[i]];        // Basically adds 1 if the mask is true thus is effectively counting collisions
            }

            if (sum >= t) {
                iteration_neighbours.insert(iteration_neighbours.end(), groups[r][b].begin(), groups[r][b].end());
            }
        }
        DEBUG_PRINT << "===============================\n" << std::endl;

        std::cout << "Number of neighbours reported in iteration " << r << ": " << iteration_neighbours.size() << std::endl;

        // Take intersection of reported neighbours and the current set of neighbours
        std::sort(iteration_neighbours.begin(), iteration_neighbours.end());
        std::sort(reported_neighbours.begin(), reported_neighbours.end());
        std::vector<uint64_t> new_reported_neighbours;
        std::set_intersection(iteration_neighbours.begin(), iteration_neighbours.end(), reported_neighbours.begin(), reported_neighbours.end(), std::back_inserter(new_reported_neighbours));
        reported_neighbours = new_reported_neighbours;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time: " << elapsed_seconds.count() << "ms\n";
    std::cout << "          Number of neighbours reported finally : " << reported_neighbours.size() << std::endl;

    // Now in order to check the correctness of the reported neighbours, we will run naive search on this as well
    int K = 10;
    ProspectiveNeighbours* ReportedNeighbours = new ProspectiveNeighbours(K);

    start = std::chrono::high_resolution_clock::now();
    
    for (auto i = 0; i < N; i++) {
        double distance = euclideanDistance(random_query.values, training_features[i].values);
        ReportedNeighbours->checkAndInsert(i, distance);
    }

    std::vector<uint64_t> golden_neighbours = ReportedNeighbours->topKNeighbours();
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Naive Search took : " << elapsed_seconds.count() << "ms\n";

    #ifdef DEBUG
    DEBUG_PRINT << "Golden neighbours: ";
    for (auto i : golden_neighbours) {
        DEBUG_PRINT << i << " ";
    }
    DEBUG_PRINT << std::endl;

    DEBUG_PRINT << "Reported neighbours: ";
    for (auto i : reported_neighbours) {
        DEBUG_PRINT << i << " ";
    }
    DEBUG_PRINT << std::endl;
    #endif

    // Computing the precision and recall of reported_neighbours wrt golden_neighbours
    double true_positives = 0.0;
    for (auto i : golden_neighbours) {
        if (std::find(reported_neighbours.begin(), reported_neighbours.end(), i) != reported_neighbours.end()) {
            true_positives++;
        }
    }
    double precision = true_positives / double(reported_neighbours.size());
    double recall = true_positives / K;
    std::cout << "-----------" << std::endl;
    std::cout << "| Precision = " << precision << std::endl;
    std::cout << "| Recall = " << recall << std::endl;
    std::cout << "-----------" << std::endl;
    // Write precision and recall into a CSV file
    std::ofstream csv_file(temp_dir + "/results_"+std::to_string(R)+".csv", std::ios::app);
    if (!csv_file) {
        throw std::runtime_error("Cannot open results CSV file");
    }
    csv_file << precision << "," << recall << "," << double(elapsed_seconds.count()) << "\n";
    csv_file.close();

    return double(elapsed_seconds.count());
}

bool checkMetadata(std::string temp_dir, int N, int B, int R, int m, int d, int l, double w) {
    std::ifstream metadata_file(temp_dir + "/metadata.json");
    if (!metadata_file) {
        return false;
    }

    nlohmann::json metadata;
    metadata_file >> metadata;
    metadata_file.close();

    if ((metadata["N"] == N) && (metadata["B"] == B) && (metadata["R"] >= R) && (metadata["m"] == m) && (metadata["d"] == d) && (metadata["l"] == l) && (metadata["w"] == w)) {
        std::cout << "Metadata matches\n";
        return true;
    }
    std::cout << "Metadata does not match\n";
    return false;
}

void updateMetadata(std::string temp_dir, int N, int B, int R, int m, int d, int l, double w, int t) {
    nlohmann::json metadata;
    metadata["N"] = N;
    metadata["B"] = B;
    metadata["m"] = m;
    metadata["d"] = d;
    metadata["l"] = l;
    metadata["w"] = w;
    metadata["t"] = t;
    
    
    std::cout << "Updating " << temp_dir + "/metadata.json\n";
    std::ifstream metadata_file(temp_dir + "/metadata.json");
    if (!metadata_file) {
        metadata["R"] = R;
    }
    else {
        nlohmann::json old_metadata;
        metadata_file >> old_metadata;
        metadata_file.close();
        bool compatible_metadata = checkMetadata(temp_dir, N, B, R, m, d, l, w);
        metadata["R"] = (compatible_metadata==true) ? static_cast<int>(old_metadata["R"]) : R;
    }
    
    std::ofstream file(temp_dir + "/metadata.json", std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Cannot open metadata file");
    }
    file << metadata.dump(4);  // Pretty print with indentation of 4 spaces
    if (!file) {
        throw std::runtime_error("Error writing to metadata file");
    }
    file.close();
}