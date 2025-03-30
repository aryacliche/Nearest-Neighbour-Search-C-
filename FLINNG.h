#include "standard_header.h"
#include "omp.h"
#include "naive_search.h"

void random_group_creation (std::vector<uint64_t> &buckets, u_int64_t R, u_int64_t B, u_int64_t N) {
    for (uint64_t i = 0; i < R * N; i++) {
        buckets[i] =
            (rand() % B + B) % B +
            (i % R) * B;
    }
}

void labelled_group_creation (std::vector<uint64_t> &buckets, u_int64_t R, u_int64_t B, u_int64_t N, std::vector<uint64_t> &training_labels) {
    std::vector<uint64_t> unique_clusters(training_labels.begin(), training_labels.end());
    std::sort(unique_clusters.begin(), unique_clusters.end());
    unique_clusters.erase(std::unique(unique_clusters.begin(), unique_clusters.end()), unique_clusters.end());
    std::mt19937 rng(std::time(nullptr));

    std::cout << "Buckets has these many elements : " << buckets.size() << std::endl;
    std::cout << "N and R are " << N << "x" << R << std::endl;
    
    uint64_t group_size = N / B;
    
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
        uint64_t num_extra_groups = outcasts.size() / group_size;
        
        for (uint64_t i = 0; i < num_extra_groups; ++i) {
            iter_groups.push_back(std::vector<uint64_t>(
                outcasts.begin() + i * group_size,
                outcasts.begin() + (i + 1) * group_size
            ));
        }

        // Now we go through each of the points in this iteration and just add it to buckets
        for (uint64_t i = 0; i < iter_groups.size(); i++) {
            for (auto point : iter_groups[i]) {
                if (point * R + r >= N * R) {
                    std::cout << "Mistah Homes\n";
                }
                buckets[point * R + r] = B * r + i;
            }
        }
    }
}

class Flinng {

    public:
      Flinng(uint64_t num_rows, uint64_t cells_per_row, uint64_t num_hashes,
             uint64_t hash_range)
          : num_rows(num_rows), cells_per_row(cells_per_row),
            num_hash_tables(num_hashes), hash_range(hash_range),
            inverted_flinng_index(hash_range * num_hashes),
            cell_membership(num_rows * cells_per_row) {}
    
      // All the hashes for point 1 come first, etc.
      // Size of hashes should be multiple of num_hash_tables
      void addPoints(std::vector<uint64_t> hashes, std::string group_formation_algorithm, std::vector<uint64_t> &training_labels) {
    
        uint64_t num_points = hashes.size() / num_hash_tables;
        std::vector<uint64_t> buckets(num_rows * num_points);   // A bucket maps which cells a single point will be part of. It consists of R such mappings per point.
        if (group_formation_algorithm == "random") {
            random_group_creation(buckets, num_rows, cells_per_row, num_points);
        }
        else if (group_formation_algorithm == "labelled") {
            labelled_group_creation(buckets, num_rows, cells_per_row, num_points, training_labels);
        }
        else {
            throw std::runtime_error("Invalid group_formation_algorithm");
        }
        
        #pragma omp parallel for
        for (uint64_t table = 0; table < num_hash_tables; table++) {
          for (uint64_t point = 0; point < num_points; point++) {
            uint64_t hash = hashes[point * num_hash_tables + table];
            uint64_t hash_id = table * hash_range + hash;
            for (uint64_t row = 0; row < num_rows; row++) {
              inverted_flinng_index[hash_id].push_back(
                  buckets[point * num_rows + row]);
            }
          }
        }
    
        for (uint64_t point = 0; point < num_points; point++) {
          for (uint64_t row = 0; row < num_rows; row++) {
            cell_membership[buckets[point * num_rows + row]].push_back(
                total_points_added + point);
          }
        }
    
        total_points_added += num_points;
    
        prepareForQueries();
      }
    
      void prepareForQueries() {
        for (uint64_t i = 0; i < inverted_flinng_index.size(); i++) {
          std::sort(inverted_flinng_index[i].begin(),
                    inverted_flinng_index[i].end());
          inverted_flinng_index[i].erase(
              std::unique(inverted_flinng_index[i].begin(),
                          inverted_flinng_index[i].end()),
              inverted_flinng_index[i].end());
        }
      }
    
      // Again all the hashes for point 1 come first, etc.
      // Size of hashes should be multiple of num_hash_tables
      // Results are similarly ordered
      std::vector<uint64_t> query(std::vector<uint64_t> hashes, uint32_t top_k, std::string temp_dir) {
        uint64_t num_queries = hashes.size() / num_hash_tables;
        std::vector<uint64_t> results(top_k * num_queries);
        
        #ifdef DEBUG
        std::cout << "Num of queries = " << hashes.size() << " / " << num_hash_tables << std::endl;
        std::cout << "               = " << num_queries << std::endl;
        std::cout << "Size of results = " << top_k << " x " << num_queries << std::endl;
        std::cout << "                = " << results.size() << std::endl;
        #endif
    
        for (uint32_t query_id = 0; query_id < num_queries; query_id++) {
    
          std::vector<uint32_t> counts(num_rows * cells_per_row, 0);
          for (uint32_t rep = 0; rep < num_hash_tables; rep++) {
            const uint32_t index =
                hash_range * rep + hashes[num_hash_tables * query_id + rep];
            const uint32_t size = inverted_flinng_index[index].size();
            for (uint32_t small_index = 0; small_index < size; small_index++) {
              // This single line takes 80% of the time, around half for the move
              // and half for the add
              ++counts[inverted_flinng_index[index][small_index]];
            }
          }

          #ifdef VISUALISE
          std::ofstream counts_file(temp_dir + "/counts_"+std::to_string(query_id)+".csv", std::ios::trunc);
          if (!counts_file) {
              throw std::runtime_error("Cannot open counts CSV file");
          }
          counts_file << "row,cell,count\n";
          for (uint32_t i = 0; i < num_rows * cells_per_row; i++) {
              counts_file << i / cells_per_row << "," << i % cells_per_row << "," << counts[i] << "\n";
          }
          counts_file.close();
          #endif
    
          std::vector<uint32_t> sorted[num_hash_tables + 1];
          uint32_t size_guess = num_rows * cells_per_row / (num_hash_tables + 1);
          for (std::vector<uint32_t> &v : sorted) {
            v.reserve(size_guess);
          }
    
          for (uint32_t i = 0; i < num_rows * cells_per_row; ++i) {
            sorted[counts[i]].push_back(i);
          }

          #ifdef VISUALISE
          std::ofstream col_file(temp_dir + "/collisions_"+std::to_string(query_id)+".csv", std::ios::trunc);
          if (!col_file) {
              throw std::runtime_error("Cannot open collisions CSV file");
          }
          col_file << "num_collisions,freq\n";
          for (uint32_t i = 0; i < num_hash_tables + 1; i++) {
            col_file << i << "," << sorted[i].size() << "\n";
          }
          col_file.close();
          #endif
          for (uint32_t i = 0; i < num_hash_tables + 1; ++i) {
            DEBUG_PRINT << "collisions = " << i << " : " << sorted[i].size() << std::endl;
          }
    
          if (num_rows > 2) {
            std::vector<uint8_t> num_counts(total_points_added, 0);
            uint32_t num_found = 0;
            for (int32_t rep = num_hash_tables; rep >= 0; --rep) {
              for (uint32_t bin : sorted[rep]) {
                for (uint32_t point : cell_membership[bin]) {
                  if (++num_counts[point] == num_rows) {
                    results[top_k * query_id + num_found] = point;
                    if (++num_found == top_k) {
                      goto end_of_query;
                    }
                  }
                }
              }
            }
          } else {
            char *num_counts =
                (char *)calloc(total_points_added / 8 + 1, sizeof(char));
            uint32_t num_found = 0;
            for (int32_t rep = num_hash_tables; rep >= 0; --rep) {
              for (uint32_t bin : sorted[rep]) {
                for (uint32_t point : cell_membership[bin]) {
                  if (num_counts[(point / 8)] & (1 << (point % 8))) {
                    results[top_k * query_id + num_found] = point;
                    if (++num_found == top_k) {
                      free(num_counts);
                      goto end_of_query;
                    }
                  } else {
                    num_counts[(point / 8)] |= (1 << (point % 8));
                  }
                }
              }
            }
          }
        end_of_query:;
        }
    
        return results;
      }
    
      //// ARYA : I ADDED THIS!
      void save_to_disk (std::string temp_dir) {
        std::ofstream index_file(temp_dir + "/index.bin", std::ios::binary);
        if (!index_file) {
          throw std::runtime_error("Cannot open index_file");
        }
    
        for (auto i = 0; i < inverted_flinng_index.size(); i++) {
          uint64_t size = inverted_flinng_index[i].size();
          index_file.write(reinterpret_cast<char*>(&size), sizeof(uint64_t));
          index_file.write(reinterpret_cast<char*>(inverted_flinng_index[i].data()), size * sizeof(uint32_t));
        }
        index_file.close();
    
        std::ofstream membership_file(temp_dir + "/membership.bin", std::ios::binary);
        if (!membership_file) {
          throw std::runtime_error("Cannot open membership_file");
        }
    
        for (auto i = 0; i < cell_membership.size(); i++) {
          uint64_t size = cell_membership[i].size();
          membership_file.write(reinterpret_cast<char*>(&size), sizeof(uint64_t));
          membership_file.write(reinterpret_cast<char*>(cell_membership[i].data()), size * sizeof(uint64_t));
        }
        membership_file.close();
    
        std::ofstream points_file(temp_dir + "/num_points.bin", std::ios::binary);
        if (!points_file) {
          throw std::runtime_error("Cannot open points_file");
        }
    
        points_file.write(reinterpret_cast<char*>(&total_points_added), sizeof(uint64_t));
        points_file.close();
      }
    
      //// ARYA : I ADDED THIS!
      void load_from_disk (std::string temp_dir) {
        std::ifstream index_file(temp_dir + "/index.bin", std::ios::binary);
        if (!index_file) {
          throw std::runtime_error("Cannot open index_file");
        }
    
        for (uint64_t i = 0; i < inverted_flinng_index.size(); i++) {
          uint64_t size;
          index_file.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
          inverted_flinng_index[i].resize(size);
          index_file.read(reinterpret_cast<char*>(inverted_flinng_index[i].data()), size * sizeof(uint32_t));
        }
        index_file.close();
    
        std::ifstream membership_file(temp_dir + "/membership.bin", std::ios::binary);
        if (!membership_file) {
          throw std::runtime_error("Cannot open membership_file");
        }
    
        for (uint64_t i = 0; i < cell_membership.size(); i++) {
          uint64_t size;
          membership_file.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
          cell_membership[i].resize(size);
          membership_file.read(reinterpret_cast<char*>(cell_membership[i].data()), size * sizeof(uint64_t));
        }
        membership_file.close();
    
        std::ifstream points_file(temp_dir + "/num_points.bin", std::ios::binary);
        if (!points_file) {
          throw std::runtime_error("Cannot open points_file");
        }
    
        points_file.read(reinterpret_cast<char*>(&total_points_added), sizeof(uint64_t));
        points_file.close();
      }
    
    private:
      const uint64_t num_rows, cells_per_row, num_hash_tables, hash_range;
      uint64_t total_points_added = 0;
      std::vector<std::vector<uint32_t>> inverted_flinng_index;
      std::vector<std::vector<uint64_t>> cell_membership;
};

void offlinePrep (std::vector<std::vector<float>> &vecs, std::vector<double> &t_vals, std::string temp_dir, std::vector<VGGNetFeature> &training_features, std::vector<uint64_t> &training_labels, std::string &group_formation_algorithm, uint64_t N, uint64_t m, int d, int l, double w, Flinng &flinng) {        
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

    //// Creating hash values
    std::cout << "Allocating hashes of size " << m * N << std::endl;
    std::vector<uint64_t> hashes(m * N);
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (auto i = 0; i < training_features.size(); i++) {
        for (auto j = 0; j < m; j++) {
            float dot_prod = std::inner_product(
                vecs[j].begin(), vecs[j].end(),
                training_features[i].values.begin(),
                0.0f
            );
            hashes[ i * m + j ] = uint64_t((int((dot_prod + t_vals[j]) / w) % (l) + l) % l);  // I need to do it this way because C++ will output -7 % 5 = -2 instead of 3.
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time for making hash values: " << elapsed_seconds.count() << "ms\n";

    // Adding the points to the FLINNG index
    flinng.addPoints(hashes, group_formation_algorithm, training_labels);

    std::cout << "FLINNG index created successfully\n";

    // Save to disk
    flinng.save_to_disk(temp_dir);
}

double evaluateQuery(std::vector<std::vector<float>> &vecs, std::vector<double> &t_vals, std::vector<int> &query_indices, std::vector<VGGNetFeature> &val_features, uint32_t K, int num_queries, int m, int l, double w, std::string temp_dir, Flinng &flinng, std::vector<VGGNetFeature> &training_features) {
    std::vector<uint64_t> query_hash_values(num_queries * m);   // This stores the hash values of the query
    auto start = std::chrono::high_resolution_clock::now();
    
    // Calculating hash values of the query
    DEBUG_PRINT << "query_hash_values: \n";
    
    for (auto i = 0; i < num_queries; i++) {
        for (int j = 0; j < m; j++) {
            float dot_prod = std::inner_product(
                vecs[j].begin(), vecs[j].end(),
                val_features[query_indices[i]].values.begin(),
                0.0f
            );
            DEBUG_PRINT << "Dot prod = " << dot_prod <<  ", t_vals[j] = " << t_vals[j] << ", w = " << w << ", l = " << l<< std::endl;
            DEBUG_PRINT << "Final = " << uint64_t((int((dot_prod + t_vals[j]) / w) % l + l) % l) << std::endl;

            query_hash_values[i * m + j] = uint64_t((int((dot_prod + t_vals[j]) / w) % l + l) % l);
        }
    }

    #ifdef DEBUG
    DEBUG_PRINT << "Query hash values: \n";
    for (auto i = 0; i < num_queries; i++) {
        DEBUG_PRINT << i << " -> " ;
        for (int j = 0; j < m; j++) {
            DEBUG_PRINT << query_hash_values[i * m + j] << " ";
        }
        DEBUG_PRINT << std::endl;
    }
    #endif

    // Checking against all the masks (we will force it to return 10 * K neighbours and then further filter it)
    std::vector<uint64_t> reported_neighbours = flinng.query(query_hash_values, K, temp_dir);
    
    // std::vector<uint64_t> reported_neighbours(K * num_queries);
    // for (auto i = 0 ; i < num_queries; i++) {   // Doing exhaustive search on the smaller "dataset"
    //   ProspectiveNeighbours* ReportedNeighbours = new ProspectiveNeighbours(K);
        
    //   for (auto j = 0; j < 10 * K; j++) {
    //       double distance = euclideanDistance(val_features[query_indices[i]].values, training_features[filtered_neighbours[i * 10 * K + j]].values);
    //       ReportedNeighbours->checkAndInsert(filtered_neighbours[i * 10 * K + j], distance);
    //   }

    //   std::vector<uint64_t> curr_reported_neighbours = ReportedNeighbours->topKNeighbours();
    //   for (auto j = 0; j < K; j++) {
    //     reported_neighbours[i * K + j] = curr_reported_neighbours[j];
    //   }
      
    // }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time: " << elapsed_seconds.count() << "ms\n";
    std::cout << "          Number of neighbours reported finally : " << reported_neighbours.size() << std::endl;

    // Now in order to check the correctness of the reported neighbours, we will run naive search on this as well
    std::vector<uint64_t> golden_neighbours(num_queries * K);
    
    #ifdef DEBUG
    std::ofstream naive_file(temp_dir + "/naive_neighbours_returned.csv", std::ios::trunc);
    if (!naive_file) {
        throw std::runtime_error("Cannot open queries CSV file");
    }
    naive_file << "query_index,";
    for (auto i = 0 ; i < K; i++) {
      naive_file << "naive_neighbour_" << i << ",";
    }
    naive_file << "\n";
    #endif
    start = std::chrono::high_resolution_clock::now();

    #ifdef VISUALISE
    #pragma omp parallel for shared(golden_neighbours) // For now we want to only output the serialised values of nearest neighbour computation
    #endif
    for (auto i=0; i < num_queries; i++) {
        ProspectiveNeighbours* ReportedNeighbours = new ProspectiveNeighbours(K);
        
        for (auto j = 0; j < training_features.size(); j++) {
            double distance = euclideanDistance(val_features[query_indices[i]].values, training_features[j].values);
            ReportedNeighbours->checkAndInsert(j, distance);
        }

        #pragma omp critical
        {        
          #ifdef DEBUG
            naive_file << query_indices[i] << ",";
          #endif
          std::vector<uint64_t> curr_golden_neighbours = ReportedNeighbours->topKNeighbours();
          #ifdef VISUALISE
          std::vector<uint64_t> smallestDistances = ReportedNeighbours->smallestKDistances();
          std::ofstream dist_file(temp_dir+"/distances_"+std::to_string(i)+".csv", std::ios::trunc);
          if (!dist_file) {
              throw std::runtime_error("Cannot open distances CSV file");
          }
          dist_file << "neighbour,distance\n";
          for (auto i = 0; i < K; i++) {
              dist_file << curr_golden_neighbours[i] << "," << smallestDistances[i] << "\n";
          }
          dist_file.close();
          #endif
          for (auto j = 0; j < K; j++) {
              golden_neighbours[i * K + j] = curr_golden_neighbours[j];
              #ifdef DEBUG
              naive_file << curr_golden_neighbours[j] << ",";
              #endif
          }
          #ifdef DEBUG
          naive_file << '\n';
          #endif
        }
    }
    #ifdef DEBUG
    naive_file.close();
    #endif
    end = std::chrono::high_resolution_clock::now();
    auto naive_elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Naive Search took : " << naive_elapsed_seconds.count() << "ms\n";

    #ifdef DEBUG
    for (auto i = 0; i < num_queries; i ++) {
        DEBUG_PRINT << "Golden neighbours: ";
        for (auto j = i * K; j < (i + 1) * K; ++j) {
            std::sort(golden_neighbours.begin() + i * K, golden_neighbours.begin() + (i + 1) * K);
            DEBUG_PRINT << golden_neighbours[j]<< " ";
        }
        DEBUG_PRINT << std::endl;

        DEBUG_PRINT << "Reported neighbours: ";
        for (auto j = i * K; j < (i + 1) * K; ++j) {
            std::sort(reported_neighbours.begin() + i * K, reported_neighbours.begin() + (i + 1) * K);
            DEBUG_PRINT << reported_neighbours[j]<< " ";
        }
        DEBUG_PRINT << std::endl;
    }
    #endif  

    // Computing the precision and recall of reported_neighbours wrt golden_neighbours
    // Write precision, recall into a CSV file
    std::vector<double> precision(num_queries);
    std::vector<double> recall(num_queries);
    
    #pragma omp parallel for
    for (auto i = 0; i < num_queries; i ++) {
        double true_positives = 0.0;
        LabelMaker* labelmaker = new LabelMaker(); // To keep track of the golden labels
        for (auto j = i * K; j < (i + 1) * K; ++j) {
          for (auto k = 0; k < K; k++) {
            labelmaker->updateCounters(golden_neighbours[j]);
          }
          
          if (std::find(reported_neighbours.begin() + i * K, reported_neighbours.begin() + (i + 1) * K, golden_neighbours[j]) != reported_neighbours.begin() + (i + 1) * K) {
                true_positives++;
          }
        }
        std::vector<size_t> distribution_of_labels = labelmaker->topKLabels(5);
        precision[i] = true_positives / double(K);
        recall[i] = true_positives / double(K);

        #pragma omp critical
        {
            std::cout << "-----------" << std::endl;
            std::cout << "| Precision = " << precision[i] << std::endl;
            std::cout << "| Recall = " << recall[i] << std::endl;
            std::cout << "| Distribution =";
            for (auto j = 0; j < distribution_of_labels.size(); j++) {
                std::cout << " " << distribution_of_labels[j] / double(K);
            }
            std::cout << std::endl;
            std::cout << "-----------" << std::endl;
        }
    }
    std::ofstream csv_file(temp_dir + "/results.csv", std::ios::app);
    if (!csv_file) {
        throw std::runtime_error("Cannot open results CSV file");
    }
    for (auto i =0; i < num_queries; i ++) {
        csv_file << precision[i] << "," << recall[i] << "," << double(elapsed_seconds.count()) << "," << double(naive_elapsed_seconds.count()) << "\n";
    }
    csv_file.close();

    return double(elapsed_seconds.count());
}

bool checkMetadata(std::string temp_dir, int N, int B, int R, int m, int d, int l, double w) {
    std::ifstream metadata_file(temp_dir + "/metadata.json");
    if (!metadata_file) { // It doesn't exist yet
        return false;
    }

    nlohmann::json metadata;
    metadata_file >> metadata;
    metadata_file.close();

    if ((metadata["N"] == N) && (metadata["B"] == B) && (metadata["R"] == R) && (metadata["m"] == m) && (metadata["d"] == d) && (metadata["l"] == l) && (metadata["w"] == w)) {
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
