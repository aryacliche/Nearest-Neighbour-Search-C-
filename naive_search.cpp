#include "standard_header.h"
#include "naive_search.h"
#include "omp.h"

/*
    Deprecated code : This function is not needed anymore
    bool compareFeatures(int index1, int index2, std::vector<VGGNetFeature>& features, VGGNetFeature& reference) {
        return euclideanDistance(features[index1].values, reference.values) < euclideanDistance(features[index2].values, reference.values);
    }
*/

int main(int argc, char const *argv[])
{
	if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <train_features_file> <val_features_file> <K> [N]" << std::endl;
        return 1;
    }

    const std::string train_filename = argv[1];//"data/vggnet_imagenet_train_features.bin";
	const std::string val_filename = argv[2];//"data/vggnet_imagenet_val_features.bin";
    const int K = std::stoi(argv[3]); // Number of neighbours we are interested in 
    int forced_N = -1;
    if (argc >= 5) {
        forced_N = std::stoi(argv[4]);
    }

	// Load the training dataset (probably VGGNET features of IMAGENET/MIRFLICKR) [Store in the heap]
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<VGGNetFeature> training_features = readVGGNetFeatures(train_filename);
	auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time for loading training dataset: " << elapsed_seconds.count() << "s\n";

    int N = (forced_N != -1) ? std::min(forced_N, static_cast<int>(training_features.size())) : training_features.size();
    training_features.resize(N);    // We only keep the first N features of the training set

	// Load the validation dataset (probably VGGNET features of IMAGENET/MIRFLICKR) [Store in the heap]
	start = std::chrono::high_resolution_clock::now();
    std::vector<VGGNetFeature> val_features = readVGGNetFeatures(val_filename);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Time for loading validation dataset: " << elapsed_seconds.count() << "s\n";

	// We also want an "index" vector for the training features
	std::vector<size_t> training_indices(training_features.size());
	std::iota(training_indices.begin(), training_indices.end(), 0);    // Fills it with 0, 1, 2, ....

    #ifdef VISUALISE_DISTANCES
    int num_threads = 24;
    omp_set_num_threads(num_threads);
    std::ofstream csv_file("ancillary_stuff/distances.csv", std::ios::trunc);
    csv_file << "Index1,Index2,Distance\n";
    #endif

	std::srand(std::time(nullptr)); // use current time as seed for random generator
    
    auto running_total_time = 0.0;

    for (auto i = 0; i < 100; i++) {
        int random_index = std::rand() % val_features.size();
        VGGNetFeature random_query = val_features[random_index];

        /*
        Deprecated code : This sorts the entire dataset to find the K-nearest neighbours. This is not efficient.
            std::vector<size_t> sorted_indices(training_indices); // Makes a proper copy of training_indices
            std::sort(sorted_indices.begin(), sorted_indices.end(), [&](size_t a, size_t b) {
                return compareFeatures(a, b, training_features, random_query);
            });
        */
        
        ProspectiveNeighbours* ReportedNeighbours = new ProspectiveNeighbours(K);

        start = std::chrono::high_resolution_clock::now();
        #ifdef VISUALISE_DISTANCES
        #pragma omp parallel for
        #endif
        for (auto i = 0; i < N; i++) {
            double distance = euclideanDistance(random_query.values, training_features[i].values);
            #ifdef VISUALISE_DISTANCES
            #pragma omp critical
            {
                csv_file << random_index << "," << i << "," << distance << "\n";
            }
            #endif
            ReportedNeighbours->checkAndInsert(i, distance);
        }

        std::vector<size_t> reported_neighbours = ReportedNeighbours->topKNeighbours();

        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        running_total_time += elapsed_seconds.count();
        std::cout << "Execution time: " << elapsed_seconds.count() << "s\n";
    }
    std::cout << "Average time: " << running_total_time / 100.0 << "s\n";
    std::cout << "Done" << std::endl;

    #ifdef VISUALISE_DISTANCES
    csv_file.close();
    #endif
}
