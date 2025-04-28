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
	int d;
	if (argc == 8) {
		forced_N = std::stoi(argv[7]);
	}

	int num_threads = 24;
	omp_set_num_threads(num_threads);

	uint32_t K = 100; // Number of neighbours we want.

	// Verifying that OpenMP is enabled
#ifdef _OPENMP
	std::cout << "OpenMP is enabled! Version: " << _OPENMP << std::endl;
#else
	std::cout << "OpenMP is NOT enabled!" << std::endl;
#endif

	// Estimate the size of the training dataset
	auto start = std::chrono::high_resolution_clock::now();
	std::vector<VGGNetFeature> training_features = readVGGNetFeatures(train_features_file_name, d);
	std::vector<uint64_t> training_labels = readLabelsAsInt(train_labels_file_name);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Time for loading training dataset: " << elapsed_seconds.count() << "s\n";

	// Load the validation dataset (probably VGGNET features of IMAGENET/MIRFLICKR) [Store in the heap]
	start = std::chrono::high_resolution_clock::now();
	std::vector<VGGNetFeature> val_features = readVGGNetFeatures(val_features_file_name, d);
	std::vector<uint64_t> val_labels = readLabelsAsInt(val_labels_file_name);
	end = std::chrono::high_resolution_clock::now();
	elapsed_seconds = end - start;
	std::cout << "Time for loading validation dataset: " << elapsed_seconds.count() << "s\n";

	uint64_t N = (forced_N != -1) ? std::min(forced_N, int(training_features.size())) : training_features.size();
	int B = 2 * int(sqrt(N));
	N = N - N % B;
	training_features.resize(N);    // We only keep the first N features of the training set

	// Parameters that should be read from a config file
	std::ifstream config_file(temp_dir + "/config.json");
	if (!config_file) {
		std::cout << "Could not open config file. Please make sure it exists.\n";
		std::cout << "Exiting...\n";
		return false;
	}
	nlohmann::json config;
	config_file >> config;
	config_file.close();
	int R = config["R"];
	int t = config["t"];
	uint64_t m = config["m"];
	int L = config["L"];
	double w = config["w"];

	int l = 1 << L; 

	// We will use p-stable LSH functions taken from the paper "Locality-Sensitive Hashing Scheme Based on p-Stable Distributions" by Piotr Indyk and Rajeev Motwani (https://dl.acm.org/doi/pdf/10.1145/997817.997857)
	std::vector<std::vector<float>> vecs(m, std::vector<float>(d));
	std::vector<double> t_vals(m);

	Flinng flinng(R, B, m, l);

	bool masks_present = checkMetadata(temp_dir, N, B, R, m, d, l, w);    

	if (masks_present == false) {
		offlinePrep(vecs, t_vals, temp_dir, training_features, training_labels, group_creation_algorithm, N, m, d, l, w, flinng);
	}
	else {
		flinng.load_from_disk(temp_dir);

		// Also update the vecs and t_vals
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
	}
	updateMetadata(temp_dir, N, B, R, m, d, l, w, t);   

	// Just before we start the query phase
	flinng.prepareForQueries();

	//// Querying with the validation set
	// Discarding contents of results.csv
	std::ofstream csv_file(temp_dir + "/results.csv", std::ios::trunc);
	if (!csv_file) {
		throw std::runtime_error("Cannot open results CSV file");
	}
	csv_file << "precision" << "," << "recall" << "," << "flinng_running_time" << "," <<  "naive_running_time\n";
	csv_file.close();

	int num_queries = 100;
	std::vector<int> query_indices;
	query_indices.reserve(num_queries);

	for (auto i = 0; i < num_queries; i++) {
		query_indices.push_back(rand() % val_features.size());
	}

	// We run the online querying phase
	double running_total_time = evaluateQuery(vecs, t_vals, query_indices, val_features, K, num_queries, m, l, w, temp_dir, flinng, training_features, training_labels);

	std::cout << "Average time: " << running_total_time / num_queries << "ms\n";
	std::cout << "Done" << std::endl;
}   
