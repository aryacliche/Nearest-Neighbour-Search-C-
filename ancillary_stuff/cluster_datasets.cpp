#include "cluster.cpp"

int main(int argc, char const *argv[])
{
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " <train_data_file> <val_data_file> <num_passes> [num_clusters] [N]" << std::endl;
		std::cerr << "num_passes : Number of time we repeat the clustering" << std::endl;
		std::cerr << "num_clusters : Number of clusters (Use -1 for automated value) (default : empty)" << std::endl;
		std::cerr << "N : Limit on the number of datapoints to consider (Use -1 for unrestricted) (default : empty)" << std::endl;
		return 1;
	}

	const std::string train_filename = argv[1];//"data/vggnet_imagenet_train_features.bin";
	const std::string val_filename = argv[2];//"data/vggnet_imagenet_val_features.bin";
	int num_passes = atoi(argv[3]);
	int num_clusters = -1;
	if (argc > 4) {
		num_clusters = std::stoi(argv[4]); // Number of clusters
	}
	uint64_t forced_N = -1;
	if (argc > 5) {
		forced_N = std::stoi(argv[5]);
	}
	int num_threads = 24;
	omp_set_num_threads(num_threads);
	std::cout << "Number of threads: " << num_threads << std::endl;
	int training_d, validation_d; 

	// Load the training dataset (probably VGGNET features of IMAGENET/MIRFLICKR) [Store in the heap]
	auto start = std::chrono::high_resolution_clock::now();
	std::vector<VGGNetFeature> training_features = readVGGNetFeatures(train_filename, training_d);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Time for loading training dataset: " << elapsed_seconds.count() << "s\n";

#ifdef DEBUG
	for (auto i = 0 ; i < 10; i ++ ) {
		for (auto j = 0; j < training_d; j ++ ) {
			std::cout << training_features[i].values[j] << " ";
		}
		std::cout << std::endl;
	}
#endif

	uint64_t N = (forced_N != -1) ? min(forced_N, static_cast<uint64_t>(training_features.size())) : training_features.size();
	training_features.resize(N);    // We only keep the first N features of the training set

	// Load the validation dataset (probably VGGNET features of IMAGENET/MIRFLICKR) [Store in the heap]
	start = std::chrono::high_resolution_clock::now();
	std::vector<VGGNetFeature> val_features = readVGGNetFeatures(val_filename, validation_d);
	end = std::chrono::high_resolution_clock::now();
	elapsed_seconds = end - start;
	std::cout << "Time for loading validation dataset: " << elapsed_seconds.count() << "s\n";

	if (training_d != validation_d) {
		std::cout << "You have gotten a problemo friendo, training thinks d = " << std::to_string(training_d) << " whereas validation thinks d = " << std::to_string(validation_d);
		return -1;
	}
	else {
		std::cout << "Both files state that d = " << std::to_string(training_d) << std::endl;
	}

	int d = training_d; 
	// Let us now convert the std::vector<VGGNETFeature> to an appropriately sized array
	uint64_t B = 2 * int(sqrt(N));
	std::cout << "B = " << B << std::endl;
	std::cout << "N = " << N << std::endl;
		
	double** data = new double*[N];
	for (uint64_t i = 0; i < N; ++i) {
		data[i] = new double[d];
	}

	if (num_clusters == -1) { // It is up to us to decide the value
		num_clusters = B / 10;
	}

	// We will now convert the training_features to a 2D array
	std::cout << "Converting the std::vector into a 2-D array" << std::endl;
	dataMatrixFromVGGNETFeatures(training_features, data);

	// We will now cluster the data
	int** mask = new int*[N];
	double* weight = new double[d];
	for (uint64_t i = 0; i < N; ++i) {
		mask[i] = new int[d];
		for (int j = 0; j < d; ++j) {
			mask[i][j] = 1;
		}
		weight[i] = 1.0;
	}
	
	int *clusterid = new int[N];
	double* error = new double[N];
	int ifound;

	start = std::chrono::high_resolution_clock::now();

	std::cout << "Clustering the data" << std::endl;    
	kcluster(num_clusters, N, d, data, mask, weight, 0, 5, 'a', 'e', clusterid, error, &ifound);

	// We will now just print the clusterid
	std::string output_filename = train_filename;
	size_t pos = output_filename.find("features.bin");
	if (pos != std::string::npos) {
		output_filename.replace(pos, std::string("features.bin").length(), "clustered_labels.txt");
	}

	std::ofstream outfile(output_filename);
	if (!outfile) {
		std::cerr << "Error: Could not open file for writing cluster IDs." << std::endl;
		return 1;
	}

	for (uint64_t i = 0; i < N; ++i) {
		outfile << clusterid[i] << std::endl;
	}
	outfile.close();

	std::cout << "Cluster IDs have been written to " << output_filename << std::endl;

	end = std::chrono::high_resolution_clock::now();
	elapsed_seconds = end - start;
	std::cout << "Clustering and saving clusters took: " << elapsed_seconds.count() << "s\n";
	return 0;
}
