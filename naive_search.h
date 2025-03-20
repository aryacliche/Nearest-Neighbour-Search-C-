class ProspectiveNeighbours {  // This linked list is good to use when we are looking at K that is close to 100. We don't want to 100 comparisons for every single datapoint.
    struct Node {
        size_t index;
        double distance;
        Node* betterNeighbour;
        Node(size_t index, double distance) : index(index), distance(distance), betterNeighbour(nullptr) {}
        Node(size_t index, double distance, Node* betterNeighbour) : index(index), distance(distance), betterNeighbour(betterNeighbour) {}
    };
    
    Node* worstNeighbour;

    public:
        ProspectiveNeighbours(size_t K){
            Node* current_neighbour = new Node(-1, MAXFLOAT);   // This is the best neighbour
            for (int i = 1; i < K - 1; i++)
                current_neighbour = new Node(-1, MAXFLOAT, current_neighbour);
            worstNeighbour = new Node(-1, MAXFLOAT, current_neighbour); // We have now created the worst neighbour
        }

        void checkAndInsert(size_t index, double distance) {
            Node* current = worstNeighbour;
            Node* previous = nullptr;
            while (current != nullptr) {
                if (distance < current->distance) {  
                    if (previous != nullptr) {
                        previous->distance = current->distance;
                        previous->index = current->index;
                    }
                    current->index = index;
                    current->distance = distance;
                    previous = current;
                    current = current->betterNeighbour; // We are now going to the next neighbour
                }
                else {
                    // This node is better than the current distance. Break
                    break;
                }
            }
        }

        std::vector<size_t> topKNeighbours() {
            std::vector<size_t> topK;
            Node* current = worstNeighbour;
            while (current != nullptr) {
                topK.push_back(current->index);
                current = current->betterNeighbour;
            }
            return topK;
        }

        std::vector<size_t> smallestKDistances() {
            std::vector<size_t> topK;
            Node* current = worstNeighbour;
            while (current != nullptr) {
                topK.push_back(current->distance);
                current = current->betterNeighbour;
            }
            return topK;
        }
};