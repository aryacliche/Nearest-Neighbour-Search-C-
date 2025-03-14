import lshashpy3 as lshash  # Taken from https://github.com/loretoparisi/lshash
import matplotlib.pyplot as plt
import sys
import numpy as np
import logging
logger = logging.getLogger(__name__)

if 'debug' in sys.argv[3].lower():
    logger.setLevel(logging.DEBUG)
else:
    logger.disable()

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def hash_value(lsh_function, data_point):
    """
    lshash.LSHash has methods to get us the binary representation of the hash values. This is just a wrapper function around it
    """
    str = lsh_function.get_hashes(data_point)[0]
    return int(str, 2)

def main():
    print("We will check how good the LSH functions at abiding to the locality sensitive hashing property.")
    print("                     sim(x, y) = Pr_\{H\}(h(x) = h(y))      for h ~ H (family of LSH functions)")

    L = int(sys.argv[1]) # hash size
    d = int(sys.argv[2]) # input dimension
    num_lsh_functions = 1000
    family_of_lsh_functions = [lshash.LSHash(hash_size=L, input_dim=d, num_hashtables=1) for _ in range(num_lsh_functions)]

    cosine_sim = []
    probability = []
    for i in range(1000):
        # Generate random vectors
        x = np.random.rand(d)
        y = np.random.rand(d)
        # We want the cosine similarity between them to be i/ 1000
        # s* = i / 1000 [our target similarity which is achieved using y*]
        # s = <x, y> / (||x|| ||y||) [cosine similarity]
        # s* = <x, y*> / (||x|| ||y*||)
        # Thus we will try to find alpha such that y* = y + alpha * x
        # Then S* = <x, y + alpha * x> / (||x|| ||y + alpha * x||)
        # Assuming alpha << 1
        # S* = <x, y> / (||x|| ||y|) + alpha * <x, x> / (||x|| ||y|)
        # S* = S + alpha * ||x|| / ||y|
        # Thus alpha = (i / 1000 - S) / (||x|| / ||y|) 
        alpha = (i / 1000 - cosine_similarity(x, y)) / (np.linalg.norm(x) / np.linalg.norm(y))
        y = y + alpha * x       # this assumes alpha is small
        print(f"            {alpha}")

        try:
            cosine_sim.append(cosine_similarity(x, y))
            counter = 0
            for h in family_of_lsh_functions:
                if hash_value(h, x) == hash_value(h, y):
                    counter += 1
            probability.append(counter / num_lsh_functions)
        except:
            print("                 Invalid value encountered")

    plt.figure(figsize=(20, 10))
    plt.plot(cosine_sim, probability, 'ro', label = "cosine vs probability")
    plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), 'b', label = "ideal")
    plt.ylabel('Probability')
    plt.xlabel('Cosine Similarity')
    plt.title('Probability vs Cosine Similarity')
    plt.legend()
    plt.savefig(f'testing_LSH_results/cosine_testing_LSH_function_{L}_{d}.png')
    
    distance = []
    probability = []
    for i in range(1, 1001):
        # Generate random vectors
        x = np.random.rand(d)
        intermediate = np.random.rand(d)
        y = intermediate * (1000/i) / np.linalg.norm(intermediate - x)  # We want the distance between x and y to be close to i/1000 

        try:
            distance.append(np.linalg.norm(x - y))
            counter = 0
            for _ in range(num_lsh_functions):
                h = family_of_lsh_functions[_]
                if hash_value(h, x) == hash_value(h, y):
                    counter += 1
            probability.append(counter / num_lsh_functions)
        except:
            print("                 Invalid value encountered")

    plt.figure(figsize=(20, 10))
    # Ideal similarity would be 1/ (1 + ||x - y||) because that is bounded to [0, 1] thus we can plot it
    plt.plot([1/(1 + x) for x in distance], probability, 'ro', label = "inverse-distance vs probability")
    plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), 'b', label = "ideal")
    plt.ylabel('Probability')
    plt.xlabel('Inverse Distance')
    plt.title('Probability vs Inverse Distance')
    plt.legend()
    plt.savefig(f'testing_LSH_results/distance_testing_LSH_function_{L}_{d}.png')

if __name__=="__main__":
    main()