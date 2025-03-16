import matplotlib.pyplot as plt
import sys
import numpy as np

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def p_stable_hash_value(vec, data_point, t, w):
    return int(np.floor((np.dot(vec, data_point) + t) / w))

hash_value = p_stable_hash_value

def main():
    print("We will check how good the LSH functions at abiding to the locality sensitive hashing property.")
    print("                     sim(x, y) = Pr_\{H\}(h(x) = h(y))      for h ~ H (family of LSH functions)")

    w = float(sys.argv[1]) 
    d = int(sys.argv[2]) # input dimension
    num_lsh_functions = 1000
    
    vecs = [np.random.randn(d) for _ in range(num_lsh_functions)]
    t = np.random.rand() * w

    distance = []
    probability = []
    for i in range(1, 50001):
        # Generate random vectors
        x = np.random.rand(d)
        intermediate = np.random.rand(d)
        y = x + (i / 1000) * intermediate / np.linalg.norm(intermediate)

        try:
            distance.append(np.linalg.norm(x - y))
            counter = 0
            for vec in vecs:
                if hash_value(vec, x, t, w) == hash_value(vec, y, t, w):
                    counter += 1
            probability.append(counter / num_lsh_functions)
        except:
            print("                 Invalid value encountered")

    plt.figure(figsize=(20, 10))
    # Ideal similarity would be 1/ (1 + ||x - y||) because that is bounded to [0, 1] thus we can plot it
    plt.plot([1/(1 + x) for x in distance], probability, 'rx', label = "1/(1+d) vs probability")
    plt.plot([1/(1 + x**2) for x in distance], probability, 'bx', label = "1/(1+d^2) vs probability")
    plt.plot([1/(1 + x**3) for x in distance], probability, 'gx', label = "1/(1+d^3) vs probability")
    plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), 'b', label = "ideal")
    plt.ylabel('Probability')
    plt.xlabel('Inverse Distance')
    plt.title('Probability vs Inverse Distance')
    plt.legend()
    plt.savefig(f'./testing_LSH_results/distance_testing_LSH_function_{d}_{w}.png')

if __name__=="__main__":
    main()
