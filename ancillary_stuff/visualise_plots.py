import pandas as pd #type: ignore
import numpy as np
import matplotlib.pyplot as plt #type: ignore
import sys
import os
from multiprocessing import Process

def count_diags(working_dir, file):
    counts_df = pd.read_csv(f'{working_dir}/{file}')
        
    R = np.max(counts_df['row'] + 1)
    B = np.max(counts_df['cell'] + 1)

    print('Number of rows:', R)
    print('Number of cells:', B)

    counts = np.zeros((R, B))
    for index, row in counts_df.iterrows():
        r = row['row']
        b = row['cell']

        counts[r][b]=row['count']

    plt.figure(figsize=(100, 5))
    plt.imshow(counts, cmap='viridis', interpolation=None)
    im_ratio = counts.shape[0]/counts.shape[1] 
    plt.colorbar(fraction=0.046*im_ratio, pad=0.004)
    # cbar.ax.set_height(20)
    plt.title('Counts')
    plt.xlabel('Cell')
    plt.ylabel('Row')
    plt.savefig(f'{working_dir}/{file.replace(".csv", ".png")}')
    plt.close()

def collision_diags(working_dir, collision_file):
    collisions_df = pd.read_csv(f'{working_dir}/{collision_file}')
    plt.figure(figsize=(10, 5))
    plt.plot(collisions_df['num_collisions'], collisions_df['freq'])
    plt.title('Collisions')
    plt.xlabel('Number of Collisions')
    plt.ylabel('Frequency')
    plt.savefig(f'{working_dir}/{collision_file.replace(".csv", ".png")}')
    plt.close()

def main():
    dataset = sys.argv[1]
    algo = sys.argv[2]
    allocated_threads = 24

    working_dir = f'/home/aryavishe/Nearest-Neighbour-Search-C-/alt_temp/{dataset}/{algo}'
    count_files = [f for f in os.listdir(working_dir) if 'counts' in f]
    
    for i in range(int(np.ceil(len(count_files) / allocated_threads))):
        num_threads = min(allocated_threads, len(count_files) - i * allocated_threads)
        
        threads = [Process(target=count_diags, args=(working_dir, count_files[i * allocated_threads + j])) for j in range(num_threads)]

        [thread.start() for thread in threads]

        [thread.join() for thread in threads]

    collision_files = [f for f in os.listdir(working_dir) if 'collisions' in f]
    for i in range(int(np.ceil(len(collision_files) / allocated_threads))):
        num_threads = min(allocated_threads, len(collision_files) - i * allocated_threads)
        
        threads = [Process(target=collision_diags, args=(working_dir, collision_files[i * allocated_threads + j])) for j in range(num_threads)]

        [thread.start() for thread in threads]

        [thread.join() for thread in threads]
        

if __name__=="__main__":
    main()
