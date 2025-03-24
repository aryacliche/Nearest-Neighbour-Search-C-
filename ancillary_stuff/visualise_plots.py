import pandas as pd #type: ignore
import numpy as np
import matplotlib.pyplot as plt #type: ignore
import sys
import os

def main():
    dataset = sys.argv[1]
    algo = sys.argv[2]

    working_dir = f'/home/aryavishe/Nearest-Neighbour-Search-C-/alt_temp/{dataset}/{algo}'
    count_files = [f for f in os.listdir(working_dir) if 'counts' in f]
    for file in count_files:
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

    collision_files = [f for f in os.listdir(working_dir) if 'collisions' in f]
    for collision_file in collision_files:    
        collisions_df = pd.read_csv(f'{working_dir}/{collision_file}')
        plt.figure(figsize=(10, 5))
        plt.plot(collisions_df['num_collisions'], collisions_df['freq'])
        plt.title('Collisions')
        plt.xlabel('Number of Collisions')
        plt.ylabel('Frequency')
        plt.savefig(f'{working_dir}/{collision_file.replace(".csv", ".png")}')

if __name__=="__main__":
    main()
