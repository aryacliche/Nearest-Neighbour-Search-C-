import pandas as pd #type: ignore
import numpy as np
import matplotlib.pyplot as plt #type: ignore
import sys
import os
from multiprocessing import Process

def count_diags(working_dir, file):
    try:
        counts_df = pd.read_csv(f'{working_dir}/{file}')
            
        R = np.max(counts_df['row'] + 1)
        B = np.max(counts_df['cell'] + 1)

        # print('Number of rows:', R)
        # print('Number of cells:', B)

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
    except:
        print(f"Error processing file: {file}. It may not exist or be in the expected format.")

def collision_diags(working_dir, collision_file):
    try:
        collisions_df = pd.read_csv(f'{working_dir}/{collision_file}')
        threshold = int(collisions_df.columns[1].split('(')[1].split(')')[0])
        collisions_df = collisions_df.rename(columns={collisions_df.columns[1]: 'freq'})
        plt.figure(figsize=(10, 5))
        fig, ax1 = plt.subplots()

        # First y-axis
        ax1.plot(collisions_df['num_collisions'], collisions_df['freq'], label='Frequency')
        ax1.set_xlabel('Number of Collisions')
        ax1.set_ylabel('Frequency', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Second y-axis
        ax2 = ax1.twinx()
        ax2.plot(collisions_df['num_collisions'], collisions_df['cumulative_points'], color='red', alpha=0.5, linestyle='-', label='Cumulative')
        ax2.plot([threshold, threshold], [0, max(collisions_df['cumulative_points'])], color='green', alpha=0.5, linestyle='--', label='Threshold')
        ax2.set_ylabel('Cumulative', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Title and save
        plt.title('Collisions')
        fig.tight_layout()
        plt.title(f'Collisions (Threshold: {threshold}, Number of graduating cells = {collisions_df["cumulative_points"].iloc[np.where(collisions_df["num_collisions"] == threshold)[0][0]]})')
        plt.savefig(f'{working_dir}/{collision_file.replace(".csv", ".png")}')
        plt.close()
    except:
        print(f"Error processing file: {collision_file}. It may not exist or be in the expected format.")

def main():
    dataset = sys.argv[1]
    algo = sys.argv[2]
    layer = sys.argv[3]
    if len(sys.argv) > 4:
        working_dir = sys.argv[4]
    else:
        working_dir = f'/home/aryavishe/Nearest-Neighbour-Search-C-/temp/{layer}/{dataset}/{algo}'
    allocated_threads = 24

    os.makedirs(working_dir, exist_ok=True)
    count_files = [f for f in os.listdir(working_dir) if ('counts' in f and f.endswith('.csv'))]
    
    for i in range(int(np.ceil(len(count_files) / allocated_threads))):
        num_threads = min(allocated_threads, len(count_files) - i * allocated_threads)
        
        threads = [Process(target=count_diags, args=(working_dir, count_files[i * allocated_threads + j])) for j in range(num_threads)]

        [thread.start() for thread in threads]

        [thread.join() for thread in threads]

    collision_files = [f for f in os.listdir(working_dir) if ('collisions' in f and f.endswith('.csv'))]
    for i in range(int(np.ceil(len(collision_files) / allocated_threads))):
        num_threads = min(allocated_threads, len(collision_files) - i * allocated_threads)
        
        threads = [Process(target=collision_diags, args=(working_dir, collision_files[i * allocated_threads + j])) for j in range(num_threads)]

        [thread.start() for thread in threads]

        [thread.join() for thread in threads]
        

if __name__=="__main__":
    main()
