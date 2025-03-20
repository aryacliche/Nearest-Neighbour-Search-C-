import pandas as pd #type: ignore
import numpy as np
import matplotlib.pyplot as plt #type: ignore
import sys

def main():
    dataset = sys.argv[1]
    algo = sys.argv[2]

    working_dir = f'../alt_temp/{dataset}/{algo}'
    counts_df = pd.read_csv(f'{working_dir}/counts.csv')
    
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
    plt.savefig('counts.png')

    groups_df = pd.read_csv(f'{working_dir}/cell_memberships.csv')
    groups = np.zeros((R,B))
    for index, row in groups_df.iterrows():
        r = row['cell'] / R
        b = row['cell'] % R
        
        groups[r][b] = row['size']

    plt.figure(figsize=(100, 100))
    plt.imshow(groups, cmap='viridis', interpolation=None)
    plt.colorbar()
    plt.title('Groups')
    plt.xlabel('Cell')
    plt.ylabel('Row')
    plt.savefig('groups.png')

if __name__=="__main__":
    main()
