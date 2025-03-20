import pandas as pd #type: ignore
import numpy as np
import matplotlib.pyplot as plt #type: ignore

def main():
    counts_df = pd.read_csv('counts.csv')
    
    R = np.max(counts_df['row'] + 1)
    B = np.max(counts_df['cell'] + 1)

    print('Number of rows:', R)
    print('Number of cells:', B)

    counts = np.zeros((R, B))
    for index, row in counts_df.iterrows():
        r = row['row']
        b = row['cell']

        counts[r][b]=row['count']

    plt.figure(figsize=(100, 100))
    plt.imshow(counts, cmap='viridis', interpolation=None)
    cbar = plt.colorbar()
    cbar.ax.set_aspect(20)
    plt.title('Counts')
    plt.xlabel('Cell')
    plt.ylabel('Row')
    plt.savefig('counts.png')

if __name__=="__main__":
    main()
