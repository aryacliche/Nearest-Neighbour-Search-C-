import pandas as pd #type: ignore
import matplotlib.pyplot as plt #type: ignore
import json
import sys
import os

def single_use(temp_dir, R, graph_dir, savefig_name = None, csv_file = None, title_info = None):
    """
    Give this the name of a dir and R value, and this will plot the precision vs recall graph
    """
    # Load CSV file
    if csv_file is None:
        csv_file = f'{temp_dir}/results.csv'

    data = pd.read_csv(csv_file)

    if savefig_name is None:
        # Load JSON file
        json_file = f'{temp_dir}/metadata.json'
        with open(json_file, 'r') as f:
            info = json.load(f)
        # Extract information from JSON
        B = info.get('B', -1)
        N = info.get('N', -1)
        R = info.get('R', -1)
        d = info.get('d', -1)
        l = info.get('l', -1)
        t = info.get('t', -1)
        m = info.get('m', -1)
        w = info.get('w', -1)
        title_info = f'B={B}, N={N}, R={R}, d={d}, l={l}, t = {t}, m={m}, w={w}'
        savefig_name = f'{graph_dir}/precision_vs_recall_{title_info}_{temp_dir.replace("/", "_")}.png'
    else:
        title_info = f'{title_info}, FLINNG time={data.iloc[0, 2]}, Exhaustive time = {data.iloc[0, 3]}'
    
    # Plot data
    plt.figure()
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
    plt.title(title_info)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.xlim(0, 1)
    plt.ylim(0, 1)  

    # Save figure
    plt.savefig(savefig_name)
    plt.close()

    return [data.iloc[:, 0].max(), data.iloc[:, 1].max()]

def batch_type(dataset, algorithm, folder_name):
    suffix = 'results.csv'
    candidate_files = [f for f in os.listdir(folder_name) if f.endswith(suffix)]
    print(len(candidate_files), " candidate files found")

    max_prec_overall = 0
    max_rec_overall = 0
    for csv_file in candidate_files:
        temp_dir = None
        R = csv_file.split('_')[0]
        w = csv_file.split('_')[1]
        graph_dir = 'graphs'
        savefig_name = os.path.join(graph_dir, f'{dataset}_{algorithm}_{csv_file}'.replace('csv', 'png'))
        csv_file_path = os.path.join(folder_name, csv_file)
        title_info = f'R={R}, w={w}'
        max_values_seen = single_use(temp_dir, R, graph_dir, csv_file=csv_file_path, title_info = title_info, savefig_name = savefig_name)
        max_prec_overall = max(max_prec_overall, max_values_seen[0])
        max_rec_overall = max(max_rec_overall, max_values_seen[1])
    
    print("Max value of precision found = ", max_prec_overall)
    print("Max value of recall found = ", max_rec_overall)


def main():
    mode = sys.argv[1]
    if mode == 'single':
        temp_dir = sys.argv[2]
        R = sys.argv[3]
    else:
        dataset = sys.argv[2]
        algorithm = sys.argv[3]
        temp_dir = f'history/{dataset}/{algorithm}'

    if mode == 'single':
        graph_dir = 'graphs'
        single_use(temp_dir, R, graph_dir)
    elif mode == 'batch':
        batch_type(dataset, algorithm, temp_dir)
    else:
        print('Invalid mode')

if __name__ == "__main__":
    main()
