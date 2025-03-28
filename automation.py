import json
import subprocess
import os
import sys
from plotter import single_use

def main():
    mode = sys.argv[1]
    dataset_mode = sys.argv[2]
    if dataset_mode == None:
        dataset_range = ['imagenet', 'mirflickr']
        w_range = [0.05, 0.1, 0.3, 0.5, 1.0, 5.0, 10.0, 30.0, 50.0]
    else:
        dataset_range = [dataset_mode]
        if dataset_mode == 'imagenet':
            w_range = [1.0, 5.0, 10.0, 30.0, 50.0]
        else:
            w_range = [1.0, 5.0, 10.0, 30.0, 50.0]
    
    json_file_path = 'config.json'
    for r in range(20, 1, -3):
        for w in w_range:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            data['R'] = r
            data['w'] = w

            with open(json_file_path, 'w') as file:
                json.dump(data, file, indent=4)

            for dataset in dataset_range:
                for group_formation in ['labelled']:
                    print(f"Running for {dataset} and {group_formation}")
                    subprocess.run(['./run_FLINNG.sh', f'{mode}', f'{dataset}', f'{group_formation}', f'{r}'])
                    if mode == 'visualise':        
                        try:
                            src_dir = f'/home/aryavishe/Nearest-Neighbour-Search-C-/temp/{dataset}/{group_formation}/'
                            dest_dir = f'/home/aryavishe/Nearest-Neighbour-Search-C-/history_400_clus/{dataset}/{group_formation}/'
                            os.makedirs(dest_dir, exist_ok=True)
                            collision_files = [f for f in os.listdir(src_dir) if 'collisions' in f] # This contains both csv and png files
                            distance_files = [f for f in os.listdir(src_dir) if 'distances' in f]
                            counts_files = [f for f in os.listdir(src_dir) if 'counts' in f] # This contains both csv and png files
                            total_files = ['results.csv', 'log.out']
                            total_files.extend(distance_files)
                            total_files.extend(counts_files)
                            total_files.extend(collision_files)
                            for filename in total_files:
                                src_file = os.path.join(src_dir, filename)
                                dest_file = os.path.join(dest_dir, f'{r}_{w}_{filename}')
                                os.rename(src_file, dest_file)

                            # Now we will plot the precision vs recall for all of the graphs
                            print(f"Printing precision vs recall graphs")
                            single_use(dest_dir, r, "graphs", savefig_name = f"{dest_dir}/{r}_{w}_precision_vs_recall.png", csv_file = f"{dest_dir}/{r}_{w}_results.csv", title_info = f"R = {r}, w= {w}")
                        except Exception as e:
                            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
