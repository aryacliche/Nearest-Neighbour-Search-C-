import json
import subprocess
import os

def main():
    w_range = [0.1, 0.3, 0.5, 1.0, 5.0, 10.0, 30.0]
    json_file_path = 'config.json'
    for r in range(3, 1, -1):
        for w in w_range:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            data['R'] = r
            data['w'] = w

            with open(json_file_path, 'w') as file:
                json.dump(data, file, indent=4)

            for dataset in ['imagenet', 'mirflickr']:
                for group_formation in ['random', 'labelled']:
                    subprocess.run(['./alt_run_FLINNG.sh' ,f'{dataset}' ,f'{group_formation}', f'{r}'])
                    try:
                        src_dir = f'/home/aryavishe/Nearest-Neighbour-Search-C-/alt_temp/{dataset}/{group_formation}/'
                        dest_dir = f'/home/aryavishe/Nearest-Neighbour-Search-C-/history/{dataset}/{group_formation}/'
                        os.makedirs(dest_dir, exist_ok=True)
                        for filename in ['counts.csv', 'collisions.csv', 'distances.csv', 'counts.png', 'results.csv', 'log.out']:
                            src_file = os.path.join(src_dir, filename)
                            dest_file = os.path.join(dest_dir, f'{r}_{w}_{filename}')
                            os.rename(src_file, dest_file)
                    except Exception as e:
                        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
