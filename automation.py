import json
import subprocess

def main():
    json_file_path = 'config.json'
    for r in range(25, 1, -1):
        for t in range(10, 95, 5):
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            data['R'] = r
            data['t'] = t

            with open(json_file_path, 'w') as file:
                json.dump(data, file, indent=4)

            for dataset in ['imagenet', 'mirflickr']:
                for group_formation in ['random', 'labelled']:
                    subprocess.run(['./run_FLINNG.sh' ,f'{dataset}' ,f'{group_formation}', f'{r}'])


if __name__ == "__main__":
    main()
