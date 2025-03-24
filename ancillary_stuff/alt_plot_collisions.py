import matplotlib.pyplot as plt #type: ignore
import numpy as np
import json
import os
import subprocess
import sys


w_range = [0.05, 0.1, 0.3, 0.5, 1.0, 5.0, 10.0, 30.0, 50.0]
R = 20

for dataset in ['imagenet', 'mirflickr']:
    for group_formation in ['random', 'labelled']:
        for w in w_range:
            # Load the data
            if os.path.exists('/home/aryavishe/Nearest-Neighbour-Search-C-/config.json'):
                with open('/home/aryavishe/Nearest-Neighbour-Search-C-/config.json', 'r+') as f:
                    config = json.load(f)
            else:
                config =    {
                                "R": 11,
                                "t": 50,
                                "m": 100,
                                "L": 12,
                                "w": 50.0
                            }
                
            config['R'] = R
            config['w'] = w

            with open('/home/aryavishe/Nearest-Neighbour-Search-C-/config.json', 'w') as f:
                json.dump(config, f)

            subprocess.run(['./alt_plot_collisions.sh' ,f'{dataset}' ,f'{group_formation}', f'{R}'])

            try:
                src_dir = f'/home/aryavishe/Nearest-Neighbour-Search-C-/alt_temp/{dataset}/{group_formation}/'
                dest_dir = f'/home/aryavishe/Nearest-Neighbour-Search-C-/history/{dataset}/{group_formation}/'
                os.makedirs(dest_dir, exist_ok=True)
                for filename in ['counts.csv', 'collisions.csv', 'distances.csv', 'counts.png', 'results.csv', 'log.out']:
                    src_file = os.path.join(src_dir, filename)
                    dest_file = os.path.join(dest_dir, f'{R}_{w}_{filename}')
                    os.rename(src_file, dest_file)
            except Exception as e:
                print(f"An error occurred: {e}")

            # Now we need to also plot the historgram of collisions
            if not os.path.exists(f"/home/aryavishe/Nearest-Neighbour-Search-C-/ancillary_stuff/collisions_trends"):
                os.makedirs(f"/home/aryavishe/Nearest-Neighbour-Search-C-/ancillary_stuff/collisions_trends")

            fig_name = "collisions_{R}_{w}.png"

            