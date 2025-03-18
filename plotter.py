import pandas as pd #type: ignore
import matplotlib.pyplot as plt #type: ignore
import json
import sys

temp_dir = sys.argv[1]
R = sys.argv[2]

graph_dir = 'graphs'

# Load CSV file
csv_file = f'{temp_dir}/results_{R}.csv'
data = pd.read_csv(csv_file)

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
plt.show()