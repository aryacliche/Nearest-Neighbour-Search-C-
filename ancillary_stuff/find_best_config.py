import os
import sys
import pandas as pd #type: ignore

def main():
    """
    This function will go through the '{r}_{w}_results.csv' files in a folder and find the configuration that provides the best value of precision and recall`
    """
    print("Please note that this method only works for cases where precision = recall")
    search_dir = sys.argv[1]
    results_files = [f for f in os.listdir(search_dir) if 'results.csv' in f]
    max_precision = 0
    max_recall = 0
    for file in results_files:
        results_df = pd.read_csv(os.path.join(search_dir, file))
        local_max_precision = results_df['precision'].max()
        # it is implied in this case that precision = recall

        if local_max_precision > max_precision:
            max_precision = local_max_precision
            max_precision_file = file
            max_precision_r = file.split('_')[0]
            max_precision_w = file.split('_')[1]

    print(f"The best configuration is {max_precision_r} and {max_precision_w} with a precision of {max_precision}")

if __name__=="__main__":
    main()
