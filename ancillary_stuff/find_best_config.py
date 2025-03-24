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
    max_avg_recall = 0
    max_avg_precision = 0
    for file in results_files:
        results_df = pd.read_csv(os.path.join(search_dir, file))
        local_max_precision = results_df['precision'].max()
        local_avg_precision = results_df['precision'].mean()
        
        try:
            naive_time = results_df['naive_running_time'][3]
            flinng_time = results_df['flinng_running_time'][3]
        except:
            naive_time = None
            flinng_time = None

        if local_max_precision > max_precision:
            max_precision = local_max_precision
            max_precision_file = file
            max_precision_r = file.split('_')[0]
            max_precision_w = file.split('_')[1]
            max_case_naive_time = naive_time
            max_case_FLINNG_time = flinng_time

        if local_avg_precision > max_avg_precision:
            max_avg_precision = local_avg_precision
            max_avg_precision_file = file
            max_avg_precision_r = file.split('_')[0]
            max_avg_precision_w = file.split('_')[1]
            max_avg_case_naive_time = naive_time
            max_avg_case_FLINNG_time = flinng_time

    print(f"The maximum precision/recall was seen with r = {max_precision_r} and w = {max_precision_w} => {max_precision}")
    if max_case_naive_time != None:
        print(f"For that case, exhaustive search took {max_case_naive_time} ms and FLINNG took {max_case_FLINNG_time} ms")
    
    print()
    print(f"The maximum average precision was seen with r = {max_avg_precision_r} and w = {max_avg_precision_w} => {max_avg_precision}")
    if max_avg_case_FLINNG_time != None:
        print(f"For that case, exhaustive search took {max_avg_case_naive_time} ms and FLINNG took {max_avg_case_FLINNG_time} ms")

if __name__=="__main__":
    main()
