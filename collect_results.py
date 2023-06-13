

from copy import deepcopy
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils import compute_score, get_results_ids_from_config, parse_collect_results_args


def main():
    root_dir, experiment_config, experiment_name, results_config = parse_collect_results_args()
    results_ids = get_results_ids_from_config(root_dir, experiment_config)
    print(f"Collecting results for {experiment_name} experiment...")
    print(f"Found {len(results_ids)} results directories.")
    run(root_dir, results_ids, experiment_name, **results_config)


def run(root_dir, results_ids, experiment_name, score="accuracy"):

    score_name = score
    all_results = []
    pbar = tqdm(results_ids, leave=False)
    for result_id in pbar:
        with open(os.path.join(root_dir,f"results/raw/{result_id}/config.json"), "r") as f:
            config = json.load(f)
        for filename in glob.glob(os.path.join(root_dir, "results/calibrated", result_id, "*.pkl")):
            with open(filename, "rb") as f:
                result = pickle.load(f)
            for k, v in result.items():
                if k == "test_labels":
                    continue
                this_result = deepcopy(config)
                this_result["boots_id"] = int(filename.split("/")[-1].split(".")[0])
                score = compute_score(np.argmax(v, axis=1), result["test_labels"], score_name)
                this_result["output_prob_type"] = k
                this_result[f"score:{score_name}"] = score
                all_results.append(this_result)
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(root_dir, "results", f"{experiment_name}_results.csv"), index=False)
    print("Done!")
            


if __name__ == "__main__":
    main()