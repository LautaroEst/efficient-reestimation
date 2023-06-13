
from tqdm import tqdm
import json
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd

from src.utils import parse_collect_results_args, get_results_ids_from_config, compute_score



def main():
    root_dir, experiment_config, experiment_name, results_config = parse_collect_results_args()
    results_ids = get_results_ids_from_config(root_dir, experiment_config)
    print(f"Collecting results for {experiment_name} experiment...")
    collect_results(root_dir, experiment_name, results_ids, n_boots=results_config["n_boots"], score_name=results_config["score"])
    print("Done!")
    


def collect_results(root_dir, experiment_name, results_ids, n_boots=None, score_name='accuracy'):

    # check if results directory exists
    if len(results_ids) == 0:
        print("No results found!")
        return
    
    results = []
    raw_results = results_ids
    for raw_result in tqdm(raw_results, leave=False):
        with open(f"{root_dir}/results/raw/{raw_result}/config.json", "r") as f:
            config = json.load(f)
        with open(f"{root_dir}/results/raw/{raw_result}/results.pkl", "rb") as f:
            result = pickle.load(f)

        true_labels = result["true_labels"]
        rs = np.random.RandomState(config["random_state"])
        iters = range(n_boots) if n_boots is not None else range(1)
        bootstrap = n_boots is not None
        for _ in iters:
            for key in ["original_probs", "probs_rescaled_train_queries"]:
                this_result = {k: v for k, v in deepcopy(config).items() if k != "content_free_inputs"}
                predictions = np.argmax(result[key], axis=1)
                score = compute_score(true_labels, predictions, bootstrap=bootstrap, score=score_name, random_state=rs)
                this_result["output_prob_type"] = key
                this_result[f"score:{score_name}"] = score
                results.append(this_result)
            for cf_probs in result["cf_probs"]:
                this_result = {k: v for k, v in deepcopy(config).items() if k != "content_free_inputs"}
                predictions = np.argmax(cf_probs["rescaled_probs"], axis=1)
                score = compute_score(true_labels, predictions, bootstrap=bootstrap, score=score_name, random_state=rs)
                this_result["output_prob_type"] = f"content_free_{cf_probs['inputs']}"
                this_result[f"score:{score_name}"] = score
                results.append(this_result)
    
    # save results
    df_results = pd.DataFrame.from_records(results)
    df_results.to_csv(f"{root_dir}/results/{experiment_name}_results.csv", index=False)


if __name__ == "__main__":
    main()