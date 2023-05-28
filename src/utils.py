


import argparse
from copy import deepcopy
import hashlib
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="./")
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--use_saved_results', action='store_true', default=False)
    args = parser.parse_args()
    root_dir = args.root_dir
    config_filename = args.config + ".json"
    with open(os.path.join(root_dir,"configs",config_filename)) as config_file:
        config = json.load(config_file)
    use_saved_results = args.use_saved_results

    return root_dir, config, use_saved_results



def create_hash_from_dict(d):
    # Create a hash name from config dict:
    m = hashlib.md5()
    d = {k: sorted(v) if isinstance(v, list) or isinstance(v, tuple) else v for k, v in d.items()}
    m.update(repr(sorted(d.items())).encode("utf-8"))
    result_name = str(int(m.hexdigest(), 16))[:12]
    return result_name


def save_results(root_dir, result, model_name, config, result_id):

    # check if results directory exists
    if not os.path.exists(f"{root_dir}/results/raw/{result_id}"):
        os.makedirs(f"{root_dir}/results/raw/{result_id}")

    # save results
    with open(f"{root_dir}/results/raw/{result_id}/results.pkl", "wb") as f:
        pickle.dump(result, f)
    
    # save config
    config_to_be_saved = deepcopy(config)
    config_to_be_saved["model"] = model_name
    config_to_be_saved["dataset"] = config_to_be_saved.pop("dataset_name")
    config_path = os.path.join(root_dir,"results", "raw", result_id, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_to_be_saved, f, indent=4, sort_keys=True)

def collect_results(root_dir, results_ids):

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
        for _ in range(100): # bootstraping in test
            for key in ["original_probs", "probs_rescaled_train_queries"]:
                this_result = {k: v for k, v in deepcopy(config).items() if k != "content_free_inputs"}
                predictions = np.argmax(result[key], axis=1)
                acc = compute_accuracy(true_labels, predictions, random_state=rs)
                this_result["output_prob_type"] = key
                this_result["acc"] = acc
                results.append(this_result)
            for cf_probs in result["cf_probs"]:
                this_result = {k: v for k, v in deepcopy(config).items() if k != "content_free_inputs"}
                predictions = np.argmax(cf_probs["rescaled_probs"], axis=1)
                acc = compute_accuracy(true_labels, predictions, random_state=rs)
                this_result["output_prob_type"] = f"content_free_{cf_probs['inputs']}"
                this_result["acc"] = acc
                results.append(this_result)
    
    # save results
    df_results = pd.DataFrame.from_records(results)
    df_results.to_csv(f"{root_dir}/results/acc_results.csv", index=False)
    df_results.groupby(["model", "dataset", "eval_split", "n_shots", "output_prob_type"]).agg({
        "acc": ["mean", "std"], 
    }).to_html(f"{root_dir}/results/results.html")
    

def compute_accuracy(true_labels, predictions, random_state=0):
    df_boots = pd.DataFrame({"true_labels": true_labels, "predictions": predictions}).sample(
        frac=1, replace=True, random_state=random_state
    )
    return accuracy_score(df_boots["true_labels"], df_boots["predictions"])

