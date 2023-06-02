


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


def parse_classification_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="./")
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--use_saved_results', action='store_true', default=False)
    args = parser.parse_args()
    root_dir = args.root_dir
    config_filename = args.config + ".json"
    with open(os.path.join(root_dir,"configs/classification",config_filename)) as config_file:
        config = json.load(config_file)
    use_saved_results = args.use_saved_results

    return root_dir, config, use_saved_results

def parse_collect_results_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="./")
    parser.add_argument('--experiment', type=str, default='experiment')
    parser.add_argument('--config', type=str, default='config')
    args = parser.parse_args()
    root_dir = args.root_dir
    experiment_filename = args.experiment + ".json"
    config_filename = args.config + ".json"
    with open(os.path.join(root_dir,"configs/classification",experiment_filename)) as experiment_file:
        experiment_config = json.load(experiment_file)

    with open(os.path.join(root_dir,"configs/collect_results",config_filename)) as results_file:
        results_config = json.load(results_file)

    return root_dir, experiment_config, args.experiment, results_config



def create_hash_from_dict(**d):
    # Create a hash name from config dict:
    m = hashlib.md5()
    d = {k: sorted(v) if isinstance(v, list) or isinstance(v, tuple) else v for k, v in d.items()}
    m.update(repr(sorted(d.items())).encode("utf-8"))
    result_name = str(int(m.hexdigest(), 16))
    return result_name


def save_results(root_dir, result, config, result_id):

    # check if results directory exists
    if not os.path.exists(f"{root_dir}/results/raw/{result_id}"):
        os.makedirs(f"{root_dir}/results/raw/{result_id}")

    # save results
    with open(f"{root_dir}/results/raw/{result_id}/results.pkl", "wb") as f:
        pickle.dump(result, f)
    
    # save config
    config_to_be_saved = deepcopy(config)
    config_path = os.path.join(root_dir,"results", "raw", result_id, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_to_be_saved, f, indent=4, sort_keys=True)

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
    

def compute_score(true_labels, predictions, bootstrap=True, score="accuracy", random_state=0):
    
    if bootstrap:
        df_boots = pd.DataFrame({"true_labels": true_labels, "predictions": predictions}).sample(
            frac=1, replace=True, random_state=random_state
        )
    else:
        df_boots = pd.DataFrame({"true_labels": true_labels, "predictions": predictions})
    
    if score == "accuracy":
        return accuracy_score(df_boots["true_labels"], df_boots["predictions"])
    else:
        raise ValueError(f"Score {score} not supported!")

