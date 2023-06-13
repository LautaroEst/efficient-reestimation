from copy import deepcopy
import json
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data import ClassificationDataset
from src.evaluation import get_content_free_input_probs
from src.models import create_model
from src.utils import parse_content_free_args, get_results_ids_from_config, save_results



def main():
    root_dir, experiment_config, experiment_name, cf_config, cf_config_name, use_saved_results = parse_content_free_args()
    results_ids = get_results_ids_from_config(root_dir, experiment_config)
    print(f"Running content-free-input calibration for {experiment_name} experiment...")

    # check if results directory exists
    if len(results_ids) == 0:
        print("No results found!")
        return
    
    for result_id in tqdm(results_ids, leave=False):

        if os.path.exists(os.path.join(root_dir,f"results/raw/{result_id}/cf_{cf_config_name}.pkl")) and use_saved_results:
            continue

        with open(f"{root_dir}/results/raw/{result_id}/config.json", "r") as f:
            config = json.load(f)
        with open(f"{root_dir}/results/raw/{result_id}/n_shots_config.json", "r") as f:
            n_shots_config = json.load(f)

        # Instantiate model
        model = create_model(root_dir, model=config["model"], max_memory=n_shots_config["max_memory"])
        result = run(
            root_dir,
            model,
            dataset_name=config["dataset"],
            n_shots=config["n_shots"],
            cf_input=cf_config["content_free_inputs"],
            batch_size=n_shots_config["batch_size"],
            random_state=config["random_state"]
        )
        save_results(root_dir, result, config, result_id, subdir="raw", results_name=f"cf_{cf_config_name}")
        with open(f"{root_dir}/results/raw/{result_id}/cf_{cf_config_name}.json", "w") as f:
            cf_config["config_name"] = cf_config_name
            json.dump(cf_config, f)
        
        del model

    print("Done!")


def run(
    root_dir,
    model,
    dataset_name = "agnews",
    n_shots = 4,
    cf_input = ["N/A"],
    batch_size = 16,
    random_state = None,
):
    
    # Instantiate dataset
    dataset = ClassificationDataset(
        root_dir,
        model.tokenizer,
        dataset_name,
        n_shot=n_shots,
        random_state=random_state
    )

    probs, queries, query_truncated, shots_truncated = get_content_free_input_probs(
        model, 
        dataset, 
        cf_input,
        batch_size=batch_size
    )

    result = {
        "probs": probs,
        "queries": queries,
        "query_truncated": query_truncated,
        "shots_truncated": shots_truncated
    }

    return result


if __name__ == "__main__":
    main()