from copy import deepcopy
import json
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data import ClassificationDataset
from src.evaluation import get_train_queries_probs
from src.models import create_model
from src.utils import parse_train_queries_args, get_results_ids_from_config, save_results



def main():
    root_dir, experiment_config, experiment_name, tq_config, tq_config_name, use_saved_results = parse_train_queries_args()
    results_ids = get_results_ids_from_config(root_dir, experiment_config)
    print(f"Running queries calibration for {experiment_name} experiment...")

    # check if results directory exists
    if len(results_ids) == 0:
        print("No results found!")
        return
    
    for result_id in tqdm(results_ids, leave=False):

        # if all([os.path.exists(os.path.join(root_dir,f"results/raw/{result_id}/train_queries_{n:04}.pkl")) for n in tq_config["num_train_samples"]]) and use_saved_results:
        #     continue

        with open(f"{root_dir}/results/raw/{result_id}/config.json", "r") as f:
            config = json.load(f)
        with open(f"{root_dir}/results/raw/{result_id}/n_shots_config.json", "r") as f:
            n_shots_config = json.load(f)

        # Instantiate model
        model = create_model(root_dir, model=config["model"], max_memory=n_shots_config["max_memory"])

        # Instantiate dataset
        dataset = ClassificationDataset(
            root_dir,
            model.tokenizer,
            config["dataset"],
            n_shot=config["n_shots"],
            random_state=config["random_state"]
        )

        results = {}
        for n in tq_config["num_train_samples"]:
            if os.path.exists(os.path.join(root_dir,f"results/raw/{result_id}/train_queries_{n:04}.pkl")):
                for batch in dataset.random_batch_loader_from_split(split="train", num_samples=n, batch_size=n_shots_config["batch_size"]):
                    pass
                with open(os.path.join(root_dir,f"results/raw/{result_id}/train_queries_{n:04}.pkl"), "rb") as f:
                    results[n] = pickle.load(f)
            else:
                results[n] = run(
                    dataset,
                    model,
                    num_train_samples=n,
                    batch_size=n_shots_config["batch_size"],
                )

        for n in tq_config["num_train_samples"]:
            save_results(root_dir, results[n], config, result_id, subdir="raw", results_name=f"train_queries_{n:04}")

        with open(f"{root_dir}/results/raw/{result_id}/train_queries_config.json", "w") as f:
            tq_config["config_name"] = tq_config_name
            json.dump(tq_config, f)
            
        del model

    print("Done!")


def run(
    dataset,
    model,
    num_train_samples = 100,
    batch_size = 16,
):

    labels, probs, queries, query_truncated, shots_truncated = get_train_queries_probs(
        model, 
        dataset, 
        num_train_samples=num_train_samples, 
        batch_size=batch_size
    )

    result = {
        "probs": probs,
        "labels": labels,
        "queries": queries,
        "query_truncated": query_truncated,
        "shots_truncated": shots_truncated
    }

    return result


if __name__ == "__main__":
    main()