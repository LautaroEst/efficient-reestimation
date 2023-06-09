
import json
import os
import numpy as np
from src.utils import parse_classification_args, create_hash_from_dict, save_results
from src.data import ClassificationDataset
from src.models import create_model
from src.evaluation import get_original_unnormalized_probs
from tqdm import tqdm



def main():
    root_dir, config, use_saved_results = parse_classification_args()
    random_state = config.pop("random_state")
    n_seeds = config.pop("n_seeds")

    model_name = config["model"] # model
    datasets = config.pop("datasets") # datasets
    rs = np.random.RandomState(random_state) # seed

    print(f"Running model {model_name} for {n_seeds} seeds and {len(datasets)} datasets on the {config['eval_split']} set.\n")

    results_ids = []

    # For each dataset
    pbar_datasets = tqdm(datasets.items(), leave=False)
    for dataset, n_shots_configs in pbar_datasets:
        pbar_datasets.set_description(f"{ClassificationDataset.dataset2short[dataset]} Dataset")
        config["dataset"] = dataset
        seeds = rs.randint(low=0,high=100000,size=(len(n_shots_configs),n_seeds)) # seeds

        pbar_nshots = tqdm(n_shots_configs, leave=False)
        for i, n_shots_config in enumerate(pbar_nshots):
            config["n_shots"] = n_shots_config["n_shots"]
            pbar_nshots.set_description(f"{n_shots_config['n_shots']}-shot")
       
            # For each seed
            pbar_seeds = tqdm(seeds[i], leave=False, total=n_seeds)
            for seed in pbar_seeds:
                config["random_state"] = int(seed)
                pbar_seeds.set_description(f"Seed {seed}")
                result_id = create_hash_from_dict(**config)
                results_ids.append(result_id)
                if os.path.exists(os.path.join(root_dir,f"results/raw/{result_id}")) and use_saved_results:
                    continue

                # Instantiate model
                model = create_model(root_dir, model=model_name, max_memory=n_shots_config["max_memory"])
                result = run(
                    root_dir=root_dir, 
                    model=model, 
                    dataset_name=dataset,
                    n_shots=n_shots_config["n_shots"],
                    eval_split=config["eval_split"],
                    batch_size=n_shots_config["batch_size"],
                    random_state=seed,
                )
                save_results(root_dir, result, config, result_id, subdir="raw", results_name="main")

                with open(os.path.join(root_dir,f"results/raw/{result_id}/n_shots_config.json"), "w") as f:
                    json.dump(n_shots_config, f)

                del model

    print("All runs finished!")


def run(
    root_dir,
    model,
    dataset_name = "agnews",
    n_shots = 4,
    eval_split = "test",
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

    # Obtain the plain (unnormed) probabilities for each label
    true_labels, original_probs, test_queries, queries_truncated, shots_truncated = get_original_unnormalized_probs(
        model, 
        dataset, 
        eval_split=eval_split, 
        num_samples=None,
        batch_size=batch_size
    )
    
    results = {
        "true_labels": true_labels,
        "original_probs": original_probs,
        "test_queries": test_queries,
        "queries_truncated": queries_truncated,
        "shots_truncated": shots_truncated,
        "prompt_shots_sentences": dataset.prompt_shots_sentences,
        "prompt_shots_labels": dataset.prompt_shots_labels,
    }

    return results



if __name__ == "__main__":
    main()