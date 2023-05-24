
import os
import numpy as np
from src.utils import parse_args, create_hash_from_dict, save_results
from src.data import ClassificationDataset
from src.models import create_model
from src.evaluation import get_original_unnormalized_logprobs, get_content_free_input_probs, get_train_queries_probs, transform_probs
from tqdm import tqdm



DATASETS = ["agnews", "trec", "cb", "rte", "sst2", "dbpedia"]
dataset2short = {
    "agnews": "AGNews",
    "trec": "TREC",
    "cb": "CB",
    "rte": "RTE",
    "sst2": "SST-2",
    "dbpedia": "DBPedia"
}


def main():
    root_dir, config, use_saved_results = parse_args()
    random_state = config.pop("random_state")
    n_seeds = config.pop("n_seeds")

    n_shots_list = config.pop("n_shots") # shots
    datasets = config.pop("datasets") # dataset
    rs = np.random.RandomState(random_state) # seed

    # Instantiate model
    model_name = config.pop("model")
    print("\nInstantiating the model...", end=" ")
    model = create_model(root_dir, model=model_name)
    print("Done!\n")
    
    print(f"Running experiments for {n_seeds} seeds, {len(n_shots_list)} n_shots and {len(datasets)} datasets with the following configuration:\n")
    print("--------------------------------------------------")
    print("Model:", model_name)
    print("Evaluation split:", config["eval_split"])
    print("Content free inputs:", config["content_free_inputs"])
    print("Number of train samples:", config["num_train_samples"])
    print("--------------------------------------------------\n")
    # For each n_shots
    pbar_shots = tqdm(n_shots_list, leave=False)
    for n_shots in pbar_shots:
        pbar_shots.set_description(f"{n_shots}-shot")
        config["n_shots"] = n_shots

        # For each dataset
        pbar_datasets = tqdm(datasets, leave=False)
        for dataset in pbar_datasets:
            pbar_datasets.set_description(f"{dataset2short[dataset]} Dataset")
            config["dataset"] = dataset
        
            # For each seed
            pbar_seeds = tqdm(range(n_seeds), leave=False, total=n_seeds)
            for _ in pbar_seeds:
                seed = rs.randint(low=0,high=100000)
                config["random_state"] = seed
                pbar_seeds.set_description(f"Seed {seed}")
                result_id = create_hash_from_dict(config)
                if os.path.exists(f"{root_dir}/results/raw/{result_id}") and use_saved_results:
                    continue
                result = run(root_dir, model, **config)
                save_results(root_dir, result, config, result_id)

    print("All runs finished!")
    print("Collecting results...", end=" ")
    collect_results()
    print("Done!\n")


def run(
    root_dir,
    model,
    dataset = "agnews",
    n_shots = 4,
    eval_split = "test",
    content_free_inputs = ["N/A"],
    num_train_samples = 1000,
    random_state = None,
):
    
    # Instantiate dataset
    dataset_obj = ClassificationDataset(
        root_dir,
        dataset,
        n_shot=n_shots,
        random_state=random_state
    )

    # Obtain the plain (unnormed) log-probabilities for each label
    true_labels, original_probs = get_original_unnormalized_logprobs(
        model, 
        dataset_obj, 
        eval_split=eval_split, 
        num_samples=None, 
        batch_size=1
    )

    # Obtain the log-probabilities for each label when using content-free inputs
    content_free_input_probs = get_content_free_input_probs(
        model, 
        dataset_obj, 
        content_free_inputs, 
        batch_size=1
    )
    probs_rescaled_content_free = transform_probs(original_probs, content_free_input_probs)

    # Obtain the log-probabilities for each label when using train samples as inputs
    train_queries_probs = get_train_queries_probs(
        model,
        dataset_obj,
        num_train_samples=num_train_samples, 
        batch_size=1
    )
    probs_rescaled_train_queries = transform_probs(original_probs, train_queries_probs)

    results = {
        "true_labels": true_labels,
        "original_probs": original_probs,
        "probs_rescaled_content_free": probs_rescaled_content_free,
        "probs_rescaled_train_queries": probs_rescaled_train_queries,
        "predictions_original": np.argmax(original_probs, axis=1),
        "predictions_content_free": np.argmax(probs_rescaled_content_free, axis=1),
        "predictions_train_queries": np.argmax(probs_rescaled_train_queries, axis=1),
    }
    return results



def collect_results():
    pass

if __name__ == "__main__":
    main()