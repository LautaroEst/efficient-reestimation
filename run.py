
import os
import numpy as np
from src.utils import parse_args, create_hash_from_dict, save_results, collect_results
from src.data import ClassificationDataset
from src.models import create_model
from src.evaluation import get_original_unnormalized_logprobs, get_content_free_input_probs, get_train_queries_probs, transform_probs
from tqdm import tqdm



def main():
    root_dir, config, use_saved_results = parse_args()
    random_state = config.pop("random_state")
    n_seeds = config.pop("n_seeds")

    datasets = config.pop("datasets") # dataset
    rs = np.random.RandomState(random_state) # seed
    config["content_free_inputs_list"] = config.pop("content_free_inputs") # cf inputs

    # Instantiate model
    model_name = config.pop("model")
    print("\nInstantiating the model...", end=" ")
    model = create_model(root_dir, model=model_name)
    print("Done!\n")
    
    print(f"Running experiments for {n_seeds} seeds, {config['n_shots']}-shots and {len(datasets)} datasets with the following configuration:\n")
    print("--------------------------------------------------")
    print("Model:", model_name)
    print("Evaluation split:", config["eval_split"])
    print("Content free inputs:", config["content_free_inputs_list"])
    print("Number of train samples:", config["num_train_samples"])
    print("--------------------------------------------------\n")
    results_ids = []

    # For each dataset
    pbar_datasets = tqdm(datasets, leave=False)
    for dataset in pbar_datasets:
        pbar_datasets.set_description(f"{ClassificationDataset.dataset2short[dataset]} Dataset")
        config["dataset_name"] = dataset
    
        # For each seed
        pbar_seeds = tqdm(range(n_seeds), leave=False, total=n_seeds)
        for _ in pbar_seeds:
            seed = rs.randint(low=0,high=100000)
            config["random_state"] = seed
            pbar_seeds.set_description(f"Seed {seed}")
            result_id = create_hash_from_dict(config)
            results_ids.append(result_id)
            if os.path.exists(f"{root_dir}/results/raw/{result_id}") and use_saved_results:
                continue
            result = run(root_dir, model, **config)
            save_results(root_dir, result, model_name, config, result_id)

    print("All runs finished!")
    print("Collecting results...", end=" ")
    collect_results(root_dir, results_ids)
    print("Done!\n")


def run(
    root_dir,
    model,
    dataset_name = "agnews",
    n_shots = 4,
    eval_split = "test",
    content_free_inputs_list = [["N/A"]],
    num_train_samples = 1000,
    batch_size = 16,
    random_state = None,
):
    
    # Instantiate dataset
    dataset = ClassificationDataset(
        root_dir,
        dataset_name,
        n_shot=n_shots,
        random_state=random_state
    )

    # Obtain the plain (unnormed) log-probabilities for each label
    true_labels, original_probs, test_queries = get_original_unnormalized_logprobs(
        model, 
        dataset, 
        eval_split=eval_split, 
        num_samples=None,
        batch_size=batch_size
    )

    # Obtain the log-probabilities for each label when using content-free inputs
    cf_probs = []
    for content_free_inputs in content_free_inputs_list:
        content_free_input_probs = get_content_free_input_probs(
            model, 
            dataset, 
            content_free_inputs, 
            batch_size=batch_size
        )
        cf_probs.append({
            "inputs": content_free_inputs,
            "probs": content_free_input_probs,
            "rescaled_probs": transform_probs(original_probs, content_free_input_probs)
        })

    # Obtain the log-probabilities for each label when using train samples as inputs
    train_queries_probs = get_train_queries_probs(
        model,
        dataset,
        num_train_samples=num_train_samples, 
        batch_size=batch_size
    )
    probs_rescaled_train_queries = transform_probs(original_probs, train_queries_probs)

    results = {
        "true_labels": true_labels,
        "original_probs": original_probs,
        "test_queries": test_queries,
        "prompt_shots_sentences": dataset.prompt_shots_sentences,
        "prompt_shots_labels": dataset.prompt_shots_labels,
        "train_queries_probs": train_queries_probs,
        "probs_rescaled_train_queries": probs_rescaled_train_queries,
        "cf_probs": cf_probs,
    }
    return results



if __name__ == "__main__":
    main()