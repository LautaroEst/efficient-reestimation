
import json
import os
import pickle
import numpy as np
from src.utils import parse_classification_args, create_hash_from_dict
from src.data import ClassificationDataset
from src.models import create_model
from src.inference import get_content_free_input_probs, get_original_unnormalized_probs
from tqdm import tqdm



def main():
    root_dir, config, use_saved_results = parse_classification_args()
    random_state = config.pop("random_state")
    n_seeds = config.pop("n_seeds")

    model_name = config["model"] # model
    datasets = config.pop("datasets") # datasets
    rs = np.random.RandomState(random_state) # seed
    content_free_inputs = config.pop("content_free_inputs", []) # Se hace acá de manera que el hash no dependa de los cf_inputs

    print(f"\nRunning model {model_name} for {n_seeds} seeds on " + ", ".join([ClassificationDataset.dataset2short[dataset] for dataset in datasets.keys()]) + " datasets.\n")

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

            # Instantiate model
            model = create_model(root_dir, model=model_name, max_memory=n_shots_config["max_memory"])
       
            # For each seed
            pbar_seeds = tqdm(seeds[i], leave=False, total=n_seeds)
            for seed in pbar_seeds:
                config["random_state"] = int(seed)
                pbar_seeds.set_description(f"Seed {seed}")
                result_id = create_hash_from_dict(**config)

                # Instantiate dataset
                dataset_obj = ClassificationDataset(
                    root_dir,
                    model.tokenizer,
                    dataset,
                    n_shot=n_shots_config["n_shots"],
                    random_state=seed
                )

                test_results, train_results, cf_results, prompt_shots = run(
                    model=model, 
                    dataset=dataset_obj,
                    batch_size=n_shots_config["batch_size"],
                    results_dir=os.path.join(root_dir,f"results/train_test/{result_id}"),
                    content_free_inputs=content_free_inputs
                )

                # Save results
                if not os.path.exists(os.path.join(root_dir,f"results/train_test/{result_id}")):
                    os.makedirs(os.path.join(root_dir,f"results/train_test/{result_id}"))
                with open(os.path.join(root_dir,f"results/train_test/{result_id}/train.pkl"), "wb") as f:
                    pickle.dump(train_results, f)
                with open(os.path.join(root_dir,f"results/train_test/{result_id}/test.pkl"), "wb") as f:
                    pickle.dump(test_results, f)
                with open(os.path.join(root_dir,f"results/train_test/{result_id}/config.json"), "w") as f:
                    json.dump(config, f, indent=4, sort_keys=True)
                for cf_name, cf_res in cf_results.items():
                    with open(os.path.join(root_dir,f"results/train_test/{result_id}/{cf_name}.pkl"), "wb") as f:
                        pickle.dump(cf_res, f)
                with open(os.path.join(root_dir,f"results/train_test/{result_id}/prompt_shots.json"), "w") as f:
                    json.dump(prompt_shots, f)
                with open(os.path.join(root_dir,f"results/train_test/{result_id}/n_shots_config.json"), "w") as f:
                    json.dump(n_shots_config, f)

            del model

    print("All runs finished!")


def run(
    model, 
    dataset,
    batch_size = 16,
    results_dir = None,
    use_saved_results = True,
    content_free_inputs = {"mask": ["[MASK]"]}
):

    # Obtain the plain (unnormed) probabilities for each label for the test split
    if os.path.exists(os.path.join(results_dir,"test.pkl")) and use_saved_results:
        with open(os.path.join(results_dir,"test.pkl"), "rb") as f:
            test_results = pickle.load(f)
    else:
        test_labels, test_probs, test_queries, test_queries_truncated, test_shots_truncated = get_original_unnormalized_probs(
            model, 
            dataset, 
            eval_split="test", 
            num_samples=None,
            batch_size=batch_size
        )
        test_results = {
            "test_labels": test_labels,
            "test_probs": test_probs,
            "test_queries": test_queries,
            "test_queries_truncated": test_queries_truncated,
            "test_shots_truncated": test_shots_truncated
        }

    # Obtain the plain (unnormed) probabilities for each label for the train split
    if os.path.exists(os.path.join(results_dir,"train.pkl")) and use_saved_results:
        with open(os.path.join(results_dir,"train.pkl"), "rb") as f:
            train_results = pickle.load(f)
    else:
        train_labels, train_probs, train_queries, train_queries_truncated, train_shots_truncated = get_original_unnormalized_probs(
            model, 
            dataset, 
            eval_split="train", 
            num_samples=600,
            batch_size=batch_size
        )
        train_results = {
            "train_labels": train_labels,
            "train_probs": train_probs,
            "train_queries": train_queries,
            "train_queries_truncated": train_queries_truncated,
            "train_shots_truncated": train_shots_truncated
        }
    
    # Obtain the probabilities for each label when using content-free inputs
    cf_results = {}
    for cf_name, cf_in in content_free_inputs.items():
        # Obtain the plain (unnormed) probabilities for each label for the train split 
        if os.path.exists(os.path.join(results_dir,f"{cf_name}.pkl")) and use_saved_results:
            with open(os.path.join(results_dir,f"{cf_name}.pkl"), "rb") as f:
                cf_results[cf_name] = pickle.load(f)
        else:
            cf_probs, cf_queries, cf_queries_truncated, cf_shots_truncated = get_content_free_input_probs(
                model, 
                dataset, 
                cf_in,
                batch_size=batch_size
            ) # Returns a tuple
            cf_results[cf_name] = {
                "inputs": cf_in,
                "probs": cf_probs,
                "queries": cf_queries,
                "queries_truncated": cf_queries_truncated,
                "shots_truncated": cf_shots_truncated
            }

    prompt_shots = {
        "prompt_shots_sentences": dataset.prompt_shots_sentences,
        "prompt_shots_labels": dataset.prompt_shots_labels,
    }

    return test_results, train_results, cf_results, prompt_shots



if __name__ == "__main__":
    main()