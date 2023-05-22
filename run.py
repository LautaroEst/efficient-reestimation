
import os
import numpy as np
from src.utils import parse_args, create_hash_from_dict, save_results
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
    
    print(f"\nRunning experiments for {n_seeds} seeds, {len(n_shots_list)} n_shots and {len(datasets)} datasets with the following configuration:")
    print("--------------------------------------------------")
    print("Model:", config["model"])
    print("Evaluation split:", config["eval_split"])
    print("Content free inputs:", config["content_free_inputs"])
    print("Number of train samples:", config["num_train_samples"])
    print("--------------------------------------------------")
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
                result = run(root_dir, **config)
                save_results(root_dir, result, config, result_id)

    print("\nAll runs finished!")
    print("Collecting results...", end=" ")
    collect_results()
    print("Done!")

def run(
    root_dir,
    dataset = "agnews",
    n_shots = 4,
    eval_split = "test",
    model = "gpt2",
    content_free_inputs = ["N/A"],
    num_train_samples = 1000,
    random_state = None,
):
    import time
    time.sleep(0.1)
    return {"esto": "es un test", "y esto": "tambien"}


def collect_results():
    pass

if __name__ == "__main__":
    main()