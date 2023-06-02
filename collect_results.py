

import os
import numpy as np
from src.utils import parse_collect_results_args, create_hash_from_dict, collect_results



def main():
    root_dir, experiment_config, experiment_name, results_config = parse_collect_results_args()
    results_ids = get_results_ids(root_dir, experiment_config)
    print(f"Collecting results for {experiment_name} experiment...")
    collect_results(root_dir, experiment_name, results_ids, n_boots=results_config["n_boots"], score_name=results_config["score"])
    print("Done!")
    



def get_results_ids(root_dir, config):
    random_state = config.pop("random_state")
    n_seeds = config.pop("n_seeds")

    datasets = config.pop("datasets") # datasets
    rs = np.random.RandomState(random_state) # seed

    results_ids = []
    for dataset, n_shots_configs in datasets.items():
        config["dataset"] = dataset
        seeds = rs.randint(low=0,high=100000,size=(len(n_shots_configs),n_seeds)) # seeds
        for i, n_shots_config in enumerate(n_shots_configs):
            config["n_shots"] = n_shots_config["n_shots"]
            this_seed_results = []
            for seed in seeds[i]:
                config["random_state"] = int(seed)
                result_id = create_hash_from_dict(**config)
                if os.path.exists(f"{root_dir}/results/raw/{result_id}"):
                    this_seed_results.append(result_id)
            
            if len(this_seed_results) == n_seeds:
                for result_id in this_seed_results:
                    results_ids.append(result_id)
    return results_ids

if __name__ == "__main__":
    main()