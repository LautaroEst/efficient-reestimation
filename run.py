

import argparse
import json
import os
import pickle
import hashlib

import numpy as np



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



def main():
    root_dir, config, use_saved_results = parse_args()
    random_state = config.pop("random_state")
    n_seeds = config.pop("n_seeds")

    rs = np.random.RandomState(random_state)
    datasets = config.pop("datasets")
    n_shots_list = config.pop("n_shots")
    for _ in range(n_seeds):
        config["random_state"] = rs.randint(low=0,high=100000)
        for dataset in datasets:
            config["dataset"] = dataset
            for n_shots in n_shots_list:
                config["n_shots"] = n_shots
                result_id = create_hash_from_dict(config)
                if os.path.exists(f"{root_dir}/results/{result_id}") and use_saved_results:
                    print(f"Skipping {result_id}")
                    continue
                result = run(root_dir, **config)
                save_results(root_dir, result, config, result_id)

def run(
    root_dir,
    dataset = "agnews",
    n_shots = 4,
    eval_split = "test",
    model = "gpt2",
    api_num_log_probs = None,
    batch_size = 32,
    approx = False,
    content_free_inputs = ["N/A"],
    num_train_samples = 1000,
    random_state = None,
):
    return {"esto": "es un test", "y esto": "tambien"}


def create_hash_from_dict(d):
    # Create a hash name from config dict:
    m = hashlib.md5()
    m.update(repr(sorted(d.items())).encode("utf-8"))
    result_name = str(int(m.hexdigest(), 16))[:12]
    return result_name


def save_results(root_dir, result, config, result_id):

    # check if results directory exists
    if not os.path.exists(f"{root_dir}/results/{result_id}"):
        os.makedirs(f"{root_dir}/results/{result_id}")

    # save results
    with open(f"{root_dir}/results/{result_id}/results.pkl", "wb") as f:
        pickle.dump(result, f)
    
    # save config
    config_path = os.path.join(root_dir,"results", result_id, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()