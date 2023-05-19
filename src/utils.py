


import argparse
import hashlib
import json
import os
import pickle


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



def create_hash_from_dict(d):
    # Create a hash name from config dict:
    m = hashlib.md5()
    d = {k: sorted(v) if isinstance(v, list) or isinstance(v, tuple) else v for k, v in d.items()}
    m.update(repr(sorted(d.items())).encode("utf-8"))
    result_name = str(int(m.hexdigest(), 16))[:12]
    return result_name


def save_results(root_dir, result, config, result_id):

    # check if results directory exists
    if not os.path.exists(f"{root_dir}/results/raw/{result_id}"):
        os.makedirs(f"{root_dir}/results/raw/{result_id}")

    # save results
    with open(f"{root_dir}/results/raw/{result_id}/results.pkl", "wb") as f:
        pickle.dump(result, f)
    
    # save config
    config_path = os.path.join(root_dir,"results", "raw", result_id, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)