
from copy import deepcopy
import json
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils import parse_calibration_args, get_results_ids_from_config, save_results, compute_score
from src.calibration import calibrate_and_evaluate_psr



def main():
    root_dir, experiment_config, experiment_name, calibration_config, use_saved_results = parse_calibration_args()
    results_ids = get_results_ids_from_config(root_dir, experiment_config)
    print(f"Running calibration for {experiment_name} experiment...")
    run_calibration(root_dir, experiment_name, results_ids, use_saved_results, **calibration_config)
    print("Done!")
    

def run_calibration(root_dir, experiment_name, results_ids, use_saved_results, **calibration_config):

    n_boots = calibration_config.pop("bootstrap")
    score_name = calibration_config.pop("score")

    # check if results directory exists
    if len(results_ids) == 0:
        print("No results found!")
        return
    
    results = []
    raw_results = results_ids
    for raw_result in tqdm(raw_results, leave=False):
        with open(f"{root_dir}/results/raw/{raw_result}/config.json", "r") as f:
            config = json.load(f)
        with open(f"{root_dir}/results/raw/{raw_result}/results.pkl", "rb") as f:
            result = pickle.load(f)

        scores = result["original_probs"]
        scores = scores / scores.sum(axis=1, keepdims=True)
        scores = np.log(scores)
        targets = result["true_labels"]
        
        rs = np.random.RandomState(config["random_state"])
        iters = range(n_boots) if n_boots is not None else range(1)
        calibration_config["bootstrap"] = n_boots is not None
        
        for i in iters:
            this_result_config = {**deepcopy(config), **deepcopy(calibration_config)}
            seed = rs.randint(0, 100000)
            if os.path.exists(f"{root_dir}/results/calibrated/{raw_result}/{i:06}.pkl") and use_saved_results:
                with open(f"{root_dir}/results/calibrated/{raw_result}/{i:06}.pkl", "rb") as f:
                    calibration_results = pickle.load(f)
            else:
                calibration_results = calibrate_and_evaluate_psr(scores, targets, random_state=seed, **calibration_config)
                save_results(root_dir, calibration_results, this_result_config, raw_result, subdir="calibrated", results_name=i, config_name="config")
            
            predictions_cal = np.argmax(calibration_results["scores_cal"], axis=1)
            score = compute_score(calibration_results["targets"], predictions_cal, bootstrap=False, score=score_name, random_state=None)
            this_result_config[f"score:{score_name}"] = score
            this_result_config["random_state"] = seed
            for key in ["overall_perf", "overall_perf_after_cal", "cal_loss", "rel_cal_loss"]:
                this_result_config[key] = calibration_results[key]
            results.append(this_result_config)

    # save results
    df_results = pd.DataFrame.from_records(results)
    df_results.to_csv(f"{root_dir}/results/{experiment_name}_calibration_results.csv", index=False)

if __name__ == "__main__":
    main()