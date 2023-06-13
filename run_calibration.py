
from copy import deepcopy
import glob
import json
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils import parse_calibration_args, get_results_ids_from_config, save_results
from src.calibration import calibration_train_on_heldout, calibration_with_crossval, calmethod_name2fn
from src.evaluation import transform_probs



def main():
    root_dir, experiment_config, experiment_name, calibration_config, cf_config_name, use_saved_results = parse_calibration_args()
    results_ids = get_results_ids_from_config(root_dir, experiment_config)
    print(f"Running calibration for {experiment_name} experiment...")

    n_boots = calibration_config["bootstrap"]
    bootstrap = n_boots is not None
    metric = calibration_config["metric"]
    calmethod = calibration_config["calmethod"]
    use_bias = calibration_config["use_bias"]
    deploy_priors = calibration_config["deploy_priors"]

    # check if results directory exists
    if len(results_ids) == 0:
        print("No results found!")
        return
    
    for result_id in tqdm(results_ids, leave=False):
        with open(f"{root_dir}/results/raw/{result_id}/config.json", "r") as f:
            main_config = json.load(f)
        with open(f"{root_dir}/results/raw/{result_id}/main.pkl", "rb") as f:
            main_result = pickle.load(f)
        
        # Read the test probs
        test_probs = main_result["original_probs"]
        test_labels = main_result["true_labels"]

        # Read the content-free probs
        cf_probs = []
        for cf_filename in glob.glob(f"{root_dir}/results/raw/{result_id}/cf_*.pkl"):
            with open(cf_filename, "rb") as f:
                cf_result = pickle.load(f)
            with open(cf_filename.replace(".pkl",".json"), "r") as f:
                cf_config = json.load(f)
            cf_probs.append({
                "probs": cf_result["probs"],
                "input": cf_config["content_free_inputs"]
            })

        # Read the train queries probs
        with open(f"{root_dir}/results/raw/{result_id}/train_queries_config.json", "r") as f:
            train_query_config = json.load(f)
        train_queries_probs = {}
        for n in train_query_config["num_train_samples"]:
            with open(f"{root_dir}/results/raw/{result_id}/train_queries_{n:04}.pkl", "rb") as f:
                train_queries_probs[n] = pickle.load(f)

        rs = np.random.RandomState(main_config["random_state"])
        iters = range(n_boots) if n_boots is not None else range(1)
        
        for i in iters:
            seed = rs.randint(0, 100000)
            if os.path.exists(f"{root_dir}/results/calibrated/{result_id}/{i:06}.pkl") and use_saved_results:
                with open(f"{root_dir}/results/calibrated/{result_id}/{i:06}.pkl", "rb") as f:
                    calibration_results = pickle.load(f)
            else:
                calibration_results = run(test_probs, test_labels, cf_probs, train_queries_probs, metric=metric, calmethod=calmethod, use_bias=use_bias, deploy_priors=deploy_priors, bootstrap=bootstrap, random_state=seed)
                save_results(root_dir, calibration_results, main_config, result_id, subdir="calibrated", results_name=i, config_name="config")
                with open(f"{root_dir}/results/calibrated/{result_id}/calibration_config.json", "w") as f:
                    calibration_config["config_name"] = cf_config_name
                    json.dump(calibration_config, f)
    print("Done!")

    
def run(test_probs, test_labels, cf_probs, train_queries_probs, metric="LogLoss", calmethod='AffineCalLogLoss', use_bias=True, deploy_priors=None, bootstrap=True, random_state=None):
    
    if bootstrap:
        rs = np.random.RandomState(random_state)
        boots_idx = rs.choice(len(test_labels), size=len(test_labels), replace=True)
        test_probs = test_probs[boots_idx]
        test_labels = test_labels[boots_idx]
    else:
        boots_idx = None

    calmethod_fn = calmethod_name2fn[calmethod]

    test_scores = test_probs.copy()
    test_scores = test_scores / test_scores.sum(axis=1, keepdims=True)
    test_scores = np.log(test_scores)

    # Calibrate cheating
    calibrated_test_probs_train_on_test = calibration_train_on_heldout(test_scores, test_scores, test_labels, use_bias=use_bias, priors=deploy_priors, calmethod=calmethod_fn, return_model=False)
    calibrated_test_probs_train_on_test = np.exp(calibrated_test_probs_train_on_test)
    # Calibrate with xval and condition_ids
    calibrated_test_probs_cross_val_condids = calibration_with_crossval(test_scores, test_labels, condition_ids=boots_idx, stratified=False, use_bias=use_bias, priors=deploy_priors, calmethod=calmethod_fn)
    calibrated_test_probs_cross_val_condids = np.exp(calibrated_test_probs_cross_val_condids)
    # Calibrate with xval
    calibrated_test_probs_cross_val = calibration_with_crossval(test_scores, test_labels, condition_ids=None, stratified=False, use_bias=use_bias, priors=deploy_priors, calmethod=calmethod_fn)
    calibrated_test_probs_cross_val = np.exp(calibrated_test_probs_cross_val)
    # Calibrate with train queries
    calibrated_test_probs_train_queries = {}
    for n, train_queries in train_queries_probs.items():
        train_scores = train_queries["probs"].copy()
        train_scores = train_scores / train_scores.sum(axis=1, keepdims=True)
        train_scores = np.log(train_scores)
        calibrated_test_probs_train_queries[n] = calibration_train_on_heldout(test_scores, train_scores, train_queries["labels"], use_bias=use_bias, priors=deploy_priors, calmethod=calmethod_fn, return_model=False)
        calibrated_test_probs_train_queries[n] = np.exp(calibrated_test_probs_train_queries[n])

    # Reestimate with content-free probs
    reestimated_cf_test_probs = []
    for cf_prob in cf_probs:
        d = {}
        d["input"] = cf_prob["input"]
        probs = cf_prob["probs"]
        mean_probs = probs.mean(axis=0)
        mean_probs_norm = mean_probs / mean_probs.sum() # Normalize
        d["probs"] = transform_probs(test_probs, mean_probs_norm)
        reestimated_cf_test_probs.append(d)

    # Reestimate with train queries probs
    reestimated_train_queries_test_probs = {}
    for n, train_queries in train_queries_probs.items():
        probs = train_queries["probs"]
        mean_probs = probs.mean(axis=0)
        mean_probs_norm = mean_probs / mean_probs.sum()
        reestimated_train_queries_test_probs[n] = transform_probs(test_probs, mean_probs_norm)
    
    result = {
        "test_probs": test_probs,
        "test_labels": test_labels,
        "calibrated_test_probs_train_on_test": calibrated_test_probs_train_on_test,
        "calibrated_test_probs_cross_val_condids": calibrated_test_probs_cross_val_condids,
        "calibrated_test_probs_cross_val": calibrated_test_probs_cross_val,
        **{f"calibrated_test_probs_train_queries_{n:04}": probs for n, probs in calibrated_test_probs_train_queries.items()},
        **{f"reestimated_test_probs_cf_{probs['input']}": probs["probs"] for probs in reestimated_cf_test_probs},
        **{f"reestimated_test_probs_train_queries_{n:04}": probs for n, probs in reestimated_train_queries_test_probs.items()},
    }

    return result


if __name__ == "__main__":
    main()