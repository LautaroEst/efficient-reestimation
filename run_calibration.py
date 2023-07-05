
from copy import deepcopy
import glob
import json
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.expected_cost.psrcal_wrappers import LogLoss, Brier
from src.utils import parse_calibration_args, get_results_ids_from_config
from src.calibration import calibrate_from_train_probs, calibrate_probs_from_trained_model, reestimate_probs_from_trained_model, train_calibrator_from_probs, train_reestimator_from_probs, train_reestimator_iter_from_probs, calibrate_from_train_probs, calibrate_probs_from_trained_model, train_reestimator_prompt_from_probs
from sklearn.metrics import accuracy_score


def main():
    root_dir, experiment_config, experiment_name, calibration_config, calibration_config_name, use_saved_results = parse_calibration_args()
    results_ids = get_results_ids_from_config(root_dir, experiment_config)
    print(f"Running calibration for {experiment_name} experiment with the {calibration_config_name} configuration...")

    evaluation_metrics = calibration_config.pop("metrics")

    # check if results directory exists
    if len(results_ids) == 0:
        print("No results found!")
        return
    
    scores_records = []
    for result_id in tqdm(results_ids, leave=False):

        with open(f"{root_dir}/results/train_test/{result_id}/config.json", "r") as f:
            main_config = json.load(f)

        if os.path.exists(os.path.join(root_dir, f"results/calibrated/{calibration_config_name}/{result_id}.pkl")) and use_saved_results:
            with open(os.path.join(root_dir, f"results/calibrated/{calibration_config_name}/{result_id}.pkl"), "rb") as f:
                results = pickle.load(f)
        else:
            with open(f"{root_dir}/results/train_test/{result_id}/train.pkl", "rb") as f:
                train_results = pickle.load(f)
            with open(f"{root_dir}/results/train_test/{result_id}/test.pkl", "rb") as f:
                test_results = pickle.load(f)
            
            cf_results = {}
            for filename in glob.glob(f"{root_dir}/results/train_test/{result_id}/*.pkl"):
                if filename.endswith("train.pkl") or filename.endswith("test.pkl"):
                    continue
                with open(filename, "rb") as f:
                    cf_results[filename.split("/")[-1].split(".")[0]] = pickle.load(f)
            
            results = run_calibration_with_bootstrap(train_results, test_results, cf_results, random_state=main_config["random_state"], **calibration_config)

            # Save Results
            if not os.path.exists(os.path.join(root_dir, f"results/calibrated/{calibration_config_name}")):
                os.makedirs(os.path.join(root_dir, f"results/calibrated/{calibration_config_name}"))
            with open(os.path.join(root_dir, f"results/calibrated/{calibration_config_name}/{result_id}.pkl"), "wb") as f:
                pickle.dump(results, f)

        if any([(result[key] == 0).sum() > 0 and "20" in key for result in results for key in result.keys() if key not in ["boot_idx", "test_labels"]]):
            print(f"WARNING: {result_id} contains zero probabilities!")
        scores = compute_metrics(results, main_config, evaluation_metrics)
        scores_records.extend(scores)

    # Save Scores
    scores_df = pd.DataFrame(scores_records)
    scores_df.to_csv(os.path.join(root_dir, f"results/{experiment_name}_{calibration_config_name}.csv"), index=False)
    print("Done!")


def compute_metrics(results, config, evaluation_metrics):
    scores = []
    for i, result in enumerate(results):
        for key in result.keys():
            if key in ["boot_idx", "test_labels"]:
                continue

            this_prob_type = {
                **deepcopy(config),
                "prob_type": key,
                "bootstrap_iter": i
            }
            for metric in evaluation_metrics:
                if metric  == "cross-entropy":
                    metric_fn = lambda probs, labels: LogLoss(np.log(probs), labels, norm=True, priors=None)
                elif metric == "brier":
                    metric_fn = lambda probs, labels: Brier(np.log(probs), labels, norm=True, priors=None)
                elif metric == "accuracy":
                    metric_fn = lambda probs, labels: accuracy_score(np.argmax(probs, axis=1), labels, normalize=True, sample_weight=None)
                else:
                    raise ValueError(f"Metric {metric} not supported!")
            
                score = metric_fn(result[key], result["test_labels"])
                this_prob_type[f"score:{metric}"] = float(score)

            scores.append(this_prob_type)

    return scores

            
def run_calibration_with_bootstrap(
    train_results,
    test_results,
    cf_results,
    num_train_samples,
    calmethod = 'AffineCalLogLoss',
    calparams={},
    bootstrap = 100,
    random_state = None
):

    test_probs = test_results["test_probs"].copy()
    test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)
    test_labels = test_results["test_labels"]
    
    train_probs = train_results["train_probs"].copy()
    train_probs = train_probs / train_probs.sum(axis=1, keepdims=True)
    train_labels = train_results["train_labels"]
    train_prompt_probs = np.exp(train_results["train_prompt_logprobs"])

    rs = np.random.RandomState(random_state)
    train_models_dict = {}
    for n in num_train_samples:
        train_idx = rs.choice(len(train_probs), n, replace=False)
        subsample_probs = train_probs[train_idx]
        subsample_prompt_probs = train_prompt_probs[train_idx]
        subsample_labels = train_labels[train_idx]
        scalecalmodel = train_calibrator_from_probs(subsample_probs, subsample_labels, calmethod=calmethod, calparams={"scale": True, "bias": True})
        noscalecalmodel = train_calibrator_from_probs(subsample_probs, subsample_labels, calmethod=calmethod, calparams={"scale": False, "bias": True})
        reestmodel = train_reestimator_from_probs(subsample_probs)
        reestmodelwithpriors = train_reestimator_from_probs(subsample_probs, subsample_labels)
        reestiterative = train_reestimator_iter_from_probs(subsample_probs)
        reestiterativewithpriors = train_reestimator_iter_from_probs(subsample_probs, subsample_labels)
        reestpromptmodel = train_reestimator_prompt_from_probs(subsample_probs, subsample_prompt_probs)
        reestpromptwithpriorsmodel = train_reestimator_prompt_from_probs(subsample_probs, subsample_prompt_probs, subsample_labels)
        train_models_dict[n] = (scalecalmodel, noscalecalmodel, reestmodel, reestmodelwithpriors, reestiterative, reestiterativewithpriors, reestpromptmodel, reestpromptwithpriorsmodel)

    cf_models_dict = {}
    for cf_name, cf_result in cf_results.items():
        cf_probs = cf_result["probs"].copy()
        cf_probs = cf_probs / cf_probs.sum(axis=1, keepdims=True)
        reestmodel = train_reestimator_from_probs(cf_probs)
        cf_models_dict[cf_name] = reestmodel
    
    if bootstrap is not None:
        boots_idx = [rs.choice(len(test_labels), len(test_labels), replace=True) for _ in range(bootstrap)]
    else:
        boots_idx = [None]

    boots_results = []
    for bi in boots_idx:
        test_probs_boots = test_probs[bi].copy() if bi is not None else test_probs.copy()
        test_labels_boots = test_labels[bi].copy() if bi is not None else test_labels.copy()
        result = calibrate_reestimate_all(
            test_probs_boots, 
            test_labels_boots, 
            cf_models_dict, 
            train_models_dict,
            boots_idx=bi,
            calmethod=calmethod, 
            calparams=calparams
        )
        result["boot_idx"] = bi
        boots_results.append(result)
    
    return boots_results        
    

def calibrate_reestimate_all(
    test_probs, 
    test_labels,
    cf_models_dict, 
    train_models_dict, 
    boots_idx, 
    calmethod='AffineCalLogLoss', 
    calparams={}
):
    
    test_probs_cal_peaky = calibrate_from_train_probs(
        test_probs, 
        test_labels, 
        test_probs, 
        test_labels, 
        calmethod=calmethod, 
        calparams=calparams, 
        cross_val=False,
        boots_idx=None
    )

    test_probs_cal_xval = calibrate_from_train_probs(
        None,
        None,
        test_probs,
        test_labels,
        calmethod=calmethod,
        calparams=calparams,
        cross_val=True,
        boots_idx=boots_idx
    )

    test_probs_scalecal_train = {}
    test_probs_noscalecal_train = {}
    test_probs_reest_train = {}
    test_probs_reestwithpriors_train = {}
    test_probs_reestiterative_train = {}
    test_probs_reestiterativewithpriors_train = {}
    test_probs_reestprompt_train = {}
    test_probs_reestpromptwithpriors_train = {}
    for n, (scalecalmodel, noscalecalmodel, reestmodel, reestmodelwithpriors, reestiterative, reestiterativewithpriors, reestpromptmodel, reestpromptwithpriorsmodel) in train_models_dict.items():
        test_probs_scalecal_train[n] = calibrate_probs_from_trained_model(test_probs,scalecalmodel)
        test_probs_noscalecal_train[n] = calibrate_probs_from_trained_model(test_probs,noscalecalmodel)
        test_probs_reest_train[n] = reestimate_probs_from_trained_model(test_probs,reestmodel)
        test_probs_reestwithpriors_train[n] = reestimate_probs_from_trained_model(test_probs,reestmodelwithpriors)
        test_probs_reestiterative_train[n] = reestimate_probs_from_trained_model(test_probs,reestiterative)
        test_probs_reestiterativewithpriors_train[n] = reestimate_probs_from_trained_model(test_probs,reestiterativewithpriors)
        test_probs_reestprompt_train[n] = reestimate_probs_from_trained_model(test_probs,reestpromptmodel)
        test_probs_reestpromptwithpriors_train[n] = reestimate_probs_from_trained_model(test_probs,reestpromptwithpriorsmodel)

    test_probs_reest_cf = {}
    for cf_name, cf_reestmodel in cf_models_dict.items():
        test_probs_reest_cf[cf_name] = reestimate_probs_from_trained_model(test_probs,cf_reestmodel)
    
    test_probs_original = test_probs.copy()

    return {
        "test_probs_original": test_probs_original,
        "test_labels": test_labels,
        "test_probs_cal_peaky": test_probs_cal_peaky,
        **{f"test_probs_scalecal_train_{n}": test_probs_scalecal_train[n] for n in train_models_dict},
        **{f"test_probs_noscalecal_train_{n}": test_probs_noscalecal_train[n] for n in train_models_dict},
        **{f"test_probs_reest_train_{n}": test_probs_reest_train[n] for n in train_models_dict},
        **{f"test_probs_reestwithpriors_train_{n}": test_probs_reestwithpriors_train[n] for n in train_models_dict},
        **{f"test_probs_reestiterative_train_{n}": test_probs_reestiterative_train[n] for n in train_models_dict},
        **{f"test_probs_reestiterativewithpriors_train_{n}": test_probs_reestiterativewithpriors_train[n] for n in train_models_dict},
        **{f"test_probs_reestprompt_train_{n}": test_probs_reestprompt_train[n] for n in train_models_dict},
        **{f"test_probs_reestpromptwithpriors_train_{n}": test_probs_reestpromptwithpriors_train[n] for n in train_models_dict},
        "test_probs_cal_xval": test_probs_cal_xval,
        **{f"test_probs_reest_cf_{cf_name}": test_probs_reest_cf[cf_name] for cf_name in cf_models_dict}
    }
    
        
if __name__ == "__main__":
    main()