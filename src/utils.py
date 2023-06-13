


import argparse
from copy import deepcopy
import hashlib
import json
import os
import pickle
from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from tqdm import tqdm
from IPython.display import HTML

from .data import ClassificationDataset


def parse_classification_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="./")
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--use_saved_results', action='store_true', default=False)
    args = parser.parse_args()
    root_dir = args.root_dir
    config_filename = args.config + ".json"
    with open(os.path.join(root_dir,"configs/main",config_filename)) as config_file:
        config = json.load(config_file)
    use_saved_results = args.use_saved_results

    return root_dir, config, use_saved_results

def parse_collect_results_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="./")
    parser.add_argument('--experiment', type=str, default='experiment')
    parser.add_argument('--config', type=str, default='config')
    args = parser.parse_args()
    root_dir = args.root_dir
    experiment_filename = args.experiment + ".json"
    config_filename = args.config + ".json"
    with open(os.path.join(root_dir,"configs/main",experiment_filename)) as experiment_file:
        experiment_config = json.load(experiment_file)

    with open(os.path.join(root_dir,"configs/collect_results",config_filename)) as results_file:
        results_config = json.load(results_file)

    return root_dir, experiment_config, args.experiment, results_config

def parse_calibration_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="./")
    parser.add_argument('--experiment', type=str, default='experiment')
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--use_saved_results', action='store_true', default=False)
    args = parser.parse_args()
    root_dir = args.root_dir
    experiment_filename = args.experiment + ".json"
    config_filename = args.config + ".json"
    with open(os.path.join(root_dir,"configs/main",experiment_filename)) as experiment_file:
        experiment_config = json.load(experiment_file)

    with open(os.path.join(root_dir,"configs/calibration",config_filename)) as results_file:
        results_config = json.load(results_file)

    use_saved_results = args.use_saved_results

    return root_dir, experiment_config, args.experiment, results_config, args.config, use_saved_results

def parse_content_free_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="./")
    parser.add_argument('--experiment', type=str, default='experiment')
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--use_saved_results', action='store_true', default=False)
    args = parser.parse_args()
    root_dir = args.root_dir
    experiment_filename = args.experiment + ".json"
    config_filename = args.config + ".json"
    with open(os.path.join(root_dir,"configs/main",experiment_filename)) as experiment_file:
        experiment_config = json.load(experiment_file)

    with open(os.path.join(root_dir,"configs/content_free",config_filename)) as results_file:
        results_config = json.load(results_file)

    use_saved_results = args.use_saved_results

    return root_dir, experiment_config, args.experiment, results_config, args.config, use_saved_results

def parse_train_queries_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="./")
    parser.add_argument('--experiment', type=str, default='experiment')
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--use_saved_results', action='store_true', default=False)
    args = parser.parse_args()
    root_dir = args.root_dir
    experiment_filename = args.experiment + ".json"
    config_filename = args.config + ".json"
    with open(os.path.join(root_dir,"configs/main",experiment_filename)) as experiment_file:
        experiment_config = json.load(experiment_file)

    with open(os.path.join(root_dir,"configs/train_queries",config_filename)) as results_file:
        results_config = json.load(results_file)

    use_saved_results = args.use_saved_results

    return root_dir, experiment_config, args.experiment, results_config, args.config, use_saved_results



def create_hash_from_dict(**d):
    # Create a hash name from config dict:
    m = hashlib.md5()
    d = {k: sorted(v) if isinstance(v, list) or isinstance(v, tuple) else v for k, v in d.items()}
    m.update(repr(sorted(d.items())).encode("utf-8"))
    result_name = str(int(m.hexdigest(), 16))
    return result_name


def save_results(root_dir, result, config, result_id, subdir="raw", results_name="results", config_name="config"):

    # check if results directory exists
    if not os.path.exists(f"{root_dir}/results/{subdir}/{result_id}"):
        os.makedirs(f"{root_dir}/results/{subdir}/{result_id}")

    # save results
    with open(f"{root_dir}/results/{subdir}/{result_id}/{results_name}.pkl", "wb") as f:
        pickle.dump(result, f)
    
    # save config
    config_to_be_saved = deepcopy(config)
    config_path = os.path.join(root_dir,"results", subdir, result_id, f"{config_name}.json")
    with open(config_path, "w") as f:
        json.dump(config_to_be_saved, f, indent=4, sort_keys=True)


def get_results_ids_from_config(root_dir, config):
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


def compute_score(true_labels, predictions, bootstrap=True, score="accuracy", random_state=0):
    
    if bootstrap:
        df_boots = pd.DataFrame({"true_labels": true_labels, "predictions": predictions}).sample(
            frac=1, replace=True, random_state=random_state
        )
    else:
        df_boots = pd.DataFrame({"true_labels": true_labels, "predictions": predictions})
    
    if score == "accuracy":
        return accuracy_score(df_boots["true_labels"], df_boots["predictions"])
    else:
        raise ValueError(f"Score {score} not supported!")



def dataset2description(dataset_name):
    dataset = ClassificationDataset("./",None,dataset=dataset_name,n_shot=0,random_state=None)
    description = dataset.dataset2short[dataset_name] + ": " + dataset.description + f"\nTotal test samples: {sum(dataset.test_samples.values())}" + "\nClasses: " + ", ".join(dataset.test_samples.keys())
    return description

def dataset2baseline(dataset_name, score="accuracy"):
    dataset = ClassificationDataset("./",None,dataset=dataset_name,n_shot=0,random_state=None)
    dummy_classifier = DummyClassifier(strategy="most_frequent")
    train_labels = dataset._data['test_labels'] 
    test_labels = dataset._data['test_labels']
    dummy_classifier.fit(np.zeros((len(train_labels),1)), train_labels)
    predictions = dummy_classifier.predict(np.zeros((len(test_labels),1)))
    score = compute_score(test_labels, predictions, bootstrap=False, score=score, random_state=None)
    return score
    


def read_results(root_dir, experiment_name):
    score_results = pd.read_csv(os.path.join(root_dir, "results", f"{experiment_name}_results.csv"), index_col=None)
    # calibrated_results = pd.read_csv(os.path.join(root_dir, "results", f"{experiment_name}_calibration_results.csv"), index_col=None)
    # calibrated_results["output_prob_type"] = "calibrated"
    # score_results = pd.concat([score_results, calibrated_results], axis=0)
    for column in score_results.columns:
        if "score:" in column:
            score_name = column.split(":")[1]
            break
    models = score_results["model"].unique()
    assert len(models) == 1, "Only one model supported"
    model_name = models[0]
    return score_results, score_name, model_name


def show_table_mean_std(score_results, score, model_name, prob_types=None, dataset=None):
    score_results = score_results.loc[score_results["model"] == model_name, :]
    score_results = score_results.sort_values(["model", "dataset", "eval_split", "n_shots", "output_prob_type"])
    score_results = score_results.groupby(["model", "dataset", "eval_split", "n_shots", "output_prob_type"]).agg({
        f"score:{score}": ["mean", "std"], 
    })
    if prob_types is not None:
        prob_types = [pt for pt in prob_types if pt in score_results.index.get_level_values("output_prob_type").unique()]
        score_results = score_results.loc[(slice(None), slice(None), slice(None), slice(None), prob_types), :].sort_index(level=["model", "dataset", "eval_split", "n_shots"])
    if dataset is not None:
        dataset = [ds for ds in dataset if ds in score_results.index.get_level_values("dataset").unique()]
        score_results = score_results.loc[(slice(None), dataset, slice(None), slice(None), slice(None)), :].sort_index(level=["model", "dataset", "eval_split", "n_shots"])

    return HTML(score_results.to_html(justify="start"))


def plot_score_vs_nshots_std(score_results, score="accuracy", model_name="gpt2-xl", prob_types=None, datasets=None):
    score_results = score_results.sort_values(["model", "dataset", "eval_split", "n_shots", "output_prob_type"])
    score_results = score_results.sort_values(["model", "dataset", "eval_split", "n_shots", "output_prob_type"])
    if prob_types is not None:
        prob_types = [pt for pt in prob_types if pt in score_results["output_prob_type"].unique()]
        score_results = score_results.loc[score_results["output_prob_type"].isin(prob_types), :].sort_index(level=["model", "dataset", "eval_split", "n_shots"])
    if datasets is not None:
        datasets = [ds for ds in datasets if ds in score_results["dataset"].unique()]
        score_results = score_results.loc[score_results["dataset"].isin(datasets), :].sort_index(level=["model", "dataset", "eval_split", "n_shots"])
    datasets = score_results["dataset"].unique()
    n_shots = score_results["n_shots"].unique()
    score_results = score_results.groupby(["model", "dataset", "eval_split", "n_shots", "output_prob_type"]).agg({
        f"score:{score}": ["mean", "std"], 
    })
    fig, ax = plt.subplots(1, len(datasets), figsize=(15, 5))
    for i, dataset in enumerate(datasets):
        # Plot acc vs nshots for each output_prob_type
        for output_prob_type in score_results.reset_index()["output_prob_type"].unique():
            mean = score_results.loc[(model_name, dataset, "test", slice(None), output_prob_type), (f"score:{score}", "mean")]
            std = score_results.loc[(model_name, dataset, "test", slice(None), output_prob_type), (f"score:{score}", "std")]
            ax[i].plot(n_shots, mean, label=output_prob_type)
            ax[i].fill_between(n_shots, mean - std, mean + std, alpha=0.2)
        ax[i].set_title(dataset2description(dataset))
        ax[i].set_xlabel("n_shots")
        ax[i].set_xticks(n_shots)
        ax[i].set_ylabel(score)
        ax[i].grid()

    # add legend
    handles, labels = ax[i].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=max([len(output_prob_type)//2,1]), bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()

def plot_score_vs_nshots_boxplot(score_results, score="accuracy", model_name="gpt2-xl", prob_types=None, datasets=None):
    score_results = score_results.sort_values(["model", "dataset", "eval_split", "n_shots", "output_prob_type"])
    if prob_types is not None:
        prob_types = [pt for pt in prob_types if pt in score_results["output_prob_type"].unique()]
        score_results = score_results.loc[score_results["prob_types"].isin(prob_types), :].sort_index(level=["model", "dataset", "eval_split", "n_shots"])
    if datasets is not None:
        datasets = [ds for ds in datasets if ds in score_results["dataset"].unique()]
        score_results = score_results.loc[score_results["dataset"].isin(datasets), :].sort_index(level=["model", "dataset", "eval_split", "n_shots"])

    datasets = score_results["dataset"].unique()
    n_shots = score_results["n_shots"].unique()
    output_prob_type = score_results["output_prob_type"].unique()
    fig, ax = plt.subplots(1, len(datasets), figsize=(15, 5), sharey=False)
    if len(datasets) == 1:
        ax = np.array([ax])
    for i, dataset in enumerate(datasets):
        data = score_results.loc[
            (score_results.model == model_name) & \
            (score_results.dataset == dataset) & \
            (score_results.eval_split == "test"),
            ["n_shots", "output_prob_type", f"score:{score}"]
        ]
        ax[i] = sns.boxplot(
            data=data, 
            x="n_shots", 
            y=f"score:{score}", 
            hue="output_prob_type", 
            ax=ax[i],
            order=np.arange(0,max(n_shots)+1)
        )
        ax[i].hlines(dataset2baseline(dataset), n_shots[0]-1, n_shots[-1]+1, linestyles="dashed", colors="gray")
        ax[i].set_title(dataset2description(dataset))
        ax[i].set_xticks(n_shots)
        ax[i].set_xlabel("N-shots")
        ax[i].set_xlim(-1, max(n_shots)+1)
        ax[i].set_ylabel(score)
        # ax[i].set_ylim(0, 1)
        ax[i].grid()
        ax[i].get_legend().remove()

    # add legend
    handles, labels = ax[i].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(output_prob_type), bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()