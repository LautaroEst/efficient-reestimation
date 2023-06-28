


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

from IPython.display import HTML
from sklearn.dummy import DummyClassifier
from .data import ClassificationDataset
from .evaluation import compute_score


def parse_classification_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="./")
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--use_saved_results', action='store_true', default=False)
    args = parser.parse_args()
    root_dir = args.root_dir
    config_filename = args.config + ".json"
    with open(os.path.join(root_dir,"configs/models",config_filename)) as config_file:
        config = json.load(config_file)
    use_saved_results = args.use_saved_results

    return root_dir, config, use_saved_results

def parse_finetuning_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="./")
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--use_saved_results', action='store_true', default=False)
    args = parser.parse_args()
    root_dir = args.root_dir
    config_filename = args.config + ".json"
    with open(os.path.join(root_dir,"configs/finetuning",config_filename)) as config_file:
        config = json.load(config_file)
    use_saved_results = args.use_saved_results

    return root_dir, config, args.config, use_saved_results

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
    with open(os.path.join(root_dir,"configs/models",experiment_filename)) as experiment_file:
        experiment_config = json.load(experiment_file)

    with open(os.path.join(root_dir,"configs/calibration",config_filename)) as results_file:
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


def get_results_ids_from_config(root_dir, config):
    random_state = config.pop("random_state")
    n_seeds = config.pop("n_seeds")

    datasets = config.pop("datasets") # datasets
    cf_inputs = config.pop("content_free_inputs") # cf_inputs
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
                if os.path.exists(f"{root_dir}/results/train_test/{result_id}"):
                    this_seed_results.append(result_id)
            
            if len(this_seed_results) == n_seeds:
                for result_id in this_seed_results:
                    results_ids.append(result_id)
    return results_ids


def read_results(root_dir, experiment_name, calibration_config):
    df_results = pd.read_csv(os.path.join(root_dir, "results", f"{experiment_name}_{calibration_config}.csv"), index_col=None)
    
    with open(os.path.join(root_dir, f"configs/calibration/{calibration_config}.json")) as f:
        calibration_config = json.load(f)
        metrics = sorted(calibration_config["metrics"])
    score_columns = sorted([column.split(":")[1] for column in df_results.columns if "score:" in column])
    assert score_columns == metrics, "Metrics in config and results do not match"

    models = df_results["model"].unique()
    assert len(models) == 1, "Only one model supported"
    model_name = models[0]
    df_results.drop(columns=["model"], inplace=True)

    return df_results, metrics, model_name


def show_table_mean_std(root_dir, experiment_name, calibration_config, prob_types=None, dataset=None):
    df_results, metrics, model_name = read_results(root_dir, experiment_name, calibration_config)
    df_results = df_results.sort_values(["dataset", "n_shots", "prob_type"])
    df_results = df_results.groupby(["dataset", "n_shots", "prob_type"]).agg({
        f"score:{metric}": ["mean", "std"] for metric in metrics
    })
    if prob_types is not None:
        prob_types = [pt for pt in prob_types if pt in df_results.index.get_level_values("prob_type").unique()]
        df_results = df_results.loc[(slice(None), slice(None), prob_types), :].sort_index(level=["dataset", "n_shots"])
    if dataset is not None:
        dataset = [ds for ds in dataset if ds in df_results.index.get_level_values("dataset").unique()]
        df_results = df_results.loc[(dataset, slice(None), slice(None)), :].sort_index(level=["dataset", "n_shots"])

    return HTML(f"<h1>Model: {model_name}</h1>\n" + df_results.to_html())



def dataset2description(dataset_name):
    dataset = ClassificationDataset("./",None,dataset=dataset_name,n_shot=0,random_state=None)
    classes_str = "".join([f'{k}, ' if i % 3 != 2 else f"{k},\n" for i, k in enumerate(dataset.test_samples.keys())])
    if classes_str.endswith(",\n"):
        classes_str = classes_str[:-2]
    if classes_str.endswith(", "):
        classes_str = classes_str[:-2]
    description = f"{dataset.dataset2short[dataset_name]}: {dataset.description}" + \
    f"\nTotal test samples: {sum(dataset.test_samples.values())}" + \
    f"\nClasses: {classes_str}"
    return description

def dataset2baseline(dataset_name, score="accuracy"):
    dataset = ClassificationDataset("./",None,dataset=dataset_name,n_shot=0,random_state=None)
    dummy_classifier = DummyClassifier(strategy="most_frequent")
    train_labels = dataset._data['test_labels'] 
    test_labels = dataset._data['test_labels']
    dummy_classifier.fit(np.zeros((len(train_labels),1)), train_labels)
    predictions = dummy_classifier.predict(np.zeros((len(test_labels),1)))
    score = compute_score(test_labels, predictions, score=score)
    return score

def _prepare_results(root_dir, experiment_name, calibration_configs, metrics=None, prob_types=None, datasets=None):
    all_results = []
    all_metrics = []
    all_model_names = []
    for calibration_config in calibration_configs:
        df_results, metrics, model_name = read_results(root_dir, experiment_name, calibration_config)
        df_results["calibration_config"] = calibration_config
        all_results.append(df_results)
        all_metrics.extend(metrics)
        all_model_names.append(model_name)
    assert len(set(all_model_names)) == 1, "Only one model supported"
    assert len(set(all_metrics)) == len(metrics), "Metrics in config and results do not match"
    df_results = pd.concat(all_results, axis=0)

    if metrics is None:
        metrics = all_metrics
    assert set(metrics).issubset(set(all_metrics)), "Metrics not in results"
    
    if prob_types is None:
        prob_types = df_results["prob_type"].unique()
    assert set(prob_types).issubset(set(df_results["prob_type"].unique())), "Prob types not in results"

    if datasets is None:
        datasets = df_results["dataset"].unique()
    assert set(datasets).issubset(set(df_results["dataset"].unique())), "Datasets not in results"

    df_results = df_results.loc[
        df_results["dataset"].isin(datasets) & df_results["prob_type"].isin(prob_types), :
    ].sort_index(level=["dataset", "n_shots", "prob_type", "calibration_config"])
    
    # df_results = df_results.groupby(["dataset", "n_shots", "prob_type", "calibration_config"]).agg({
    #     f"score:{metric}": ["mean", "std"] for metric in metrics
    # })
    # df_results = df_results.reset_index()
    df_results["result"] = df_results["prob_type"] + "_" + df_results["calibration_config"]
    df_results = df_results.sort_index(axis=1)
    df_results = df_results.drop(columns=["prob_type", "calibration_config"])
    # df_results = df_results.set_index(["dataset", "n_shots", "result"]).sort_index(level=["dataset", "n_shots", "result"])

    return df_results, all_metrics, model_name


def plot_score_vs_nshots_std(root_dir, experiment_name, calibration_configs, metrics=None, prob_types=None, datasets=None, results=None, result2kwargs=None):
    df_results, all_metrics, model_name = _prepare_results(root_dir, experiment_name, calibration_configs, metrics=metrics, prob_types=prob_types, datasets=datasets)
    df_results = df_results.groupby(["dataset", "n_shots", "result"]).agg({
        f"score:{metric}": ["mean", "std"] for metric in metrics
    }).sort_index(level=["dataset", "n_shots", "result"])

    fig, ax = plt.subplots(len(metrics), len(datasets), figsize=(15, 10))
    if len(metrics) == 1 and len(datasets) == 1:
        ax = np.array([[ax]])
    elif len(metrics) == 1:
        ax = ax.reshape(1, -1)
    elif len(datasets) == 1:
        ax = ax.reshape(-1, 1)

    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(datasets):
            # Plot acc vs nshots for each output_prob_type
            for result in df_results.loc[(dataset, slice(None)), :].index.get_level_values("result").unique():
                if result not in results:
                    continue
                mean = df_results.loc[(dataset, slice(None), result), (f"score:{metric}", "mean")]
                std = df_results.loc[(dataset, slice(None), result), (f"score:{metric}", "std")]
                n_shots = mean.index.get_level_values("n_shots")
                kwargs = result2kwargs(result)
                ax[i,j].plot(n_shots, mean, **kwargs)
                ax[i,j].fill_between(n_shots, mean - std, mean + std, alpha=0.1, color=kwargs["color"])
            if i == 0:
                ax[i,j].set_title(dataset2description(dataset),ha='left', x=0.0)
            if metric == "accuracy":
                ax[i,j].hlines(dataset2baseline(dataset), n_shots[0]-1, n_shots[-1]+1, linestyles="dashed", colors="gray", label="Baseline")
            ax[i,j].set_xlabel("n-shots")
            ax[i,j].set_xticks(n_shots)
            ax[i,j].set_ylabel(metric)
            ax[i, j].set_xlim([n_shots[0], n_shots[-1]])
            ax[i,j].grid()

    # add legend
    handles, labels = ax[i,j].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=max([len(results)//3,1]), bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()


def plot_score_vs_nshots_boxplot(root_dir, experiment_name, calibration_configs, metrics=None, prob_types=None, datasets=None, results=None, result2kwargs=None):
    df_results, all_metrics, model_name = _prepare_results(root_dir, experiment_name, calibration_configs, metrics=metrics, prob_types=prob_types, datasets=datasets)
    # df_results = df_results.reset_index()
    n_shots = sorted(df_results["n_shots"].unique())

    fig, ax = plt.subplots(len(metrics), len(datasets), figsize=(20, 10))
    if len(metrics) == 1 and len(datasets) == 1:
        ax = np.array([[ax]])
    elif len(metrics) == 1:
        ax = ax.reshape(1, -1)
    elif len(datasets) == 1:
        ax = ax.reshape(-1, 1)
    
    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(datasets):
            data = df_results.loc[(df_results.dataset == dataset) & df_results.result.isin(results), ["n_shots", "result", f"score:{metric}"]]
            ax[i,j] = sns.boxplot(
                data=data, 
                x="n_shots", 
                y=f"score:{metric}", 
                hue="result", 
                ax=ax[i,j],
                order=np.arange(0,max(n_shots)+1)
            )
            ax[i,j].hlines(dataset2baseline(dataset), n_shots[0]-1, n_shots[-1]+1, linestyles="dashed", colors="gray")
            ax[i,j].set_title(dataset2description(dataset))
            ax[i,j].set_xticks(n_shots)
            ax[i,j].set_xlabel("N-shots")
            ax[i,j].set_xlim(-1, max(n_shots)+1)
            ax[i,j].set_ylabel(metric)
            ax[i,j].grid()
            ax[i,j].get_legend().remove()

    # add legend
    handles, labels = ax[i,j].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=max([len(results)//3,1]), bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()

def plot_score_vs_nshots(root_dir, experiment_name, calibration_configs, plot="line", metrics=None, prob_types=None, datasets=None, results=None, result2kwargs=None):
    if plot == "line":
        plot_score_vs_nshots_std(root_dir, experiment_name, calibration_configs, metrics=metrics, prob_types=prob_types, datasets=datasets, results=results, result2kwargs=result2kwargs)
    elif plot == "boxplot":
        plot_score_vs_nshots_boxplot(root_dir, experiment_name, calibration_configs, metrics=metrics, prob_types=prob_types, datasets=datasets, results=results, result2kwargs=result2kwargs)
    