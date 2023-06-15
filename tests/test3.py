from src.models import create_model
from src.inference import get_content_free_input_probs
from src.data import ClassificationDataset
from tqdm import tqdm
import json
import os
import pickle

import numpy as np


def create_model_dataset():
    root_dir = "./"
    model_name = "gpt2-xl"
    model = create_model(root_dir, model=model_name)
    n_shots = 0
    dataset_name = "cb"
    random_state = 123456
    dataset = ClassificationDataset(
        root_dir,
        model.tokenizer,
        dataset_name,
        n_shot=n_shots,
        random_state=random_state
    )
    return model, dataset

def read_results_predicted(results_gold):
    for d in os.listdir("results/raw/"):
        with open(f"results/raw/{d}/config.json", "rb") as f:
            config = json.load(f)
        if config["n_shots"] == 0 and config["dataset"] == "cb":
            break
    with open(f"results/raw/{d}/results.pkl", "rb") as f:
        results = pickle.load(f)

    model, dataset = create_model_dataset()

    original_probs = []
    predicted_probs = []
    for i, (response, test_sentence) in enumerate(tqdm(zip(results_gold["raw_resp_test"], results_gold["test_sentences"]),total=len(results_gold["raw_resp_test"]))):
        test_prompt, query_truncated, shots_truncated = dataset.construct_prompt_with_train_shots(test_sentence)
        for idx, batch in enumerate(dataset.random_batch_loader_from_split(split="test", num_samples=None, batch_size=1)):
            if batch["prompt"][0] != test_prompt:
                continue
            probs = model.get_label_probs(batch["prompt"], dataset.label_dict)
            original_probs.append(probs)
            predicted_probs.append(results["original_probs"][idx,:])
            break
    original_probs = np.vstack(original_probs)
    predicted_probs = np.vstack(predicted_probs)
    return original_probs, predicted_probs



def read_results_gold():
    results_path = "cb_gpt2-xl_0shot_None_subsample_seed0.pkl"
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    return results


def main():
    
    results_gold = read_results_gold()
    original_probs, predicted_probs = read_results_predicted(results_gold)
    print(np.mean(np.argmax(results_gold["all_label_probs"],axis=-1) == results_gold["test_labels"]))
    print(np.mean(np.argmax(original_probs,axis=-1) == results_gold["test_labels"]))
    print(np.mean(np.argmax(predicted_probs,axis=-1) == results_gold["test_labels"]))

    



    
    # true_labels = results["true_labels"]
    # original_probs = results["original_probs"]
    # print(results["prompt_shots_sentences"])
    # print("Accuracy: ", np.mean(np.argmax(original_probs, axis=-1) == true_labels))

    # results_path = "cb_gpt2-xl_0shot_None_subsample_seed0.pkl"
    # with open(results_path, "rb") as f:
    #     gold_results = pickle.load(f)
    
    # idx = np.argsort(results["test_queries"])
    # idx_gold = np.argsort(gold_results["test_sentences"])
    # probs = original_probs[idx]
    # probs_gold = gold_results["all_label_probs"][idx_gold]
    # # print(probs == probs_gold)
    # # print(gold_results["all_label_probs"])
    # # print(probs)
    # print(probs_gold)
    # # for i, i_g in zip(idx, idx_gold):
    # #     print(results["test_queries"][i])
    # #     print(gold_results["test_sentences"][i_g])
    # #     if i > 2:
    # #         break
    # print(sum([1 for i, i_g in zip(idx, idx_gold) if results["test_queries"][i] != gold_results["test_sentences"][i_g]]))
    


if __name__ == "__main__":
    main()