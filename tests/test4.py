from src.models import create_model
from src.inference import get_original_unnormalized_probs
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


def get_original_unnormalized_probs(model, dataset, eval_split="test", num_samples=None, batch_size=1, prompt_func=None):
    if eval_split == "test":
        all_sentences = dataset._data['test_sentences']
        all_labels = dataset._data['test_labels']
    elif eval_split == "dev":
        ## TODO: implement dev
        raise NotImplementedError
    elif eval_split == "train":
        all_sentences = dataset._data['train_sentences']
        all_labels = dataset._data['train_labels']
    
    total_samples = len(all_sentences)
    if num_samples is None:
        num_samples = total_samples
    else:
        num_samples = min(num_samples,total_samples)
    test_idx = dataset._rs.permutation(total_samples)[:num_samples]
    test_idx = sorted(test_idx, key=lambda x: len(all_sentences[x]), reverse=True) # sort by length of sentence
    print(test_idx)

    label_dict = dataset.label_dict
    new_probs = []
    new_labels = []
    new_queries = []
    new_query_truncated  = []
    new_shots_truncated  = []

    pbar = tqdm(range(0, num_samples, batch_size),total=num_samples//batch_size, leave=False, desc=f"")
    for i in pbar:
        batch_idx = test_idx[i:i+batch_size]
        batch = {'prompt': [], 'label': [], 'query': [], 'query_truncated': [], 'shots_truncated': []}
        for idx in batch_idx:
            query = all_sentences[idx]
            prompt, query_truncated, shots_truncated  = dataset.construct_prompt_with_train_shots(query, prompt_func=prompt_func)
            batch['prompt'].append(prompt)
            batch['query_truncated'].append(query_truncated)
            batch['shots_truncated'].append(shots_truncated)
            batch['label'].append(all_labels[idx])
            batch['query'].append(query)
        
        probs = model.get_label_probs(batch["prompt"], label_dict)
        new_probs.append(probs)
        new_labels.append(batch["label"])
        new_query_truncated.extend(batch["query_truncated"])
        new_shots_truncated.extend(batch["shots_truncated"])
        new_queries.extend(batch['query'])
    new_probs = np.vstack(new_probs)
    new_labels = np.hstack(new_labels)
    new_query_truncated = np.array(new_query_truncated, dtype=int)
    new_shots_truncated = np.array(new_shots_truncated, dtype=int)
    return new_labels, new_probs, new_queries, new_query_truncated, new_shots_truncated


def main():
    model, dataset = create_model_dataset()
    labels1, probs1, _, _, _ = get_original_unnormalized_probs(model, dataset, eval_split="test", num_samples=None, batch_size=1)
    print(np.mean(np.argmax(probs1, axis=1) == labels1))
    labels2, probs2, _, _, _ = get_original_unnormalized_probs(model, dataset, eval_split="test", num_samples=None, batch_size=2)
    print(np.mean(np.argmax(probs2, axis=1) == labels2))
    labels4, probs4, _, _, _ = get_original_unnormalized_probs(model, dataset, eval_split="test", num_samples=None, batch_size=4)
    print(np.mean(np.argmax(probs4, axis=1) == labels4))
    labels8, probs8, _, _, _ = get_original_unnormalized_probs(model, dataset, eval_split="test", num_samples=None, batch_size=8)
    print(np.mean(np.argmax(probs8, axis=1) == labels8))
    labels16, probs16, _, _, _ = get_original_unnormalized_probs(model, dataset, eval_split="test", num_samples=None, batch_size=16)
    print(np.mean(np.argmax(probs16, axis=1) == labels16))
    labels32, probs32, _, _, _ = get_original_unnormalized_probs(model, dataset, eval_split="test", num_samples=None, batch_size=32)
    print(np.mean(np.argmax(probs32, axis=1) == labels32))


if __name__ == "__main__":
    main()


