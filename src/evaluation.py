import numpy as np
from tqdm import tqdm

def _get_probs_from_split(model, dataset, eval_split, num_samples, batch_size, prompt_func=None):
    label_dict = dataset.label_dict
    all_probs = []
    all_labels = []
    all_queries = []
    all_query_truncated  = []
    all_shots_truncated  = []
    for batch in dataset.random_batch_loader_from_split(split=eval_split, num_samples=num_samples, batch_size=batch_size, prompt_func=prompt_func):
        probs = model.get_label_probs(batch["prompt"], label_dict)
        all_probs.append(probs)
        all_labels.append(batch["label"])
        all_query_truncated.append(batch["query_truncated"])
        all_shots_truncated.append(batch["shots_truncated"])
        all_queries.extend(batch['query'])
    all_probs = np.vstack(all_probs)
    all_labels = np.hstack(all_labels)
    all_query_truncated = np.hstack(all_query_truncated).astype(int)
    all_shots_truncated = np.hstack(all_shots_truncated).astype(int)
    return all_labels, all_probs, all_queries, all_query_truncated, all_shots_truncated
    

def get_original_unnormalized_probs(model, dataset, eval_split="test", num_samples=None, batch_size=1, prompt_func=None):
    if eval_split not in ["dev", "test"]:
        raise ValueError("eval_split must be either 'dev' or 'test'")
    labels, probs, queries, all_query_truncated, all_shots_truncated = _get_probs_from_split(model, dataset, eval_split=eval_split, num_samples=num_samples, batch_size=batch_size, prompt_func=prompt_func)
    return labels, probs, queries, all_query_truncated, all_shots_truncated


def get_content_free_input_probs(model, dataset, content_free_inputs, batch_size=1, prompt_func=None):
    label_dict = dataset.label_dict
    all_probs = []
    all_queries = []
    all_query_truncated  = []
    all_shots_truncated  = []
    for batch in dataset.random_batch_loader_from_list(content_free_inputs, num_samples=None, batch_size=batch_size, prompt_func=prompt_func):
        probs = model.get_label_probs(batch["prompt"], label_dict)
        all_probs.append(probs)
        all_query_truncated.append(batch["query_truncated"])
        all_shots_truncated.append(batch["shots_truncated"])
        all_queries.extend(batch['query'])
    all_probs = np.vstack(all_probs)
    all_query_truncated = np.hstack(all_query_truncated).astype(int)
    all_shots_truncated = np.hstack(all_shots_truncated).astype(int)
    return all_probs, all_queries, all_query_truncated, all_shots_truncated


def get_train_queries_probs(model, dataset, num_train_samples=100, batch_size=32, prompt_func=None):
    all_labels, all_probs, all_queries, all_query_truncated, all_shots_truncated = _get_probs_from_split(model, dataset, eval_split="train", num_samples=num_train_samples, batch_size=batch_size, prompt_func=prompt_func)
    # mean_probs = all_probs.mean(axis=0)
    # mean_probs_norm = mean_probs / mean_probs.sum() # Normalize
    return all_labels, all_probs, all_queries, all_query_truncated, all_shots_truncated


def transform_probs(original_probs, rescale_factor):
    # W = np.linalg.inv(np.diag(rescale_factor))
    # transformed_probs = np.matmul(W, original_probs.T).T
    num_classes = original_probs.shape[1]
    W = np.linalg.inv(np.identity(num_classes) * rescale_factor)
    b = np.zeros([num_classes, 1])
    transformed_probs = np.matmul(W, original_probs.T).T + b.T
    return transformed_probs




