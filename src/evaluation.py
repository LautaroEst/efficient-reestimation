import numpy as np
from tqdm import tqdm

def _get_logprobs_from_split(model, dataset, eval_split, num_samples, batch_size):
    label_dict = dataset.label_dict
    all_logprobs = []
    all_labels = []
    all_queries = []
    all_query_truncated  = []
    all_shots_truncated  = []
    for batch in dataset.random_batch_loader_from_split(split=eval_split, num_samples=num_samples, batch_size=batch_size):
        logprobs = model.get_label_probs(batch["prompt"], label_dict)
        all_logprobs.append(logprobs)
        all_labels.append(batch["label"])
        all_query_truncated.append(batch["query_truncated"])
        all_shots_truncated.append(batch["shots_truncated"])
        all_queries.extend(batch['query'])
    all_logprobs = np.vstack(all_logprobs)
    all_labels = np.hstack(all_labels)
    all_query_truncated = np.hstack(all_query_truncated).astype(int)
    all_shots_truncated = np.hstack(all_shots_truncated).astype(int)
    return all_labels, all_logprobs, all_queries, all_query_truncated, all_shots_truncated
    

def get_original_unnormalized_logprobs(model, dataset, eval_split="test", num_samples=None, batch_size=1):
    if eval_split not in ["dev", "test"]:
        raise ValueError("eval_split must be either 'dev' or 'test'")
    labels, logprobs, queries, all_query_truncated, all_shots_truncated = _get_logprobs_from_split(model, dataset, eval_split=eval_split, num_samples=num_samples, batch_size=batch_size)
    return labels, logprobs, queries, all_query_truncated, all_shots_truncated


def get_content_free_input_probs(model, dataset, content_free_inputs, batch_size=1):
    all_logprobs = []
    for prompt in dataset.random_batch_loader_from_list(content_free_inputs, num_samples=None, batch_size=batch_size):
        logprobs = model.get_label_probs(prompt, dataset.label_dict)
        all_logprobs.append(logprobs)
    all_logprobs = np.vstack(all_logprobs)
    probs = np.exp(all_logprobs)
    probs /= probs.sum(axis=1, keepdims=True) # Normalize
    mean_probs = probs.mean(axis=0)
    return mean_probs


def get_train_queries_probs(model, dataset, num_train_samples=100, batch_size=32):
    _, logprobs, _, _, _ = _get_logprobs_from_split(model, dataset, eval_split="train", num_samples=num_train_samples, batch_size=batch_size)
    probs = np.exp(logprobs)
    probs /= probs.sum(axis=1, keepdims=True) # Normalize
    mean_probs = probs.mean(axis=0)
    return mean_probs


def transform_probs(original_probs, rescale_factor):
    W = np.diag(rescale_factor)
    transformed_probs = np.matmul(original_probs, np.linalg.inv(W))
    return transformed_probs

