import numpy as np
from tqdm import tqdm

def get_original_unnormalized_probs(model, dataset, eval_split="test", num_samples=None, batch_size=1, prompt_func=None):
    label_dict = dataset.label_dict
    all_probs = []
    all_prompt_logprobs = []
    all_labels = []
    all_queries = []
    all_query_truncated  = []
    all_shots_truncated  = []
    for batch in dataset.random_batch_loader_from_split(split=eval_split, num_samples=num_samples, batch_size=batch_size, prompt_func=prompt_func):
        probs, prompt_logprobs = model.get_label_probs(batch["prompt"], label_dict)
        all_probs.append(probs)
        all_prompt_logprobs.append(prompt_logprobs)
        all_labels.append(batch["label"])
        all_query_truncated.append(batch["query_truncated"])
        all_shots_truncated.append(batch["shots_truncated"])
        all_queries.extend(batch['query'])
    all_probs = np.vstack(all_probs)
    all_prompt_logprobs = np.hstack(all_prompt_logprobs)
    all_labels = np.hstack(all_labels)
    all_query_truncated = np.hstack(all_query_truncated).astype(int)
    all_shots_truncated = np.hstack(all_shots_truncated).astype(int)
    return all_labels, all_probs, all_prompt_logprobs, all_queries, all_query_truncated, all_shots_truncated
    

def get_content_free_input_unnormalized_probs(model, dataset, content_free_inputs, batch_size=1, prompt_func=None):
    label_dict = dataset.label_dict
    all_probs = []
    all_prompt_logprobs = []
    all_queries = []
    all_query_truncated  = []
    all_shots_truncated  = []
    for batch in dataset.random_batch_loader_from_list(content_free_inputs, num_samples=None, batch_size=batch_size, prompt_func=prompt_func):
        probs, prompt_logprobs = model.get_label_probs(batch["prompt"], label_dict)
        all_probs.append(probs)
        all_prompt_logprobs.append(prompt_logprobs)
        all_query_truncated.append(batch["query_truncated"])
        all_shots_truncated.append(batch["shots_truncated"])
        all_queries.extend(batch['query'])
    all_probs = np.vstack(all_probs)
    all_prompt_logprobs = np.hstack(all_prompt_logprobs)
    all_query_truncated = np.hstack(all_query_truncated).astype(int)
    all_shots_truncated = np.hstack(all_shots_truncated).astype(int)
    return all_probs, all_prompt_logprobs, all_queries, all_query_truncated, all_shots_truncated



