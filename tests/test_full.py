import pickle
import numpy as np
from src.models import create_model
from src.inference import get_content_free_input_probs
from src.data import ClassificationDataset
from tqdm import tqdm

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

def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    correctness_list = []
    assert len(all_label_probs) == len(test_labels)
    for label_probs, true_label in zip(all_label_probs, test_labels):
        label_probs = label_probs / np.sum(label_probs) # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b

        ans_label = np.argmax(calibrate_label_probs)
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    return np.mean(correctness_list)

def main():
    
    model, dataset = create_model_dataset()

    results_path = "cb_gpt2-xl_0shot_None_subsample_seed0.pkl"
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    gold_label_probs = np.zeros((len(results["test_labels"]), len(dataset.label_dict)))
    predicted_label_probs = np.zeros((len(results["test_labels"]), len(dataset.label_dict)))
    for i, (response, test_sentence) in enumerate(tqdm(zip(results["raw_resp_test"], results["test_sentences"]),total=len(results["raw_resp_test"]))):
        gold_logprobs = {key.split(" ")[1]: response["logprobs"]["top_logprobs"][0][key] for key in [' false', ' true', ' neither']}
        test_prompt, query_truncated, shots_truncated = dataset.construct_prompt_with_train_shots(test_sentence)
        for batch in dataset.random_batch_loader_from_split(split="test", num_samples=None, batch_size=1):
            if batch["prompt"][0] != test_prompt:
                continue
            probs = model.get_label_probs(batch["prompt"], dataset.label_dict)
            logprobs = np.log(probs)
            break
        for k, v in dataset.label_dict.items():
            gold_label_probs[i,k] = np.exp(gold_logprobs[v[0]])
            predicted_label_probs[i,k] = np.exp(logprobs[0,k])
            if abs(logprobs[0,k] - gold_logprobs[v[0]]) > 0.01:
                print(test_sentence)
                print(f"Pred: {logprobs[0,k]:.4f}, Gold: {gold_logprobs[v[0]]:.4f}")
                print()
            
    gold_accs = eval_accuracy(gold_label_probs, results["test_labels"], mode=None, p_cf=None)
    pred_accs = eval_accuracy(predicted_label_probs, results["test_labels"], mode=None, p_cf=None)
    goldgold_accs = eval_accuracy(results["all_label_probs"], results["test_labels"], mode=None, p_cf=None)
    mypred_accs = my_eval_accuracy(results["all_label_probs"], results["test_labels"], mode=None, p_cf=None)
    print(f"Gold acc: {gold_accs:.4f}, Pred acc: {pred_accs:.4f}", f"GoldGold acc: {goldgold_accs:.4f}, MyPred acc: {mypred_accs:.4f}")


def my_eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False
    
    calibrate_label_probs = np.matmul(W, all_label_probs.T).T + b.T
    ans_labels = np.argmax(calibrate_label_probs, axis=1)
    return np.mean(ans_labels == test_labels)


if __name__ == "__main__":
    main()