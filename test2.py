import numpy as np
from src.models import create_model
from src.evaluation import get_content_free_input_probs
from src.data import ClassificationDataset


def main():
    root_dir = "./"
    model_name = "gpt2-xl"
    model = create_model(root_dir, model=model_name)
    n_shots = 1
    dataset_name = "cb"
    random_state = 123456
    dataset = ClassificationDataset(
        root_dir,
        model.tokenizer,
        dataset_name,
        n_shot=n_shots,
        random_state=random_state
    )
    train_sentences = ["A: No, not really. I spends a lot of time with our income tax, though. especially, this year and last year. Um, I have been married for just a few years, so I've had to really switch around from the EZ form to the, uh, B: Schedule A. A: Right. B: Well, yeah. A: All the deductions and all that. B: Did you notice that when they passed the new simplified tax act, it seemed like it made everything harder?\nquestion: when they passed the new simplified tax act  it seemed like it made everything harder. true, false, or neither?"]
    train_labels = [2]
    old_train_sentences = dataset.prompt_shots_sentences
    old_train_labels = dataset.prompt_shots_labels
    dataset.prompt_shots_sentences = train_sentences
    dataset.prompt_shots_labels = train_labels
    dataset._data["train_sentences"].extend(old_train_sentences)
    dataset._data["train_labels"].extend(old_train_labels)
    test_sentence = "Valence the void-brain, Valence the virtuous valet. Why couldn't the figger choose his own portion of titanic anatomy to shaft? Did he think he was helping?\nquestion: Valence was helping. true, false, or neither?"
    gold_probs = {' true': -0.6178356409072876,' false': -1.5358854532241821,' neither': -3.714348554611206}
    test_prompt, query_truncated, shots_truncated = dataset.construct_prompt_with_train_shots(test_sentence)
    for batch in dataset.random_batch_loader_from_split(split="test", num_samples=None, batch_size=1):
        if batch["prompt"][0] != test_prompt:
            continue
        probs = model.get_label_probs(batch["prompt"], dataset.label_dict)
        logprobs = np.log(probs)
        break
    print([(v, logprobs[0,k]) for k, v in dataset.label_dict.items()])
    print(gold_probs)
    wt = model.tokenizer(" Sports", return_tensors="pt")
    print(wt)

    content_free_input_probs = get_content_free_input_probs(
        model, 
        dataset, 
        content_free_inputs=["[MASK]","N/A",""],
        batch_size=1
    )
    print(content_free_input_probs)
    content_free_input_probs = get_content_free_input_probs(
        model, 
        dataset, 
        content_free_inputs=["idk"],
        batch_size=1
    )
    print(content_free_input_probs)
    print(np.array([0.6759636 , 0.09522448, 0.22881192]))
    


if __name__ == "__main__":
    main()