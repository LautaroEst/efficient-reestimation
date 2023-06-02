import numpy as np
from src.models import create_model
from src.evaluation import get_content_free_input_probs
from src.data import ClassificationDataset


def main():
    root_dir = "./"
    model_name = "gpt2-xl"
    model = create_model(root_dir, model=model_name)
    n_shots = 0
    dataset_name = "agnews"
    random_state = 123456
    dataset = ClassificationDataset(
        root_dir,
        dataset_name,
        n_shot=n_shots,
        random_state=random_state
    )
    test_sentence = 'DEFOE DRIVES SPURS HOME. Jermain Defoe underlined his claims for an improved contract as he inspired Tottenham to a 2-0 win against 10-man Middlesbrough. New coach Martin Jol, who secured his first win in charge, may have been helped '
    gold_probs = {' World': -3.681288719177246, ' Sports': -5.388176441192627, ' Business': -5.5484514236450195, ' Technology': -6.235733985900879}
    test_prompt, truncated = dataset.construct_prompt_with_train_shots(test_sentence)
    for batch in dataset.random_batch_loader_from_split(split="test", num_samples=None, batch_size=1):
        if batch["prompt"][0] != test_prompt:
            continue
        logprobs = model.get_label_probs(batch["prompt"], dataset.label_dict)
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
    print(np.array([0.45427626, 0.1490043 , 0.19061938, 0.20610012]))
    


if __name__ == "__main__":
    main()