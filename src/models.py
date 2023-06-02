import os
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
import numpy as np

def create_model(root_dir, model="text-davinci-003", max_memory=None):
    if model in GPT3LanguageModel.MODELS:
        return GPT3LanguageModel(root_dir, model_name=model)
    elif model in HFLanguageModel.MODELS:
        return HFLanguageModel(root_dir, model_name=model, max_memory=max_memory)
    raise ValueError(f"Model {model} not supported.")



class GPT3LanguageModel:

    MODELS = [
        "text-davinci-003", 
        "text-davinci-002", 
        "text-curie-001", 
        "text-babbage-001", 
        "text-ada-001"
    ]

    def __init__(self, root_dir, model_name="text-davinci-003"):
        self.root_dir = root_dir
        self.model_name = model_name
        self._setup_model()

    def _setup_model(self):
        if self.model_name in self.MODELS:
            with open(os.path.join(self.root_dir, 'openai_key.txt'), 'r') as f:
                key = f.readline().strip()
                openai.api_key = key
        raise ValueError(f"Model {self.model_name} not supported.")
    
    def get_label_probs(self, prompts_batch, labels_dict):
        """
        Return the probability of the label given the prompt for the given model.

        Parameters
        ----------
        prompt_batch : List[str]
            The prompt to be completed.
        label_batch : List[str]
            The label to be predicted.

        Returns
        -------
        torch.FloatTensor
            The probability P(label|prompt) of the label given the prompt.

        Comments
        --------
        con estos parámetros, completition["choices"][0] devuelve un diccionario:
        {
            "finish_reason": "length",
            "index": 0,
            "logprobs": {
                "text_offset": [
                    ... # lista de offsets de cada token de entrada
                ],
                "token_logprobs": [
                    ... # lista de [None, P(w2|w1), ..., P(wn|w1,...,wn-1)]
                ],
                "tokens": [
                    ... # lista de [w1, w2, ..., wn]
                ],
                "top_logprobs": None # Lista de los logprobs tokens más probables para [(w1), (w1,w2), ..., (w1,...,wn-1)]
            },
            "text": ...,
        }
        """
        labels_probs = np.zeros((len(prompts_batch), len(labels_dict)))
        for prompt_idx, prompt in enumerate(prompts_batch):
            for label_idx, label_list in labels_dict.items():
                for label in label_list:
                    prompt = f"{prompt} {label}"
                    completiton = openai.Completion.create(
                        model=self.model_name,
                        prompt=prompt,
                        logprobs=0,
                        temperature=0,
                        max_tokens=0,
                        top_p=1,
                        echo=True
                    )
                    logprob = completiton["choices"][0]["logprobs"]["token_logprobs"][-1]
                    labels_probs[prompt_idx, label_idx] += np.exp(logprob)
        return labels_probs
    
class HFLanguageModel:

    MODELS = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl"
    ]

    def __init__(self, root_dir, model_name="gpt2", max_memory=None):
        self.root_dir = root_dir
        self.model_name = model_name
        self.model, self.tokenizer = self._setup_model(max_memory=max_memory)
                
    def _setup_model(self, max_memory=None):
        if max_memory is None:
            max_memory = {"cuda:0": "10GiB", "cpu": "30GiB"}
        if not isinstance(max_memory, dict):
            raise ValueError(f"max_memory must be a dict, got {type(max_memory)}")
        if self.model_name in self.MODELS:
            model_dir = os.path.join(self.root_dir, "models", self.model_name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            try:
                config = AutoConfig.from_pretrained(self.model_name, cache_dir=model_dir, local_files_only=True)
                with init_empty_weights():
                    model = AutoModelForCausalLM.from_config(config)
                model.config.pad_token_id = model.config.eos_token_id
                model.tie_weights()
                device_map = infer_auto_device_map(model, max_memory=max_memory)
                model = load_checkpoint_and_dispatch(
                    model, model_dir, device_map=device_map, no_split_module_classes=["GPT2Block"]
                )
                model.eval()

                tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=model_dir, local_files_only=True)
                tokenizer.padding_side = "left"
                tokenizer.pad_token = tokenizer.eos_token
                
            except OSError as e:
                model = AutoModelForCausalLM.from_pretrained(self.model_name)
                model.config.pad_token_id = model.config.eos_token_id
                model.save_pretrained(model_dir,max_shard_size="500MB")
                
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                tokenizer.padding_side = "left"
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.save_pretrained(model_dir)
            
            return model, tokenizer
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
        

    def get_label_probs(self, prompts_batch, labels_dict):
        """
        Return the probability of the label given the prompt for the given model.

        Parameters
        ----------
        prompts_batch : List[str]
            The prompts to be completed.
        labels : str
            The labels to be predicted.
        model : str, optional
            The model to be used. The default is "gpt2".

        Returns
        -------
        torch.FloatTensor[batch_size, number_of_labels]
            The probability P(label|prompt) of the label given the prompt.

        """
        labels_probs = torch.zeros(len(prompts_batch), len(labels_dict))
        with torch.no_grad():
            for idx, label_list in labels_dict.items():
                probs = torch.zeros(len(prompts_batch),1, device=self.model.device)
                for label in label_list:
                    label_start_idx = sum(self.tokenizer(f" {label}")["attention_mask"])
                    prompts = [f"{prompt} {label}" for prompt in prompts_batch]
                    encoded_input = self.tokenizer(prompts, return_tensors="pt", padding=True)
                    encoded_input = {k: v.to(self.model.device) for k, v in encoded_input.items()}
                    logits = self.model(**encoded_input).logits
                    probs += torch.gather(
                        torch.softmax(logits[:,-label_start_idx-1:-1,:], dim=-1), 
                        dim=-1, 
                        index=encoded_input["input_ids"][:, -label_start_idx:].unsqueeze(-1)
                    ).squeeze(-1)
                labels_probs[:, idx] = probs[:, 0].cpu()
            labels_probs = labels_probs.numpy()
        return labels_probs

