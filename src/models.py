import os
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPT2LMHeadModel
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
from torch.optim import AdamW
import numpy as np
import sys
import time

def create_model(root_dir, model="text-davinci-003", max_memory=None):
    if model in GPT3LanguageModel.MODELS:
        return GPT3LanguageModel(root_dir, model_name=model, max_memory=max_memory)
    elif model in HFLanguageModel.MODELS:
        return HFLanguageModel(root_dir, model_name=model, max_memory=max_memory)
    else:
        raise ValueError(f"Model {model} not supported.")



class GPT3LanguageModel:

    MODELS = [
        "text-davinci-003", 
        "text-davinci-002", 
        "text-curie-001", 
        "text-babbage-001", 
        "text-ada-001"
    ]

    def __init__(self, root_dir, model_name="text-davinci-003", max_memory=None):
        self.root_dir = root_dir
        self.model_name = model_name
        self.max_memory = max_memory
        self._setup_model()

    def _setup_model(self):
        self.tokenizer = None
        if self.model_name in self.MODELS:
            with open(os.path.join(self.root_dir, 'openai_key.txt'), 'r') as f:
                for line in f.readlines():
                    if line.startswith("key: "):
                        key = line.split(": ")[1].strip()
                        openai.api_key = key
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
    
    def _complete_gpt3(self, prompt, logprobs=0, temperature=0, max_tokens=0, top_p=1, echo=True):
        response = None
        received = False
        while not received:
            try:
                response = openai.Completion.create(
                    model=self.model_name,
                    prompt=prompt,
                    logprobs=logprobs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    echo=echo
                )
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError: 
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False

                print("API error:", error)
                time.sleep(1)

        return response
    
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
                    completiton = self._complete_gpt3(prompt, logprobs=0, temperature=0, max_tokens=0, top_p=1, echo=True)
                    logprob = completiton["choices"][0]["logprobs"]["token_logprobs"][-1]
                    labels_probs[prompt_idx, label_idx] += np.exp(logprob)
        return labels_probs
    

class HFLanguageModel:

    MODELS = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "gpt2-xl_trec_600"
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
                config = AutoConfig.from_pretrained(model_dir)
                with init_empty_weights():
                    model = AutoModelForCausalLM.from_config(config)
                # model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=model_dir, local_files_only=True, device_map="auto") 
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
        with torch.no_grad():
            labels_probs = torch.zeros(len(prompts_batch), len(labels_dict), device=self.model.device)
            for idx, label_list in labels_dict.items():
                probs = torch.zeros(len(prompts_batch),1, device=self.model.device)
                for label in label_list:
                    label_start_idx = sum(self.tokenizer(f" {label}")["attention_mask"])
                    prompts = [f"{prompt} {label}" for prompt in prompts_batch]
                    encoded_input = self.tokenizer(prompts, return_tensors="pt", padding=True)
                    encoded_input = {k: v.to(self.model.device) for k, v in encoded_input.items()}
                    encoded_input["position_ids"] = self.create_position_ids(encoded_input["attention_mask"])
                    logits = self.model(**encoded_input).logits
                    probs += torch.exp(torch.gather(
                        torch.log_softmax(logits[:,-label_start_idx-1:-1,:], dim=-1),
                        dim=-1, 
                        index=encoded_input["input_ids"][:, -label_start_idx:].unsqueeze(-1)
                    ).squeeze(-1).sum(dim=-1, keepdim=True))
                labels_probs[:, idx] = probs[:, 0]
            labels_probs = labels_probs.cpu().numpy()
        return labels_probs

    @staticmethod
    def create_position_ids(attention_mask):
        position_ids = torch.cumsum(attention_mask, dim=1).long() - 1
        position_ids.masked_fill_(position_ids < 0, 0)
        return position_ids


def create_optimizer(model,layers_to_be_trained,learning_rate=1e-5):
    trainable_parameters = {}
    for layer in layers_to_be_trained:
        trainable_parameters = {
            **trainable_parameters,
            **{name: param for name, param in model.model.named_parameters() if layer in name}
        }
    for name, param in model.model.named_parameters():
        if name not in trainable_parameters:
            param.requires_grad = False
    optimizer = AdamW((param for param in trainable_parameters.values()), lr=learning_rate)
    return optimizer