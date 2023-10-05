import os
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPT2LMHeadModel, T5ForConditionalGeneration
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
from torch.optim import AdamW, SGD
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

    MODELS = {
        "gpt2": GPT2LMHeadModel,
        "gpt2-medium": GPT2LMHeadModel,
        "gpt2-large": GPT2LMHeadModel,
        "gpt2-xl": GPT2LMHeadModel,
        "gpt2-xl_trec_600": GPT2LMHeadModel,
        "google/flan-t5-small": T5ForConditionalGeneration
    }

    def __init__(self, root_dir, model_name="gpt2", max_memory=None):
        self.root_dir = root_dir
        self.model_name = model_name
        self.model, self.tokenizer = self._setup_model(max_memory=max_memory)
                
    def _setup_model_and_tokenizer(self, model, tokenizer):
        if self.model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "gpt2-xl_trec_600"]:
            model.config.pad_token_id = model.config.eos_token_id
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            self._model_architecture = "decoder_only"
        elif self.model_name in ["google/flan-t5-small"]:
            model.config.pad_token_id = model.config.eos_token_id
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            self._model_architecture = "encoder_decoder"
        return model, tokenizer

    def _setup_model(self, max_memory=None):
        if max_memory is None:
            max_memory = {"cuda:0": "10GiB", "cpu": "30GiB"}
        if not isinstance(max_memory, dict):
            raise ValueError(f"max_memory must be a dict, got {type(max_memory)}")
        if self.model_name in self.MODELS:
            model_dir = os.path.join(self.root_dir, "models", self.model_name)
            model_cls = self.MODELS[self.model_name]
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            try:
                config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
                with init_empty_weights():
                    model = model_cls(config=config)
                tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=model_dir, local_files_only=True)
                model, tokenizer = self._setup_model_and_tokenizer(model, tokenizer)
                model.tie_weights()
                device_map = infer_auto_device_map(model, max_memory=max_memory)
                model = load_checkpoint_and_dispatch(
                    model, model_dir, device_map=device_map, no_split_module_classes=["GPT2Block"]
                )
            except OSError as e:
                model = model_cls.from_pretrained(self.model_name, local_files_only=True)
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                model, tokenizer = self._setup_model_and_tokenizer(model, tokenizer)
                model.save_pretrained(model_dir,max_shard_size="100MB")
                tokenizer.save_pretrained(model_dir)
            
            model.eval()
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
        if self._model_architecture == "decoder_only":
            return self._get_label_probs_for_decoder_only(prompts_batch, labels_dict)
        elif self._model_architecture == "encoder_decoder":
            return self._get_label_probs_for_encoder_decoder(prompts_batch, labels_dict)
        
    def _get_label_probs_for_encoder_decoder(self, prompts_batch, labels_dict):
        batch_size = len(prompts_batch)
        with torch.no_grad():
            labels_probs = torch.zeros(batch_size, len(labels_dict), device=self.model.device)
            encoded_prompts = self.tokenizer(prompts_batch, return_tensors="pt", padding=True)
            encoded_prompts["position_ids"] = self.create_position_ids(encoded_prompts["attention_mask"])
            encoded_prompts = {k: v.to(self.model.device) for k, v in encoded_prompts.items()}
            prompt_output = self.model(**encoded_prompts, use_cache=True, output_attentions=False, output_hidden_states=False)
            last_token_logprobs = torch.log_softmax(prompt_output.logits[:,-1,:], dim=-1)
            sequence_lens = encoded_prompts["attention_mask"].sum(dim=-1,keepdim=True).cpu()
            for idx, label_list in labels_dict.items():
                probs = torch.zeros(batch_size,1, device=self.model.device)
                for label in label_list:
                    encoded_label = self.tokenizer([f" {label}" for _ in range(batch_size)], return_tensors="pt", padding=True)
                    label_len = encoded_label["attention_mask"].shape[1]
                    encoded_label["position_ids"] = torch.arange(label_len).repeat(batch_size,1) + sequence_lens
                    encoded_label["attention_mask"] = torch.cat((encoded_prompts["attention_mask"].cpu(),torch.ones((batch_size,label_len),dtype=torch.long)),dim=1)
                    encoded_label = {k: v.to(self.model.device) for k, v in encoded_label.items()}
                    logprobs = torch.log_softmax(self.model(**encoded_label, past_key_values=prompt_output.past_key_values, output_attentions=False, output_hidden_states=False).logits,dim=-1)
                    gathered_logprobs = torch.gather(
                        logprobs[:,:-1,:],
                        dim=-1,
                        index=encoded_label["input_ids"][:, 1:].unsqueeze(-1)
                    ).squeeze(-1).sum(dim=1,keepdim=True) + torch.gather(last_token_logprobs,dim=-1,index=encoded_label["input_ids"][:,-1].unsqueeze(-1))
                    probs += torch.exp(gathered_logprobs)
                labels_probs[:, idx] = probs[:, 0]
            labels_probs = labels_probs.cpu().numpy()
        return labels_probs

    def _get_label_probs_for_decoder_only(self, prompts_batch, labels_dict):
        batch_size = len(prompts_batch)
        with torch.no_grad():
            labels_probs = torch.zeros(batch_size, len(labels_dict), device=self.model.device)
            encoded_prompts = self.tokenizer(prompts_batch, return_tensors="pt", padding=True)
            encoded_prompts["position_ids"] = self.create_position_ids(encoded_prompts["attention_mask"])
            encoded_prompts = {k: v.to(self.model.device) for k, v in encoded_prompts.items()}
            prompt_output = self.model(**encoded_prompts, use_cache=True, output_attentions=False, output_hidden_states=False)
            prompt_logprob = torch.gather(
                torch.log_softmax(prompt_output.logits[:,:-1,:],dim=-1),
                dim=-1,
                index=encoded_prompts["input_ids"][:,1:].unsqueeze(-1)
            ).squeeze(-1).mean(dim=1).cpu().numpy()
            last_token_logprobs = torch.log_softmax(prompt_output.logits[:,-1,:], dim=-1)
            sequence_lens = encoded_prompts["attention_mask"].sum(dim=-1,keepdim=True).cpu()
            for idx, label_list in labels_dict.items():
                probs = torch.zeros(batch_size,1, device=self.model.device)
                for label in label_list:
                    encoded_label = self.tokenizer([f" {label}" for _ in range(batch_size)], return_tensors="pt", padding=True)
                    label_len = encoded_label["attention_mask"].shape[1]
                    encoded_label["position_ids"] = torch.arange(label_len).repeat(batch_size,1) + sequence_lens
                    encoded_label["attention_mask"] = torch.cat((encoded_prompts["attention_mask"].cpu(),torch.ones((batch_size,label_len),dtype=torch.long)),dim=1)
                    encoded_label = {k: v.to(self.model.device) for k, v in encoded_label.items()}
                    logprobs = torch.log_softmax(self.model(**encoded_label, past_key_values=prompt_output.past_key_values, output_attentions=False, output_hidden_states=False).logits,dim=-1)
                    gathered_logprobs = torch.gather(
                        logprobs[:,:-1,:],
                        dim=-1,
                        index=encoded_label["input_ids"][:, 1:].unsqueeze(-1)
                    ).squeeze(-1).sum(dim=1,keepdim=True) + torch.gather(last_token_logprobs,dim=-1,index=encoded_label["input_ids"][:,-1].unsqueeze(-1))
                    probs += torch.exp(gathered_logprobs)
                labels_probs[:, idx] = probs[:, 0]
            labels_probs = labels_probs.cpu().numpy()
        return labels_probs, prompt_logprob

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
    # optimizer = SGD((param for param in trainable_parameters.values()), lr=learning_rate)
    return optimizer