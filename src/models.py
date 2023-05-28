import os
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
import numpy as np

def create_model(root_dir, model="text-davinci-003"):
    if model in GPT3LanguageModel.MODELS:
        return GPT3LanguageModel(root_dir, model)
    elif model in HFLanguageModel.MODELS:
        return HFLanguageModel(root_dir, model)
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
    
    def get_label_probs(self, prompt_batch, label_batch):
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
        logprobs = []
        for prompt, label in zip(prompt_batch, label_batch):
            prompt = prompt + " " + label
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
            logprobs.append(logprob)
        logprobs = torch.tensor(logprobs).views(-1, 1)
        return logprob
    
class HFLanguageModel:

    MODELS = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl"
    ]

    def __init__(self, root_dir, model_name="gpt2"):
        self.root_dir = root_dir
        self.model_name = model_name
        self.model, self.tokenizer = self._setup_model()
                
    def _setup_model(self):
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
                device_map = infer_auto_device_map(model, max_memory={0: "10GiB", "cpu": "30GiB"})
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
        encoded_input = self.tokenizer(prompts_batch, return_tensors="pt", padding=True)
        # prompt_lengths = encoded_input["attention_mask"].sum(dim=1)
        labels_probs = torch.zeros(len(prompts_batch), len(labels_dict))

        with torch.no_grad():
            for idx, label_list in labels_dict.items():
                probs = torch.zeros(len(prompts_batch),1)
                for label in label_list:
                    label_start_idx = sum(self.tokenizer(" " + label)["attention_mask"])
                    prompts = [prompt + " " + label for prompt in prompts_batch]
                    encoded_input = self.tokenizer(prompts, return_tensors="pt", padding=True)
                    logits = self.model(**encoded_input).logits
                    probs += torch.gather(
                        torch.softmax(logits[:,-label_start_idx-1:-1,:], dim=-1), 
                        dim=-1, 
                        index=encoded_input["input_ids"][:, -label_start_idx:].unsqueeze(-1)
                    ).squeeze(-1)
                labels_probs[:, idx] = probs[:, 0]
            labels_logprobs = torch.log(labels_probs).numpy()
        return labels_logprobs


    # def get_label_probs2(self, prompt, label, l=10, num_log_probs=None, echo=False):
    #     """
    #     Return the probability of the label given the prompt for the given model.

    #     Parameters
    #     ----------
    #     prompt : str
    #         The prompt to be completed.
    #     label : str
    #         The label to be predicted.
    #     model : str, optional
    #         The model to be used. The default is "gpt2".

    #     Returns
    #     -------
    #     float
    #         The probability P(label|prompt) of the label given the prompt.

    #     """
        
    #     ## TODO: Sumar las logprobs de cada token que conforma el label.
    #     # Es decir, si el label es "politics", calcular logP(" politics"|prompt) = logP(" po"|prompt) + logP("litics"|prompt," po")
        
    #     ''' This function runs GPT-2 locally but places the outputs into an json that looks just like the one
    #     provided by the OpenAI API. '''
    #     if isinstance(prompt, str):
    #         prompt = [prompt] # the code below assumes a list
    #     input_ids = self.tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
        
    #     # greedily generate l tokens
    #     if l > 0:
    #         # the generate function can handle left padded inputs automatically in HF
    #         # total_sequences is now the input + possible generated output
    #         # total_sequences = self.model.generate(input_ids=input_ids['input_ids'].cuda(), attention_mask=input_ids['attention_mask'].cuda(), max_length=l + len(input_ids['input_ids'][0]), do_sample=False)
    #         total_sequences = self.model.generate(input_ids=input_ids['input_ids'], attention_mask=input_ids['attention_mask'], max_length=l + len(input_ids['input_ids'][0]), do_sample=False)
    #     else:
    #         assert echo == True and l == 0
    #         total_sequences = input_ids['input_ids']#.cuda()

    #     # they want the probs of the top tokens
    #     if num_log_probs is not None:
    #         # we are left padding, so we need to adjust the position IDs
    #         attention_mask = (total_sequences != 50256).float()
    #         position_ids = attention_mask.long().cumsum(-1) - 1
    #         position_ids.masked_fill_(attention_mask == 0, 1)
    #         # get the logits for the context and the next l tokens
    #         logits = self.model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits.detach().cpu()
    #         if not echo:
    #             # get the top tokens and probs for the generated l tokens
    #             probs = torch.softmax(logits[:,-l-1:], dim=2).cpu()
    #         else:
    #             # get the top tokens and probs for the context and the generated l tokens
    #             probs = torch.softmax(logits, dim=2).cpu()
    #         top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
    #         logprobs = torch.log(probs)
    #         top_log_probs = torch.log(top_probs)

    #     # create the return value to resemble OpenAI
    #     return_json = {}
    #     choices = []
    #     for batch_id in range(len(prompt)):
    #         curr_json = {}
    #         # text is just the optional context and next l tokens
    #         if not echo:
    #             curr_json['text'] = self.tokenizer.decode(total_sequences[batch_id][-l:], skip_special_tokens=True)
    #         else:
    #             curr_json['text'] = self.tokenizer.decode(total_sequences[batch_id], skip_special_tokens=True)

    #         # fill the return json with the top tokens and probs to match the OpenAI return value.
    #         if num_log_probs is not None:
    #             curr_json['logprobs'] = {}
    #             curr_json['logprobs']['top_logprobs'] = []
    #             curr_json['logprobs']['token_logprobs'] = []
    #             curr_json['logprobs']['tokens'] = []
    #             if not echo:
    #                 # cutoff the -1 here because the probs are shifted one over for LMs
    #                 for current_element_top_log_probs, current_element_top_tokens in zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1]):
    #                     # tokens is a list of the top token at each position
    #                     curr_json['logprobs']['tokens'].append(self.tokenizer.decode([current_element_top_tokens[0]]))
    #                     # token_logprobs is a list of the logprob of the top token at each position
    #                     curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
    #                     # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
    #                     temp = {}
    #                     for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
    #                         temp[self.tokenizer.decode(token.item())] = log_prob.item()
    #                     curr_json['logprobs']['top_logprobs'].append(temp)
    #             else:
    #                 # same as not above but small tweaks
    #                 # we add null to the front because for the GPT models, they have null probability for the first token
    #                 # (for some reason they don't have an beginning of sentence token)
    #                 curr_json['logprobs']['top_logprobs'].append('null')
    #                 # cutoff the -1 here because the probs are shifted one over for LMs
    #                 for index, (current_element_top_log_probs, current_element_top_tokens) in enumerate(zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1])):
    #                     # skip padding tokens
    #                     if total_sequences[batch_id][index].item() == 50256:
    #                         continue
    #                     temp = {}
    #                     for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
    #                         temp[self.tokenizer.decode(token.item())] = log_prob.item()
    #                     curr_json['logprobs']['top_logprobs'].append(temp)
    #                 for index in range(len(probs[batch_id])):
    #                     curr_json['logprobs']['tokens'].append(self.tokenizer.decode([total_sequences[batch_id][index]]))
    #                 curr_json['logprobs']['token_logprobs'].append('null')
    #                 for index, log_probs_token_position_j in enumerate(logprobs[batch_id][:-1]):
    #                     # probs are left shifted for LMs 
    #                     curr_json['logprobs']['token_logprobs'].append(log_probs_token_position_j[total_sequences[batch_id][index+1]])

    #         choices.append(curr_json)
    #     return_json['choices'] = choices
    #     return return_json
    

# def construct_prompt(params, train_sentences, train_labels, test_sentence):
#     """construct a single prompt to be fed into the model"""
#     # special case when the user defines a custom prompt function. 
#     if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
#         return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

#     # take the prompt template and fill in the training and test example
#     prompt = params["prompt_prefix"]
#     q_prefix = params["q_prefix"]
#     a_prefix = params["a_prefix"]
#     for s, l in zip(train_sentences, train_labels):
#         prompt += q_prefix
#         prompt += s + "\n"
#         if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
#             assert params['task_format'] == 'classification'
#             l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
#         else:
#             assert isinstance(l, str) # string labels
#             assert params['task_format'] == 'qa'
#             l_str = l

#         prompt += a_prefix
#         prompt += l_str + "\n\n"

#     prompt += q_prefix
#     prompt += test_sentence + "\n"
#     assert a_prefix[-1] == ' '
#     prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
#     return prompt


# if __name__ == "__main__":
#     from .data import ClassificationDataset
#     dataset = ClassificationDataset(
#         "./",
#         "sst2",
#         n_shot=4,
#         random_state=10137
#     )
#     lm = HFLanguageModel(root_dir="./", model_name="gpt2-xl")
#     params = {}
#     params['prompt_prefix'] = ""
#     params["q_prefix"] = "Review: "
#     params["a_prefix"] = "Sentiment: "
#     params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
#     params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
#     params['task_format'] = 'classification'
#     params['num_tokens_to_predict'] = 1
#     train_sentences = [
#         'peppered with witty dialogue and inventive moments .',
#         'a quasi-documentary by french filmmaker karim dridi that celebrates the hardy spirit of cuban music .',
#         "despite the film 's shortcomings , the stories are quietly moving .",
#         "... has about 3\\/4th the fun of its spry 2001 predecessor -- but it 's a rushed , slapdash , sequel-for-the-sake - of-a-sequel with less than half the plot and ingenuity ."
#     ]
#     train_labels = [1, 1, 1, 0]
#     test_sentence = "smith 's point is simple and obvious -- people 's homes are extensions of themselves , and particularly eccentric people have particularly eccentric living spaces -- but his subjects are charmers ."
#     gold_labels = {"Positive": -0.3488830626010895, 'Negative': -1.7250447273254395}
#     prompt = construct_prompt(params, train_sentences, train_labels, test_sentence)
#     for batch in dataset.random_batch_loader_from_split(split="test", num_samples=None, batch_size=1):
#         test = [dataset.construct_prompt_with_train_shots("smith 's point is simple and obvious -- people 's homes are extensions of themselves , and particularly eccentric people have particularly eccentric living spaces -- but his subjects are charmers .")]
#         if batch["prompt"] == test:
#             print(prompt)
#             print("-----------------------------------------------")
#             print(batch["prompt"][0])
#             break
#     # pred_labels = {label: lm.get_label_probs2(prompt, label=[], l=1, num_log_probs=100, echo=True)["choices"][0]["logprobs"]["top_logprobs"][-1][" " + label] for label in gold_labels.keys()}
#     pred_labels = {label: lm.get_label_probs([prompt], labels_dict={0: ["Positive"], 1: ["Negative"]}) for label in gold_labels.keys()}
#     for label, pred in pred_labels.items():
#         print(f"Pred {label}: {pred}")
#         print(f"True {label}: {gold_labels[label]}")
