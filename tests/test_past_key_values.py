

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt2")
    model = GPT2LMHeadModel.from_pretrained("./models/gpt2")
    model.eval()

    prompt = ["Hello, my name is"]
    encoded_input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    print(encoded_input["input_ids"])
    output = model(**encoded_input, use_cache=True, output_attentions=True, output_hidden_states=False)
    seq_len = encoded_input["input_ids"].shape[1]
    print(output.logits[:,:,:10])

    prompt = [" Lautaro Estienne"]
    encoded_input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    print(encoded_input["input_ids"])
    encoded_input["attention_mask"] = torch.cat((encoded_input["attention_mask"],torch.ones((1,seq_len),dtype=torch.long,device=encoded_input["attention_mask"].device)),dim=1)
    output = model(**encoded_input, past_key_values=output.past_key_values, output_attentions=False, output_hidden_states=False)
    print(output.logits[:,:,:10])

    prompt = ["Hello, my name is Lautaro Estienne"]
    encoded_input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    print(encoded_input["input_ids"])
    output = model(**encoded_input, output_attentions=False, output_hidden_states=False)
    print(output.logits[:,:,:10])






if __name__ == "__main__":
    main()