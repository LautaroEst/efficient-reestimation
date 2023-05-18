from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MODELS = ["gpt2", "gpt2-xl"]

def main():

    cwd = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(cwd)
    models_dir = os.path.join(parent_dir, "models")
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    
    for model_name in MODELS:
        this_model_path = os.path.join(models_dir,model_name.replace("/","_"))
        model = AutoModelForCausalLM.from_pretrained(f"{model_name}", cache_dir=this_model_path)
        model.save_pretrained(this_model_path,max_shard_size="500MB")
        tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
        tokenizer.save_pretrained(this_model_path)


if __name__ == "__main__":
    main()