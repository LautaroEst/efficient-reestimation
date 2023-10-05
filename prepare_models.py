import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    models_dir = os.path.join(args.root_directory, "models")
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    
    model_name = args.model
    model_dir = os.path.join(models_dir,model_name.replace("/","--"))
    model = AutoModelForCausalLM.from_pretrained(f"{model_name}", cache_dir=model_dir)
    model.save_pretrained(model_dir,max_shard_size="500MB")
    tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
    tokenizer.save_pretrained(model_dir)


if __name__ == "__main__":
    main()