
import os
from src.models import create_model, create_optimizer
from src.utils import parse_finetuning_args
from src.data import prepare_training_dataloader
from torch.utils.tensorboard import SummaryWriter
from accelerate import cpu_offload, hooks, dispatch_model, infer_auto_device_map

from tqdm import tqdm


def main():
    root_dir, config, config_name, use_saved_results = parse_finetuning_args()

    if not os.path.exists(os.path.join(root_dir,"results/finetuning",config_name)):
        os.makedirs(os.path.join(root_dir,"results/finetuning",config_name))
    writer = SummaryWriter(log_dir=os.path.join(root_dir,"results/finetuning",config_name))
    
    # Instantiate model
    model = create_model(root_dir, model=config["model"], max_memory=config["max_memory"])
    model.model.train()

    # Prepare training data
    training_dataloader = prepare_training_dataloader(
        root_dir, 
        model, 
        config["num_samples"], 
        batch_size=1, 
        dataset=config["dataset"],
        random_state=config["random_state"]
    )

    # Prepare optimizer
    optimizer = create_optimizer(model, config["layers_to_train"], learning_rate=config["learning_rate"])

    # Train
    batch_accumulation = config["batch_size"]
    epochs_pbar = tqdm(range(config["epochs"]),desc="Epochs",leave=False)
    for epoch in epochs_pbar:
        training_pbar = tqdm(training_dataloader,desc="Batch",leave=False)
        optimizer.zero_grad()
        for i, batch in enumerate(training_pbar):
            loss = model.model(**batch,output_attentions=False,output_hidden_states=False).loss
            loss = loss / batch_accumulation
            loss.backward()
            training_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if (i + 1) % batch_accumulation == 0 or i == len(training_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                writer.add_scalar("Loss/train", loss.item(), epoch * len(training_dataloader) + i)
    writer.flush()
    writer.close()

    # Save model
    if not os.path.exists(f"{root_dir}/models/{model.model_name}_{config_name}"):
        os.makedirs(f"{root_dir}/models/{model.model_name}_{config_name}")
    hooks.remove_hook_from_submodules(model.model)
    model.model = dispatch_model(model.model,device_map=infer_auto_device_map(model.model, max_memory={"cpu": "30GiB"}))
    model.model.save_pretrained(f"{root_dir}/models/{model.model_name}_{config_name}",max_shard_size="500MB")
    model.tokenizer.save_pretrained(f"{root_dir}/models/{model.model_name}_{config_name}")
    







if __name__ == "__main__":
    main()