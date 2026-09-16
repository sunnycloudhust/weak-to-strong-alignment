import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import CONFIG
from data import load_preference_pairs, make_loaders, tokenize_pairs
from train import train


def select_device(config):
    requested = config.get("device", "auto").lower()
    if requested not in {"auto", "cpu", "cuda", "tpu"}:
        raise ValueError("config['device'] must be one of: auto, cpu, cuda, tpu")

    xla_available = False
    if requested in {"auto", "tpu"}:
        try:
            import torch_xla.core.xla_model as xm

            xla_available = xm.xla_device_hw() == "TPU"
        except (ImportError, RuntimeError):
            if requested == "tpu":
                raise RuntimeError(
                    "TPU requested but torch-xla is unavailable or no TPU runtime "
                    "was detected. Install the torch-xla version matching PyTorch."
                )

    if xla_available:
        import torch_xla.core.xla_model as xm

        return xm.xla_device(), True
    if requested == "tpu":
        raise RuntimeError("TPU requested but no TPU runtime was detected.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available.")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), False


def main():
    config = CONFIG
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting reward-model training with {config['reward_model_name']}")

    train_dataset, eval_dataset, _ = load_preference_pairs(config)
    tokenizer = AutoTokenizer.from_pretrained(config["reward_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = tokenize_pairs(train_dataset, tokenizer, config["max_length"])
    eval_dataset = tokenize_pairs(eval_dataset, tokenizer, config["max_length"])
    train_loader, eval_loader = make_loaders(
        train_dataset,
        eval_dataset,
        tokenizer,
        config["batch_size"],
        num_workers=config.get("dataloader_num_workers", 0),
        pin_memory=False,
    )

    device, is_tpu = select_device(config)
    gpu_ids = config["gpu_ids"] if device.type == "cuda" else []
    if gpu_ids and max(gpu_ids) >= torch.cuda.device_count():
        raise ValueError(
            f"Requested GPU ids {gpu_ids}, but only {torch.cuda.device_count()} "
            "CUDA device(s) are available."
        )
    device_name = "tpu" if is_tpu else (gpu_ids or device.type)
    print(
        f"Using device={device}, accelerator={device_name}, "
        f"train_pairs={len(train_dataset)}, "
        f"eval_pairs={len(eval_dataset)}"
    )
    
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config["reward_model_name"], num_labels=1
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    history = train(
        model,
        train_loader,
        eval_loader,
        optimizer,
        device,
        config,
        is_tpu=is_tpu,
    )

    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    if is_tpu:
        model_to_save = model_to_save.cpu()
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        "dataset": config["dataset_name"],
        "model": config["reward_model_name"],
        "device": "tpu" if is_tpu else str(device),
        "train_pairs": len(train_dataset),
        "eval_pairs": len(eval_dataset),
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Training complete; metrics saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()