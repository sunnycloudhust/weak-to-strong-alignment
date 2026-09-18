from pathlib import Path

import torch
from tqdm.auto import tqdm
from loss import pairwise_preference_loss, score


def train(model, train_loader, eval_loader, optimizer, device, config, tokenizer, checkpoint_dir):
    history = []
    accumulation_steps = config["gradient_accumulation_steps"]
    checkpoint_dir = Path(checkpoint_dir)

    for epoch in range(config["epochs"]):
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_pairs = 0
        optimizer.zero_grad(set_to_none=True)

        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config['epochs']} train",
            unit="batch",
        )
        for step, batch in enumerate(train_progress):
            chosen = {key: value.to(device) for key, value in batch["chosen"].items()}
            rejected = {key: value.to(device) for key, value in batch["rejected"].items()}
            loss, margins = pairwise_preference_loss(
                score(model, chosen), score(model, rejected)
            )
            (loss / accumulation_steps).backward()
            if (step + 1) % accumulation_steps == 0 or step + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            pair_count = len(margins)
            train_loss_total += loss.item() * pair_count
            train_correct += (margins > 0).sum().item()
            train_pairs += pair_count

        model.eval()
        eval_loss_total = 0.0
        eval_correct = 0
        eval_pairs = 0
        with torch.no_grad():
            eval_progress = tqdm(
                eval_loader,
                desc=f"Epoch {epoch + 1}/{config['epochs']} eval",
                unit="batch",
            )
            for step, batch in enumerate(eval_progress):
                chosen = {key: value.to(device) for key, value in batch["chosen"].items()}
                rejected = {key: value.to(device) for key, value in batch["rejected"].items()}
                loss, margins = pairwise_preference_loss(
                    score(model, chosen), score(model, rejected)
                )
                pair_count = len(margins)
                eval_loss_total += loss.item() * pair_count
                eval_correct += (margins > 0).sum().item()
                eval_pairs += pair_count


        train_metrics = {
            "loss": train_loss_total / train_pairs,
            "accuracy": train_correct / train_pairs,
        }
        eval_metrics = {
            "loss": eval_loss_total / eval_pairs,
            "accuracy": eval_correct / eval_pairs,
        }
        metrics = {"epoch": epoch + 1, "train": train_metrics, "eval": eval_metrics}
        history.append(metrics)
        print(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_accuracy={train_metrics['accuracy']:.4f} | "
            f"eval_loss={eval_metrics['loss']:.4f} | "
            f"eval_accuracy={eval_metrics['accuracy']:.4f}"
        )

        epoch_checkpoint_dir = checkpoint_dir / f"epoch_{epoch + 1}"
        epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = (
            model.module if isinstance(model, torch.nn.DataParallel) else model
        )
        model_to_save.save_pretrained(epoch_checkpoint_dir)
        tokenizer.save_pretrained(epoch_checkpoint_dir)
        print(f"Checkpoint saved to {epoch_checkpoint_dir}")

    return history
