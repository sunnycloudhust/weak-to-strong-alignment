from datasets import load_dataset
from torch.utils.data import DataLoader

def load_preference_pairs(config):
    """
        This function loads the dataset and devides into train/val/test
    """
    
    dataset = load_dataset(config["dataset_name"], config["dataset_config"])
    train = dataset["train"]
    required = {"chosen", "rejected"}
    if not required.issubset(train.column_names):
        raise ValueError(f"Dataset must contain {sorted(required)}; got {train.column_names}")

    heldout_ratio = config["eval_ratio"] + config["test_ratio"]
    split = train.train_test_split(test_size=heldout_ratio)
    heldout = split["test"].train_test_split(
        test_size=config["test_ratio"] / heldout_ratio,
    )
    train, evaluation, test = (
        split["train"],
        heldout["train"],
        heldout["test"],
    )
    # Check number of samples (not necessary)
    if config["max_train_samples"]:
        train = train.select(range(min(config["max_train_samples"], len(train))))
    if config["max_eval_samples"]:
        evaluation = evaluation.select(range(min(config["max_eval_samples"], len(evaluation))))
    if config["max_test_samples"]:
        test = test.select(range(min(config["max_test_samples"], len(test))))
    return train, evaluation, test


def tokenize_example(example, tokenizer, max_length):
    chosen = tokenizer(example["chosen"], truncation=True, max_length=max_length)
    rejected = tokenizer(example["rejected"], truncation=True, max_length=max_length)
    
    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
    }


def tokenize_pairs(dataset, tokenizer, max_length):
    return dataset.map(
        tokenize_example,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        remove_columns=dataset.column_names,
    )


def pad_batch(batch, tokenizer, prefix):
    return tokenizer.pad(
        {
            "input_ids": [item[f"{prefix}_input_ids"] for item in batch],
            "attention_mask": [item[f"{prefix}_attention_mask"] for item in batch],
        },
        return_tensors="pt",
    )


def collate_pairs(batch, tokenizer):
    chosen = pad_batch(batch, tokenizer, "chosen")
    rejected = pad_batch(batch, tokenizer, "rejected")
    return {"chosen": chosen, "rejected": rejected}


class PairCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        return collate_pairs(batch, self.tokenizer)


def make_loaders(
    train,
    evaluation,
    tokenizer,
    batch_size,
    num_workers=0,
    pin_memory=False,
):
    collate = PairCollator(tokenizer)
    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(train, shuffle=True, **loader_kwargs)
    eval_loader = DataLoader(evaluation, shuffle=False, **loader_kwargs)
    return train_loader, eval_loader
