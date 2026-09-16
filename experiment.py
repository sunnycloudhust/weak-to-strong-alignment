import argparse
import json
from pathlib import Path
from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import AutoTokenizer

from config import CONFIG
from data import load_preference_pairs


def extract_prompt(conversation):
    marker = "\n\nAssistant:"
    if marker not in conversation:
        return conversation
    return conversation.rsplit(marker, 1)[0] + marker


def score_texts(model, tokenizer, texts, device, max_length):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        return model(**inputs).logits.squeeze(-1).float().cpu().tolist()


def generate_candidates(model, tokenizer, prompt, count, config, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generation_kwargs = {
        "max_new_tokens": config["generation_max_new_tokens"],
        "pad_token_id": tokenizer.pad_token_id,
        "num_return_sequences": count,
    }
    if count == 1:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": config["generation_temperature"],
                "top_p": config["generation_top_p"],
            }
        )
    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)
    prompt_length = inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(
        generated[:, prompt_length:], skip_special_tokens=True
    )


def evaluate_base_model(
    config,
    evaluation,
    reward_model,
    reward_tokenizer,
    base_model_name,
    device,
):
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    base_model.to(device)
    base_model.eval()

    results = []
    for index, example in enumerate(evaluation):
        prompt = extract_prompt(example["chosen"])
        baseline_response = generate_candidates(
            base_model, base_tokenizer, prompt, 1, config, device
        )[0]
        baseline_text = prompt + baseline_response
        baseline_reward = score_texts(
            reward_model,
            reward_tokenizer,
            [baseline_text],
            device,
            config["max_length"],
        )[0]
        max_candidates = max(config["num_candidates"])
        all_responses = generate_candidates(
            base_model, base_tokenizer, prompt, max_candidates, config, device
        )
        candidate_data = {}
        for count in config["num_candidates"]:
            responses = all_responses[:count]
            rewards = score_texts(
                reward_model,
                reward_tokenizer,
                [prompt + response for response in responses],
                device,
                config["max_length"],
            )
            best_index = max(range(len(rewards)), key=rewards.__getitem__)
            candidate_data[str(count)] = {
                "responses": responses,
                "rewards": rewards,
                "selected_response": responses[best_index],
                "selected_reward": rewards[best_index],
            }
        results.append(
            {
                "prompt": prompt,
                "reference_chosen": example["chosen"],
                "baseline_response": baseline_response,
                "baseline_reward": baseline_reward,
                "by_num_candidates": candidate_data,
            }
        )
        print(
            f"{base_model_name}: processed {index + 1}/{len(evaluation)} prompts"
        )

    aggregate = {
        "baseline_reward_mean": sum(
            result["baseline_reward"] for result in results
        ) / max(len(results), 1),
        "by_num_candidates": {},
    }
    for count in config["num_candidates"]:
        selected_rewards = [
            result["by_num_candidates"][str(count)]["selected_reward"]
            for result in results
        ]
        aggregate["by_num_candidates"][str(count)] = {
            "selected_reward_mean": sum(selected_rewards)
            / max(len(selected_rewards), 1),
        }

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "base_model": base_model_name,
        "num_prompts": len(results),
        "num_candidates": config["num_candidates"],
        "aggregate": aggregate,
        "results": results,
    }


def run_experiment(
    config,
    reward_model_path,
    base_model_names,
    max_prompts,
    output_path,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(range(min(torch.cuda.device_count(), 2))) if torch.cuda.is_available() else []
    _, _, evaluation = load_preference_pairs(config)
    if max_prompts is not None:
        evaluation = evaluation.select(range(min(max_prompts, len(evaluation))))

    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path
    )
    reward_model.to(device)
    if len(gpu_ids) > 1:
        reward_model = torch.nn.DataParallel(reward_model, device_ids=gpu_ids)
    reward_model.eval()

    if isinstance(base_model_names, str):
        base_model_names = [base_model_names]
    if not isinstance(base_model_names, Sequence) or not base_model_names:
        raise ValueError("base_model_names must contain at least one model name")
    experiments = {
        model_name: evaluate_base_model(
            config,
            evaluation,
            reward_model,
            reward_tokenizer,
            model_name,
            device,
        )
        for model_name in base_model_names
    }

    summary = {
        "reward_model": str(reward_model_path),
        "device": str(device),
        "experiments": experiments,
    }
    if len(experiments) == 1:
        summary.update(next(iter(experiments.values())))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Experiment results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run best-of-N test-time alignment.")
    parser.add_argument("--reward-model", default=None)
    parser.add_argument("--base-model", nargs="+", default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reward_model_ref = CONFIG.get("reward_model_hf_repo", CONFIG["output_dir"])
    base_model_names = args.base_model or CONFIG.get(
        "base_model_names", [CONFIG["base_model_name"]]
    )
    run_experiment(
        CONFIG,
        args.reward_model or reward_model_ref,
        base_model_names,
        args.max_prompts if args.max_prompts is not None else CONFIG["max_test_samples"],
        args.output
        or Path(CONFIG["experiment_output_dir"]) / "results.json",
    )