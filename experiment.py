import argparse
import json
from pathlib import Path
from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import AutoTokenizer, BitsAndBytesConfig

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


def load_base_model(model_name, device, quantized=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if quantized:
        if device.type != "cuda":
            raise ValueError("Quantized base models require a CUDA device")
        model_kwargs.update(
            {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                ),
                "device_map": {"": device.index or 0},
            }
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not quantized:
        model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_base_model(
    config,
    evaluation,
    reward_model,
    reward_tokenizer,
    base_model_name,
    device,
    quantized=True,
):
    base_model, base_tokenizer = load_base_model(
        base_model_name, device, quantized
    )

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
    quantized=True,
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_model_dir = output_path.parent / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    experiments = {}
    for model_name in base_model_names:
        experiment = evaluate_base_model(
            config,
            evaluation,
            reward_model,
            reward_tokenizer,
            model_name,
            device,
            quantized,
        )
        experiments[model_name] = experiment
        safe_model_name = model_name.replace("/", "__")
        per_model_path = per_model_dir / f"{safe_model_name}.json"
        per_model_summary = {
            "reward_model": str(reward_model_path),
            "device": str(device),
            "quantized": quantized,
            "experiment": experiment,
        }
        per_model_path.write_text(
            json.dumps(per_model_summary, indent=2) + "\n"
        )
        print(f"Base-model result saved to {per_model_path}")

    summary = {
        "reward_model": str(reward_model_path),
        "device": str(device),
        "experiments": experiments,
    }
    if len(experiments) == 1:
        summary.update(next(iter(experiments.values())))
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Experiment results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run best-of-N test-time alignment.")
    parser.add_argument("--reward-model", default=None)
    parser.add_argument("--base-model", nargs="+", default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--no-quantized",
        action="store_true",
        help="Disable the default 4-bit NF4 base-model loading (CUDA only).",
    )
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
        not args.no_quantized,
    )