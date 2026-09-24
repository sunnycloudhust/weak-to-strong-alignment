CONFIG = {
    "dataset_name": "Anthropic/hh-rlhf",
    "dataset_config": None,
    "base_model_names": [
    
    # Qwen
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",

    # Llama
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",

    # Phi
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3-mini-4k-instruct",

    # Mistral
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.2",

    # Gemma
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",

    # DeepSeek
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ],
    "reward_model_name": "sunnycloudhust/Qwen2.5-0.5B-Instruct-Reward-Model",
    "output_dir": "outputs/reward_model_hh_rlhf",
    "reward_model_hf_repo": "sunnycloudhust/Qwen2.5-0.5B-Instruct-Reward-Model",
    "experiment_output_dir": "outputs/test_time_alignment",
    "gpu_ids": [0,1],
    "max_length": 256,
    "epochs": 5,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "batch_size": 8,
    "gradient_accumulation_steps": 2,
    "eval_ratio": 0.10,
    "test_ratio": 0.10,
    "max_train_samples": 20000,
    "max_eval_samples": 1000,
    "max_test_samples": 1000,
    "num_candidates": [1, 2, 4, 8],
    "generation_max_new_tokens": 128,
    "generation_temperature": 0.8,
    "generation_top_p": 0.95,
}
