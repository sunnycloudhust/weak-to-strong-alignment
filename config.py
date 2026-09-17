CONFIG = {
    "dataset_name": "Anthropic/hh-rlhf",
    "dataset_config": None,
    "base_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "base_model_names": [
        # Qwen
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        # TinyLlama
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # SmolLM2
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        # Phi
        "microsoft/Phi-3.5-mini-instruct",
        # Mistral
        "mistralai/Mistral-7B-Instruct-v0.3",
        # Gemma
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        # Falcon
        "tiiuae/Falcon3-7B-Instruct",
    ],
    "reward_model_name": "sunnycloudhust/Qwen2.5-0.5B-Instruct-Reward-Model",
    "output_dir": "outputs/reward_model_hh_rlhf",
    "reward_model_hf_repo": "sunnycloudhust/Qwen2.5-0.5B-Instruct-Reward-Model",
    "experiment_output_dir": "outputs/test_time_alignment",
    # Leave empty to use the first available GPU; set [0, 1] for multi-GPU.
    "gpu_ids": [],
    "max_length": 256,
    "epochs": 2,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "eval_ratio": 0.10,
    "test_ratio": 0.10,
    "max_train_samples": 25000,
    "max_eval_samples": 100,
    "max_test_samples": 100,
    "num_candidates": [1, 2, 4, 8],
    "generation_max_new_tokens": 128,
    "generation_temperature": 0.8,
    "generation_top_p": 0.95,
}
