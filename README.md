# Weak-to-strong generalization

We train Reward Model `Qwen/Qwen2.5-0.5B-Instruct`on dataset `Anthropic/hh-rlhf`

## Reward Model Results

The current reward-model checkpoint was trained from `Qwen/Qwen2.5-0.5B-Instruct` on `Anthropic/hh-rlhf` using 25,000 training pairs, 1,000 evaluation pairs, and 4 epochs. The latest metrics stored in `outputs/reward_model_hh_rlhf/metrics.json` were extended with the old epoch-1 and epoch-2 records for continuity of the documented model card history.

| Epoch | Train loss | Train accuracy | Eval loss | Eval accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.696 | 48.510% | 0.639 | 51.980% |
| 2 | 0.583 | 58.000% | 0.646 | 52.770% |
| 3 | 0.567 | 59.516% | 0.568 | 59.900% |
| 4 | 0.230 | 78.348% | 0.813 | 60.100% |

Accuracy is the fraction of pairs for which the model scores `chosen` higher than `rejected`. 

The checkpoint and its Model Card are available on [Hugging Face](https://huggingface.co/sunnycloudhust/Qwen2.5-0.5B-Instruct-Reward-Model). The raw metrics are stored in `outputs/reward_model_hh_rlhf/metrics.json`.

## Installation

Python 3.10+ and a CUDA-capable GPU are recommended for full training.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Train the Reward Model

The reward model uses pairwise preference loss:

`-log(sigmoid(score(chosen) - score(rejected)))`

The `Anthropic/hh-rlhf` dataset already provides `chosen` and `rejected` columns, so no additional normalization is required.

All training settings are defined in `config.py`. The default configuration uses a maximum sequence length of 256, batch size 2, gradient accumulation, and two GPUs through `DataParallel`.

```bash
python main.py
```

The script saves the model, tokenizer, and aggregate metrics to `outputs/reward_model_hh_rlhf/`. For a smaller smoke run, set `max_train_samples`, `max_eval_samples`, and `max_test_samples` in `config.py` before starting training.

## Best-of-N Test-Time Alignment

After training a reward model, run:

```bash
python experiment.py \
  --reward-model outputs/reward_model_hh_rlhf \
  --max-prompts 100 \
  --output outputs/test_time_alignment/results.json
```

The experiment generates candidate responses with a base model, scores them with the reward model, and selects the highest-scoring response for `N = 1, 2, 4`. The output contains prompts, candidates, scores, baseline responses, and selected responses.

Verified test-time alignment runs from `outputs/test_time_alignment/results.json`, `outputs/test_time_alignment/results 2.json`, and the latest `outputs/test_time_alignment/results.txt`:

- Base models: `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, and `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Reward model: `sunnycloudhust/Qwen2.5-0.5B-Instruct-Reward-Model`
- Device: `cuda:0`
- Number of prompts: `1000`
- Candidate counts: `1, 2, 4, 8`

| Base model | Baseline mean reward | N = 1 selected reward | N = 2 selected reward | N = 4 selected reward | N = 8 selected reward |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 2.7313 | 1.9785 | 3.0605 | 3.8488 | 4.4574 |
| Qwen2.5-3B-Instruct | 2.1727 | 1.9026 | 2.8844 | 3.6344 | 4.2196 |
| TinyLlama-1.1B-Chat-v1.0 | 0.9657 | 0.4764 | 1.5255 | 2.3693 | 3.1039 |

This measures reward-model selection rather than independent response quality. Use human evaluation or a fixed external judge to estimate win rate; the reward model itself should not be treated as ground truth.


