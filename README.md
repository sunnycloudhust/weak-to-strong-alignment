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


## Best-of-N Test-Time Alignment

The experiment does not train a reward model. It uses the reward model repository
from `config.py` to score responses, and Transformers downloads it automatically
from Hugging Face on the first run. No manual clone is required.

Run all base models listed in `config.py` with:

```bash
./run_experiments.sh
```

The number of prompts is controlled by `max_test_samples` in `config.py`, and
the base model list is controlled by `base_model_names` in the same file.
Results are written to `outputs/test_time_alignment/results.json`.

```bash
REWARD_MODEL=/path/to/local/reward_model ./run_experiments.sh
```

The optional `REWARD_MODEL` environment variable can point to a local reward
model checkpoint instead of the Hugging Face repository. The experiment
generates candidate responses with each base model, scores them with the reward
model, and selects the highest-scoring response for `N = 1, 2, 4, 8`.

All base models are loaded in 4-bit NF4 quantization on CUDA by default:

```bash
./run_experiments.sh
```

Quantized base-model loading is CUDA-only and requires `accelerate` and `bitsandbytes`. It changes base-model loading only; the reward model remains in its normal precision. Use `--no-quantized` to disable it when running a non-CUDA test.

Verified test-time alignment results:

- Base models: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, and `Qwen/Qwen2.5-7B-Instruct`
- Reward model: `sunnycloudhust/Qwen2.5-0.5B-Instruct-Reward-Model`
- Device: `cuda:0`
- Number of prompts: `1000`
- Candidate counts: `1, 2, 4, 8`

| Base model | Baseline mean reward | N = 1 selected reward | N = 2 selected reward | N = 4 selected reward | N = 8 selected reward |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 0.1448 | -1.0837 | 1.2118 | 2.3467 | 3.3858 |
| Qwen2.5-1.5B-Instruct | 1.8874 | 0.2070 | 1.5848 | 2.4772 | 3.3069 |
| Qwen2.5-3B-Instruct | 2.0674 | 1.7284 | 2.6216 | 3.5232 | 4.3100 |
| Qwen2.5-7B-Instruct | 1.5498 | 1.5615 | 2.7051 | 3.4168 | 4.2600 |

This measures reward-model selection rather than independent response quality. Use human evaluation or a fixed external judge to estimate win rate; the reward model itself should not be treated as ground truth.

