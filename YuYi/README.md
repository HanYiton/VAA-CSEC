# YuYi — VAA-CSEC

This directory contains the reference implementation for

> **VAA-CSEC: Vote-guided Advantage Allocation for Chinese Semantic Error Correction**
>
> Yitong Han, Nankai Lin†, Juan Luo, Hongyan Wu, Lianxi Wang, Shengyi Jiang
>
> † corresponding author

It provides the task-specific reward function, the **Group-Level Relative Policy Optimization (GLPO)** plugin for ms-swift, the SFT/RL configuration files, and the ChERRANT-based inference & evaluation scripts used in the paper.

## Abstract

Chinese Semantic Error Correction (CSEC) targets semantic errors in Chinese text, which are typically more subtle and complex than spelling and grammatical errors. Existing LLM-based approaches face two recurring obstacles: **over-correction**, and an **unclear interaction between Chain-of-Thought (CoT) reasoning and self-consistency decoding**. We propose **VAA-CSEC**, a multi-stage framework combining CoT distillation, Supervised Fine-Tuning (SFT), Reinforcement Learning (RL), and self-consistency decoding. During RL we design a task-specific reward function directly aligned with the minimal-editing principle of CSEC, and introduce **GLPO**, which reallocates GRPO advantages according to the margin between individual rollout rewards and the vote-aggregated group reward. On CSED-C and NaSGEC-Exam, VAA-CSEC reaches **47.72% F0.5** and **41.55% F0.5** under 32-vote decoding, respectively.

## Method

### Pipeline

1. **CoT distillation.** Multi-reference training pairs are unfolded into single-reference pairs. For each pair, Qwen3.5-27B generates a reasoning rationale and DeepSeek-V3.2 verifies it (up to three regeneration attempts).
2. **SFT.** Qwen3.5-4B is fine-tuned with LoRA via LLaMA-Factory to produce outputs in the structured `<think>...</think><answer>...</answer>` format.
3. **RL (GLPO).** The policy is optimized with a group-level relative policy objective on top of GRPO, using a CSEC-specific reward (below).
4. **Self-consistency decoding.** At inference, `N` stochastic generations (temperature `1.0`) are aggregated by majority voting.

### Reward function

The reward has two components: a format reward `R_f` and a correctness reward `R_c`.

- `R_f = 0.3` if the output contains exactly one `<think>` block and one `<answer>` block, and `0` otherwise.
- `R_c` is a piecewise function over five mutually exclusive cases, inspired by the F0.5 scoring scheme.

Let `d_sr`, `d_sp`, and `d_pr` be the edit distances between source→reference, source→prediction, and prediction→reference, respectively. The number of useful edits is

```text
u = (d_sr + d_sp - d_pr) / 2
```

with edit-based precision `P = min(u / d_sp, 1)`, recall `R̂ = min(u / d_sr, 1)`, and

```text
F0.5 = 1.25 * P * R̂ / (0.25 * P + R̂)
```

| Case | Correctness reward `R_c` |
| --- | --- |
| Exact match (`y == y*`) | `+3.4` |
| Effective & minimal (`u > 0`, `d_sp <= d_sr`) | `2.0 * F0.5` |
| Effective & over-corrects (`u > 0`, `d_sp > d_sr`) | `2.0 * F0.5 - 0.8 * min(ρ, 2)`, with `ρ = (d_sp - d_sr) / d_sr` |
| No effective edit (`u == 0`) | `0.0` |
| Degrades the source (`u < 0`) | `-1.5 * min(-R̂, 1)` |

`R_c` is clipped to `[-1.5, 3.4]`. The total training reward is `R = R_f + R_c`.

The implementation lives in [`src/rewards/zh_grammar_reward.py`](src/rewards/zh_grammar_reward.py) and registers the ms-swift ORM `zh_gec_reward`.

### GLPO

Standard GRPO normalizes advantages within each rollout group:

```text
A^{GRPO}_{i,j} = (r_{i,j} - r̄_{G_i}) / σ_{G_i}
```

GLPO additionally adds a one-sided **vote-margin bonus**: a rollout whose individual reward exceeds the vote-aggregated group reward `R(G_i)` receives extra positive advantage proportional to that margin.

```text
A^{GLPO}_{i,j} = (r_{i,j} - r̄_{G_i}) / σ_{G_i} + max(0, r_{i,j} - R(G_i))
```

Here `R(G_i)` is the reward of the majority-vote answer among the `K` rollouts of group `i`. This makes the RL training objective consistent with the self-consistency objective used at inference.

The implementation lives in [`glpo/glpo_swift_plugin.py`](glpo/glpo_swift_plugin.py). It registers two reward functions (`zh_gec_individual_reward` and `zh_gec_group_reward`) and monkey-patches `GRPOTrainer._compute_advantages`. A simplified `M=1` variant is also available at [`glpo_swift_plugin.py`](glpo_swift_plugin.py).

## Repository structure

```text
YuYi/
├── README.md
├── LICENSE
├── environment.yml
├── .gitignore
├── .gitmodules
├── configs/
│   ├── ds_zero2.json            # DeepSpeed ZeRO-2 config
│   └── sft_config.yaml          # reference SFT config
├── src/
│   └── rewards/
│       └── zh_grammar_reward.py # format + edit-distance correctness reward
├── scripts/
│   ├── CSED_test.py             # CSED-C inference + ChERRANT evaluation
│   └── NaSGEC_test.py           # NaSGEC/MuCGEC inference + evaluation (M2 refs)
├── glpo/
│   ├── glpo_swift_plugin.py     # GLPO advantage reallocation (M-group variant)
│   ├── prepare_glpo_data.py     # replicate each prompt M times
│   ├── run_glpo.sh              # GLPO launch script
│   └── README.md
├── glpo_swift_plugin.py         # simplified GLPO (M=1) variant
├── docs/
│   └── legacy/
│       └── CHINESE_GEC_SFT_GRPO.md
├── third_party/
│   └── ms-swift/                # RL framework (ms-swift)
├── MuCGEC/                      # MuCGEC benchmark + ChERRANT scorer
├── data/                        # raw data (not committed)
└── processed_data/              # processed data (not committed)
```

Local-only artifacts that are **not** part of the release: `.venv/`, `errant_env/`, `wandb/`, `.idea/`, `__pycache__/`, and `*.ipynb_checkpoints/`.

## Quick start

### 1. Environment

```bash
# Linux + conda
conda env create -f environment.yml
conda activate swift

# Install ms-swift (the RL framework). Use the bundled copy or pip.
pip install -e third_party/ms-swift
```

For vLLM / flash-attention and other optional dependencies, follow the [ms-swift](https://github.com/modelscope/ms-swift) installation guide.

### 2. Data

The datasets (`CSED-C` and `NaSGEC-Exam`) are not included in this release. Prepare them under `data/` and `processed_data/` following the dataset format described in the paper and in the evaluation scripts:

- JSON array / JSONL with `source` and `target` (multi-reference allowed), e.g. `{"id": 0, "source": "...", "target": ["..."]}`;
- NaSGEC/MuCGEC official `.m2` reference files are supported via `--reference_m2`.

### 3. CoT distillation

The CoT distillation prompts (generator + verifier) are provided in the paper appendix. The distillation step uses Qwen3.5-27B as generator and DeepSeek-V3.2 as verifier, and produces the structured training data consumed by SFT.

### 4. SFT (LLaMA-Factory)

SFT is performed with [LLaMA-Factory](../LLaMAFactory/LLaMA-Factory), located in the sibling directory:

```bash
cd ../LLaMAFactory/LLaMA-Factory

# Fill in model_name_or_path and dataset in the config, then:
llamafactory-cli train examples/train_lora/qwen3.5_lora_sft.yaml
llamafactory-cli export examples/merge_lora/qwen3.5_lora_grpo.yaml
```

Paper SFT settings: Qwen3.5-4B, LoRA rank `128`, batch size `4`, learning rate `2.0e-4`, `3` epochs (≈4 h per run).

### 5. RL (GLPO with ms-swift)

Optionally replicate each prompt `M` times so each prompt forms `M` independent groups:

```bash
python glpo/prepare_glpo_data.py \
    --input processed_data/grpo_train.jsonl \
    --output processed_data/grpo_train_M4.jsonl \
    --M 4
```

Launch GLPO with the individual reward (`zh_gec_reward`) and the group/vote reward (`zh_gec_group_reward`):

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 swift rlhf \
    --rlhf_type grpo \
    --model <SFT_MERGED_MODEL_DIR> \
    --external_plugins glpo/glpo_swift_plugin.py src/rewards/zh_grammar_reward.py \
    --reward_funcs zh_gec_reward zh_gec_group_reward \
    --reward_weights 1.0 1.0 \
    --enable_thinking True \
    --use_vllm true \
    --template qwen3 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --vllm_mode colocate \
    --train_type lora \
    --torch_dtype bfloat16 \
    --dataset <GLPO_TRAIN_JSONL> \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --num_generations 8 \
    --temperature 1.0 \
    --beta 0.001 \
    --log_completions true \
    --output_dir output_glpo
```

The provided [`glpo/run_glpo.sh`](glpo/run_glpo.sh) is a template for this launch; adapt its `--external_plugins` / `--reward_funcs` to the paths above if needed.

Paper RL settings: LoRA rank `8`, batch size `8` (2 GPUs × per-device `4`), learning rate `1e-6`, rollout number `N=8`, temperature `1.0`, `β=0.001`, `5` epochs (≈80 h on 2×A30).

### 6. Inference & evaluation

Both scripts run inference and ChERRANT evaluation in one pass. Use `--voting_samples 32 --voting_temperature 1.0` for the self-consistency (32-vote) setting.

The released checkpoint at `../model` is trained **only on CSED-C**; it is not the NaSGEC-Exam checkpoint.

```bash
# CSED-C (or MuCGEC)
python scripts/CSED_test.py \
    --model_path ../model \
    --input <test.json> \
    --output <pred.jsonl> \
    --use_vllm \
    --cherrant_dir MuCGEC/scorers/ChERRANT \
    --voting_samples 32 --voting_temperature 1.0

# NaSGEC-Exam with official M2 references
python scripts/NaSGEC_test.py \
    --model_path ../model \
    --input <nasgec.exam.test.input> \
    --reference_m2 <nasgec.exam.test.m2> \
    --output <pred.jsonl> \
    --use_vllm \
    --cherrant_dir MuCGEC/scorers/ChERRANT \
    --voting_samples 32 --voting_temperature 1.0
```

Add `--eval_only` to skip generation and only run ChERRANT on an existing prediction file.

## Results

### Main results (F0.5 / P / R)

**CSED-C**

| Method | P | R | F0.5 |
| --- | ---: | ---: | ---: |
| mT5-small | 33.70 | 5.40 | 16.50 |
| mT5-base | 57.00 | 19.00 | 40.70 |
| BART-large-Chinese | 53.80 | 38.30 | 49.70 |
| SynGEC | 53.00 | 39.50 | 49.60 |
| ChatGLM3-6B | 33.18 | 27.80 | 31.94 |
| CSEC-LLM (Baichuan2-7B) | 35.90 | 32.95 | 35.26 |
| Zero-shot (greedy) | 11.95 | 10.30 | 11.58 |
| Ours-SFT (greedy) | 37.28 | 35.80 | 36.97 |
| Ours-SFT (+32 vote) | 47.73 | 41.55 | 46.35 |
| Ours-GRPO (greedy) | 39.76 | 34.98 | 38.70 |
| Ours-GRPO (+32 vote) | 48.71 | 40.88 | 46.91 |
| **Ours-VAA-CSEC (greedy)** | 39.19 | 35.50 | 38.39 |
| **Ours-VAA-CSEC (+32 vote)** | **49.34** | **42.15** | **47.72** |

**NaSGEC-Exam**

| Method | P | R | F0.5 |
| --- | ---: | ---: | ---: |
| BART-large-Chinese | 23.01 | 11.31 | 19.06 |
| GECToR | 20.93 | 8.80 | 16.41 |
| ChatGLM3-6B | 30.51 | 24.30 | 29.03 |
| CSEC-LLM (Baichuan2-7B) | 34.46 | 22.69 | 31.22 |
| Zero-shot (greedy) | 11.92 | 15.10 | 12.44 |
| Ours-SFT (greedy) | 19.91 | 23.02 | 20.46 |
| Ours-SFT (+32 vote) | 47.43 | 26.78 | 41.09 |
| Ours-GRPO (greedy) | 24.61 | 22.49 | 24.15 |
| Ours-GRPO (+32 vote) | 46.81 | 26.43 | 40.55 |
| **Ours-VAA-CSEC (greedy)** | 27.66 | 24.20 | 26.89 |
| **Ours-VAA-CSEC (+32 vote)** | **48.43** | **26.50** | **41.55** |

### Ablation (CSED-C, +32 vote)

| Method | P | R | F0.5 |
| --- | ---: | ---: | ---: |
| VAA-CSEC | 49.34 | 42.15 | 47.72 |
| w/o CoT | 44.27 | 39.54 | 43.23 |
| w/o GLPO | 48.71 | 40.88 | 46.91 |
| w/o CoT & GLPO | 44.66 | 39.99 | 43.64 |

### Cross-domain evaluation (+32 vote)

| Train set | Test set | P | R | F0.5 |
| --- | --- | ---: | ---: | ---: |
| CSED-C | CSED-C | 49.34 | 42.15 | 47.72 |
| CSED-C | NaSGEC-Exam | 35.22 | 42.14 | 36.42 |
| NaSGEC-Exam | CSED-C | 50.54 | 27.88 | 43.47 |
| NaSGEC-Exam | NaSGEC-Exam | 48.43 | 26.50 | 41.55 |

## Hyperparameters

| Stage | Configuration |
| --- | --- |
| CoT distillation | Generator Qwen3.5-27B, verifier DeepSeek-V3.2 |
| SFT | Qwen3.5-4B, LoRA rank 128, batch 4, lr 2.0e-4, 3 epochs |
| RL (GLPO) | ms-swift, LoRA rank 8, batch 8, lr 1e-6, rollout N=8, T=1.0, β=0.001, 5 epochs |
| Hardware | 2×NVIDIA A30 |

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Citation

```bibtex
@inproceedings{vaa-csec,
  title     = {VAA-CSEC: Vote-guided Advantage Allocation for Chinese Semantic Error Correction},
  author    = {Han, Yitong and Lin, Nankai and Luo, Juan and Wu, Hongyan and Wang, Lianxi and Jiang, Shengyi},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```
