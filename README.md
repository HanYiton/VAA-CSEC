# VAA-CSEC: Vote-guided Advantage Allocation for Chinese Semantic Error Correction

Official code accompanying the paper:

> **VAA-CSEC: Vote-guided Advantage Allocation for Chinese Semantic Error Correction**
>
> Anonymous submission

## Overview

VAA-CSEC is a multi-stage framework for **Chinese Semantic Error Correction (CSEC)**. It combines four stages:

1. **CoT distillation** — a strong teacher (Qwen3.5-27B) generates chain-of-thought rationales, which are verified by DeepSeek-V3.2;
2. **Supervised Fine-Tuning (SFT)** — Qwen3.5-4B is fine-tuned with LoRA via LLaMA-Factory to follow the `<think>...</think><answer>...</answer>` format;
3. **Group-Level Relative Policy Optimization (GLPO)** — a GRPO variant whose advantages are reallocated by the margin between each rollout's reward and the vote-aggregated group reward;
4. **Self-consistency decoding** — the final correction is selected by majority voting over `N` stochastic samples.

Under 32-vote decoding, VAA-CSEC reaches **47.72% F0.5 on CSED-C** (highest recall, 42.15%) and **41.55% F0.5 on NaSGEC-Exam** (new state of the art).

## Repository layout

| Path | Contents |
| --- | --- |
| `YuYi/` | Main pipeline: task-specific reward function, GLPO plugin, training configs, and ChERRANT-based inference/evaluation scripts. See [`YuYi/README.md`](YuYi/README.md). |
| `LLaMAFactory/` | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), used for the SFT stage. |

## Quick start

The complete reproduction guide is in [`YuYi/README.md`](YuYi/README.md). At a high level:

```bash
# 1) SFT with LLaMA-Factory (fill in model + dataset in the config first)
cd LLaMAFactory/LLaMA-Factory
llamafactory-cli train examples/train_lora/qwen3.5_lora_sft.yaml

# 2) GLPO reinforcement learning with ms-swift (see YuYi/README.md for the command)
cd ../../YuYi

# 3) Inference + ChERRANT evaluation (self-consistency, 32 votes)
python scripts/CSED_test.py --model_path <MODEL> --input <test.json> \
    --output <pred.jsonl> --use_vllm \
    --cherrant_dir MuCGEC/scorers/ChERRANT \
    --voting_samples 32 --voting_temperature 1.0
```

## License

The code in this release is provided under the Apache-2.0 license. See `YuYi/LICENSE` and the licenses of the bundled third-party projects (`LLaMAFactory/`, `YuYi/third_party/`, `YuYi/MuCGEC/`).

## Citation

```bibtex
@inproceedings{vaa-csec,
  title     = {VAA-CSEC: Vote-guided Advantage Allocation for Chinese Semantic Error Correction},
  author    = {Anonymous},
  booktitle = {Proceedings of the Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  note      = {Anonymous submission}
}
```
