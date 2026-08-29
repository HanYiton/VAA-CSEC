[English](README_EN.md) | [中文](README.md)

# <img src="bullseye2.png" width="40"> VAA-CSEC：Vote-guided Advantage Allocation for Chinese Semantic Error Correction

**VAA-CSEC** is a multi-stage training framework for Chinese Semantic Error Correction (CSEC), combining Chain-of-Thought (CoT) distillation, Supervised Fine-Tuning (SFT), Reinforcement Learning (RL), and self-consistency decoding. During the reinforcement learning stage, we introduce **Group-Level Relative Policy Optimization (GLPO)**, which reallocates GRPO advantages according to the margin between individual rollout rewards and the vote-aggregated group reward, aligning the RL training objective with the self-consistency objective used at inference time.

Experiments on **CSED-C** and **NaSGEC-Exam** show that VAA-CSEC outperforms all LLM-based baselines on CSED-C with an F0.5 of **47.72%**, achieves the highest recall of **42.15%** among all methods, and establishes a new state of the art of **41.55% F0.5** on NaSGEC-Exam.

<p align="center">
  <img src="FrameWork.png" alt="VAA-CSEC Framework" width="900">
</p>

## 🙌 A Note from the Authors

Hello everyone! Since the framework was developed incrementally through a series of experiments, the readability of the original implementation was relatively poor. To facilitate reproduction, I substantially refactored the codebase before open-sourcing it. However, the refactoring may have introduced issues that I have not yet discovered, such as inconsistent paths or missing code.

If you encounter any problems during reproduction or discover a bug, please feel free to open an issue or contact me directly via [email](mailto:20231003317@mail.gdufs.edu.cn). I will try my best to address the issue when I have time.

## Environment Setup

We recommend creating two separate Conda environments: one for **SFT training with LLaMAFactory**, and the other for **GRPO/GLPO training with ms-swift**.

For environment setup, please refer to the official documentation of [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory) and [ms-swift](https://github.com/modelscope/ms-swift), respectively.

You may also find the following resources useful:

1. The [Qwen3.5 Best Practices](https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html#rl) for RL training in the ms-swift documentation.
2. For compatibility issues between **vLLM** and **Transformers** when using Qwen3.5, please refer to [ms-swift/issues/8188](https://github.com/modelscope/ms-swift/issues/8188).

## Data Processing

Taking **CSED-C** as an example, the distilled Chain-of-Thought data is stored in:

```text
data/CSED-C/cot_sft_deidentified.json
```

Before processing the data, please download the original dataset from the [CSED-C repository](https://github.com/wyxstriker/CSED/tree/main/CSED-C). Then, use:

```text
YuYi/scripts/build_alpaca_sft.py
```

to reconstruct the data into the format required for SFT training.

If you would like to create the data from scratch, please refer to **Section 3.3 (CoT Distillation)** and **Appendix C** of our paper. Due to the inherent instability of large language model generation, newly generated data may differ slightly from the data used in our experiments.

**The original NaSGEC-Exam data is not in the Alpaca format, and we have not yet found a suitable method for reconstructing the original data. If needed, please contact me to obtain the distilled Chain-of-Thought data.**

## SFT Training

Please follow the standard training procedure provided in the official **LLaMAFactory** documentation.

The model configuration and hyperparameter settings used in our experiments are described in the **Experiments** section of the paper.

## GRPO / GLPO Training

For the detailed training procedure, please refer to:

```text
YuYi/README.md
```
