[中文](README.md) | [English](README_EN.md)

## 🙌 A Note from the Authors

Hello everyone! Since the framework was developed incrementally through a series of experiments, the readability of the original implementation was relatively poor. To facilitate reproduction, I substantially refactored the codebase before open-sourcing it. However, the refactoring may have introduced issues that I have not yet discovered, such as inconsistent paths or missing code.

If you encounter any problems during reproduction or discover a bug, please feel free to open an issue or [contact me](https://hanyiton.github.io/) directly via email. I will try my best to address the issue when I have time.

# <img src="bullseye2.png" width="40"> VAA-CSEC：Vote-guided Advantage Allocation for Chinese Semantic Error Correction

<p align="center">
  <b>Yitong Han<sup>1</sup>, Nankai Lin<sup>1,2</sup><sup>†</sup>, Juan Luo<sup>1</sup>, Hongyan Wu<sup>3</sup>, Lianxi Wang<sup>1</sup>, Shengyi Jiang<sup>1</sup></b>
</p>
<p align="center">
  <sup>1</sup>School of Information Science and Technology, Guangdong University of Foreign Studies<br>
  <sup>2</sup>Guangdong Engineering Research Center of Data Security Governance and Privacy Computing<br>
  <sup>3</sup>College of Computer Science and Technology, National University of Defense Technology
</p>
<p align="center">
  <sup>†</sup>Corresponding author
</p>
<p align="center">
  <a href="mailto:20231003317@mail.gdufs.edu.cn">20231003317@mail.gdufs.edu.cn</a> ·
  <a href="mailto:neakail@outlook.com">neakail@outlook.com</a>
</p>

**VAA-CSEC** is a multi-stage training framework for Chinese Semantic Error Correction (CSEC), combining Chain-of-Thought (CoT) distillation, Supervised Fine-Tuning (SFT), Reinforcement Learning (RL), and self-consistency decoding. During the reinforcement learning stage, we introduce **Group-Level Relative Policy Optimization (GLPO)**, which reallocates GRPO advantages according to the margin between individual rollout rewards and the vote-aggregated group reward, aligning the RL training objective with the self-consistency objective used at inference time.

Experiments on **CSED-C** and **NaSGEC-Exam** show that VAA-CSEC outperforms all LLM-based baselines on CSED-C with an F0.5 of **47.72%**, achieves the highest recall of **42.15%** among all methods, and establishes a new state of the art of **41.55% F0.5** on NaSGEC-Exam.

<p align="center">
  <img src="FrameWork.png" alt="VAA-CSEC Framework" width="900">
</p>

## Model Release

The model weights have been released on Hugging Face. Before using them, please make sure that they correspond to the appropriate training dataset. [Click here to access the model weights](https://huggingface.co/Hanyiton/VAA-CSEC/tree/main).

## Environment Setup

We recommend creating two separate Conda environments: one for **SFT training with LLaMAFactory**, and the other for **GRPO/GLPO training with ms-swift**.

For environment setup, please refer to the official documentation of [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory) and [ms-swift](https://swift.readthedocs.io/en/latest/GetStarted/Quick-start.html), respectively.

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

Please follow the standard training procedure provided in the [official **LLaMAFactory** documentation](https://llamafactory.readthedocs.io/en/latest/).

The model configuration and hyperparameter settings used in our experiments are described in the **Experiments** section of the paper.

## GRPO / GLPO Training

For the detailed training procedure, please refer to [YuYi/README.md](https://github.com/HanYiton/VAA-CSEC/blob/main/YuYi/README.md)
