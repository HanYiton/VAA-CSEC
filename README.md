# <img src="bullseye2.png" width="40"> VAA-CSEC：Vote-guided Advantage Allocation for Chinese Semantic Error Correction
-----------------
**VAA-CSEC** 是一个面向中文语义纠错任务的多阶段训练框架，它结合了CoT蒸馏、监督微调（SFT）、强化学习（RL）和自一致性解码（self-consistency）。 在强化学习阶段，我们进一步引入了GLPO，它根据个体展开奖励与投票聚合组奖励之间的差距重新分配GRPO优势，使强化学习训练目标与推理时使用的自一致性目标对齐。

在CSED-C和NaSGEC-Exam数据集上的实验表明，VAA-CSEC在CSED-C数据集上优于所有基于LLM的基线方法，F0.5值为47.72%，在所有方法中实现了最高的召回率42.15%，并在NaSGEC-Exam数据集上创造了41.55%的F0.5值的新最佳结果。¹

## 来自作者
大家好，由于本框架流程是在一步一步探索中实现的，因此最初版本的可读性和可复现性较差，尽管现在有所改进，但可能还存在各种问题。若大家复现过程中有任何问题，欢迎提issue或直接通过[作者邮箱](20231003317@mail.gdufs.edu.cn)讨论

## 环境配置
推荐创建两个conda环境，分别进行LLaMAFactory的SFT训练和ms-swift的GRPO/GLPO训练

环境创建流程请分别参考LLaMAFactory和ms-swift的官方文档。

你可能会用到：
1. ms-swift官方文档中[Qwen3.5最佳实践说明](https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html#rl)

2. 对于Qwen3.5中vllm与transformers版本不兼容问题，可见[issue](https://github.com/modelscope/ms-swift/issues/8188)

## 数据处理
以CSED-C为例，蒸馏出的思维链保存在data\CSED-C\cot_sft_deidentified.json

为避免侵权，请自行前往[CSED-C仓库](https://github.com/wyxstriker/CSED/tree/main/CSED-C)下载原数据，然后通过YuYi\scripts\deidentify_cot_sft.py还原为SFT训练数据。

若想从0开始创建数据，请参考论文中的3.3 CoT Distillation以及Appendix C，但由于大模型生成的不稳定性，新数据可能会与论文有一定差别。

由于NaSGEC数据初始并非alpaca格式，暂时还没想到好的还原方法，若有需要可以联系我获取蒸馏出的思维链部分。

## SFT训练
请参考LLaMAFactory官方文档的标准训练流程，所使用模型以及超参数设置均在论文Experiment部分说明。

## GRPO/GLPO训练
详细流程请参考YuYi\README.md
