## 中文语义纠错：LLaMA-Factory SFT + verl GRPO 操作说明

本仓库原生支持 MultiGEC 2025 多语言语法纠错。为了在 **中文语义纠错** 任务上使用
**LLaMA-Factory 做 SFT**，并继续使用 **verl 进行 GRPO 强化学习**，这里给出一套推荐流程。

目标设定：
- 基座模型：**Qwen3-4B**（或兼容的 Qwen3 系列模型）
- 硬件：两张 A30（2×24GB）
- 训练流程：  
  1）整理中文纠错数据 →  
  2）LLaMA-Factory SFT →  
  3）使用本仓库 + verl 做 GRPO →  
  4）统一评估与对比。

---

### 1. 准备中文语义纠错原始数据

假设你已经有中文纠错数据（句对：原句 + 纠正句），且已经分好 train/dev/test，例如：

- `data/train.json`
- `data/dev.json`
- `data/test.json`

当前脚本支持三种格式（自动识别）：

- **JSON 数组（推荐，与你现在数据一致）**：

```json
[
  {
    "id": 0,
    "source": "原句……",
    "target": ["纠正句……"]
  },
  {
    "id": 1,
    "source": "原句……",
    "target": ["纠正句……"]
  }
]
```

- **JSONL**：每行一个 JSON，对象中包含 `source`/`target`（或 `src`/`tgt`/`input`/`output` 等别名）；
- **TSV**：每行 `source<TAB>target`。

---

### 2. 使用 `prepare_zh_gec.py` 统一预处理

脚本位置：`src/data/prepare_zh_gec.py`

功能：
- 从原始 JSON 数组 / JSONL / TSV 中读取中文纠错样本；
- 生成：
  - 本仓库通用的 chat JSONL（`processed_data/zh_gec_*.jsonl`）；
  - LLaMA-Factory SFT 数据（`llamafactory_data/zh_gec_*.json`，`instruction/input/output`）；
  - GRPO 训练用 Parquet（`processed_data/zh_gec_*_grpo.parquet`，可直接给 verl 使用）。

一次性处理 train/dev/test 的示例命令：

```bash
python -m src.data.prepare_zh_gec \
  --train data/train.json \
  --dev data/dev.json \
  --test data/test.json \
  --output_dir processed_data \
  --llamafactory_dir ../llamafactory_data
```

运行后你将得到（在默认目录下）：

- `processed_data/zh_gec_train.jsonl`
- `processed_data/zh_gec_dev.jsonl`
- `processed_data/zh_gec_test.jsonl`
- `processed_data/zh_gec_train_grpo.parquet`
- `processed_data/zh_gec_dev_grpo.parquet`
- `processed_data/zh_gec_test_grpo.parquet`
- `llamafactory_data/zh_gec_train.json`
- `llamafactory_data/zh_gec_dev.json`
- `llamafactory_data/zh_gec_test.json`

---

### 3. 用 LLaMA-Factory 对 Qwen3-4B 做中文 SFT

`llamafactory_data/zh_gec_*.json` 已经是 LLaMA-Factory 支持的 `instruction/input/output` 格式：

```json
{
  "instruction": "你是一个中文语法与语义纠错助手。请对下面给出的句子进行纠错，尽量保持原句含义不变，只做最必要的修改。请直接输出修改后的句子，不要添加任何解释、标注或前后缀文字。",
  "input": "原句……",
  "output": "纠正句……"
}
```

你只需要在 LLaMA-Factory 的数据配置里引用这些文件，然后设置模型与训练参数即可。典型的配置要点（伪 YAML，仅示意）：

```yaml
model_name_or_path: Qwen/Qwen3-4B
template: qwen
finetuning_type: lora
lora_rank: 64
lora_alpha: 64
dtype: bfloat16  # A30 支持时推荐，否则 float16
train_file: llamafactory_data/zh_gec_train.json
validation_file: llamafactory_data/zh_gec_dev.json
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
num_train_epochs: 3
max_source_length: 256
max_target_length: 128
learning_rate: 1e-5
```

完成 SFT 后，在 LLaMA-Factory 中 **merge LoRA 并导出一个完整 HuggingFace 模型目录**，例如：

- `checkpoints/zh_qwen3_4b_sft_merged/`

确认该目录可以被 Transformers 正常加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("checkpoints/zh_qwen3_4b_sft_merged")
tok = AutoTokenizer.from_pretrained("checkpoints/zh_qwen3_4b_sft_merged")
```

---

### 4. 使用 verl + GRPO 强化中文纠错行为

#### 4.1 奖励函数：`src/rewards/zh_grammar_reward.py`

该文件在通用 `grammar_reward.py` 的基础上，为中文做了轻微定制：

- 复用编辑距离相似度 + chrF 的组合；
- 加了一层中文“指令泄露”检测（如果模型输出中出现类似“请帮我纠正下列句子的语法错误”等内容，则直接视为格式失败，给负分）。

接口保持不变：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    ...
```

在 GRPO 训练时，通过以下参数告诉 verl 使用这个 reward：

- `custom_reward_function.path=src/rewards/zh_grammar_reward.py`
- `custom_reward_function.name=compute_score`

#### 4.2 中文专用 GRPO 启动脚本

脚本位置：`scripts/train_grpo_zh_qwen3_4b.sh`

默认设置：

- 模型：`checkpoints/zh_qwen3_4b_sft_merged`
- 训练集：`processed_data/zh_gec_train_grpo.parquet`
- 验证集：`processed_data/zh_gec_dev_grpo.parquet`
- 奖励函数：`src/rewards/zh_grammar_reward.py`

使用示例：

```bash
# 使用默认路径（推荐先按上面的预处理 & SFT 约定）
bash scripts/train_grpo_zh_qwen3_4b.sh

# 或显式指定 SFT 模型与 Parquet 数据
bash scripts/train_grpo_zh_qwen3_4b.sh \
  checkpoints/zh_qwen3_4b_sft_merged \
  processed_data/zh_gec_train_grpo.parquet \
  processed_data/zh_gec_dev_grpo.parquet
```

如果需要微调显存/性能相关参数（如 batch size、KL 系数、`max_prompt_length`、`gpu_memory_utilization` 等），可以在这个脚本中直接修改相应的命令行参数，无需改动 verl 源码。

---

### 5. 推理与评估（中文）

#### 5.1 推理

你可以复用本仓库的 `scripts/inference.py`，让它加载：

- 仅 SFT 模型：`checkpoints/zh_qwen3_4b_sft_merged`；
- 或 GRPO 后生成的权重目录。

输入 JSONL 格式建议与 `processed_data/zh_gec_*.jsonl` 对齐，至少包含：

- `id`
- `source`
- `lang`: `"zh"`

`inference.py` 会基于 tokenizer 的 chat template 构造 prompt，并输出预测 JSONL。

#### 5.2 评估

评估脚本：`src/eval/evaluate.py`  
这是语言无关的，只要：

- 预测文件每行包含：`{"id": ..., "prediction": ...}`；
- 参考文件每行包含：`{"id": ..., "target": ..., "source": ..., "lang": "zh"}`；

即可计算：

- Exact Match（严格相等）
- Edit Distance Similarity（字符级编辑距离相似度）
- GLEU（简化版 GLEU）

示例命令：

```bash
python -m src.eval.evaluate \
  --predictions outputs/zh_gec_pred_dev.jsonl \
  --references processed_data/zh_gec_dev.jsonl \
  --output outputs/zh_gec_metrics_dev.json
```

你可以分别对以下三种模型跑评估并对比：

- 原始 Qwen3-4B；
- Qwen3-4B + LLaMA-Factory SFT；
- Qwen3-4B + LLaMA-Factory SFT + verl GRPO。

---

### 6. 调参与迭代建议（在显卡和时间预算内榨干性能）

- **先保证 SFT 基线足够好**：  
  如果 SFT 后 dev 集指标明显偏弱，优先调 SFT（学习率、epoch、样本清洗/采样）再上 GRPO。

- **GRPO 初始设置稍保守**：  
  使用较小学习率、合理 KL penalty 和较短序列长度（例如 `max_prompt_length=512`, `max_response_length=128`），观察 reward 曲线是否稳定。

- **逐步增强“纠错质量”权重**：  
  训练前期更偏重 edit distance（避免乱改句子），后期逐步提高 chrF/语义一致性相关的权重。

- **难度分层 / curriculum（可选）**：  
  如果数据带有错误类型/难度标签，可以在 RL 阶段对难样本加权或分阶段加入，进一步提升模型在复杂错误上的表现。

通过上述流程，你可以在不修改 verl 源码的前提下，将 **LLaMA-Factory SFT** 与 **verl GRPO** 紧密衔接，在两张 A30 的算力与时间预算内尽可能逼近中文语义纠错性能上限。

