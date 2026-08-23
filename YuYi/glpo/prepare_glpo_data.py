"""
GLPO 数据预处理: 每个 prompt 复制 M 份

目的:
    让 swift 把每个 prompt 的 M 个副本当成 M 个独立 prompt 处理,
    各自采 K=8 次, 从而实现 "同一 prompt 的 M 个独立 group"。

关键要求:
    - M 个副本必须在数据集里连续出现 (swift 默认不 shuffle batch 内部)
    - 保持 dataset 的 shuffle 能力 (每个 epoch 重新打乱的粒度是"整组 M 个副本",不是单份)

用法:
    python prepare_glpo_data.py \
        --input  processed_data/v1/final_grpo_train.jsonl \
        --output processed_data/v1/final_grpo_train_M4.jsonl \
        --M 4
"""

import argparse
import json
import os
import random


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="原始grpo训练数据 (jsonl)")
    parser.add_argument("--output", type=str, required=True,
                        help="复制后的数据 (jsonl)")
    parser.add_argument("--M", type=int, default=4,
                        help="每个prompt复制的份数 (独立group数)")
    parser.add_argument("--shuffle_prompts", action="store_true", default=True,
                        help="shuffle原始prompt顺序 (保持M个副本连续)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 读取原数据
    original = load_jsonl(args.input)
    print(f"[INFO] 原数据: {len(original)} 条")

    # 可选: shuffle 整体顺序 (每个prompt的M个副本会一起被打乱,保持连续)
    if args.shuffle_prompts:
        random.shuffle(original)
        print(f"[INFO] 已 shuffle 原始 prompt 顺序")

    # 复制
    expanded = []
    for item in original:
        for m in range(args.M):
            new_item = dict(item)  # shallow copy
            new_item["_group_idx"] = m  # debug字段, 不影响训练
            expanded.append(new_item)
    
    print(f"[INFO] 复制后: {len(expanded)} 条 = {len(original)} × {args.M}")
    
    # 保存
    save_jsonl(expanded, args.output)
    print(f"[DONE] 输出: {args.output}")
    
    print(f"\n训练配置建议:")
    print(f"  --num_generations 8")
    print(f"  --per_device_train_batch_size 1  (每卡1个prompt,避免跨prompt混淆)")
    print(f"  --gradient_accumulation_steps {4*4}  (原来gas=8 × M=4 / 2卡 = 16, 或原值×M)")
    print(f"\n或者保持 per_device_train_batch_size=4:")
    print(f"  ⚠️ 这种情况下 batch 里会有 4/M=1 个不同prompt的4份副本,行为正确")


if __name__ == "__main__":
    main()