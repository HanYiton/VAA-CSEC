#!/usr/bin/env python3
"""Build Alpaca-format SFT data (CoT + non-CoT) from a CoT dataset only.

Given an Alpaca-format CoT dataset (instruction / input / output, where output is
``<think>...</think><answer>...</answer>``), emit two Alpaca-format JSON arrays
aligned 1:1:

  * ``--output_cot``    : the CoT records unchanged (pass-through);
  * ``--output_no_cot`` : non-CoT records with a "no reasoning" prompt and
    output ``<answer>...</answer>``.

The non-CoT prompt is a fixed constant (no ``<think>`` requirement).

Usage:
    python build_alpaca_from_cot.py \
        --cot_sft data/NaSGEC/nas_sft_cot.json \
        --output_cot data/NaSGEC/nas_sft_cot_alpaca.json \
        --output_no_cot data/NaSGEC/nas_sft_no_cot_alpaca.json
"""

import argparse
import json
import re


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


NO_COT_INSTRUCTION = (
    "你是一名中文语义纠错专家。\n\n"
    "【任务说明】\n\n"
    "下面给出的句子可能存在语义层面的问题，包括但不限于：\n"
    "用词不当、语义搭配不合理、概念使用不准确、语义逻辑不通顺或存在歧义。\n"
    "若不存在问题，直接输出原句\n\n"
    "【修改原则】\n\n"
    "保持原句核心含义不变\n"
    "仅做最少且必要的修改\n"
    "优先删除冗余或冲突表达\n"
    "避免重写句子\n\n"
    "【输出格式要求】\n\n"
    "不要输出任何思考过程，直接输出修改后的句子。严格按照以下格式：\n"
    "<answer>修改后的句子或原句</answer>\n\n"
    "你要修改的句子为："
)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot_sft", required=True, help="input CoT data (Alpaca)")
    parser.add_argument("--output_cot", required=True, help="CoT Alpaca output json")
    parser.add_argument(
        "--output_no_cot", required=True, help="non-CoT Alpaca output json"
    )
    args = parser.parse_args()

    cot = load_json(args.cot_sft)

    cot_records = []
    no_cot_records = []

    for item in cot:
        match = ANSWER_RE.search(item["output"])
        if not match:
            raise ValueError("output does not contain <answer>...</answer>")
        answer = match.group(1)

        cot_records.append(
            {
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"],
            }
        )
        no_cot_records.append(
            {
                "instruction": NO_COT_INSTRUCTION,
                "input": item["input"],
                "output": f"<answer>{answer}</answer>",
            }
        )

    with open(args.output_cot, "w", encoding="utf-8") as f:
        json.dump(cot_records, f, ensure_ascii=False, indent=2)
    with open(args.output_no_cot, "w", encoding="utf-8") as f:
        json.dump(no_cot_records, f, ensure_ascii=False, indent=2)

    print(f"CoT samples:     {len(cot_records)} -> {args.output_cot}")
    print(f"non-CoT samples: {len(no_cot_records)} -> {args.output_no_cot}")


if __name__ == "__main__":
    main()
