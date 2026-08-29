#!/usr/bin/env python3
"""Build Alpaca-format SFT data from the original CSED-C train and the
de-identified CoT data.

The script restores the CoT samples (id -> source, reference index -> target)
and emits two Alpaca-format JSON arrays aligned 1:1:

  * ``--output_cot``    : CoT samples, output = <think>...</think><answer>...</answer>
  * ``--output_no_cot`` : non-CoT samples, output = <answer>...</answer>

The two variants use different prompts (see the constants below).

Usage:
    python build_alpaca_sft.py \
        --train data/CSED-C/train.json \
        --cot_sft_deidentified data/CSED-C/cot_sft_deidentified.json \
        --output_cot data/CSED-C/sft_cot_alpaca.json \
        --output_no_cot data/CSED-C/sft_no_cot_alpaca.json
"""

import argparse
import json
import re


ANSWER_IDX_RE = re.compile(r"<answer>(\d+)</answer>")


COT_INSTRUCTION = (
    "你是一名中文语义纠错专家。请先进行简要思考，再完成任务。\n\n"
    "【任务说明】\n\n"
    "下面给出的句子可能存在语义层面的问题，包括但不限于：\n"
    "用词不当、语义搭配不合理、概念使用不准确、语义逻辑不通顺或存在歧义。\n\n"
    "【思考要求】\n\n"
    "请先进行简要分析：\n"
    "- 判断句子是否存在语义问题\n"
    "- 若存在，指出问题类型（如用词不当、搭配问题、逻辑问题、冗余等）\n"
    "- 给出最小修改方案\n"
    "- 若不存在问题，说明无需修改,并输出原句\n\n"
    "【修改原则】\n\n"
    "保持原句核心含义不变\n"
    "仅做最少且必要的修改\n"
    "优先删除冗余或冲突表达\n"
    "避免重写句子\n\n"
    "【输出格式要求】\n\n"
    "严格按照以下格式：\n"
    "<think>你的分析过程</think>\n"
    "<answer>修改后的句子或原句</answer>\n\n"
    "你要修改的句子为："
)


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
    parser.add_argument("--train", required=True, help="original CSED-C train.json")
    parser.add_argument(
        "--cot_sft_deidentified", required=True, help="de-identified CoT data"
    )
    parser.add_argument("--output_cot", required=True, help="CoT Alpaca output json")
    parser.add_argument(
        "--output_no_cot", required=True, help="non-CoT Alpaca output json"
    )
    args = parser.parse_args()

    train = load_json(args.train)
    deid = load_json(args.cot_sft_deidentified)
    by_id = {item["id"]: item for item in train}

    cot_records = []
    no_cot_records = []

    for item in deid:
        sample_id = int(item["input"])
        sample = by_id[sample_id]
        source = sample["source"]
        targets = sample["target"]

        match = ANSWER_IDX_RE.search(item["output"])
        if not match:
            raise ValueError(f"no numeric answer for sample id={sample_id}")
        ref_index = int(match.group(1))
        if not 1 <= ref_index <= len(targets):
            raise ValueError(
                f"index {ref_index} out of range for sample id={sample_id}"
            )

        target = targets[ref_index - 1]
        restored_output = (
            item["output"][: match.start(1)]
            + target
            + item["output"][match.end(1):]
        )

        cot_records.append(
            {
                "instruction": COT_INSTRUCTION,
                "input": source,
                "output": restored_output,
            }
        )
        no_cot_records.append(
            {
                "instruction": NO_COT_INSTRUCTION,
                "input": source,
                "output": f"<answer>{target}</answer>",
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
