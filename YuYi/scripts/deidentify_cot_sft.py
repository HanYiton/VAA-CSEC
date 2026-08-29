#!/usr/bin/env python3
"""De-identify the unfolded CoT-SFT data so the release does not reproduce the
original CSED-C sentences.

For each sample in ``cot_sft.json``:
  * ``input`` (the original source sentence) is replaced with the integer id of
    the corresponding sample in ``train.json`` (serialized as a string);
  * the payload of the ``<answer>...</answer>`` tag in ``output`` is replaced
    with the 1-based index of that reference inside the sample's ``target`` list.

Only the ``<answer>`` payload and ``input`` are changed; the ``<think>``
reasoning and the ``instruction`` are kept unchanged.

Usage:
    python deidentify_cot_sft.py \
        --train data/CSED-C/train.json \
        --cot_sft data/CSED-C/cot_sft.json \
        --output data/CSED-C/cot_sft_deidentified.json
"""

import argparse
import json
import re


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="original CSED-C train.json")
    parser.add_argument("--cot_sft", required=True, help="unfolded CoT SFT data")
    parser.add_argument("--output", required=True, help="output de-identified json")
    args = parser.parse_args()

    train = load_json(args.train)
    cot_sft = load_json(args.cot_sft)

    source_to_id = {item["source"]: item["id"] for item in train}
    id_to_targets = {item["id"]: item["target"] for item in train}

    out = []
    for item in cot_sft:
        src = item["input"]
        if src not in source_to_id:
            raise ValueError(f"input not found in train: {src[:40]!r}")
        sample_id = source_to_id[src]

        targets = id_to_targets[sample_id]
        match = ANSWER_RE.search(item["output"])
        if not match:
            raise ValueError(f"<answer> tag not found for sample id={sample_id}")

        answer = match.group(1)
        if answer not in targets:
            raise ValueError(
                f"answer not in target list for sample id={sample_id}: {answer[:40]!r}"
            )

        # 1-based index of the reference ("第 x 个").
        ref_index = targets.index(answer) + 1

        new_output = (
            item["output"][: match.start(1)]
            + str(ref_index)
            + item["output"][match.end(1):]
        )

        out.append(
            {
                "instruction": item["instruction"],
                "input": str(sample_id),
                "output": new_output,
            }
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"processed {len(out)} samples -> {args.output}")


if __name__ == "__main__":
    main()
