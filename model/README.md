# VAA-CSEC checkpoint (CSED-C)

This directory contains the released VAA-CSEC checkpoint used for the paper's CSED-C experiments.

- **Task**: Chinese Semantic Error Correction (CSEC).
- **Training data**: trained **only on the CSED-C training set**. This checkpoint was **not** trained on NaSGEC-Exam.
- **Method**: CoT SFT → GLPO (Group-Level Relative Policy Optimization) on top of GRPO.
- **Base model**: Qwen3.5 (multimodal), fine-tuned with ms-swift.
- **Checkpoint**: GLPO `v3` (`M=4`, `alpha=0.5`); see `args.json` for the full training configuration.
- **Format**: HuggingFace `safetensors`, `bfloat16`, sharded into 2 files (~8.66 GB in total).

## Files

| File | Description |
| --- | --- |
| `config.json`, `generation_config.json` | model configuration |
| `model.safetensors.index.json` | weight-shard index |
| `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors` | bfloat16 weights |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | tokenizer + chat template |
| `args.json` | ms-swift training arguments |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("model", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("model")
```

From the `YuYi/` directory, run the bundled inference/evaluation scripts with `--model_path ../model`:

```bash
cd YuYi
python scripts/CSED_test.py --model_path ../model --input <test.json> \
    --output <pred.jsonl> --use_vllm \
    --cherrant_dir MuCGEC/scorers/ChERRANT \
    --voting_samples 32 --voting_temperature 1.0
```

## Expected results

Because this checkpoint is trained only on CSED-C, its in-domain result is the CSED-C row of the paper's main table (VAA-CSEC, +32 vote): **P 49.34 / R 42.15 / F0.5 47.72**.
