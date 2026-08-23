"""
中文 GEC 推理 + 评测一体化脚本（含自动重试 + 多样本投票 + 多参考评测）

功能：
  1. 单次贪心解码 (t=0, 默认)
  2. 多样本投票：生成 N 个候选 (t=1)，选出现次数最多的作为最终答案
  3. 自动重试提取失败的样本
  4. ChERRANT 评测（支持多参考）

用法：
  # 贪心解码（默认）
  python 2.py \
      --model_path model/merged \
      --input processed_data/test.json \
      --output output_data/pred.jsonl \
      --use_vllm

  # 多样本投票（8 个候选，temperature=1）
  python 2.py \
      --model_path model/merged \
      --input processed_data/test.json \
      --output output_data/pred.jsonl \
      --use_vllm \
      --voting_samples 8

  # 跳过推理，只评测已有预测文件
  python CSED_test.py \
      --input data/test.json \
      --output data/baichuan_vote1.jsonl \
      --cherrant_dir MuCGEC/scorers/ChERRANT \
      --eval_only
"""
import unicodedata
import json
import re
import argparse
import os
import subprocess
import tempfile
from tqdm import tqdm
from collections import Counter

try:
    import editdistance
except ImportError:
    editdistance = None


# ==================== Prompt 定义 ====================

SYSTEM_PROMPT = ""  # 如果不需要 system prompt，留空即可

USER_PROMPT_TEMPLATE = """你是一名中文语义纠错专家。请先进行简要思考，再完成任务。\n\n【任务说明】\n\n下面给出的句子可能存在语义层面的问题，包括但不限于：\n用词不当、语义搭配不合理、概念使用不准确、语义逻辑不通顺或存在歧义。\n\n【思考要求】\n\n请先进行简要分析：\n- 判断句子是否存在语义问题\n- 若存在，指出问题类型（如用词不当、搭配问题、逻辑问题、冗余等）\n- 给出最小修改方案\n- 若不存在问题，说明无需修改,并输出原句\n\n【修改原则】\n\n保持原句核心含义不变\n仅做最少且必要的修改\n优先删除冗余或冲突表达\n避免重写句子\n\n【输出格式要求】\n\n严格按照以下格式：\n<think>简要分析</think>\n<answer>修改后的句子或原句</answer>\n\n你要修改的句子为：
{source}"""
# USER_PROMPT_TEMPLATE = """你是一名中文语义纠错专家。 【任务说明】 下面给出的句子可能存在语义层面的问题，包括但不限于： 用词不当、语义搭配不合理、概念使用不准确、语义逻辑不通顺或存在歧义。 若不存在问题，直接输出原句 【修改原则】 保持原句核心含义不变 仅做最少且必要的修改 优先删除冗余或冲突表达 避免重写句子 【输出格式要求】 严格按照以下格式： <answer>修改后的句子或原句</answer> 你要修改的句子为：
# {source}"""

# ==================== 文本提取工具 ====================

def strip_template_tokens(text: str) -> str:
    """去除 chat template 特殊 token 和 <think> 标签。"""
    if not text:
        return ""
    text = re.sub(r"<\|im_start\|>.*?\n?", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"<\|endoftext\|>", "", text)
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    text = unicodedata.normalize("NFKC", text)
    return text.strip()


def extract_prediction(raw_output: str, source: str = ""):
    """
    从模型原始输出中提取纠正后的文本。

    返回: (提取文本, 提取方式, 是否需要重试)
    """
    if not raw_output:
        return source.replace(" ", ""), "empty_output", True

    cleaned = strip_template_tokens(raw_output)
    if not cleaned:
        return source.replace(" ", ""), "empty_output", True

    src_len = len(source.replace(" ", ""))

    def _check_too_long(text, reason):
        """提取成功但输出比原句长 2 倍以上，视为异常"""
        if src_len > 0 and len(text) > src_len * 2:
            return text, "too_long", True
        return text, reason, False

    # ── 格式1: <answer>...</answer> (完整标签) ──
    m = re.search(r"<answer>(.*?)</answer>", cleaned, flags=re.DOTALL)
    if m:
        answer = m.group(1).strip()
        placeholders = ["修改后的句子", "纠错后的句子", "正确的句子", "corrected sentence"]
        if answer in placeholders or len(answer) < 2:
            return source.replace(" ", ""), "placeholder", True
        return _check_too_long(answer.replace(" ", ""), "answer_tag")

    # ── 格式1.5: <answer>存在但缺少</answer>（模型截断） ──
    m2 = re.search(r"<answer>(.*)", cleaned, flags=re.DOTALL)
    if m2:
        answer = m2.group(1).strip()
        if len(answer) >= 2:
            return answer.replace(" ", ""), "answer_tag_unclosed", True

    # ── 格式2: 纯文本回退 ──
    text_only = re.sub(r"<thinking>[\s\S]*?</thinking>\s*", "", cleaned)
    text_only = re.sub(r"<thinking>[\s\S]*$", "", text_only)
    text_only = text_only.strip()

    if len(text_only) >= 2:
        if "<thinking>" in text_only:
            return source.replace(" ", ""), "dirty_text", True
        return _check_too_long(text_only.replace(" ", ""), "plain_text")

    return source.replace(" ", ""), "no_answer", True


# ==================== 数据 IO ====================

def normalize_target(target):
    """
    将 target 统一为列表格式（多参考）。
    - 如果是 str，变成 [str]
    - 如果是 list，保持原样
    - 其他情况返回 [""]
    """
    if isinstance(target, list):
        return target if target else [""]
    elif isinstance(target, str):
        return [target] if target else [""]
    else:
        return [""]


def load_test_data(path):
    """
    加载测试集（支持 JSON 列表和 JSONL 格式）。
    target 保留为列表（多参考）。
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.splitlines() if line.strip()]
    for item in data:
        item["target"] = normalize_target(item.get("target", ""))
    return data


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    if os.path.isdir(path) or path.endswith("/") or path.endswith("\\"):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "pred.jsonl")
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


# ==================== Prompt 构建 ====================

def build_messages(source_text):
    """根据代码中定义的 prompt 模板构建 messages。"""
    user_content = USER_PROMPT_TEMPLATE.format(source=source_text)
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_content})
    return messages


# ==================== 推理部分 ====================

def inference_hf(model, tokenizer, samples, max_new_tokens, temperature, batch_size):
    import torch

    results = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Generating"):
        batch = samples[i:i + batch_size]
        prompts = []
        for s in batch:
            messages = build_messages(s["source"])
            text = tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False
            )
            prompts.append(text)

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, s in enumerate(batch):
            input_len = inputs["input_ids"].shape[1]
            generated = outputs[j][input_len:]
            raw_pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append({
                "id": s["id"],
                "source": s["source"],
                "target": s["target"],
                "raw_output": raw_pred,
            })

    return results


def inference_vllm(llm, tokenizer, samples, max_new_tokens, temperature, lora_path=None):
    from vllm import SamplingParams

    prompts = []
    for s in samples:
        messages = build_messages(s["source"])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False
        )
        prompts.append(text)

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else 0,
    )

    lora_request = None
    if lora_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(
            lora_name="gec_lora",
            lora_int_id=1,
            lora_local_path=lora_path,
        )

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    results = []
    for s, output in zip(samples, outputs):
        raw_pred = output.outputs[0].text.strip()
        results.append({
            "id": s["id"],
            "source": s["source"],
            "target": s["target"],
            "raw_output": raw_pred,
        })

    return results


# ==================== 多样本投票推理 ====================

def majority_vote(candidates, source):
    """
    从多个候选预测中选出出现次数最多的作为最终答案。
    如果所有候选都提取失败，回退到 source。

    返回: (最终预测, 投票详情 dict)
    """
    extracted_list = []
    valid_preds = []

    for raw in candidates:
        pred, reason, should_retry = extract_prediction(raw, source)
        extracted_list.append((pred, reason, should_retry))
        if not should_retry:
            valid_preds.append(pred)

    # 如果有提取成功的，在成功的里面投票
    if valid_preds:
        counter = Counter(valid_preds)
        winner, win_count = counter.most_common(1)[0]
        return winner, {
            "total_candidates": len(candidates),
            "valid_candidates": len(valid_preds),
            "winner_count": win_count,
            "unique_preds": len(counter),
            "agreement_rate": round(win_count / len(valid_preds), 4),
        }

    # 全部失败，在所有提取结果里投票（包括失败的）
    all_preds = [e[0] for e in extracted_list]
    counter = Counter(all_preds)
    winner, win_count = counter.most_common(1)[0]
    return winner, {
        "total_candidates": len(candidates),
        "valid_candidates": 0,
        "winner_count": win_count,
        "unique_preds": len(counter),
        "agreement_rate": 0.0,
    }


def inference_vllm_voting(llm, tokenizer, samples, max_new_tokens,
                          voting_samples, voting_temperature, lora_path=None):
    """
    vLLM 多样本投票推理：利用 SamplingParams(n=N) 一次生成多个候选。
    """
    from vllm import SamplingParams

    prompts = []
    for s in samples:
        messages = build_messages(s["source"])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False
        )
        prompts.append(text)

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=voting_temperature,
        n=voting_samples,
    )

    lora_request = None
    if lora_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(
            lora_name="gec_lora",
            lora_int_id=1,
            lora_local_path=lora_path,
        )

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    results = []
    for s, output in zip(samples, outputs):
        candidates = [o.text.strip() for o in output.outputs]
        winner, vote_info = majority_vote(candidates, s["source"])

        results.append({
            "id": s["id"],
            "source": s["source"],
            "target": s["target"],
            "prediction": winner,
            "raw_output": candidates[0],  # 保留第一个候选作为 raw_output
            "all_candidates": candidates,
            "vote_info": vote_info,
            "extract_reason": "voting",
        })

    return results


def inference_hf_voting(model, tokenizer, samples, max_new_tokens,
                        voting_samples, voting_temperature, batch_size):
    """
    HF 多样本投票推理：对每个样本生成 N 次，然后投票。
    """
    import torch

    results = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Voting"):
        batch = samples[i:i + batch_size]
        prompts = []
        for s in batch:
            messages = build_messages(s["source"])
            text = tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False
            )
            prompts.append(text)

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 生成 voting_samples 轮
        all_candidates = [[] for _ in batch]
        for _ in range(voting_samples):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=voting_temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for j in range(len(batch)):
                input_len = inputs["input_ids"].shape[1]
                generated = outputs[j][input_len:]
                raw_pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
                all_candidates[j].append(raw_pred)

        for j, s in enumerate(batch):
            winner, vote_info = majority_vote(all_candidates[j], s["source"])
            results.append({
                "id": s["id"],
                "source": s["source"],
                "target": s["target"],
                "prediction": winner,
                "raw_output": all_candidates[j][0],
                "all_candidates": all_candidates[j],
                "vote_info": vote_info,
                "extract_reason": "voting",
            })

    return results


def run_voting_inference(samples, model_or_llm, tokenizer, use_vllm,
                         max_new_tokens, voting_samples, voting_temperature,
                         batch_size, lora_path=None):
    """
    多样本投票推理主入口。

    返回: [{id, source, target, prediction, raw_output, extract_reason, vote_info}, ...]
    """
    print(f"\n{'='*60}")
    print(f"[投票推理] 样本数: {len(samples)} | "
          f"每条生成 {voting_samples} 个候选 | temperature={voting_temperature}")
    print(f"{'='*60}")

    if use_vllm:
        results = inference_vllm_voting(
            model_or_llm, tokenizer, samples, max_new_tokens,
            voting_samples, voting_temperature, lora_path=lora_path,
        )
    else:
        results = inference_hf_voting(
            model_or_llm, tokenizer, samples, max_new_tokens,
            voting_samples, voting_temperature, batch_size,
        )

    # 统计投票质量
    total_agreement = 0.0
    total_valid = 0
    total_unique = 0
    for r in results:
        vi = r["vote_info"]
        total_agreement += vi["agreement_rate"]
        total_valid += vi["valid_candidates"]
        total_unique += vi["unique_preds"]

    n = len(results)
    print(f"\n{'='*60}")
    print(f"[投票统计] 总计 {n} 条")
    print(f"{'='*60}")
    print(f"  平均一致率: {total_agreement/n*100:.1f}%")
    print(f"  平均有效候选: {total_valid/n:.1f}/{voting_samples}")
    print(f"  平均不同预测数: {total_unique/n:.1f}")

    # 一致率分布
    agreement_buckets = {"100%": 0, ">=75%": 0, ">=50%": 0, "<50%": 0}
    for r in results:
        rate = r["vote_info"]["agreement_rate"]
        if rate >= 1.0:
            agreement_buckets["100%"] += 1
        elif rate >= 0.75:
            agreement_buckets[">=75%"] += 1
        elif rate >= 0.50:
            agreement_buckets[">=50%"] += 1
        else:
            agreement_buckets["<50%"] += 1

    print(f"  一致率分布:")
    for bucket, count in agreement_buckets.items():
        print(f"    {bucket:<10}: {count} 条 ({count/n*100:.1f}%)")

    return results


def load_model_hf(model_path, lora_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )

    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def load_model_vllm(model_path, lora_path, lora_rank=64):
    from vllm import LLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        language_model_only=True,
        tensor_parallel_size=4,
    )
    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = lora_rank

    llm = LLM(**llm_kwargs)
    return llm, tokenizer


# ==================== 推理 + 重试逻辑 ====================

def run_inference_with_retry(
    samples, model_or_llm, tokenizer, use_vllm,
    max_new_tokens, temperature, batch_size, max_retries, retry_temperature,
    lora_path=None,
):
    """
    推理并自动重试提取失败的样本。

    返回: [{id, source, target, prediction, raw_output, extract_reason, retries}, ...]
    """
    final_results = {}
    pending_samples = list(samples)

    for attempt in range(max_retries + 1):
        if not pending_samples:
            break

        is_retry = attempt > 0
        temp = retry_temperature if is_retry else temperature

        print(f"\n{'='*60}")
        if is_retry:
            print(f"[重试 {attempt}/{max_retries}] 待重试: {len(pending_samples)} 条 | temperature={temp}")
        else:
            print(f"[首次推理] 样本数: {len(pending_samples)} | temperature={temp}")
        print(f"{'='*60}")

        if use_vllm:
            raw_results = inference_vllm(
                model_or_llm, tokenizer, pending_samples,
                max_new_tokens, temp, lora_path=lora_path,
            )
        else:
            raw_results = inference_hf(
                model_or_llm, tokenizer, pending_samples, max_new_tokens, temp, batch_size
            )

        need_retry = []
        stats = {}

        for raw_res, sample in zip(raw_results, pending_samples):
            sid = raw_res["id"]
            source = raw_res["source"]
            raw_output = raw_res["raw_output"]

            extracted, reason, should_retry = extract_prediction(raw_output, source)
            stats[reason] = stats.get(reason, 0) + 1

            if should_retry and attempt < max_retries:
                if sid not in final_results or final_results[sid]["_should_retry"]:
                    final_results[sid] = {
                        "id": sid,
                        "source": source,
                        "target": raw_res["target"],
                        "prediction": extracted,
                        "raw_output": raw_output,
                        "extract_reason": reason,
                        "retries": attempt,
                        "_should_retry": True,
                        "_sample": sample,
                    }
                need_retry.append(sample)
            else:
                final_results[sid] = {
                    "id": sid,
                    "source": source,
                    "target": raw_res["target"],
                    "prediction": extracted,
                    "raw_output": raw_output,
                    "extract_reason": reason,
                    "retries": attempt,
                    "_should_retry": False,
                    "_sample": None,
                }

        print(f"\n[提取统计] (第 {attempt+1} 轮)")
        for reason, count in sorted(stats.items(), key=lambda x: -x[1]):
            marker = " ✓" if reason in ("answer_tag", "plain_text") else " ✗ 需重试"
            print(f"  {reason:<25}: {count} 条{marker}")

        if need_retry:
            print(f"\n[结果] 成功: {len(pending_samples)-len(need_retry)} | 待重试: {len(need_retry)}")
        else:
            print(f"\n[结果] 全部提取成功！")

        pending_samples = need_retry

    ordered_results = []
    for sample in samples:
        sid = sample["id"]
        if sid in final_results:
            r = final_results[sid]
            ordered_results.append({
                "id": r["id"],
                "source": r["source"],
                "target": r["target"],
                "prediction": r["prediction"],
                "raw_output": r["raw_output"],
                "extract_reason": r["extract_reason"],
                "retries": r["retries"],
            })

    final_stats = {}
    retry_counts = {}
    for r in ordered_results:
        final_stats[r["extract_reason"]] = final_stats.get(r["extract_reason"], 0) + 1
        retry_counts[r["retries"]] = retry_counts.get(r["retries"], 0) + 1

    print(f"\n{'='*60}")
    print(f"[最终统计] 总计 {len(ordered_results)} 条")
    print(f"{'='*60}")
    print(f"提取方式分布：")
    for reason, count in sorted(final_stats.items(), key=lambda x: -x[1]):
        print(f"  {reason:<25}: {count} 条 ({count/len(ordered_results)*100:.1f}%)")
    print(f"重试次数分布：")
    for retries, count in sorted(retry_counts.items()):
        label = "首次成功" if retries == 0 else f"第{retries}次重试成功"
        print(f"  {label:<25}: {count} 条")

    return ordered_results


# ==================== ChERRANT 评测部分 ====================

def write_parallel_file(sources, corrections, path, multi_ref=False):
    """
    写入 ChERRANT 所需的平行文件格式：每行 id\tsrc\tcor。

    如果 multi_ref=True，corrections 是列表的列表，
    同一个 id 的多个参考会写成多行（ChERRANT 多参考格式）。
    """
    with open(path, "w", encoding="utf-8") as f:
        for i, (src, cor) in enumerate(zip(sources, corrections)):
            src_clean = src.replace("\t", " ").replace("\n", " ").replace("\r", "")

            if multi_ref and isinstance(cor, list):
                for ref in cor:
                    ref_clean = ref.replace("\t", " ").replace("\n", " ").replace("\r", "")
                    if not ref_clean.strip():
                        ref_clean = src_clean
                    f.write(f"{i}\t{src_clean}\t{ref_clean}\n")
            else:
                if isinstance(cor, list):
                    cor = cor[0] if cor else src
                cor_clean = cor.replace("\t", " ").replace("\n", " ").replace("\r", "")
                if not cor_clean.strip():
                    cor_clean = src_clean
                f.write(f"{i}\t{src_clean}\t{cor_clean}\n")


def run_cmd(cmd, desc="", cwd=None):
    print(f"[ChERRANT] {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    if cwd:
        print(f"  CWD: {cwd}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        stderr_lines = result.stderr.split("\n") if result.stderr else []
        real_errors = [l for l in stderr_lines if not l.strip().startswith(("\r", "0%", "|", " ")) and "it/s" not in l and l.strip()]
        if real_errors:
            print(f"  ERROR: {chr(10).join(real_errors[:20])}")
        else:
            print(f"  ERROR (returncode={result.returncode}): 无明显错误信息，可能是进度条输出")
        return None
    return result.stdout


def parse_cherrant_metrics(stdout, beta):
    if not stdout:
        return None
    pattern = r"Span-Based Correction.*?\nTP\s+FP\s+FN\s+Prec\s+Rec\s+F[\d.]+\n" \
              r"(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    m = re.search(pattern, stdout, re.DOTALL)
    if not m:
        print("[WARNING] 未匹配到 ChERRANT 核心指标")
        print(stdout)
        return None
    tp, fp, fn, prec, rec, f_beta = m.groups()
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "Precision": float(prec), "Recall": float(rec), f"F{beta}": float(f_beta)
    }


def find_cherrant_scripts(cherrant_dir=None):
    candidates = []
    if cherrant_dir:
        candidates.append(cherrant_dir)
    env_dir = os.environ.get("CHERRANT_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates.append(os.path.join(os.getcwd(), "ChERRANT"))
    candidates.append(os.path.join(os.getcwd(), "cherrant"))

    for d in candidates:
        d = os.path.abspath(d)
        p2m = os.path.join(d, "parallel_to_m2.py")
        cmp = os.path.join(d, "compare_m2_for_evaluation.py")
        if os.path.isfile(p2m) and os.path.isfile(cmp):
            return d, "parallel_to_m2.py", "compare_m2_for_evaluation.py"

    raise FileNotFoundError(
        "找不到 ChERRANT 脚本 (parallel_to_m2.py / compare_m2_for_evaluation.py)。\n"
        "请通过以下方式之一指定:\n"
        "  --cherrant_dir /path/to/ChERRANT\n"
        "  export CHERRANT_DIR=/path/to/ChERRANT\n"
        "  或将 ChERRANT 目录放在当前工作目录下"
    )


def run_cherrant_evaluation(sources, targets, pred_texts, beta=0.5,
                            save_errant_dir=None, cherrant_dir=None):
    """
    ChERRANT 评测，支持多参考。

    参数:
        sources: [str, ...] 原句列表
        targets: [[str, ...], ...] 多参考列表（每个元素是一个参考句列表）
        pred_texts: [str, ...] 预测列表
    """
    cherrant_base, p2m_name, cmp_name = find_cherrant_scripts(cherrant_dir)
    print(f"[ChERRANT] 脚本目录: {cherrant_base}")

    # 统一 NFKC 归一化
    sources = [unicodedata.normalize("NFKC", s).replace(" ", "") for s in sources]
    targets_norm = []
    for tgt_list in targets:
        if isinstance(tgt_list, list):
            targets_norm.append([
                unicodedata.normalize("NFKC", t).replace(" ", "") for t in tgt_list
            ])
        else:
            targets_norm.append([unicodedata.normalize("NFKC", tgt_list).replace(" ", "")])
    targets = targets_norm
    pred_texts = [unicodedata.normalize("NFKC", p).replace(" ", "") for p in pred_texts]

    # 检测是否有多参考
    has_multi_ref = any(len(tgt_list) > 1 for tgt_list in targets)
    if has_multi_ref:
        multi_ref_counts = [len(t) for t in targets]
        print(f"[ChERRANT] 多参考模式: 参考数分布 min={min(multi_ref_counts)} "
              f"max={max(multi_ref_counts)} avg={sum(multi_ref_counts)/len(multi_ref_counts):.1f}")

    tmp_dir  = os.path.abspath(tempfile.mkdtemp())
    ref_para = os.path.join(tmp_dir, "ref.para")
    hyp_para = os.path.join(tmp_dir, "hyp.para")
    ref_m2   = os.path.join(tmp_dir, "ref.m2")
    hyp_m2   = os.path.join(tmp_dir, "hyp.m2")

    # ref 用多参考格式写入，hyp 正常写入
    write_parallel_file(sources, targets, ref_para, multi_ref=True)
    write_parallel_file(sources, pred_texts, hyp_para, multi_ref=False)

    run_cmd(
        ["python", p2m_name, "-f", ref_para, "-o", ref_m2, "-g", "word"],
        "生成 ref.m2（多参考）", cwd=cherrant_base
    )
    run_cmd(
        ["python", p2m_name, "-f", hyp_para, "-o", hyp_m2, "-g", "word"],
        "生成 hyp.m2", cwd=cherrant_base
    )

    compare_stdout = run_cmd(
        ["python", cmp_name, "-hyp", hyp_m2, "-ref", ref_m2, "-b", str(beta)],
        "对比 hyp.m2 vs ref.m2", cwd=cherrant_base
    )

    core_metrics = parse_cherrant_metrics(compare_stdout, beta)

    # 过校正率：只要预测匹配任一参考，就不算过校正
    total_correct = modified_correct = 0
    for src, tgt_list, pred in zip(sources, targets, pred_texts):
        # 原句本身就是正确的（所有参考都等于原句）
        if all(t == src for t in tgt_list):
            total_correct += 1
            if pred != src:
                modified_correct += 1
    oc_rate = modified_correct / total_correct if total_correct > 0 else 0.0

    # 编辑距离（相对于原句）
    total_ed = 0
    if editdistance:
        for src, pred in zip(sources, pred_texts):
            total_ed += editdistance.eval(src, pred)
    avg_ed = total_ed / len(sources) if sources else 0.0

    if save_errant_dir:
        import shutil
        os.makedirs(save_errant_dir, exist_ok=True)
        for fname in ["ref.para", "hyp.para", "ref.m2", "hyp.m2"]:
            p = os.path.join(tmp_dir, fname)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(save_errant_dir, fname))
        with open(os.path.join(save_errant_dir, "compare.txt"), "w", encoding="utf-8") as f:
            for i in range(len(sources)):
                tgt_display = " | ".join(targets[i])
                f.write(f"[{i}]\n  SRC: {sources[i]}\n  REF: {tgt_display}\n  HYP: {pred_texts[i]}\n\n")
        print(f"[INFO] ChERRANT 文件已保存到：{save_errant_dir}/")

    return {
        "core_metrics": core_metrics,
        "overcorrection": {
            "total_correct_src": total_correct,
            "modified_correct_src": modified_correct,
            "rate": round(oc_rate, 4),
        },
        "edit_distance": {
            "total": total_ed,
            "average": round(avg_ed, 4),
        },
    }


def print_metrics(metrics, beta):
    print(f"\n{'='*60}")
    print(f"  中文 GEC 评估结果（ChERRANT 词级别，beta={beta}）")
    print(f"{'='*60}")

    cm = metrics["core_metrics"]
    if cm:
        print(f"\n【核心指标】")
        for k, v in cm.items():
            print(f"  {k:<12}: {v}")
    else:
        print("\n【核心指标】解析失败（请检查 ChERRANT 是否已安装）")

    oc = metrics["overcorrection"]
    print(f"\n【过校正率】")
    print(f"  正确原句: {oc['total_correct_src']} | 被误改: {oc['modified_correct_src']} | 率: {oc['rate']*100:.2f}%")

    ed = metrics["edit_distance"]
    print(f"\n【编辑距离】")
    print(f"  总计: {ed['total']} | 平均: {ed['average']}")
    print(f"{'='*60}")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="中文 GEC 推理 + ChERRANT 评测（含自动重试 + 多参考）")

    parser.add_argument("--model_path",   type=str, default=None, help="模型路径（base 或 merged）")
    parser.add_argument("--lora_path",    type=str, default=None, help="LoRA adapter 路径（不合并时使用）")
    parser.add_argument("--lora_rank",    type=int, default=64,   help="训练时的 lora_rank")
    parser.add_argument("--input",        type=str, required=True, help="测试集 JSON/JSONL 文件")
    parser.add_argument("--output",       type=str, required=True, help="预测输出 JSONL")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature",  type=float, default=0.0, help="首次推理温度")
    parser.add_argument("--batch_size",   type=int, default=256)
    parser.add_argument("--use_vllm",     action="store_true")
    parser.add_argument("--max_retries",  type=int, default=3)
    parser.add_argument("--retry_temperature", type=float, default=1)
    parser.add_argument("--voting_samples",  type=int, default=0,
                        help="多样本投票候选数（0=关闭，用贪心+重试；>0=投票模式，如 1,4,8,16,32）")
    parser.add_argument("--voting_temperature", type=float, default=1.0,
                        help="投票模式下的采样温度")
    parser.add_argument("--beta",         type=float, default=0.5)
    parser.add_argument("--save_errant_dir", type=str, default=None)
    parser.add_argument("--cherrant_dir", type=str, default=None, help="ChERRANT 脚本目录路径")
    parser.add_argument("--save_metrics", type=str, default=None)
    parser.add_argument("--eval_only",    action="store_true")
    parser.add_argument("--skip_eval",    action="store_true")

    args = parser.parse_args()

    # ── 加载测试集 ──
    samples = load_test_data(args.input)
    print(f"[INIT] 加载测试集：{len(samples)} 条 from {args.input}")

    # 统计多参考情况
    multi_ref_count = sum(1 for s in samples if len(s["target"]) > 1)
    if multi_ref_count > 0:
        print(f"[INIT] 多参考样本: {multi_ref_count}/{len(samples)} 条")

    print(f"[INIT] Prompt 模板:\n{USER_PROMPT_TEMPLATE[:100]}...")

    # ── 阶段1: 推理 ──
    if args.eval_only:
        output_path = args.output
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, "pred.jsonl")
        print(f"[MODE] eval_only: 加载已有预测 {output_path}")
        raw_results = load_jsonl(output_path)

        # 从测试集获取 target（多参考）
        id_to_target = {s["id"]: s["target"] for s in samples}

        results = []
        for r in raw_results:
            source = r.get("source", "")
            raw_output = r.get("raw_output", r.get("prediction", r.get("response", "")))
            extracted, reason, _ = extract_prediction(raw_output, source)
            rid = r.get("id", "")
            # 优先用测试集中的 target（列表格式）
            target = id_to_target.get(rid, r.get("target", ""))
            target = normalize_target(target)
            results.append({
                "id": rid,
                "source": source,
                "target": target,
                "prediction": extracted,
                "raw_output": raw_output,
                "extract_reason": reason,
                "retries": 0,
            })

        stats = {}
        for r in results:
            stats[r["extract_reason"]] = stats.get(r["extract_reason"], 0) + 1
        print(f"[提取统计]")
        for reason, count in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"  {reason:<25}: {count} 条 ({count/len(results)*100:.1f}%)")

    else:
        if not args.model_path:
            parser.error("推理模式需要 --model_path")

        print(f"[INIT] 加载模型: {args.model_path}")
        if args.lora_path:
            print(f"[INIT] LoRA adapter: {args.lora_path}  lora_rank={args.lora_rank}")

        if args.use_vllm:
            model_or_llm, tokenizer = load_model_vllm(
                args.model_path, args.lora_path, lora_rank=args.lora_rank
            )
        else:
            model_or_llm, tokenizer = load_model_hf(args.model_path, args.lora_path)

        # ── 投票模式 vs 贪心+重试模式 ──
        if args.voting_samples > 0:
            print(f"[MODE] 多样本投票: {args.voting_samples} 个候选, temperature={args.voting_temperature}")
            results = run_voting_inference(
                samples=samples,
                model_or_llm=model_or_llm,
                tokenizer=tokenizer,
                use_vllm=args.use_vllm,
                max_new_tokens=args.max_new_tokens,
                voting_samples=args.voting_samples,
                voting_temperature=args.voting_temperature,
                batch_size=args.batch_size,
                lora_path=args.lora_path,
            )
        else:
            print(f"[MODE] 贪心解码 + 自动重试 (max_retries={args.max_retries})")
            results = run_inference_with_retry(
                samples=samples,
                model_or_llm=model_or_llm,
                tokenizer=tokenizer,
                use_vllm=args.use_vllm,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                batch_size=args.batch_size,
                max_retries=args.max_retries,
                retry_temperature=args.retry_temperature,
                lora_path=args.lora_path,
            )

        actual_path = save_jsonl(results, args.output)
        args.output = actual_path
        print(f"\n[SAVE] 预测结果已保存：{actual_path} ({len(results)} 条)")

    if args.skip_eval:
        print("[MODE] skip_eval: 跳过评测")
        return

    # ── 阶段2: 评测 ──
    print(f"\n{'='*60}")
    print("[EVAL] 开始 ChERRANT 评测")
    print(f"{'='*60}")

    sources    = [r["source"] for r in results]
    targets    = [r["target"] for r in results]  # 列表的列表
    pred_texts = [r["prediction"] for r in results]

    metrics = run_cherrant_evaluation(
        sources, targets, pred_texts,
        beta=args.beta,
        save_errant_dir=args.save_errant_dir,
        cherrant_dir=args.cherrant_dir,
    )

    print_metrics(metrics, args.beta)

    if args.save_metrics:
        os.makedirs(os.path.dirname(args.save_metrics) or ".", exist_ok=True)
        with open(args.save_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        print(f"[SAVE] 指标已保存：{args.save_metrics}")


if __name__ == "__main__":
    main()