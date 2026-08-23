"""
中文 GEC 推理 + 评测一体化脚本

【关键修复】对原版的修改:
  1. write_parallel_file 多参考改为 ChERRANT 官方格式: id\\tsrc\\tref1\\tref2 (单行 tab 分隔)
  2. 新增 to_cn_punct(): 半角标点 -> 中文全角标点(，；：！？（）), 全/半角为 1:1 字符替换,
     统一到"中文默认标点"。
  3. 预测在 extract_prediction 出口即全角化 -> 投票时"只差标点宽度"的候选自动合并,
     同时保证 hyp 与 ref 字符规范一致。
  4. 评测端 (run_cherrant_evaluation) 对 source/pred/target 无条件全角化;
     参考 M2 用 normalize_ref_m2_punct() 全角化并丢弃"纯标点宽度"的 identity edit,
     从而不再把全/半角差异误判为漏纠/误纠。
  5. NaSGEC/MuCGEC 等字符级 M2 数据集自动用 -g char (而非 -g word)
  6. 新增 --reference_m2 选项: 直接用官方 M2 文件作为参考(推荐用于 NaSGEC/MuCGEC)

用法:
  # 用官方 M2 直接评测（推荐）
  python inference_eval.py \\
      --model_path model/merged --use_vllm \\
      --input data/NaSGEC-Exam/nasgec.exam.test.input \\
      --reference_m2 data/NaSGEC-Exam/nasgec.exam.test.m2 \\
      --output output_data/nas.jsonl \\
      --cherrant_dir MuCGEC/scorers/ChERRANT

  # 多参考 JSON 评测
  python inference_eval.py \\
      --model_path model/merged --use_vllm \\
      --input processed_data/test.json \\
      --output output_data/pred.jsonl \\
      --cherrant_dir MuCGEC/scorers/ChERRANT

  # 仅评测已有预测
  python inference_eval.py \\
      --input data/test.json \\
      --output output_data/pred.jsonl \\
      --cherrant_dir MuCGEC/scorers/ChERRANT \\
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


# ==================== 标点统一（中文默认全角） ====================

# 半角 -> 全角 中文标点映射。全/半角为 1:1 字符替换, 不改变字符数,
# 因此可安全用于 M2 (A 行 offset 保持有效)。
# 注意:
#   - 不映射 '.' (会破坏小数 3.14 / 英文缩写)
#   - 不映射引号 (需区分左右 “ ” ‘ ’, 无法用简单字符表)
#   - ':' 会影响时间写法如 3:00; 若你的数据里存在此类情况, 可从下表删掉 ':'
_HALF2FULL_PUNCT = {
    ",": "，",
    ";": "；",
    ":": "：",
    "!": "！",
    "?": "？",
    "(": "（",
    ")": "）",
}
_H2F_TABLE = str.maketrans(_HALF2FULL_PUNCT)


def to_cn_punct(text: str) -> str:
    """把半角标点统一为中文全角标点（中文默认规范）。
    全/半角 1:1 替换, 字符数不变, 可安全用于 M2 (offset 不变)。"""
    if not text:
        return text
    return text.translate(_H2F_TABLE)


# ==================== Prompt 定义 ====================

SYSTEM_PROMPT = ""

USER_PROMPT_TEMPLATE = """你是一名中文语义纠错专家。请先进行简要思考，再完成任务。\n\n【任务说明】\n\n下面给出的句子可能存在语义层面的问题，包括但不限于：\n用词不当、语义搭配不合理、概念使用不准确、语义逻辑不通顺或存在歧义。\n\n【思考要求】\n\n请先进行简要分析：\n- 判断句子是否存在语义问题\n- 若存在，指出问题类型（如用词不当、搭配问题、逻辑问题、冗余等）\n- 给出最小修改方案\n- 若不存在问题，说明无需修改,并输出原句\n\n【修改原则】\n\n保持原句核心含义不变\n仅做最少且必要的修改\n优先删除冗余或冲突表达\n避免重写句子\n\n【输出格式要求】\n\n严格按照以下格式：\n<think>简要分析</think>\n<answer>修改后的句子或原句</answer>\n\n你要修改的句子为：
{source}"""


# ==================== 文本提取工具 ====================

def strip_template_tokens(text: str) -> str:
    """
    去除 chat template 特殊 token 和 <think> 标签。
    不做 NFKC（NFKC 会把 ，？！；：等全角转半角），标点统一交给 to_cn_punct。
    """
    if not text:
        return ""
    text = re.sub(r"<\|im_start\|>.*?\n?", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"<\|endoftext\|>", "", text)
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    return text.strip()


def _extract_prediction_core(raw_output: str, source: str = ""):
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
        if src_len > 0 and len(text) > src_len * 2:
            return text, "too_long", True
        return text, reason, False

    m = re.search(r"<answer>(.*?)</answer>", cleaned, flags=re.DOTALL)
    if m:
        answer = m.group(1).strip()
        placeholders = ["修改后的句子", "纠错后的句子", "正确的句子", "corrected sentence"]
        if answer in placeholders or len(answer) < 2:
            return source.replace(" ", ""), "placeholder", True
        return _check_too_long(answer.replace(" ", ""), "answer_tag")

    m2 = re.search(r"<answer>(.*)", cleaned, flags=re.DOTALL)
    if m2:
        answer = m2.group(1).strip()
        if len(answer) >= 2:
            return answer.replace(" ", ""), "answer_tag_unclosed", True

    text_only = re.sub(r"<thinking>[\s\S]*?</thinking>\s*", "", cleaned)
    text_only = re.sub(r"<thinking>[\s\S]*$", "", text_only)
    text_only = text_only.strip()

    if len(text_only) >= 2:
        if "<thinking>" in text_only:
            return source.replace(" ", ""), "dirty_text", True
        return _check_too_long(text_only.replace(" ", ""), "plain_text")

    return source.replace(" ", ""), "no_answer", True


def extract_prediction(raw_output: str, source: str = ""):
    """extract + 标点全角化。全角化放在最外层, 保证:
       (1) 投票 Counter 时"只差标点宽度"的候选合并 -> 减少稀释, 提升召回;
       (2) 保存到 jsonl / 评测的字符规范统一为中文全角。"""
    text, reason, retry = _extract_prediction_core(raw_output, source)
    return to_cn_punct(text), reason, retry


# ==================== 数据 IO ====================

def normalize_target(target):
    """统一为 list 格式（多参考）"""
    if isinstance(target, list):
        return target if target else [""]
    elif isinstance(target, str):
        return [target] if target else [""]
    else:
        return [""]


def _normalize_for_match(s: str) -> str:
    """
    用于 source <-> M2 匹配的归一化（容忍全/半角差异）。
    [说明] 仅作为字典 key，不影响保存到 jsonl 的实际字符。
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace(" ", "").replace("\u3000", "").strip()
    return s


def _detect_format(path):
    with open(path, "r", encoding="utf-8") as f:
        first_char = ""
        for _ in range(100):
            ch = f.read(1)
            if not ch:
                break
            if ch.strip():
                first_char = ch
                break
    if first_char in ("{", "["):
        return "json"
    return "plain"


def load_targets_from_m2(m2_path):
    """
    解析 M2 文件,返回 {source_match_key: [target_match_key, ...]}
    （source/target 都用 NFKC + 去空格做 key,仅用于匹配）
    """
    if not os.path.exists(m2_path):
        raise FileNotFoundError(f"M2 文件不存在: {m2_path}")

    with open(m2_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())

    result = {}
    n_sentences = 0
    n_total_refs = 0

    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().split("\n")
        source = None
        targets = []
        for line in lines:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("S "):
                source = _normalize_for_match(line[2:].strip())
            elif re.match(r"T\d+-A\d+\s", line) or re.match(r"T\d+\s", line):
                parts = line.split(" ", 1)
                if len(parts) >= 2:
                    target = _normalize_for_match(parts[1])
                    if target and target != "没有错误":
                        targets.append(target)
        if source:
            if not targets:
                targets = [source]
            result[source] = targets
            n_sentences += 1
            n_total_refs += len(targets)

    print(f"[M2-LOAD] 从 {m2_path} 解析: {n_sentences} 句, 共 {n_total_refs} 个参考 "
          f"(平均 {n_total_refs/max(n_sentences,1):.2f}/句)")
    return result


def _load_json_format(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.splitlines() if line.strip()]
    for i, item in enumerate(data):
        item["target"] = normalize_target(item.get("target", ""))
        item.setdefault("id", str(i))
    return data


def _load_plain_format(path, m2_targets=None):
    with open(path, "r", encoding="utf-8") as f:
        sources = [line.rstrip("\n\r").strip() for line in f if line.strip()]

    data = []
    n_matched = n_unmatched = 0
    unmatched_examples = []
    for i, src in enumerate(sources):
        src_key = _normalize_for_match(src)
        if m2_targets is not None and src_key in m2_targets:
            target = m2_targets[src_key]
            n_matched += 1
        else:
            target = [src]
            if m2_targets is not None:
                n_unmatched += 1
                if len(unmatched_examples) < 3:
                    unmatched_examples.append(src[:50])
        data.append({"id": str(i), "source": src, "target": target})

    if m2_targets is not None:
        print(f"[INPUT-MATCH] 与 M2 匹配: {n_matched}/{len(sources)} ({n_matched/len(sources)*100:.1f}%)")
        if n_unmatched:
            print(f"[WARN] {n_unmatched} 条 input 在 M2 中找不到")
            for ex in unmatched_examples:
                print(f"  未匹配示例: {ex}...")
    return data


def load_test_data(path, reference_m2=None):
    fmt = _detect_format(path)
    print(f"[INPUT] 文件格式: {fmt} ({path})")

    m2_targets = None
    if reference_m2:
        m2_targets = load_targets_from_m2(reference_m2)

    if fmt == "json":
        data = _load_json_format(path)
        if m2_targets is not None:
            n = 0
            for item in data:
                key = _normalize_for_match(item["source"])
                if key in m2_targets:
                    item["target"] = m2_targets[key]
                    n += 1
            print(f"[INPUT] 从 M2 覆盖 target: {n}/{len(data)} 条")
        return data
    return _load_plain_format(path, m2_targets=m2_targets)


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
    """仅去空格并统一为中文全角标点, 让模型看到自然、规范的中文输入。"""
    clean = to_cn_punct(source_text.replace(" ", "").replace("\u3000", ""))
    user_content = USER_PROMPT_TEMPLATE.format(source=clean)
    msgs = []
    if SYSTEM_PROMPT:
        msgs.append({"role": "system", "content": SYSTEM_PROMPT})
    msgs.append({"role": "user", "content": user_content})
    return msgs


# ==================== 推理 ====================

def inference_hf(model, tokenizer, samples, max_new_tokens, temperature, batch_size):
    import torch
    results = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Generating"):
        batch = samples[i:i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(s["source"]), tokenize=False,
                add_generation_prompt=True, enable_thinking=False
            ) for s in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0, pad_token_id=tokenizer.pad_token_id,
            )
        for j, s in enumerate(batch):
            input_len = inputs["input_ids"].shape[1]
            generated = outputs[j][input_len:]
            raw_pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append({
                "id": s["id"], "source": s["source"], "target": s["target"], "raw_output": raw_pred,
            })
    return results


def inference_vllm(llm, tokenizer, samples, max_new_tokens, temperature, lora_path=None):
    from vllm import SamplingParams
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(s["source"]), tokenize=False,
            add_generation_prompt=True, enable_thinking=False
        ) for s in samples
    ]
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=temperature if temperature > 0 else 0)
    lora_request = None
    if lora_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("gec_lora", 1, lora_path)
    outputs = llm.generate(prompts, sp, lora_request=lora_request)
    results = []
    for s, out in zip(samples, outputs):
        results.append({
            "id": s["id"], "source": s["source"], "target": s["target"],
            "raw_output": out.outputs[0].text.strip(),
        })
    return results


# ==================== 投票推理 ====================

def majority_vote(candidates, source):
    # extract_prediction 已做全角化, 故"仅差标点宽度"的候选会落进同一票桶
    extracted = [extract_prediction(raw, source) for raw in candidates]
    valid = [p for p, _, retry in extracted if not retry]
    if valid:
        c = Counter(valid)
        winner, n = c.most_common(1)[0]
        return winner, {
            "total_candidates": len(candidates), "valid_candidates": len(valid),
            "winner_count": n, "unique_preds": len(c),
            "agreement_rate": round(n / len(valid), 4),
        }
    all_preds = [p for p, _, _ in extracted]
    c = Counter(all_preds)
    winner, n = c.most_common(1)[0]
    return winner, {
        "total_candidates": len(candidates), "valid_candidates": 0,
        "winner_count": n, "unique_preds": len(c), "agreement_rate": 0.0,
    }


def inference_vllm_voting(llm, tokenizer, samples, max_new_tokens, voting_samples, voting_temperature, lora_path=None):
    from vllm import SamplingParams
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(s["source"]), tokenize=False,
            add_generation_prompt=True, enable_thinking=False
        ) for s in samples
    ]
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=voting_temperature, n=voting_samples)
    lora_request = None
    if lora_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("gec_lora", 1, lora_path)
    outputs = llm.generate(prompts, sp, lora_request=lora_request)
    results = []
    for s, out in zip(samples, outputs):
        cands = [o.text.strip() for o in out.outputs]
        winner, vi = majority_vote(cands, s["source"])
        results.append({
            "id": s["id"], "source": s["source"], "target": s["target"],
            "prediction": winner, "raw_output": cands[0],
            "all_candidates": cands, "vote_info": vi, "extract_reason": "voting",
        })
    return results


def inference_hf_voting(model, tokenizer, samples, max_new_tokens, voting_samples, voting_temperature, batch_size):
    import torch
    results = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Voting"):
        batch = samples[i:i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(s["source"]), tokenize=False,
                add_generation_prompt=True, enable_thinking=False
            ) for s in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        all_cands = [[] for _ in batch]
        for _ in range(voting_samples):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=voting_temperature, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for j in range(len(batch)):
                input_len = inputs["input_ids"].shape[1]
                generated = outputs[j][input_len:]
                raw_pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
                all_cands[j].append(raw_pred)
        for j, s in enumerate(batch):
            winner, vi = majority_vote(all_cands[j], s["source"])
            results.append({
                "id": s["id"], "source": s["source"], "target": s["target"],
                "prediction": winner, "raw_output": all_cands[j][0],
                "all_candidates": all_cands[j], "vote_info": vi, "extract_reason": "voting",
            })
    return results


def run_voting_inference(samples, model_or_llm, tokenizer, use_vllm,
                         max_new_tokens, voting_samples, voting_temperature, batch_size, lora_path=None):
    print(f"\n{'='*60}\n[投票推理] 样本数: {len(samples)} | 候选: {voting_samples} | t={voting_temperature}\n{'='*60}")
    if use_vllm:
        results = inference_vllm_voting(model_or_llm, tokenizer, samples, max_new_tokens,
                                        voting_samples, voting_temperature, lora_path)
    else:
        results = inference_hf_voting(model_or_llm, tokenizer, samples, max_new_tokens,
                                      voting_samples, voting_temperature, batch_size)
    n = len(results)
    avg_agree = sum(r["vote_info"]["agreement_rate"] for r in results) / n
    avg_valid = sum(r["vote_info"]["valid_candidates"] for r in results) / n
    avg_unique = sum(r["vote_info"]["unique_preds"] for r in results) / n
    print(f"\n[投票统计] n={n} | 平均一致率: {avg_agree*100:.1f}% | "
          f"平均有效候选: {avg_valid:.1f}/{voting_samples} | 平均不同预测: {avg_unique:.1f}")
    return results


def load_model_hf(model_path, lora_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
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
    kwargs = dict(model=model_path, trust_remote_code=True, dtype="bfloat16",
                  max_model_len=2048, language_model_only=True, tensor_parallel_size=4)
    if lora_path:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = lora_rank
    return LLM(**kwargs), tokenizer


# ==================== 推理 + 重试 ====================

def run_inference_with_retry(samples, model_or_llm, tokenizer, use_vllm,
                             max_new_tokens, temperature, batch_size,
                             max_retries, retry_temperature, lora_path=None):
    final_results = {}
    pending = list(samples)
    for attempt in range(max_retries + 1):
        if not pending:
            break
        is_retry = attempt > 0
        temp = retry_temperature if is_retry else temperature
        print(f"\n{'='*60}")
        print(f"[{'重试 ' + str(attempt) + '/' + str(max_retries) if is_retry else '首次推理'}] "
              f"样本: {len(pending)} | t={temp}\n{'='*60}")

        if use_vllm:
            raw_results = inference_vllm(model_or_llm, tokenizer, pending,
                                         max_new_tokens, temp, lora_path=lora_path)
        else:
            raw_results = inference_hf(model_or_llm, tokenizer, pending,
                                       max_new_tokens, temp, batch_size)

        need_retry = []
        stats = {}
        for raw_res, sample in zip(raw_results, pending):
            sid = raw_res["id"]
            extracted, reason, should_retry = extract_prediction(raw_res["raw_output"], raw_res["source"])
            stats[reason] = stats.get(reason, 0) + 1
            entry = {
                "id": sid, "source": raw_res["source"], "target": raw_res["target"],
                "prediction": extracted, "raw_output": raw_res["raw_output"],
                "extract_reason": reason, "retries": attempt,
                "_should_retry": should_retry and attempt < max_retries,
            }
            if entry["_should_retry"]:
                if sid not in final_results or final_results[sid]["_should_retry"]:
                    final_results[sid] = entry
                need_retry.append(sample)
            else:
                final_results[sid] = entry

        print(f"\n[提取统计] (第 {attempt+1} 轮)")
        for reason, count in sorted(stats.items(), key=lambda x: -x[1]):
            mark = " ✓" if reason in ("answer_tag", "plain_text") else " ✗ 需重试"
            print(f"  {reason:<25}: {count}{mark}")
        print(f"\n[结果] 成功: {len(pending)-len(need_retry)} | 待重试: {len(need_retry)}")
        pending = need_retry

    ordered = []
    for s in samples:
        if s["id"] in final_results:
            r = final_results[s["id"]]
            ordered.append({k: v for k, v in r.items() if not k.startswith("_")})

    final_stats = {}
    for r in ordered:
        final_stats[r["extract_reason"]] = final_stats.get(r["extract_reason"], 0) + 1
    print(f"\n{'='*60}\n[最终统计] 总计 {len(ordered)} 条\n{'='*60}")
    for reason, c in sorted(final_stats.items(), key=lambda x: -x[1]):
        print(f"  {reason:<25}: {c} ({c/len(ordered)*100:.1f}%)")
    return ordered


# ==================== ChERRANT 评测 ====================

def write_parallel_file(sources, corrections, path, multi_ref=False):
    """
    多参考: ChERRANT 官方格式 id\\tsrc\\tref1\\tref2\\tref3 (单行 tab 分隔多个 ref)
    """
    def _clean(s):
        return s.replace("\t", " ").replace("\n", " ").replace("\r", "")

    with open(path, "w", encoding="utf-8") as f:
        for i, (src, cor) in enumerate(zip(sources, corrections)):
            src_c = _clean(src)
            if multi_ref and isinstance(cor, list) and len(cor) > 0:
                refs = [_clean(r) if r and r.strip() else src_c for r in cor]
                f.write(f"{i}\t{src_c}\t" + "\t".join(refs) + "\n")
            else:
                if isinstance(cor, list):
                    cor = cor[0] if cor else src
                cor_c = _clean(cor)
                if not cor_c.strip():
                    cor_c = src_c
                f.write(f"{i}\t{src_c}\t{cor_c}\n")


def normalize_ref_m2_punct(in_m2, out_m2):
    """
    把官方 M2 统一为中文全角标点, 写到 out_m2:
      - S 行、A 行 correction 全角化 (全/半角 1:1, offset 不变);
      - 归一后 correction == 源 span 的 edit (纯全/半角宽度差异) 丢弃;
      - 某标注者所有 edit 被清空时补一条 noop 行, 避免评测报错。
    真正的语义/标点(如 ，->。)edit 全部原样保留。
    """
    with open(in_m2, "r", encoding="utf-8") as f:
        blocks = re.split(r"\n\s*\n", f.read().strip())

    out_blocks = []
    n_drop = 0
    for block in blocks:
        lines = [l for l in block.split("\n") if l.strip()]
        s_toks = None
        block_out = []
        ann_has_edit = {}   # annotator_id -> 是否仍保留至少一条 edit
        for l in lines:
            if l.startswith("S "):
                s_toks = [to_cn_punct(t) for t in l[2:].split(" ")]
                block_out.append("S " + " ".join(s_toks))
            elif l.startswith("A "):
                parts = l[2:].split("|||")
                if len(parts) < 3:
                    block_out.append(l)
                    continue
                off = parts[0].split()
                aid = parts[-1].strip() or "0"
                try:
                    st, en = int(off[0]), int(off[1])
                except (ValueError, IndexError):
                    block_out.append(l)
                    continue
                # noop 行: 先记账, 稍后按需补
                if parts[1] == "noop" or st < 0:
                    ann_has_edit.setdefault(aid, False)
                    continue
                corr = to_cn_punct(parts[2].replace(" ", ""))
                src_span = "".join(s_toks[st:en]) if s_toks is not None else None
                # 纯宽度差异 -> 归一后变 identity -> 丢弃
                if src_span is not None and corr == src_span:
                    n_drop += 1
                    ann_has_edit.setdefault(aid, False)
                    continue
                new_parts = parts[:]
                new_parts[2] = to_cn_punct(parts[2])
                block_out.append("A " + "|||".join(new_parts))
                ann_has_edit[aid] = True
            else:
                block_out.append(l)
        # 为被清空的标注者补 noop
        for aid, has in ann_has_edit.items():
            if not has:
                block_out.append(f"A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||{aid}")
        # 整句一条 A 都没有时补默认 noop
        if len(block_out) == 1 and block_out[0].startswith("S "):
            block_out.append("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0")
        out_blocks.append("\n".join(block_out))

    with open(out_m2, "w", encoding="utf-8") as f:
        f.write("\n\n".join(out_blocks) + "\n")
    print(f"[M2-NORM] 参考 M2 全角化, 丢弃纯标点宽度 edit: {n_drop} 条 -> {out_m2}")
    return out_m2


def run_cmd(cmd, desc="", cwd=None):
    print(f"[ChERRANT] {desc}\n  CMD: {' '.join(cmd)}")
    if cwd:
        print(f"  CWD: {cwd}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        errs = [l for l in (result.stderr or "").split("\n")
                if l.strip() and not l.strip().startswith(("\r", "0%", "|", " ")) and "it/s" not in l]
        if errs:
            print(f"  ERROR: {chr(10).join(errs[:20])}")
        else:
            print(f"  ERROR (returncode={result.returncode})")
        return None
    return result.stdout


def parse_cherrant_metrics(stdout, beta):
    if not stdout:
        return None
    pat = r"Span-Based Correction.*?\nTP\s+FP\s+FN\s+Prec\s+Rec\s+F[\d.]+\n" \
          r"(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    m = re.search(pat, stdout, re.DOTALL)
    if not m:
        print("[WARNING] 未匹配到 ChERRANT 核心指标")
        print(stdout)
        return None
    tp, fp, fn, prec, rec, fb = m.groups()
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn),
            "Precision": float(prec), "Recall": float(rec), f"F{beta}": float(fb)}


def find_cherrant_scripts(cherrant_dir=None):
    cands = []
    if cherrant_dir:
        cands.append(cherrant_dir)
    if os.environ.get("CHERRANT_DIR"):
        cands.append(os.environ["CHERRANT_DIR"])
    cands.append(os.path.join(os.getcwd(), "ChERRANT"))
    cands.append(os.path.join(os.getcwd(), "cherrant"))
    for d in cands:
        d = os.path.abspath(d)
        p2m = os.path.join(d, "parallel_to_m2.py")
        cmp = os.path.join(d, "compare_m2_for_evaluation.py")
        if os.path.isfile(p2m) and os.path.isfile(cmp):
            return d, "parallel_to_m2.py", "compare_m2_for_evaluation.py"
    raise FileNotFoundError("找不到 ChERRANT 脚本，请用 --cherrant_dir 指定")


def load_m2_raw_sources(m2_path):
    """从 M2 直接读 S 行, 全角化后返回 {NFKC化key: 全角src(去空格)}
       用于把 hyp 的 src 对齐到与(全角化后)参考 M2 完全一致的字符规范。"""
    if not os.path.isfile(m2_path):
        return {}
    lookup = {}
    with open(m2_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("S "):
                s_orig = line[2:].rstrip("\n\r").replace(" ", "").replace("\u3000", "")
                if s_orig:
                    key = unicodedata.normalize("NFKC", s_orig)
                    lookup[key] = to_cn_punct(s_orig)
    return lookup


def run_cherrant_evaluation(sources, targets, pred_texts, beta=0.5,
                            save_errant_dir=None, cherrant_dir=None,
                            ref_m2_path=None, input_path=None, granularity=None):
    """
    统一为中文全角标点后再评测:
      - source/pred/target 全角化;
      - 参考 M2 用 normalize_ref_m2_punct 全角化 + 丢弃纯宽度 edit;
      - NaSGEC/MuCGEC 自动 -g char。
    """
    cherrant_base, p2m, cmp = find_cherrant_scripts(cherrant_dir)
    print(f"[ChERRANT] 脚本目录: {cherrant_base}")

    is_char_dataset = False
    for kw in ("nasgec", "mucgec"):
        if (input_path and kw in input_path.lower()) or (ref_m2_path and kw in ref_m2_path.lower()):
            is_char_dataset = True

    if granularity is None:
        granularity = "char" if is_char_dataset else "word"
    print(f"[ChERRANT] 粒度: -g {granularity}")

    # 去空格
    sources    = [s.replace(" ", "").replace("\u3000", "") for s in sources]
    pred_texts = [p.replace(" ", "").replace("\u3000", "") for p in pred_texts]
    targets = [
        [t.replace(" ", "").replace("\u3000", "") for t in (tl if isinstance(tl, list) else [tl])]
        for tl in targets
    ]

    # 用 ref.m2 的(全角化)src 覆盖 sources, 保证 hyp / ref 字符规范一致
    n_aligned = 0
    if ref_m2_path:
        m2_src = load_m2_raw_sources(ref_m2_path)
        if m2_src:
            new_src = []
            for s in sources:
                key = unicodedata.normalize("NFKC", s)
                if key in m2_src:
                    new_src.append(m2_src[key])
                    n_aligned += 1
                else:
                    new_src.append(s)
            sources = new_src
            print(f"[ChERRANT] 从 M2 同步 src 字符规范: {n_aligned}/{len(sources)}")

    # 统一为中文全角标点 (hyp 侧: source/pred/target)
    sources    = [to_cn_punct(s) for s in sources]
    pred_texts = [to_cn_punct(p) for p in pred_texts]
    targets    = [[to_cn_punct(t) for t in tl] for tl in targets]

    tmp = os.path.abspath(tempfile.mkdtemp())
    hyp_para = os.path.join(tmp, "hyp.para")
    hyp_m2   = os.path.join(tmp, "hyp.m2")

    if ref_m2_path:
        raw_ref = os.path.abspath(ref_m2_path)
        if not os.path.isfile(raw_ref):
            raise FileNotFoundError(f"参考 M2 文件不存在: {raw_ref}")
        # 全角化参考 M2 (丢弃纯宽度 edit)
        ref_m2 = os.path.join(tmp, "ref.norm.m2")
        normalize_ref_m2_punct(raw_ref, ref_m2)
        print(f"[ChERRANT] 使用(全角化后)参考 M2: {ref_m2}")
    else:
        ref_para = os.path.join(tmp, "ref.para")
        ref_m2   = os.path.join(tmp, "ref.m2")
        has_multi = any(len(t) > 1 for t in targets)
        if has_multi:
            cnts = [len(t) for t in targets]
            print(f"[ChERRANT] 多参考: min={min(cnts)} max={max(cnts)} avg={sum(cnts)/len(cnts):.1f}")
        write_parallel_file(sources, targets, ref_para, multi_ref=True)
        run_cmd(["python", p2m, "-f", ref_para, "-o", ref_m2, "-g", granularity],
                f"生成 ref.m2 (-g {granularity})", cwd=cherrant_base)

    write_parallel_file(sources, pred_texts, hyp_para, multi_ref=False)
    run_cmd(["python", p2m, "-f", hyp_para, "-o", hyp_m2, "-g", granularity],
            f"生成 hyp.m2 (-g {granularity})", cwd=cherrant_base)
    compare_stdout = run_cmd(
        ["python", cmp, "-hyp", hyp_m2, "-ref", ref_m2, "-b", str(beta)],
        "对比 hyp.m2 vs ref.m2", cwd=cherrant_base
    )
    core = parse_cherrant_metrics(compare_stdout, beta)

    # 过校正率 / 编辑距离 (全角化后已一致, 这里直接比较)
    total_correct = modified_correct = 0
    for s, tl, p in zip(sources, targets, pred_texts):
        if all(t == s for t in tl):
            total_correct += 1
            if p != s:
                modified_correct += 1
    oc = modified_correct / total_correct if total_correct else 0.0

    total_ed = 0
    if editdistance:
        for s, p in zip(sources, pred_texts):
            total_ed += editdistance.eval(s, p)
    avg_ed = total_ed / len(sources) if sources else 0

    if save_errant_dir:
        import shutil
        os.makedirs(save_errant_dir, exist_ok=True)
        for fn in ["ref.para", "hyp.para", "ref.m2", "ref.norm.m2", "hyp.m2"]:
            p = os.path.join(tmp, fn)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(save_errant_dir, fn))
        with open(os.path.join(save_errant_dir, "compare.txt"), "w", encoding="utf-8") as f:
            for i in range(len(sources)):
                f.write(f"[{i}]\n  SRC: {sources[i]}\n  REF: {' | '.join(targets[i])}\n  HYP: {pred_texts[i]}\n\n")
        print(f"[INFO] ChERRANT 文件保存到: {save_errant_dir}/")

    return {
        "core_metrics": core,
        "overcorrection": {"total_correct_src": total_correct,
                           "modified_correct_src": modified_correct,
                           "rate": round(oc, 4)},
        "edit_distance": {"total": total_ed, "average": round(avg_ed, 4)},
    }


def print_metrics(metrics, beta):
    print(f"\n{'='*60}\n  中文 GEC 评估结果（ChERRANT, beta={beta}）\n{'='*60}")
    if metrics["core_metrics"]:
        print("\n【核心指标】")
        for k, v in metrics["core_metrics"].items():
            print(f"  {k:<12}: {v}")
    else:
        print("\n【核心指标】解析失败")
    oc = metrics["overcorrection"]
    print(f"\n【过校正率】\n  正确原句: {oc['total_correct_src']} | "
          f"被误改: {oc['modified_correct_src']} | 率: {oc['rate']*100:.2f}%")
    ed = metrics["edit_distance"]
    print(f"\n【编辑距离】\n  总计: {ed['total']} | 平均: {ed['average']}\n{'='*60}")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="中文 GEC 推理 + ChERRANT 评测")

    parser.add_argument("--model_path",     type=str, default=None)
    parser.add_argument("--lora_path",      type=str, default=None)
    parser.add_argument("--lora_rank",      type=int, default=64)
    parser.add_argument("--input",          type=str, required=True,
                        help="测试集: JSON/JSONL（含 source/target）或纯文本（每行一句）")
    parser.add_argument("--reference_m2",   type=str, default=None,
                        help="官方 M2 文件，提供后直接作为评测参考（推荐 NaSGEC/MuCGEC 用）")
    parser.add_argument("--output",         type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature",    type=float, default=0.0)
    parser.add_argument("--batch_size",     type=int, default=512)
    parser.add_argument("--use_vllm",       action="store_true")
    parser.add_argument("--max_retries",    type=int, default=3)
    parser.add_argument("--retry_temperature", type=float, default=0.3)
    parser.add_argument("--voting_samples", type=int, default=0)
    parser.add_argument("--voting_temperature", type=float, default=1.0)
    parser.add_argument("--beta",           type=float, default=0.5)
    parser.add_argument("--save_errant_dir", type=str, default=None)
    parser.add_argument("--cherrant_dir",   type=str, default=None)
    parser.add_argument("--save_metrics",   type=str, default=None)
    parser.add_argument("--eval_only",      action="store_true")
    parser.add_argument("--skip_eval",      action="store_true")
    parser.add_argument("--granularity",    type=str, default=None, choices=[None, "char", "word"],
                        help="ChERRANT 粒度,默认 NaSGEC/MuCGEC=char,其他=word")

    args = parser.parse_args()

    samples = load_test_data(args.input, reference_m2=args.reference_m2)
    print(f"[INIT] 加载测试集：{len(samples)} 条")
    multi = sum(1 for s in samples if len(s["target"]) > 1)
    if multi:
        print(f"[INIT] 多参考: {multi}/{len(samples)}")
    print(f"[INIT] Prompt 模板: {USER_PROMPT_TEMPLATE[:80]}...")

    # ── 推理 ──
    if args.eval_only:
        out_path = args.output
        if os.path.isdir(out_path):
            out_path = os.path.join(out_path, "pred.jsonl")
        print(f"[MODE] eval_only: 加载 {out_path}")
        raw = load_jsonl(out_path)
        id2tgt = {s["id"]: s["target"] for s in samples}
        src2tgt = {_normalize_for_match(s["source"]): s["target"] for s in samples}
        results = []
        for r in raw:
            src = r.get("source", "")
            ro = r.get("raw_output", r.get("prediction", r.get("response", "")))
            extracted, reason, _ = extract_prediction(ro, src)
            rid = r.get("id", "")
            tgt = id2tgt.get(rid)
            if tgt is None:
                tgt = src2tgt.get(_normalize_for_match(src), r.get("target", ""))
            tgt = normalize_target(tgt)
            results.append({"id": rid, "source": src, "target": tgt,
                           "prediction": extracted, "raw_output": ro,
                           "extract_reason": reason, "retries": 0})
        stats = {}
        for r in results:
            stats[r["extract_reason"]] = stats.get(r["extract_reason"], 0) + 1
        print("[提取统计]")
        for reason, c in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"  {reason:<25}: {c} ({c/len(results)*100:.1f}%)")
    else:
        if not args.model_path:
            parser.error("推理模式需要 --model_path")
        print(f"[INIT] 加载模型: {args.model_path}")
        if args.lora_path:
            print(f"[INIT] LoRA: {args.lora_path}  rank={args.lora_rank}")
        if args.use_vllm:
            mll, tok = load_model_vllm(args.model_path, args.lora_path, lora_rank=args.lora_rank)
        else:
            mll, tok = load_model_hf(args.model_path, args.lora_path)

        if args.voting_samples > 0:
            print(f"[MODE] 投票: {args.voting_samples} 候选, t={args.voting_temperature}")
            results = run_voting_inference(samples, mll, tok, args.use_vllm,
                                           args.max_new_tokens, args.voting_samples,
                                           args.voting_temperature, args.batch_size,
                                           args.lora_path)
        else:
            print(f"[MODE] 贪心 + 重试 (max_retries={args.max_retries})")
            results = run_inference_with_retry(
                samples, mll, tok, args.use_vllm, args.max_new_tokens, args.temperature,
                args.batch_size, args.max_retries, args.retry_temperature, args.lora_path,
            )
        actual = save_jsonl(results, args.output)
        args.output = actual
        print(f"\n[SAVE] 预测保存: {actual} ({len(results)} 条)")

    if args.skip_eval:
        print("[MODE] skip_eval")
        return

    # ── 评测 ──
    print(f"\n{'='*60}\n[EVAL] 开始 ChERRANT 评测\n{'='*60}")
    sources = [r["source"] for r in results]
    targets = [r["target"] for r in results]
    preds   = [r["prediction"] for r in results]

    metrics = run_cherrant_evaluation(
        sources, targets, preds, beta=args.beta,
        save_errant_dir=args.save_errant_dir, cherrant_dir=args.cherrant_dir,
        ref_m2_path=args.reference_m2, input_path=args.input,
        granularity=args.granularity,
    )
    print_metrics(metrics, args.beta)

    if args.save_metrics:
        os.makedirs(os.path.dirname(args.save_metrics) or ".", exist_ok=True)
        with open(args.save_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        print(f"[SAVE] 指标: {args.save_metrics}")


if __name__ == "__main__":
    main()