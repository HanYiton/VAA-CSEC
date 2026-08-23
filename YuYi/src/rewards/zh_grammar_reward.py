"""
中文语法纠错 (Chinese GEC) 奖励函数 - 适用于 ms-swift GRPO 训练

模型输出格式: <think>简要分析</think><answer>纠正后的句子</answer>

奖励组成（单项）:
  格式奖励 (0.0 或 0.3):
    - 严格格式 (恰好1个think + 1个answer):  0.3
    - 其他一律:                               0.0

  正确性奖励 (-1.5 ~ 3.4):
    - 完全匹配:                 3.4
    - useful > 0 且无过度修改:  2.0 × F0.5  (最高 2.0)
    - useful > 0 但过度修改:    2.0 × F0.5 - over_edit_penalty  (可能为负)
    - useful == 0:              0.0
    - useful < 0 (改坏):       -1.5 × min(|useful|/ed_sr, 1.0)

  总分范围: -1.5 ~ 3.7

  过度修改惩罚说明:
    当 ed_sp > ed_sr 时，触发过度修改惩罚。
    excess_ratio = (ed_sp - ed_sr) / ed_sr，上限 2.0
    over_edit_penalty = min(excess_ratio, 2.0) × 0.8
"""

import re
import unicodedata
from typing import List

try:
    from swift.rewards import ORM, orms
except ImportError:
    try:
        from swift.plugin import ORM, orms
    except ImportError:
        class ORM:
            pass
        orms = {}


# ==================== 文本处理工具 ====================

def strip_template_tokens(text: str) -> str:
    """去除 chat template 特殊 token（不处理 think/answer 标签）。"""
    if not text:
        return ""
    text = re.sub(r"<\|im_start\|>.*?\n?", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"<\|endoftext\|>", "", text)
    return text.strip()


def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_answer(raw_text: str):
    """
    从模型原始输出中提取纠正后的文本。

    格式奖励: 严格 1个<think>+1个<answer> 得 0.3，其他一律 0.0

    返回: (提取文本, 格式奖励分数)
    """
    cleaned = strip_template_tokens(raw_text)
    if not cleaned:
        return "", 0.0

    # 检查是否严格格式
    think_open = len(re.findall(r"<think>", cleaned))
    think_close = len(re.findall(r"</think>", cleaned))
    answer_open = len(re.findall(r"<answer>", cleaned))
    answer_close = len(re.findall(r"</answer>", cleaned))
    perfect = (think_open == 1 and think_close == 1 and
               answer_open == 1 and answer_close == 1)

    # 提取 answer 内容
    m = re.search(r"<answer>(.*?)</answer>", cleaned, flags=re.DOTALL)
    if m:
        answer = normalize(m.group(1))
        return answer, (0.3 if perfect else 0.0)

    # answer 未闭合
    m2 = re.search(r"<answer>(.*)", cleaned, flags=re.DOTALL)
    if m2:
        answer = normalize(m2.group(1))
        if answer:
            return answer, 0.0

    # 纯文本回退：剥离 think 内容
    text_only = re.sub(r"<think>[\s\S]*?</think>\s*", "", cleaned)
    text_only = re.sub(r"<think>[\s\S]*$", "", text_only)
    text_only = normalize(text_only)
    if text_only:
        return text_only, 0.0

    return "", 0.0


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]


# ==================== 正确性奖励 ====================

def _score_single_ref(pred: str, ref: str, src: str, format_reward: float) -> float:
    """对单个参考计算正确性奖励 + 格式奖励。"""
    if not ref:
        return format_reward

    if pred == ref:
        return 3.4 + format_reward

    ed_sr = edit_distance(src, ref)
    ed_sp = edit_distance(src, pred)
    ed_pr = edit_distance(pred, ref)

    if ed_sr == 0 or ed_sp == 0:
        return format_reward

    useful = (ed_sr + ed_sp - ed_pr) / 2.0

    # 改坏了：pred 比 src 离 ref 更远
    if useful < 0:
        worsening = (-useful) / ed_sr
        return -1.5 * min(worsening, 1.0) + format_reward

    if useful == 0:
        return format_reward

    precision = min(useful / ed_sp, 1.0)
    recall = min(useful / ed_sr, 1.0)

    denominator = 0.25 * precision + recall
    if denominator <= 0:
        return format_reward

    f05 = 1.25 * precision * recall / denominator
    base_score = 2.0 * f05

    # 过度修改惩罚
    if ed_sp > ed_sr:
        excess_ratio = (ed_sp - ed_sr) / ed_sr
        over_edit_penalty = min(excess_ratio, 2.0) * 0.8
        return max(-1.5, base_score - over_edit_penalty) + format_reward

    return base_score + format_reward


def score_correctness(raw_text: str, solution, source: str) -> float:
    """
    正确性奖励 + 格式奖励。支持多参考（取最高分）。
    总分范围: -1.5 ~ 3.7
    """
    pred, format_reward = extract_answer(raw_text)
    src = normalize(source)

    # 统一为列表处理多参考
    if isinstance(solution, str):
        refs = [solution]
    elif isinstance(solution, (list, tuple)):
        refs = list(solution)
    else:
        refs = [str(solution)]

    # 过滤空参考
    refs = [r for r in refs if normalize(r)]
    if not refs:
        return format_reward

    # 对每个参考算分，取最高
    best_score = float('-inf')
    for sol in refs:
        ref = normalize(sol)
        score = _score_single_ref(pred, ref, src, format_reward)
        if score > best_score:
            best_score = score
    return best_score


# ==================== Swift ORM ====================

class ZhGecReward(ORM):

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        solutions: list = kwargs.get("solution", [])
        sources: list = kwargs.get("source", [])

        rewards = []
        for i, completion in enumerate(completions):
            sol = self._safe_get(solutions, i)
            src = self._safe_get(sources, i)
            rewards.append(score_correctness(completion, sol, src))

        return rewards

    @staticmethod
    def _safe_get(data, idx: int, default: str = "") -> str:
        if data is None:
            return default
        if isinstance(data, str):
            return data
        if isinstance(data, (list, tuple)):
            if idx < len(data):
                return data[idx]
            if len(data) == 1:
                return data[0]
            if len(data) > 0:
                return data[idx % len(data)]
        return default


orms["zh_gec_reward"] = ZhGecReward


# ==================== 独立测试 ====================

if __name__ == "__main__":

    print("=" * 70)
    print("奖励函数测试（<answer> 标签格式）")
    print("=" * 70)

    SRC = '王羲之被后世称为"书圣"，王献之则被后世评价认为可与其父比肩的书法家。'
    REF = '王羲之被后世称为"书圣"，王献之则被后世评价认为是可与其父比肩的书法家。'

    cases = [
        {
            "desc": "严格格式: think + answer",
            "completion": f"<think>需要在\"认为\"后加\"是\"</think><answer>{REF}</answer>",
        },
        {
            "desc": "严格格式 + 模板 token",
            "completion": f"<think>加\"是\"</think><answer>{REF}</answer><|im_end|>",
        },
        {
            "desc": "只有 answer 无 think（不完美格式）",
            "completion": f"<answer>{REF}</answer>",
        },
        {
            "desc": "think 未闭合 + answer 完整",
            "completion": f"<think>分析中...<answer>{REF}</answer>",
        },
        {
            "desc": "多个 think + answer 完整",
            "completion": f"<think>第一次</think><think>第二次</think><answer>{REF}</answer>",
        },
        {
            "desc": "answer 未闭合（截断）",
            "completion": f"<think>分析</think><answer>{REF}",
        },
        {
            "desc": "纯文本无标签",
            "completion": REF,
        },
        {
            "desc": "严格格式 + 部分正确",
            "completion": '<think>需要改</think><answer>王羲之被后世称为"书圣"，王献之则被后世评价为是可与其父比肩的书法家。</answer>',
        },
        {
            "desc": "严格格式 + 过度修改",
            "completion": '<think>改</think><answer>王羲之被后世称为"书圣"，而王献之则被后人普遍评价认为是完全可以与其父相比肩的著名书法家。</answer>',
        },
        {
            "desc": "严格格式 + 没改",
            "completion": f"<think>无需修改</think><answer>{SRC}</answer>",
        },
        {
            "desc": "空输出",
            "completion": "",
        },
        {
            "desc": "只有 think 截断，无 answer",
            "completion": "<think>这个句子存在主语缺失的问题，需要在",
        },
    ]

    print(f"\nSRC: {SRC}")
    print(f"REF: {REF}")
    print(f"ed(src,ref) = {edit_distance(normalize(SRC), normalize(REF))}")

    for tc in cases:
        src_n = normalize(SRC)
        ref_n = normalize(REF)
        pred_n, fmt_r = extract_answer(tc["completion"])
        score = score_correctness(tc["completion"], REF, SRC)

        ed_sr = edit_distance(src_n, ref_n)
        ed_sp = edit_distance(src_n, pred_n) if pred_n else 0
        ed_pr = edit_distance(pred_n, ref_n) if pred_n else 0
        useful = (ed_sr + ed_sp - ed_pr) / 2.0 if pred_n else 0

        print(f"\n【{tc['desc']}】")
        print(f"  output  : '{tc['completion'][:80]}{'...' if len(tc['completion']) > 80 else ''}'")
        print(f"  pred    : '{pred_n[:60]}{'...' if len(pred_n) > 60 else ''}'")
        print(f"  fmt_r={fmt_r}  ed_sr={ed_sr}  ed_sp={ed_sp}  ed_pr={ed_pr}  useful={useful:.1f}")
        print(f"  总分    : {score:+.4f}")

    # ── 多参考测试 ──
    print(f"\n\n{'='*70}")
    print("多参考测试")
    print("=" * 70)

    SRC2 = '经过深入开展学雷锋创"三好"活动，使我们的思想有了很大的提高。'
    REFS2 = [
        '经过深入开展学雷锋创"三好"活动，我们的思想有了很大的提高。',
        '深入开展学雷锋创"三好"活动，使我们的思想有了很大的提高。',
    ]

    multi_cases = [
        {
            "desc": "匹配第1个参考",
            "completion": f"<think>删\"使\"</think><answer>{REFS2[0]}</answer>",
        },
        {
            "desc": "匹配第2个参考",
            "completion": f"<think>删\"经过\"</think><answer>{REFS2[1]}</answer>",
        },
        {
            "desc": "两个都不匹配但部分正确",
            "completion": '<think>改</think><answer>经过开展学雷锋创"三好"活动，我们的思想有了很大的提高。</answer>',
        },
    ]

    print(f"\nSRC: {SRC2}")
    for i, r in enumerate(REFS2):
        print(f"REF[{i}]: {r}")

    for tc in multi_cases:
        pred_n, fmt_r = extract_answer(tc["completion"])
        score = score_correctness(tc["completion"], REFS2, SRC2)
        print(f"\n【{tc['desc']}】")
        print(f"  output: '{tc['completion'][:80]}...'")
        print(f"  pred  : '{pred_n[:60]}'")
        print(f"  总分  : {score:+.4f}")