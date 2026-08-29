import os
import re
import unicodedata
import torch
from collections import Counter
from typing import List, Tuple, Dict, Any

from swift.rewards import ORM, orms


# ==================== 全局状态 ====================

_STEP_COUNTER = {"count": 0}
_PATCH_APPLIED = False


# ==================== 文本处理 ====================

def _strip(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<\|im_start\|>.*?\n?", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"<\|endoftext\|>", "", text)
    return text.strip()


def _norm(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_answer(raw: str) -> Tuple[str, float]:
    c = _strip(raw)
    if not c:
        return "", 0.0
    to = len(re.findall(r"", c))
    tc = len(re.findall(r"", c))
    ao = len(re.findall(r"<answer>", c))
    ac = len(re.findall(r"</answer>", c))
    perfect = (to == 1 and tc == 1 and ao == 1 and ac == 1)
    m = re.search(r"<answer>(.*?)</answer>", c, flags=re.DOTALL)
    if m:
        return _norm(m.group(1)), (0.3 if perfect else 0.0)
    m2 = re.search(r"<answer>(.*)", c, flags=re.DOTALL)
    if m2:
        a = _norm(m2.group(1))
        if a:
            return a, 0.0
    text_only = re.sub(r"[\s\S]*?\s*", "", c)
    text_only = re.sub(r"[\s\S]*$", "", text_only)
    text_only = _norm(text_only)
    return (text_only, 0.0) if text_only else ("", 0.0)


def _edit_dist(a: str, b: str) -> int:
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


def _score(pred: str, ref: str, src: str, fmt_r: float = 0.0) -> float:
    if not ref:
        return fmt_r
    if pred == ref:
        return 3.4 + fmt_r
    ed_sr = _edit_dist(src, ref)
    ed_sp = _edit_dist(src, pred)
    ed_pr = _edit_dist(pred, ref)
    if ed_sr == 0 or ed_sp == 0:
        return fmt_r
    useful = (ed_sr + ed_sp - ed_pr) / 2.0
    if useful < 0:
        w = (-useful) / ed_sr
        return -1.5 * min(w, 1.0) + fmt_r
    if useful == 0:
        return fmt_r
    prec = min(useful / ed_sp, 1.0)
    rec = min(useful / ed_sr, 1.0)
    denom = 0.25 * prec + rec
    if denom <= 0:
        return fmt_r
    f05 = 1.25 * prec * rec / denom
    base = 2.0 * f05
    if ed_sp > ed_sr:
        excess = (ed_sp - ed_sr) / ed_sr
        penalty = min(excess, 2.0) * 0.8
        return max(-1.5, base - penalty) + fmt_r
    return base + fmt_r


def _normalize_refs(sol):
    if isinstance(sol, str):
        refs = [sol]
    elif isinstance(sol, (list, tuple)):
        refs = list(sol)
    else:
        refs = [str(sol)]
    return [_norm(r) for r in refs if _norm(r)]


def _safe_get(data, idx: int, default=""):
    if data is None:
        return default
    if isinstance(data, str):
        return data
    if isinstance(data, (list, tuple)):
        if idx < len(data):
            return data[idx]
        if len(data) == 1:
            return data[0]
        return data[idx % len(data)] if data else default
    return default


# ==================== 分组逻辑 ====================

def _group_by_source(sources: List[str]) -> List[List[int]]:
    if not sources:
        return []
    groups = []
    cur = [0]
    cur_src = sources[0]
    for i in range(1, len(sources)):
        if sources[i] == cur_src:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
            cur_src = sources[i]
    groups.append(cur)
    return groups


# ==================== Reward 计算 ====================

def _compute_individual_per_sample(completions, solutions, sources) -> List[float]:
    n = len(completions)
    rewards = [0.0] * n
    for i in range(n):
        pred, fmt_r = _extract_answer(completions[i])
        src = _norm(_safe_get(sources, i))
        refs = _normalize_refs(_safe_get(solutions, i))
        if not refs:
            rewards[i] = fmt_r
            continue
        best = float("-inf")
        for ref in refs:
            s = _score(pred, ref, src, fmt_r)
            best = max(best, s)
        rewards[i] = best
    return rewards


def _compute_group_per_sample(completions, solutions, sources) -> Tuple[List[float], Dict]:
    """每组 K 个 sample 计算 R(G) = r(Vote(G)), broadcast 到组内"""
    n = len(completions)
    K = int(os.environ.get("GLPO_K", "8"))

    if n % K != 0:
        # 兜底
        src_list = [_safe_get(sources, i) for i in range(n)]
        sol_list = [_safe_get(solutions, i) for i in range(n)]
        groups = _group_by_source(src_list)
    else:
        groups = [list(range(i, i + K)) for i in range(0, n, K)]
        src_list = [_safe_get(sources, i) for i in range(n)]
        sol_list = [_safe_get(solutions, i) for i in range(n)]

    rewards = [0.0] * n
    stats = {"num_groups": len(groups), "K_list": [], "R_G_list": [],
             "unique_list": [], "vote_count_list": []}

    for group_ids in groups:
        group_src = _norm(src_list[group_ids[0]])
        group_sol = sol_list[group_ids[0]]
        refs = _normalize_refs(group_sol)

        answers = []
        fmt_flags = []
        for i in group_ids:
            ans, fmt_r = _extract_answer(completions[i])
            answers.append(ans)
            fmt_flags.append(fmt_r > 0)
        valid = [a for a in answers if a]
        vote_ans = Counter(valid).most_common(1)[0][0] if valid else ""

        vote_fmt_bonus = 0.0
        if vote_ans:
            vote_ans_normed = _norm(vote_ans)
            for idx, _ in enumerate(group_ids):
                if fmt_flags[idx] and _norm(answers[idx]) == vote_ans_normed:
                    vote_fmt_bonus = 0.3
                    break

        if not refs or not vote_ans:
            g_r = 0.0
        else:
            best = float("-inf")
            for ref in refs:
                s = _score(vote_ans, ref, group_src, vote_fmt_bonus)
                best = max(best, s)
            g_r = best

        for i in group_ids:
            rewards[i] = g_r

        counter = Counter(valid)
        stats["K_list"].append(len(group_ids))
        stats["R_G_list"].append(g_r)
        stats["unique_list"].append(len(counter))
        stats["vote_count_list"].append(counter.most_common(1)[0][1] if counter else 0)

    return rewards, stats


# ==================== Reward Functions ====================

class ZhGecIndividualReward(ORM):
    def __init__(self, args=None, **kwargs):
        super().__init__()
        self._call_counter = 0

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        if self._call_counter == 0:
            _ensure_patched()
        self._call_counter += 1
        solutions = kwargs.get("solution", [])
        sources = kwargs.get("source", [])
        return _compute_individual_per_sample(completions, solutions, sources)


class ZhGecGroupReward(ORM):
    def __init__(self, args=None, **kwargs):
        super().__init__()
        self._call_counter = 0

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        self._call_counter += 1
        solutions = kwargs.get("solution", [])
        sources = kwargs.get("source", [])
        rewards, stats = _compute_group_per_sample(completions, solutions, sources)

        if os.environ.get("GLPO_DEBUG") == "1" and self._call_counter % 5 == 1:
            n_g = stats["num_groups"]
            if n_g > 0:
                K = stats["K_list"][0]
                avg_R = sum(stats["R_G_list"]) / n_g
                avg_u = sum(stats["unique_list"]) / n_g
                avg_vc = sum(stats["vote_count_list"]) / n_g
                print(f"[GLPO-GroupRwd] call={self._call_counter} | groups={n_g} | K={K} | "
                      f"R(G)_avg={avg_R:+.3f} | unique={avg_u:.1f}/{K} | "
                      f"vote_cnt={avg_vc:.1f}/{K}")
        return rewards


orms["zh_gec_individual_reward"] = ZhGecIndividualReward
orms["zh_gec_group_reward"] = ZhGecGroupReward


# ==================== Monkey-patch _compute_advantages ====================

def _log_glpo_metrics(trainer, **data):
    try:
        metrics = getattr(trainer, '_metrics', None)
        if metrics is None:
            return
        if isinstance(metrics, dict) and 'train' in metrics:
            metrics = metrics['train']
        if not isinstance(metrics, dict):
            return

        def log(k, v):
            if k not in metrics:
                metrics[k] = []
            if hasattr(v, 'item'):
                v = v.item()
            try:
                metrics[k].append(float(v))
            except Exception:
                pass

        for k, v in data.items():
            if v is not None:
                log(k, v)
    except Exception as e:
        if os.environ.get("GLPO_DEBUG") == "1" and _STEP_COUNTER["count"] % 50 == 1:
            print(f"[GLPO] metrics log err: {e}")


def _apply_glpo_patch():
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
    _original = GRPOTrainer._compute_advantages

    def _glpo_compute_advantages(self, inputs, rewards_per_func, batch_encoded_inputs):
        # ============ Step 1: 跑原版 (仅诊断, 不再用结果) ============
        try:
            swift_adv = _original(self, inputs, rewards_per_func, batch_encoded_inputs)
        except Exception as e:
            if os.environ.get("GLPO_DEBUG") == "1":
                print(f"[GLPO] WARN: original _compute_advantages failed: {e}")
            swift_adv = None

        # ============ Step 2: 超参 ============
        K = int(os.environ.get("GLPO_K", "8"))
        bonus_scale = float(os.environ.get("GLPO_BONUS_SCALE", "1.0"))
        std_eps = float(os.environ.get("GLPO_STD_EPS", "1e-4"))
        group_col = int(os.environ.get("GLPO_GROUP_COL_IDX", "1"))
        debug = os.environ.get("GLPO_DEBUG") == "1"
        debug_first_n = int(os.environ.get("GLPO_DEBUG_FIRST_STEPS", "3"))

        # ============ Step 3: 形状检查 ============
        if not hasattr(rewards_per_func, 'shape') or rewards_per_func.dim() < 2:
            if swift_adv is not None:
                return swift_adv
            raise RuntimeError("[GLPO] rewards_per_func shape invalid and no fallback")

        n_samples, n_funcs = rewards_per_func.shape
        if n_funcs <= group_col:
            print(f"[GLPO] ERROR: need ≥{group_col+1} reward funcs, got {n_funcs}, fallback")
            return swift_adv if swift_adv is not None else torch.zeros(n_samples)

        if n_samples % K != 0:
            if debug:
                print(f"[GLPO] WARN: n_samples={n_samples} not divisible by K={K}, fallback")
            return swift_adv if swift_adv is not None else torch.zeros(n_samples)

        num_groups = n_samples // K

        # ============ Step 4: 取 rewards ============
        ind_col = 1 - group_col if group_col in (0, 1) else 0
        group_rewards = rewards_per_func[:, group_col].float()  # R_G, 同组内常数
        ind_rewards = rewards_per_func[:, ind_col].float()      # R_ij

        # ============ Step 5: K 组内归一化 ============
        ind_grouped = ind_rewards.view(num_groups, K)
        grp_grouped = group_rewards.view(num_groups, K)

        group_mean = ind_grouped.mean(dim=1, keepdim=True)
        group_std = ind_grouped.std(dim=1, keepdim=True, unbiased=True).clamp(min=std_eps)
        base_adv_grouped = (ind_grouped - group_mean) / group_std
        base_adv = base_adv_grouped.reshape(-1)

        # ============ Step 6: bonus = max(0, R_ij - R_G) ============
        R_G_per_group = grp_grouped[:, 0:1]  # 组内常数, 取第一个值
        margin = ind_grouped - R_G_per_group
        bonus_grouped = torch.clamp(margin, min=0.0) * bonus_scale
        bonus = bonus_grouped.reshape(-1)

        # ============ Step 7: 最终 advantage ============
        glpo_adv = base_adv + bonus

        _STEP_COUNTER["count"] += 1
        cur_step = _STEP_COUNTER["count"]

        # ============ Step 8: 详细诊断日志 ============
        if debug and (cur_step <= debug_first_n or cur_step % 20 == 1):
            with torch.no_grad():
                first_group_ind = ind_grouped[0].cpu().tolist()
                first_group_grp = grp_grouped[0].cpu().tolist()
                first_group_base = base_adv_grouped[0].cpu().tolist()
                first_group_bonus = bonus_grouped[0].cpu().tolist()
                first_group_glpo = (base_adv_grouped[0] + bonus_grouped[0]).cpu().tolist()

                print(f"\n{'='*70}")
                print(f"[GLPO-DEBUG] step={cur_step} | n={n_samples} K={K} "
                      f"num_groups={num_groups}")
                print(f"{'='*70}")
                print(f"Group[0] R_G        = {first_group_grp[0]:+.4f} (组内常数)")
                print(f"Group[0] mean(R_ij) = {group_mean[0].item():+.4f}")
                print(f"Group[0] std(R_ij)  = {group_std[0].item():.4f}")
                print(f"{'idx':>4} | {'R_ij':>8} | {'R_G':>8} | {'base_adv':>10} | "
                      f"{'bonus':>8} | {'glpo_adv':>10}")
                print(f"{'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")
                for j in range(K):
                    print(f"{j:>4} | {first_group_ind[j]:>+8.3f} | "
                          f"{first_group_grp[j]:>+8.3f} | "
                          f"{first_group_base[j]:>+10.4f} | "
                          f"{first_group_bonus[j]:>8.4f} | "
                          f"{first_group_glpo[j]:>+10.4f}")

                if swift_adv is not None:
                    swift_first = swift_adv.view(num_groups, K)[0].cpu().tolist()
                    print(f"\nswift_adv (原版, 仅诊断): {[f'{x:+.4f}' for x in swift_first]}")
                    swift_grp_mean = swift_adv.view(num_groups, K)[0].mean().item()
                    swift_grp_std = swift_adv.view(num_groups, K)[0].std(unbiased=True).item()
                    print(f"swift_adv group[0] mean={swift_grp_mean:+.4f} "
                          f"std={swift_grp_std:.4f} "
                          f"({'✓ 组内归一' if abs(swift_grp_mean) < 0.01 else '✗ 非组内归一'})")
                print(f"{'='*70}\n")

        # ============ Step 9: wandb 指标 ============
        with torch.no_grad():
            r_g_mean = group_rewards.mean().item()
            r_g_std = group_rewards.std().item()
            r_g_max = group_rewards.max().item()
            r_g_min = group_rewards.min().item()

            ind_mean_global = ind_rewards.mean().item()
            ind_std_global = ind_rewards.std().item()
            ind_max_global = ind_rewards.max().item()
            ind_min_global = ind_rewards.min().item()

            base_adv_mean = base_adv.mean().item()
            base_adv_std = base_adv.std().item()
            bonus_mean = bonus.mean().item()
            bonus_max = bonus.max().item()
            bonus_nonzero_ratio = (bonus > 0).float().mean().item()
            final_adv_mean = glpo_adv.mean().item()
            final_adv_std = glpo_adv.std().item()

            ind_mean_per_group = ind_grouped.mean(dim=1)
            ind_max_per_group = ind_grouped.max(dim=1).values
            R_G_per_group_flat = grp_grouped[:, 0]
            collective_gain = (R_G_per_group_flat - ind_mean_per_group).mean().item()
            R_G_vs_best_of_K = (R_G_per_group_flat - ind_max_per_group).mean().item()

            pct_vote_perfect = (R_G_per_group_flat >= 3.4).float().mean().item()
            pct_vote_good = (R_G_per_group_flat >= 2.0).float().mean().item()
            pct_vote_failed = (R_G_per_group_flat <= 0.0).float().mean().item()

            swift_adv_mean = swift_adv.mean().item() if swift_adv is not None else 0.0
            swift_adv_std = swift_adv.std().item() if swift_adv is not None else 0.0

        _log_glpo_metrics(
            self,
            **{
                "glpo/R_G_mean": r_g_mean,
                "glpo/R_G_std": r_g_std,
                "glpo/R_G_max": r_g_max,
                "glpo/R_G_min": r_g_min,
                "glpo/R_ind_mean": ind_mean_global,
                "glpo/R_ind_std": ind_std_global,
                "glpo/R_ind_max": ind_max_global,
                "glpo/R_ind_min": ind_min_global,
                "glpo/collective_gain": collective_gain,
                "glpo/R_G_vs_best_of_K": R_G_vs_best_of_K,
                "glpo/pct_vote_perfect": pct_vote_perfect,
                "glpo/pct_vote_good": pct_vote_good,
                "glpo/pct_vote_failed": pct_vote_failed,
                "glpo/base_adv_mean": base_adv_mean,
                "glpo/base_adv_std": base_adv_std,
                "glpo/bonus_mean": bonus_mean,
                "glpo/bonus_max": bonus_max,
                "glpo/bonus_nonzero_ratio": bonus_nonzero_ratio,
                "glpo/bonus_scale": bonus_scale,
                "glpo/final_adv_mean": final_adv_mean,
                "glpo/final_adv_std": final_adv_std,
                "glpo/swift_adv_mean_diagnostic": swift_adv_mean,
                "glpo/swift_adv_std_diagnostic": swift_adv_std,
            }
        )

        if debug and cur_step % 20 == 1:
            print(f"[GLPO-patch] step~{cur_step} | n={n_samples} groups={num_groups} K={K} | "
                  f"R(G) μ={r_g_mean:+.3f} σ={r_g_std:.3f} | "
                  f"R_ind μ={ind_mean_global:+.3f} | "
                  f"collective_gain={collective_gain:+.3f} | "
                  f"base_adv: μ={base_adv_mean:+.3f} σ={base_adv_std:.3f} | "
                  f"bonus: μ={bonus_mean:.3f} max={bonus_max:.3f} "
                  f"nz={bonus_nonzero_ratio:.2f} | "
                  f"final_adv: μ={final_adv_mean:+.3f} σ={final_adv_std:.3f}")

        return glpo_adv

    GRPOTrainer._compute_advantages = _glpo_compute_advantages
    _PATCH_APPLIED = True

    K = int(os.environ.get("GLPO_K", "8"))
    bonus_scale = os.environ.get("GLPO_BONUS_SCALE", "1.0")
    print(f"[GLPO-SIMPLIFIED] Patched _compute_advantages.")
    print(f"[GLPO-SIMPLIFIED] K={K}, bonus_scale={bonus_scale}")
    print(f"[GLPO-SIMPLIFIED] base_adv = (R_ij - mean_K)/std_K  (K组内归一化)")
    print(f"[GLPO-SIMPLIFIED] bonus    = max(0, R_ij - R_G) * scale")
    print(f"[GLPO-SIMPLIFIED] glpo_adv = base_adv + bonus")


def _ensure_patched():
    global _PATCH_APPLIED
    if not _PATCH_APPLIED:
        try:
            _apply_glpo_patch()
        except Exception as e:
            print(f"[GLPO] patch failed: {e}")


try:
    _apply_glpo_patch()
except Exception as e:
    print(f"[GLPO] initial patch deferred: {e}")


# ==================== 自测 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Advantage 计算单元测试")
    print("=" * 70)

    R = torch.tensor([[0.3, 0.3, 0.8, 1.2, 3.7, 2.16, 0.95, 0.88]])
    R_G_val = 0.3
    R_G = torch.full_like(R, R_G_val)
    K = 8
    std_eps = 1e-4

    group_mean = R.mean(dim=1, keepdim=True)
    group_std = R.std(dim=1, keepdim=True, unbiased=True).clamp(min=std_eps)
    base_adv = (R - group_mean) / group_std
    margin = R - R_G[:, 0:1]
    bonus = torch.clamp(margin, min=0.0)
    glpo_adv = base_adv + bonus

    print(f"R           = {R[0].tolist()}")
    print(f"R_G         = {R_G_val}")
    print(f"R 均值      = {group_mean.item():.4f}")
    print(f"R 标准差    = {group_std.item():.4f}")
    print(f"base_adv    = {[f'{x:+.4f}' for x in base_adv[0].tolist()]}")
    print(f"bonus       = {[f'{x:+.4f}' for x in bonus[0].tolist()]}")
    print(f"glpo_adv    = {[f'{x:+.4f}' for x in glpo_adv[0].tolist()]}")
    print()
    print("预期:")
    print("base_adv ≈ [-0.87, -0.87, -0.43, -0.08, +2.12, +0.77, -0.30, -0.36]")
    print("bonus    ≈ [0.00, 0.00, 0.50, 0.90, 3.40, 1.86, 0.65, 0.58]")
    print("glpo_adv ≈ [-0.87, -0.87, +0.07, +0.82, +5.52, +2.63, +0.35, +0.22]")

    # ==================== Pipeline 测试 ====================
    print("\n" + "=" * 70)
    print("完整 reward pipeline 测试")
    print("=" * 70)

    SRC = '王羲之被后世称为"书圣"，王献之则被后世评价认为可与其父比肩的书法家。'
    REF = '王羲之被后世称为"书圣"，王献之则被后世评价认为是可与其父比肩的书法家。'

    os.environ["GLPO_K"] = "4"
    os.environ["GLPO_DEBUG"] = "1"

    completions = [
        f't<answer>{REF}</answer>',
        f't<answer>{REF}</answer>',
        f't<answer>{SRC}</answer>',
        f't<answer>{REF}</answer>',
        f't<answer>{REF}</answer>',
        f't<answer>{SRC}</answer>',
        f't<answer>{SRC}</answer>',
        f't<answer>{SRC}</answer>',
    ]
    sols = [REF] * 8
    srcs = [SRC] * 8

    ir = ZhGecIndividualReward()
    r_ind = ir(completions=completions, solution=sols, source=srcs)
    print(f"Individual rewards: {[f'{r:.2f}' for r in r_ind]}")

    gr = ZhGecGroupReward()
    r_grp = gr(completions=completions, solution=sols, source=srcs)
    print(f"Group rewards     : {[f'{r:.2f}' for r in r_grp]}")
    assert r_grp[0] == r_grp[1] == r_grp[2] == r_grp[3]
    assert r_grp[4] == r_grp[5] == r_grp[6] == r_grp[7]
    print("✓ Group reward 在组内一致")