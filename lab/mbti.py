"""MBTI 人格对话数据（用户自己 v6 的真实训练集）的数据流水线和专用评估。

要回答的问题：v6 人工盲审"跑题 123/160"，是不是因为模型在背题？
  · v6 的 4,214 条数据只有 149 个不同的核心开场问题，每个问题被 16 型反复覆盖；
  · v6 按"条"随机切 8% 当验证集 → 验证集里的问题训练时都见过，val loss 看不出死记硬背。
所以这里切两种验证集：
  · 见过的问题（按条随机切，复刻 v6 的做法）
  · 没见过的问题（按开场问题整组留出，训练中从不出现）
并在每次评估时测（都不需要 LLM 判官，结果可复现）：
  · 两种验证损失及其差距（背题程度）
  · 型可分度：同一条回答放在 16 种人格标签下，哪个标签让模型觉得它最可能 → 命中率（随机 6.25%）
  · 生成回答的背诵率：回答里的 8 字片段有多少能在训练集回答里原样找到
  · 生成回答的切题度：用基座给回答打分，配自己的问题 vs 配别人的问题，概率高多少
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

TYPES = [a + b + c + d for a in "EI" for b in "SN" for c in "TF" for d in "JP"]
TAG_RE = re.compile(r"^\[以 ([EI][SN][TF][JP]) 人格的身份回复(，语气符合该人格)?\] ?")
FILES = ["persona_full_v5.jsonl", "persona_v2_extra_v5.jsonl"]


def tag(T, body, suffix=""):
    """两种标签写法都原样保留（数据里有 377 条带「，语气符合该人格」）。"""
    return f"[以 {T} 人格的身份回复{suffix}] {body}"


def core_question(body):
    """开场问题的"核心"：去掉「好友说：/朋友说：」前缀和括号里的补充说明。
    数据里有「深夜有点emo，想找人说说话（深夜独处型…）」这类变体，不归一化会把同一个问题分到训练和留出两边（实测 24 组里有 9 组泄漏）。"""
    first = body.split("\n")[0].strip()
    first = re.sub(r"^(好友|朋友)说：", "", first)
    return re.split(r"[（(]", first)[0].strip("，。！？ ")


SEP = "\n\n请按以上说话风格回复好友。\n"  # v8 的用户消息格式：人设卡 + 这句引导语 + 好友的话


def make_prompt(r, T, cards):
    """按这条数据原本的格式，换成人格 T 的提示：v6 = 人格标签，v8 = 整张人设卡。"""
    if r.get("fmt") == "card":
        return cards[T] + SEP + r["body"]
    return tag(T, r["body"], r.get("suffix", ""))


def _load_v8(src_dir, cards):
    rows = []
    by_card = {v.strip(): k for k, v in cards.items()}
    for f in ("train.jsonl", "valid.jsonl"):
        for l in (src_dir / f).read_text(encoding="utf-8").splitlines():
            m = json.loads(l)["messages"]
            u, a = m[0]["content"], m[-1]["content"]
            card, body = u.split(SEP, 1)
            rows.append({"instruction": u, "input": "", "output": a, "_type": by_card.get(card.strip()), "_body": body})
    return rows


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_data(c, tok, hub, root):
    lin = {"stages": [], "mbti": {}}
    src_dir = hub / c["dataset_file"]
    rows, srcs = [], []
    v8 = c["dataset"] == "mbti-v8"
    cards = {T: (src_dir / "cards" / f"{T}.txt").read_text(encoding="utf-8").strip() for T in TYPES}
    for f in (["train.jsonl", "valid.jsonl"] if v8 else FILES):
        p = src_dir / f
        rs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        srcs.append({"file": str(p.relative_to(root)), "rows": len(rs), "sha256": _sha(p), "bytes": p.stat().st_size})
        rows += rs
    if v8:
        rows = _load_v8(src_dir, cards)
    lin["source"] = {
        "repo": "你自己的 v6 训练集", "url": "",
        "file": " + ".join(s["file"] for s in srcs), "bytes": sum(s["bytes"] for s in srcs),
        "sha256": "; ".join(f"{s['file'].split('/')[-1]}={s['sha256'][:16]}…" for s in srcs),
        "license": c["dataset_license"], "rows": len(rows), "columns": list(rows[0].keys()),
        "origin": ("MBTI 项目 v8 noGate 训练集（mlx_data train 2,637 + valid 229）：教师 MiniMax-M3 + 人设卡 v1 生成，只做长度和泄漏过滤；提示 = 整张人设卡 + 好友的话，不含人格字母"
                   if v8 else "MBTI 项目 v6 实际训练集（persona_full_v5 3,636 + persona_v2_extra_v5 578）：教师模型生成、规则降噪后的 16 型人格陪伴对话"),
        "raw_examples": rows[:3], "files": srcs,
    }
    stage = lambda name, why, kept, dropped=None: lin["stages"].append(
        {"name": name, "why": why, "kept": kept, "dropped_examples": (dropped or [])[:3]})
    stage("原始数据", "两个 jsonl 文件合并", len(rows))

    bad, ok = [], []
    for r in rows:
        if v8:
            if r["_type"] and r["output"].strip():
                ok.append({**r, "type": r["_type"], "body": r["_body"], "fmt": "card", "group": core_question(r["_body"])})
            else:
                bad.append(r)
            continue
        m = TAG_RE.match(r["instruction"])
        if not m or not r["output"].strip():
            bad.append(r); continue
        body = r["instruction"][m.end():]
        ok.append({**r, "type": m.group(1), "body": body, "suffix": m.group(2) or "", "group": core_question(body)})
    rows = ok
    stage("解析人格", "v8：用人设卡全文反查是哪一型；v6：读指令开头的「[以 XXXX 人格的身份回复]」。认不出人格或空回答的丢掉", len(rows), bad)

    seen, dup, uniq = set(), [], []
    for r in rows:
        k = (r["instruction"], r["output"])
        (dup if k in seen else uniq).append(r); seen.add(k)
    rows = uniq
    stage("完全重复去重", "指令和回答都一样的行只留一条", len(rows), dup)

    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    rng = random.Random(c["seed"])
    gnames = sorted(groups)
    rng.shuffle(gnames)
    n_unseen = max(1, round(len(gnames) * c["unseen_group_frac"]))
    unseen_g, seen_g = set(gnames[:n_unseen]), set(gnames[n_unseen:])
    stage("按开场问题分组", f"去掉人格标签、「好友说/朋友说」前缀和括号里的变体说明后，{len(rows)} 条只有 {len(groups)} 个不同的开场问题（最大的一组 {sizes[0]} 条）。"
          f"整组留出 {n_unseen} 个问题（{round(100 * c['unseen_group_frac'])}%）当「没见过的问题」，训练中从不出现",
          sum(len(groups[g]) for g in seen_g))

    def fits(r):
        p, a = tok.encode(render_prompt(r["instruction"])), tok.encode(r["output"].strip() + "<|im_end|>")
        return (p, a) if len(p) + len(a) <= c["max_len"] else None

    seen_rows = [r for g in seen_g for r in groups[g]]
    rng.shuffle(seen_rows)
    long_, kept = [], []
    for r in seen_rows:
        pa = fits(r)
        if pa is None:
            long_.append(r); continue
        kept.append({**r, "_p": pa[0], "_a": pa[1]})
    n_vs = round(len(kept) * c["seen_val_frac"])
    va_seen, tr = kept[:n_vs], kept[n_vs:]
    unseen = []
    for g in sorted(unseen_g):
        for r in groups[g]:
            pa = fits(r)
            if pa is not None:
                unseen.append({**r, "_p": pa[0], "_a": pa[1]})
    rng.shuffle(unseen)
    stage(f"去掉超过 {c['max_len']} token 的", "本机显存与速度的折中", len(kept), long_)
    tr_groups = {r["group"] for r in tr}
    stage("切分：训练 / 验证（见过的问题）/ 验证（没见过的问题）",
          f"训练 {len(tr)} 条；「见过的问题」验证 {len(va_seen)} 条按条随机切（复刻 v6 的 8% 自切分），"
          f"其中 {sum(r['group'] in tr_groups for r in va_seen)} 条的开场问题在训练集里出现过；"
          f"「没见过的问题」{len(unseen)} 条，与训练集的开场问题重叠 0 条", len(tr))

    # 回答的 8 字片段库：用来判断生成的回答是不是在背训练集
    grams = set()
    for r in tr:
        s = r["output"]
        grams.update(s[i:i + 8] for i in range(len(s) - 7))

    tc = {T: 0 for T in TYPES}
    for r in tr:
        tc[r["type"]] += 1
    gc = {}
    for r in tr:
        gc[r["group"]] = gc.get(r["group"], 0) + 1
    lin["mbti"] = {
        "n_groups": len(groups), "n_unseen_groups": n_unseen, "group_sizes": sizes,
        "type_counts": tc, "train_groups": len(tr_groups),
        "top_groups": sorted(gc.items(), key=lambda kv: -kv[1])[:8],
        "mean_rows_per_group": round(len(tr) / max(1, len(tr_groups)), 1),
        "seen_val_overlap": sum(r["group"] in tr_groups for r in va_seen), "seen_val_n": len(va_seen),
        "unseen_n": len(unseen), "unseen_groups_examples": sorted(unseen_g)[:8],
    }

    def tokview(ex):
        return {"prompt": [tok.decode([i]) for i in ex["_p"]], "answer": [tok.decode([i]) for i in ex["_a"]],
                "raw": {"instruction": ex["instruction"], "input": "", "output": ex["output"]}}
    pl = np.array([len(e["_p"]) for e in tr]); al = np.array([len(e["_a"]) for e in tr])
    lin["format"] = {
        "template": ("<|im_start|>user\\n{整张人设卡}\\n\\n请按以上说话风格回复好友。\\n{好友说的话}<|im_end|>\\n<|im_start|>assistant\\n{回答}<|im_end|>" if v8 else "<|im_start|>user\\n[以 {人格} 人格的身份回复] {好友说的话}<|im_end|>\\n<|im_start|>assistant\\n{回答}<|im_end|>"),
        "examples": [tokview(e) for e in tr[:3]],
        "prompt_len_hist": np.histogram(pl, bins=20, range=(0, c["max_len"]))[0].tolist(),
        "answer_len_hist": np.histogram(al, bins=20, range=(0, c["max_len"]))[0].tolist(),
        "hist_range": [0, c["max_len"]],
        "trained_token_frac": round(float(al.sum() / (al.sum() + pl.sum())), 4),
        "train_tokens": int(pl.sum() + al.sum()), "answer_tokens": int(al.sum()),
        "epochs": round(c["max_steps"] * c["batch_size"] * c["grad_accum"] / len(tr), 2),
    }
    return lin, tr, va_seen, unseen, grams, cards


def render_prompt(instruction):
    return f"<|im_start|>user\n{instruction.strip()}<|im_end|>\n<|im_start|>assistant\n"


def pick_disc_set(unseen, n_per_type, seed):
    """型可分度用的固定子集：每型取 n 条单轮回答（多轮的上下文本身就带人格口吻，会泄露答案）。"""
    rng = random.Random(seed + 7)
    by = {T: [r for r in unseen if r["type"] == T and "\n" not in r["body"]] for T in TYPES}
    out = []
    for T in TYPES:
        rs = by[T][:]
        rng.shuffle(rs)
        out += rs[:n_per_type]
    return out


def pick_gen_set(unseen, n, seed):
    """生成用：n 个不同的没见过的问题，人格轮流取，单轮。"""
    rng = random.Random(seed + 11)
    singles = [r for r in unseen if "\n" not in r["body"]]
    rng.shuffle(singles)
    out, used = [], set()
    for r in singles:
        if r["group"] in used:
            continue
        used.add(r["group"]); out.append(r)
        if len(out) >= n:
            break
    for i, r in enumerate(out):
        r = out[i] = {**r, "gen_type": TYPES[(i * 5) % 16]}  # 5 与 16 互质：12 个问题覆盖 12 种人格
    return out


@torch.no_grad()
def answer_logprob(model, tok, device, prompts, answers, bs=16):
    """每条回答在给定提示下的平均每 token 对数概率。"""
    out = []
    for i in range(0, len(prompts), bs):
        P = [tok.encode(p) for p in prompts[i:i + bs]]
        A = [tok.encode(a) for a in answers[i:i + bs]]
        L = max(len(p) + len(a) for p, a in zip(P, A))
        L = int(np.ceil(L / 64) * 64)
        ids = torch.full((len(P), L), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(P), L), -100, dtype=torch.long)
        att = torch.zeros((len(P), L), dtype=torch.long)
        for j, (p, a) in enumerate(zip(P, A)):
            s = p + a
            ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
            lab[j, len(p):len(s)] = torch.tensor(a)
        ids, lab, att = ids.to(device), lab.to(device), att.to(device)
        # 只在回答位置过输出层（15 万词表的完整 logits 会占好几 GB）
        h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
        y = lab[:, 1:]
        m = y != -100
        row = torch.arange(y.size(0), device=device)[:, None].expand_as(y)[m]
        lp = model.lm_head(h[m]).float().log_softmax(-1).gather(1, y[m][:, None])[:, 0]
        tot = torch.zeros(y.size(0), device=device).index_add_(0, row, lp)
        out += (tot / m.sum(1).clamp(min=1)).tolist()
    return out


def type_discrimination(model, tok, device, rows, prompt_fn):
    """每条回答放进 16 种人格的提示里打分，概率最高的那个人格算模型的"判型"。"""
    prompts, answers = [], []
    for r in rows:
        for T in TYPES:
            prompts.append(render_prompt(prompt_fn(T, r)))
            answers.append(r["output"].strip())
    lp = np.array(answer_logprob(model, tok, device, prompts, answers)).reshape(len(rows), 16)
    pred = [TYPES[int(i)] for i in lp.argmax(1)]
    truth = [r["type"] for r in rows]
    top1 = float(np.mean([p == t for p, t in zip(pred, truth)]))
    axes = [float(np.mean([p[k] == t[k] for p, t in zip(pred, truth)])) for k in range(4)]
    # 真实标签的排名（1 = 第一）
    ranks = [int((lp[i] > lp[i, TYPES.index(t)]).sum()) + 1 for i, t in enumerate(truth)]
    conf = {T: 0 for T in TYPES}
    for p in pred:
        conf[p] += 1
    return {"top1": round(top1, 4), "axes": [round(a, 4) for a in axes], "mean_rank": round(float(np.mean(ranks)), 3),
            "pred_counts": conf}


def memo_rate(text, grams):
    """生成文本中，有多少比例的 8 字片段能在训练集回答里原样找到。"""
    s = text.strip()
    n = len(s) - 7
    if n <= 0:
        return 0.0
    return sum(s[i:i + 8] in grams for i in range(n)) / n


def relevance(model, tok, device, gens, set_base):
    """切题度：用基座给生成的回答打分。配自己的问题 vs 配其它问题，平均每 token 对数概率高多少。"""
    own_p, oth_p, ans = [], [], []
    k = len(gens)
    for i, g in enumerate(gens):
        if not g["output"].strip():
            continue
        other = gens[(i + k // 2) % k]
        own_p.append(render_prompt(g["body"])); oth_p.append(render_prompt(other["body"])); ans.append(g["output"].strip())
    if not ans:
        return [], None
    with set_base():
        a = answer_logprob(model, tok, device, own_p, ans)
        b = answer_logprob(model, tok, device, oth_p, ans)
    d = [x - y for x, y in zip(a, b)]
    return d, round(float(np.mean(d)), 4)
