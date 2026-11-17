"""真实基座模型的 LoRA 指令微调（SFT），全过程可追溯、可视化。

流水线（每一步都记录到 runs/<name>/lineage.json，看板「数据血缘」页逐步展示）：
  ① 来源   公开数据集文件 → 校验和、行数、许可证、原始样例
  ② 清洗   去空、去重、去超长 → 每一步剩多少、被删的样例和原因
  ③ 抽样   固定随机种子切成 训练 / 验证 / 测试
  ④ 模板   套用对话模板（ChatML），把"指令+回答"变成一段带特殊标记的文本
  ⑤ 分词   变成 token；只有"回答"部分计算损失（loss mask）
  ⑥ 训练   冻结基座 6 亿参数，只训练插入的 LoRA 小矩阵
  ⑦ 去向   产出 adapter（几 MB），可随时开关、对比基座

影响评估（每 eval_every 步，第 0 步 = 原始基座）：
  · 验证集回答损失（学会了没有）
  · 通用文本损失：你的中文文档 + Python 代码（有没有"遗忘"）
  · 与基座的 KL 偏离（模型整体被改动了多少）
  · 固定探针问题的生成结果（行为变化），是否学会"答完就停"
  · 参考答案逐 token 概率变化（哪些词变得更可能）
  · 每层每个模块的 LoRA 改动幅度 |ΔW|/|W|（改了哪里）

用法：python -m lab.finetune --preset sft-alpaca --name my-sft
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np

# 给苹果 GPU 显存设上限（约为系统建议上限的 60%）：超出就报错，而不是把整台机器拖进内存交换
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
import torch
import torch.nn as nn
import torch.nn.functional as F

from .train import RUNS, ROOT, Recorder, pick_device

HUB = ROOT / "data" / "hub"

FT_DEFAULTS = dict(
    kind="finetune",
    base="Qwen3-0.6B-Base", base_repo="Qwen/Qwen3-0.6B-Base",
    dataset="alpaca-zh", dataset_repo="shibing624/alpaca-zh",
    dataset_file="alpaca-zh/default/train/0000.parquet",
    dataset_license="卡片标注 CC BY-NC 4.0：仅限研究 / 非商用（由 GPT-4 生成）",
    n_train=2400, n_val=200, n_test=50, max_len=384,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    train_special_tokens=True, special_tokens=["<|im_start|>", "<|im_end|>"],
    lr=2e-4, min_lr=2e-5, warmup=20, weight_decay=0.0, grad_clip=1.0,
    batch_size=4, grad_accum=2, max_steps=300, eval_every=30, log_every=5, seed=42,
    hypothesis="", notes="",
    unseen_group_frac=0.1, seen_val_frac=0.08, disc_per_type=3, n_gen=12, save_every_eval=False,
)

MBTI = dict(dataset="mbti-v6", dataset_repo="你自己的 v6 训练集", dataset_file="mbti-v6",
            dataset_license="你自己的项目数据（教师模型生成），仅本地使用", max_len=320,
            lr=1e-4, min_lr=1e-5, warmup=30, batch_size=8, grad_accum=1)

FT_PRESETS = {
    "sft-alpaca": dict(hypothesis="基座只会复读问题、不会作答。用 2400 条中文指令做 LoRA SFT，并训练 2 个特殊标记的嵌入后，它会按问答格式回答并在答完时停下；"
                                  "代价是对通用文本（代码、普通中文）的损失略微上升（轻度遗忘）。"),
    "sft-no-special": dict(train_special_tokens=False,
                           hypothesis="对照组：只训练 LoRA、不训练特殊标记的嵌入。预期模型学会回答，但学不会答完就停（<|im_end|> 在基座里是未训练的占位向量）。"),
    "sft-alpaca-attn-only": dict(targets=["q_proj", "k_proj", "v_proj", "o_proj"],
                                 hypothesis="只在注意力层插 LoRA（可训练参数少约 60%）：同样能学会格式，但验证损失略差。"),
    "sft-alpaca-r4": dict(lora_r=4, lora_alpha=8, hypothesis="秩从 16 降到 4：参数少 4 倍，学指令格式这种简单任务应该差别不大。"),
    "sft-alpaca-lr-high": dict(lr=2e-3, min_lr=2e-4, hypothesis="学习率提高 10 倍：学得更快，但遗忘（通用文本损失上升、KL 偏离）会明显更大。"),
    "mbti-4ep": dict(**MBTI, max_steps=1760, eval_every=110,
                     hypothesis="复刻 v6 的做法（同一份数据、4 个 epoch）：「见过的问题」验证损失会一直降，看起来越训越好；"
                                "但「没见过的问题」验证损失先降后升，生成的回答越来越多地照搬训练集原句、切题度下降——这就是 v6 人工盲审跑题 75% 的来源。"),
    "mbti-v8-1ep": dict(**{**MBTI, "dataset": "mbti-v8", "dataset_file": "mbti-v8", "dataset_repo": "你自己的 v8 noGate 训练集", "max_len": 768, "batch_size": 4, "grad_accum": 2},
                        max_steps=330, eval_every=110,
                        hypothesis="对照实验：7B 上的 v8 adapter 训 1 轮就崩了（复读、背人设卡里的指令词）。用同一份 v8 数据、TinyLab 的配方在 0.6B 上训 1 轮："
                                   "如果也崩 → 问题出在数据（教师口头禅、人设卡格式）；如果不崩 → 问题出在 7B 那边的训练配方。"),
    "mbti-new-0.6b": dict(**{**MBTI, "dataset": "mbti-new", "dataset_file": "data/mbti-new", "dataset_repo": "新 MBTI 数据集（DeepSeek 生成）",
                             "dataset_license": "DeepSeek 生成（开放平台条款 4.2(3) 允许用于训练其他模型）", "max_len": 384},
                          max_steps=160, eval_every=40,
                          hypothesis="新数据（核心问题多、无口头禅、人格设定放 system）训练 0.6B：没见过的问题上型贴近度明显高于基座+人设卡，生成不复读、不泄漏设定词，答完会停。"),
    "mbti-new-1.7b": dict(**{**MBTI, "dataset": "mbti-new", "dataset_file": "data/mbti-new", "dataset_repo": "新 MBTI 数据集（DeepSeek 生成）",
                             "dataset_license": "DeepSeek 生成（开放平台条款 4.2(3) 允许用于训练其他模型）", "max_len": 384,
                             "base": "Qwen3-1.7B-Base", "base_repo": "Qwen/Qwen3-1.7B-Base", "batch_size": 4, "grad_accum": 2},
                          max_steps=160, eval_every=40,
                          hypothesis="同样的新数据，基座换成 1.7B：没见过的问题上型贴近度和切题度都比 0.6B 高；1.7B 是端侧（RK3576 约 11 tok/s、手机 18–36 tok/s）能流畅运行的主力尺寸。"),
    "mbti-1ep": dict(**MBTI, max_steps=440, eval_every=110,
                     hypothesis="同样的数据只训 1 个 epoch：没见过的问题上的损失和型可分度不比 4 个 epoch 差，背诵率明显更低，回答更切题。"),
}

PROBES = [  # 固定探针：每次评估都用它们生成，看行为如何变化
    {"kind": "指令", "prompt": "用三句话解释什么是机器学习。"},
    {"kind": "指令", "prompt": "给我一个周末学习计划，要求包含运动和阅读。"},
    {"kind": "指令", "prompt": "把下面这句话翻译成英文：今天的天气非常好，适合出去散步。"},
    {"kind": "指令", "prompt": "写一首关于秋天的四行短诗。"},
    {"kind": "知识", "prompt": "中国的首都是哪里？"},
    {"kind": "数学", "prompt": "小明有 12 个苹果，给了小红 5 个，又买了 8 个，现在有几个？"},
]


# ---------------------------------------------------------------- LoRA（手写，约 40 行）

class LoRALinear(nn.Module):
    """y = W x  +  (alpha / r) · B A x
    W 是冻结的原始权重（不改动）；A (r×in)、B (out×r) 是新加的小矩阵，只训练它们。
    B 初始化为 0，所以训练开始时模型和基座完全一样。"""

    def __init__(self, base: nn.Linear, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        d = (self.drop(x).float() @ self.A.T @ self.B.T) * self.scale
        return y + d.to(y.dtype)

    def delta_ratio(self):
        """|ΔW| / |W|：这个模块被改动的相对幅度（ΔW = scale · B A）"""
        dW = (self.B @ self.A) * self.scale
        return (dW.norm() / self.base.weight.float().norm()).item()


class SpecialTokens(nn.Module):
    """给少数特殊标记（如 <|im_end|>）的嵌入加一个可训练的修正量 D。
    Qwen 的输入嵌入和输出层共享同一个矩阵，所以 D 同时作用在"读到这个标记"和"说出这个标记"两处。
    为什么需要：基座里 <|im_start|>、<|im_end|> 和另外约 290 个保留标记的向量一模一样——它们是
    从未被训练过的占位向量。冻结状态下，模型无法把"结束"和其他占位标记区分开，几乎说不出它，答完就不会停。"""
    def __init__(self, model, ids):
        super().__init__()
        self.ids = torch.tensor(ids)
        self.D = nn.Parameter(torch.zeros(len(ids), model.config.hidden_size, dtype=torch.float32))
        self.emb, self.head = model.model.embed_tokens, model.lm_head
        sp = self

        class Emb(nn.Module):
            def __init__(self): super().__init__(); self.base = sp.emb
            @property
            def weight(self): return self.base.weight
            def forward(self, ids):
                e = self.base(ids)
                hit = (ids[..., None] == sp.ids.to(ids.device)).to(sp.D.dtype)  # [.., k]
                return e + (hit @ sp.D).to(e.dtype)

        class Head(nn.Module):
            def __init__(self): super().__init__(); self.base = sp.head
            @property
            def weight(self): return self.base.weight
            def forward(self, h):
                y = self.base(h)
                add = (h.float() @ sp.D.T).to(y.dtype)
                idx = sp.ids.to(y.device)
                return y.index_add(-1, idx, add)

        model.model.embed_tokens = Emb()
        model.lm_head = Head()

    def norms(self):
        base = self.emb.weight[self.ids.to(self.emb.weight.device)].float()
        return {"before": base.norm(dim=-1).tolist(), "after": (base + self.D).norm(dim=-1).tolist()}


def inject_lora(model, c):
    hits = []
    for li, layer in enumerate(model.model.layers):
        for parent_name in ("self_attn", "mlp"):
            parent = getattr(layer, parent_name)
            for t in c["targets"]:
                if hasattr(parent, t) and isinstance(getattr(parent, t), nn.Linear):
                    lin = LoRALinear(getattr(parent, t), c["lora_r"], c["lora_alpha"], c["lora_dropout"]).to(getattr(parent, t).weight.device)
                    setattr(parent, t, lin)
                    hits.append((li, t, lin))
    return hits


def set_lora(hits, on):
    for _, _, m in hits:
        m.enabled = on


# ---------------------------------------------------------------- 数据流水线（带血缘记录）

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def render(ex, tok, with_answer=True):
    """ChatML 对话模板。注意：基座里这两个特殊标记的嵌入是未经训练的占位向量（见 SpecialTokens）。"""
    q = ex["instruction"].strip() + (("\n" + ex["input"].strip()) if ex.get("input", "").strip() else "")
    prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    if not with_answer:
        return prompt
    return prompt, ex["output"].strip() + "<|im_end|>"


def build_data(c, tok):
    import pyarrow.parquet as pq
    lin = {"stages": []}
    src = HUB / c["dataset_file"]
    rows = pq.read_table(src).to_pylist()
    lin["source"] = {
        "repo": c["dataset_repo"], "url": f"https://huggingface.co/datasets/{c['dataset_repo']}",
        "file": str(src.relative_to(ROOT)), "bytes": src.stat().st_size, "sha256": sha256(src),
        "license": c["dataset_license"], "rows": len(rows), "columns": list(rows[0].keys()),
        "origin": "Alpaca 方法：用 GPT-4 按种子指令自动生成的中文问答（self-instruct）",
        "raw_examples": rows[:3],
    }
    stage = lambda name, why, kept, dropped=None: lin["stages"].append(
        {"name": name, "why": why, "kept": kept, "dropped_examples": (dropped or [])[:3]})
    stage("原始数据", "从 parquet 读入全部行", len(rows))

    empty = [r for r in rows if not r["instruction"].strip() or not r["output"].strip()]
    rows = [r for r in rows if r["instruction"].strip() and r["output"].strip()]
    stage("去掉空指令 / 空回答", "空样本只会教模型输出空白", len(rows), empty)

    seen, dup, uniq = set(), [], []
    for r in rows:
        k = (r["instruction"].strip(), r["input"].strip())
        (dup if k in seen else uniq).append(r)
        seen.add(k)
    rows = uniq
    stage("按指令去重", "重复样本会让模型对少数问题过度记忆", len(rows), dup)

    rng = random.Random(c["seed"])
    rng.shuffle(rows)
    need = c["n_train"] + c["n_val"] + c["n_test"]
    pool, too_long, out = rows[: need * 2], [], []
    for r in pool:
        p, a = render(r, tok)
        pi, ai = tok.encode(p), tok.encode(a)
        if len(pi) + len(ai) > c["max_len"]:
            too_long.append({**r, "_tokens": len(pi) + len(ai)})
            continue
        out.append({**r, "_p": pi, "_a": ai})
        if len(out) >= need:
            break
    stage(f"随机打乱并取样（种子 {c['seed']}），去掉超过 {c['max_len']} token 的",
          f"本机显存与速度的折中；按比例估算全量约 {round(100 * len(too_long) / max(1, len(too_long) + len(out)))}% 的样本会因超长被丢弃",
          len(out), too_long)
    tr, va, te = out[: c["n_train"]], out[c["n_train"]: c["n_train"] + c["n_val"]], out[c["n_train"] + c["n_val"]:]
    stage("切分：训练 / 验证 / 测试", f"训练 {len(tr)} 条用于更新参数；验证 {len(va)} 条只用于打分；测试 {len(te)} 条留给最终对比，训练中从不看",
          len(tr))

    # 模板与 loss mask 的可视化样例
    def tokview(ex):
        return {"prompt": [tok.decode([i]) for i in ex["_p"]], "answer": [tok.decode([i]) for i in ex["_a"]],
                "raw": {k: ex[k] for k in ("instruction", "input", "output")}}
    pl = np.array([len(e["_p"]) for e in tr]); al = np.array([len(e["_a"]) for e in tr])
    lin["format"] = {
        "template": "<|im_start|>user\\n{指令}\\n{输入}<|im_end|>\\n<|im_start|>assistant\\n{回答}<|im_end|>",
        "examples": [tokview(e) for e in tr[:3]],
        "prompt_len_hist": np.histogram(pl, bins=20, range=(0, c["max_len"]))[0].tolist(),
        "answer_len_hist": np.histogram(al, bins=20, range=(0, c["max_len"]))[0].tolist(),
        "hist_range": [0, c["max_len"]],
        "trained_token_frac": round(float(al.sum() / (al.sum() + pl.sum())), 4),
        "train_tokens": int(pl.sum() + al.sum()), "answer_tokens": int(al.sum()),
        "epochs": round(c["max_steps"] * c["batch_size"] * c["grad_accum"] / len(tr), 2),
    }
    return lin, tr, va, te


def collate(batch, pad_id, device, L=None):
    # 补齐到固定长度：苹果 GPU 上每次形状不同会让显存缓存不断膨胀（实测锁住 20GB+）
    L = L or max(len(e["_p"]) + len(e["_a"]) for e in batch)
    L = int(math.ceil(L / 64) * 64)  # 只用少数几种长度（64 的倍数），既省计算又不让显存缓存膨胀
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, e in enumerate(batch):
        s = e["_p"] + e["_a"]
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
        lab[i, len(e["_p"]): len(s)] = torch.tensor(e["_a"])  # 只对回答部分算损失
    return ids.to(device), lab.to(device), att.to(device)


def lm_loss(model, ids, lab, att, chunk=256):
    """只在"要算损失"的位置过输出层，并分块计算：15 万词表的 logits 很占显存。"""
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    y = lab[:, 1:]
    keep = y != -100
    h, y = h[keep], y[keep]
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.size(0), chunk):
        logits = model.lm_head(h[i:i + chunk]).float()
        total = total + F.cross_entropy(logits, y[i:i + chunk], reduction="sum")
    return total / max(1, y.numel())


# ---------------------------------------------------------------- 影响评估

GENERAL_DESC = {"中文文档": "中文文档 8 段 × 256 个 token（默认取本项目自己的 README 和 docs/；可用环境变量 TINYLAB_ZH_DIR 指向你自己的文件夹）",
                "Python 代码": "Python 标准库源码，8 段 × 256 个 token"}


def _zh_files():
    d = os.environ.get("TINYLAB_ZH_DIR")
    if d:
        return sorted(p for p in Path(d).expanduser().rglob("*") if p.suffix in (".md", ".txt"))
    return [ROOT / "README.md"] + sorted((ROOT / "docs").rglob("*.md"))


def general_texts(tok, n=8, L=256):
    """遗忘监测用的"通用文本"：一批中文文档 + Python 标准库代码，都不在微调数据里。
    每份文件按 256 token 切段，轮流从各文件取，凑够 8 段。"""
    chunks = []
    for p in _zh_files():
        if p.exists():
            ids = tok.encode(p.read_text(encoding="utf-8"))
            chunks.append([ids[i:i + L] for i in range(0, len(ids) - L + 1, L)][:n])
    zh = []
    for k in range(n):  # 轮流取，避免 8 段都来自同一份文件
        for c in chunks:
            if k < len(c) and len(zh) < n:
                zh.append(c[k])
    from .data import corpus_pycode
    py_ids = tok.encode(corpus_pycode(200_000)[50_000:120_000])
    return {"中文文档": zh, "Python 代码": [py_ids[i * L:(i + 1) * L] for i in range(n) if (i + 1) * L <= len(py_ids)]}


@torch.no_grad()
def evaluate(model, hits, tok, va, gen_sets, kl_prompts, device, c):
    model.eval()
    out = {}
    losses = []
    for i in range(0, len(va), 8):
        ids, lab, att = collate(va[i:i + 8], tok.pad_token_id, device)
        losses.append(lm_loss(model, ids, lab, att).item())
    out["val_loss"] = float(np.mean(losses))
    for name, chunks in gen_sets.items():
        x = torch.tensor(chunks, device=device)
        out[f"general:{name}"] = lm_loss(model, x, x.clone(), torch.ones_like(x)).item()
    # KL(当前 ‖ 基座)：在通用文本上，两个模型下一个 token 分布的平均差异
    x = torch.tensor(kl_prompts, device=device)
    lp = model(input_ids=x).logits.float().log_softmax(-1)
    sp = getattr(model, "_tl_special", None)
    set_lora(hits, False)
    if sp is not None:
        saved = sp.D.data.clone(); sp.D.data.zero_()
    lq = model(input_ids=x).logits.float().log_softmax(-1)
    set_lora(hits, True)
    if sp is not None:
        sp.D.data.copy_(saved)
    out["kl_to_base"] = (lp.exp() * (lp - lq)).sum(-1).mean().item()
    model.train()
    return out


@torch.no_grad()
def probe_generate(model, tok, device, prompts, max_new=96):
    model.eval()
    res = []
    for p in prompts:
        ids = tok.encode(p["raw"] if p.get("raw") else render({"instruction": p["prompt"], "input": ""}, tok, with_answer=False))
        x = torch.tensor([ids], device=device)
        g = model.generate(x, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True,
                           eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id])
        new = g[0, len(ids):].tolist()
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        # 只有输出 <|im_end|> 才算"答完会停"；基座常在复读之后输出 <|endoftext|>（文档结束），那不算
        stop = "im_end" if im_end in new else "endoftext" if tok.eos_token_id in new else None
        res.append({"kind": p["kind"], "prompt": p["prompt"], "output": tok.decode(new, skip_special_tokens=True),
                    "stopped": stop == "im_end", "stop": stop, "n_tokens": len(new)})
    model.train()
    return res


@torch.no_grad()
def token_logprobs(model, tok, ex, device):
    """参考答案每个 token 的对数概率：和第 0 步对比，就能看到"哪些词变得更可能了"。"""
    model.eval()
    s = ex["_p"] + ex["_a"]
    x = torch.tensor([s], device=device)
    lp = model(input_ids=x).logits[0, :-1].float().log_softmax(-1)
    tgt = x[0, 1:]
    v = lp.gather(1, tgt[:, None])[:, 0][len(ex["_p"]) - 1:]
    model.train()
    return [round(a, 3) for a in v.tolist()]


# ---------------------------------------------------------------- 主流程

def lr_at(step, c):
    if step < c["warmup"]:
        return c["lr"] * (step + 1) / c["warmup"]
    t = (step - c["warmup"]) / max(1, c["max_steps"] - c["warmup"])
    return c["min_lr"] + 0.5 * (c["lr"] - c["min_lr"]) * (1 + math.cos(math.pi * min(1.0, t)))


def run(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    c = {**FT_DEFAULTS, **cfg}
    name = c.get("name") or time.strftime("sft-%m%d-%H%M%S")
    c["name"] = name
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    rec = Recorder(d)
    rec.status(state="preparing", step=0, max_steps=c["max_steps"], pid=os.getpid(), started=time.time())
    random.seed(c["seed"]); torch.manual_seed(c["seed"])
    device = pick_device()
    bp = HUB / c["base"]
    tok = AutoTokenizer.from_pretrained(bp)

    rec.status(phase="数据流水线")
    is_mbti = c["dataset"].startswith("mbti")
    if is_mbti:
        from . import mbti
        lineage, tr, va, te, grams, cards = mbti.build_data(c, tok, HUB, ROOT)
        va_unseen = te[:200]
        disc_rows = mbti.pick_disc_set(te, c["disc_per_type"], c["seed"])
        gen_rows = mbti.pick_gen_set(te, c["n_gen"], c["seed"])
    else:
        lineage, tr, va, te = build_data(c, tok)

    rec.status(phase="加载基座")
    model = AutoModelForCausalLM.from_pretrained(bp, dtype=torch.bfloat16).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    mc = model.config
    n_base = sum(p.numel() for p in model.parameters())
    comp = {"词嵌入（与输出层共享）": model.model.embed_tokens.weight.numel(), "注意力": 0, "MLP": 0, "归一化": 0}
    for n, p in model.named_parameters():
        if "self_attn" in n and "norm" not in n: comp["注意力"] += p.numel()
        elif "mlp" in n: comp["MLP"] += p.numel()
        elif "norm" in n: comp["归一化"] += p.numel()
    hits = inject_lora(model, c)
    params = [p for _, _, m in hits for p in (m.A, m.B)]
    special = None
    if c["train_special_tokens"]:
        special = SpecialTokens(model, [tok.convert_tokens_to_ids(t) for t in c["special_tokens"]]).to(device)
        params.append(special.D)
        object.__setattr__(model, "_tl_special", special)
    n_lora = sum(p.numel() for p in params)
    if c.get("grad_ckpt", True):
        # 梯度检查点：前向时不保存每层中间结果，反向时重算。实测把一个 batch 的显存从约 7GB 降到 1–2GB，代价是约 30% 的时间
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    opt = torch.optim.AdamW(params, lr=c["lr"], weight_decay=c["weight_decay"], betas=(0.9, 0.999))

    gen_sets = general_texts(tok)
    kl_prompts = gen_sets["中文文档"][:4] + gen_sets["Python 代码"][:4]
    ref = te[0]
    probes = PROBES + [{"kind": "测试集", "prompt": (e["instruction"] + ("\n" + e["input"] if e["input"].strip() else "")).strip(),
                        "reference": e["output"]} for e in te[:2]]
    if is_mbti:
        probes = [{"kind": g["gen_type"], "prompt": g["body"], "raw": mbti.full_prompt(g, g["gen_type"], cards), "body": g["body"],
                   "reference": g["output"] if g["type"] == g["gen_type"] else ""} for g in gen_rows]
    lineage["destination"] = {
        "adapter": f"runs/{name}/adapter.pt", "note": "只保存 LoRA 的 A、B 矩阵；基座文件不改动。加载 = 基座 + adapter；随时可以关掉 adapter 回到基座。"}
    (d / "lineage.json").write_text(json.dumps(lineage, ensure_ascii=False))
    (d / "config.json").write_text(json.dumps({
        **c, "device": device, "n_params": n_base, "n_trainable": n_lora,
        "base_arch": {"layers": mc.num_hidden_layers, "hidden": mc.hidden_size, "heads": mc.num_attention_heads,
                      "kv_heads": mc.num_key_value_heads, "head_dim": getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads),
                      "ffn": mc.intermediate_size, "vocab": mc.vocab_size, "context": mc.max_position_embeddings},
        "base_components": comp, "lora_modules": sorted(set(t for _, t, _ in hits)),
        "special_info": {"tokens": c["special_tokens"], "trainable": special.D.numel() if special else 0,
                         "norm_before": special.norms()["before"] if special else None,
                         "avg_token_norm": round(model.model.embed_tokens.weight.float().norm(dim=-1).mean().item(), 4)},
        "tokens_per_step": c["batch_size"] * c["grad_accum"] * c["max_len"],
        "data_report": {"name": c["dataset"], "train_examples": len(tr)},
        "mbti": {"disc_n": len(disc_rows), "gen_n": len(gen_rows), "unseen_val_n": len(va_unseen)} if is_mbti else None,
        "probes": probes, "general_desc": GENERAL_DESC, "ref_example": {"prompt": [tok.decode([i]) for i in ref["_p"]], "answer": [tok.decode([i]) for i in ref["_a"]]},
        "created": time.time(),
    }, ensure_ascii=False, indent=1))
    rec.status(state="running", phase="训练", n_params=n_base, device=device)

    def save_adapter(path):
        torch.save({"lora": {f"{li}.{t}": {"A": m.A.detach().cpu(), "B": m.B.detach().cpu()} for li, t, m in hits},
                    "special": special.D.detach().cpu() if special is not None else None,
                    "config": {k: c[k] for k in ("base", "lora_r", "lora_alpha", "targets", "train_special_tokens", "special_tokens")}}, path)

    def checkpoint_eval(step):
        if c.get("save_every_eval") and step > 0:
            # 每次评估都存一份 adapter：训练结束后可以按"没见过的问题"上的指标回头挑最好的一步
            (d / "ckpts").mkdir(exist_ok=True)
            save_adapter(d / "ckpts" / f"adapter_{step:06d}.pt")
        if device == "mps":
            torch.mps.empty_cache()
        ev = evaluate(model, hits, tok, va, gen_sets, kl_prompts, device, c)
        gens = probe_generate(model, tok, device, probes, max_new=128 if is_mbti else 96)
        if is_mbti:
            ev.update(mbti_eval(step, gens))
        rec.log("eval", {"step": step, **ev})
        snap = {"step": step, "probes": gens,
                "ref_logprobs": token_logprobs(model, tok, ref, device),
                "delta": [[0.0] * len(c["targets"]) for _ in range(mc.num_hidden_layers)], "targets": c["targets"]}
        for li, t, m in hits:
            snap["delta"][li][c["targets"].index(t)] = round(m.delta_ratio(), 6)
        if special is not None:
            snap["special_norm"] = [round(v, 4) for v in special.norms()["after"]]
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        snap["stop_prob"] = round(math.exp(snap["ref_logprobs"][-1]), 6) if ref["_a"][-1] == im_end else None
        (d / "snapshots").mkdir(exist_ok=True)
        (d / "snapshots" / f"{step:07d}.json").write_text(json.dumps(snap, ensure_ascii=False))
        if device == "mps":
            torch.mps.empty_cache()

    from contextlib import contextmanager

    @contextmanager
    def base_mode():
        """临时关掉 LoRA 和特殊标记的改动，得到原始基座。"""
        set_lora(hits, False)
        saved = special.D.data.clone() if special is not None else None
        if special is not None:
            special.D.data.zero_()
        try:
            yield
        finally:
            set_lora(hits, True)
            if special is not None:
                special.D.data.copy_(saved)

    def mbti_eval(step, gens):
        model.eval()
        out = {}
        with torch.no_grad():
            lu = []
            for i in range(0, len(va_unseen), 8):
                ids, lab, att = collate(va_unseen[i:i + 8], tok.pad_token_id, device)
                lu.append(lm_loss(model, ids, lab, att).item())
        out["val_unseen_loss"] = float(np.mean(lu))
        out["disc"] = mbti.type_discrimination(model, tok, device, disc_rows, lambda T, r: mbti.full_prompt(r, T, cards))
        if step == 0:
            # 对照臂：基座 + 人设卡（不用人格标签，把整张卡放在问题前面），只在第 0 步测一次
            out["disc_card"] = mbti.type_discrimination(model, tok, device, disc_rows, lambda T, r: mbti.card_prompt(r, T, cards))
        for g in gens:
            g["memo"] = round(mbti.memo_rate(g["output"], grams), 4)
        rel, rel_mean = mbti.relevance(model, tok, device, [{**g, "body": p["body"]} for g, p in zip(gens, probes)], base_mode)
        it = iter(rel)
        for g in gens:
            if g["output"].strip():
                g["rel"] = round(next(it), 4)
        out["memo_rate"] = round(float(np.mean([g["memo"] for g in gens])), 4)
        out["memo_high"] = sum(g["memo"] >= 0.5 for g in gens)
        out["relevance"] = rel_mean
        out["stop_rate"] = round(float(np.mean([g["stopped"] for g in gens])), 4)
        model.train()
        return out

    t0 = time.time(); t_last = t0; tok_seen = 0; order = []
    model.train()
    for step in range(c["max_steps"] + 1):
        if (d / "STOP").exists():
            rec.status(state="stopped", step=step); break
        if step % c["eval_every"] == 0 or step == c["max_steps"]:
            rec.status(phase=f"评估（第 {step} 步）")
            checkpoint_eval(step)
            rec.status(phase="训练")
        if step == c["max_steps"]:
            break
        lr = lr_at(step, c)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_sum, ntok = 0.0, 0
        for _ in range(c["grad_accum"]):
            if len(order) < c["batch_size"]:
                order += random.sample(range(len(tr)), len(tr))
            batch = [tr[order.pop()] for _ in range(c["batch_size"])]
            ids, lab, att = collate(batch, tok.pad_token_id, device)
            loss = lm_loss(model, ids, lab, att)
            (loss / c["grad_accum"]).backward()
            loss_sum += loss.item(); ntok += int(att.sum())
        gn = torch.nn.utils.clip_grad_norm_(params, c["grad_clip"]).item()
        opt.step(); opt.zero_grad(set_to_none=True)
        if device == "mps" and step % 10 == 0:
            torch.mps.empty_cache()
        tok_seen += ntok
        if step % c["log_every"] == 0:
            now = time.time()
            rec.log("metrics", {"step": step, "tokens": tok_seen, "loss": round(loss_sum / c["grad_accum"], 5), "lr": lr,
                                "grad_norm": round(gn, 5), "tok_per_s": round(ntok * c["log_every"] / max(now - t_last, 1e-6)) if step else 0,
                                "elapsed": round(now - t0, 2)})
            t_last = now
            rec.status(step=step, elapsed=round(now - t0, 1))
            if not math.isfinite(loss_sum):
                rec.status(state="failed", error="loss 变成 NaN：训练发散"); return
    save_adapter(d / "adapter.pt")
    s = json.loads((d / "status.json").read_text())
    if s.get("state") != "stopped":
        rec.status(state="done", step=c["max_steps"], phase="完成", elapsed=round(time.time() - t0, 1),
                   adapter_bytes=(d / "adapter.pt").stat().st_size)


# ---------------------------------------------------------------- 对话：基座 vs 微调后

_chat = {}
import threading
_chat_lock = threading.Lock()


def chat(*a, **k):
    with _chat_lock:  # 看板可能并发请求；同一份模型一次只服务一个
        return _chat_impl(*a, **k)


@torch.no_grad()
def _chat_impl(run_name, prompt, use_lora=True, max_new=160, temperature=0.0):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = RUNS / run_name
    ad = torch.load(d / "adapter.pt", map_location="cpu", weights_only=True)
    key = (run_name, (d / "adapter.pt").stat().st_mtime)
    if _chat.get("key") != key:
        _chat.clear()
        c = {**FT_DEFAULTS, **ad["config"], "lora_dropout": 0.0}
        bp = HUB / c["base"]
        tok = AutoTokenizer.from_pretrained(bp)
        model = AutoModelForCausalLM.from_pretrained(bp, dtype=torch.bfloat16).to(pick_device())
        hits = inject_lora(model, c)
        for li, t, m in hits:
            w = ad["lora"][f"{li}.{t}"]
            m.A.data.copy_(w["A"]); m.B.data.copy_(w["B"])
        sp = None
        if ad.get("special") is not None:
            sp = SpecialTokens(model, [tok.convert_tokens_to_ids(t) for t in c["special_tokens"]]).to(model.device)
            sp.D.data.copy_(ad["special"])
        model.eval()
        _chat.update(key=key, tok=tok, model=model, hits=hits, sp=sp)
    tok, model, hits, sp = _chat["tok"], _chat["model"], _chat["hits"], _chat["sp"]
    max_new = max(1, min(int(max_new), 400))
    temperature = max(0.0, min(float(temperature), 2.0))
    set_lora(hits, use_lora)
    saved = None
    if sp is not None and not use_lora:  # "基座"模式：同时关掉特殊标记修正
        saved = sp.D.data.clone(); sp.D.data.zero_()
    try:
        return _generate(tok, model, prompt, max_new, temperature)
    finally:  # 无论成功与否都恢复，避免把"微调后"的模型状态弄坏
        set_lora(hits, True)
        if saved is not None:
            sp.D.data.copy_(saved)


def _generate(tok, model, prompt, max_new, temperature):
    ids = tok.encode(render({"instruction": prompt, "input": ""}, tok, with_answer=False))
    x = torch.tensor([ids], device=model.device)
    kw = dict(do_sample=temperature > 0, temperature=temperature or None, top_p=0.9 if temperature > 0 else None)
    g = model.generate(x, max_new_tokens=int(max_new), pad_token_id=tok.pad_token_id, output_scores=True, use_cache=True, return_dict_in_generate=True,
                       eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id], **{k: v for k, v in kw.items() if v is not None})
    new = g.sequences[0, len(ids):].tolist()
    steps = []
    for t, sc in zip(new, g.scores):
        pr = sc[0].float().softmax(-1)
        tp, ti = pr.topk(5)
        steps.append({"token": tok.decode([t]), "p": round(pr[t].item(), 4),
                      "alts": [{"t": tok.decode([i]), "p": round(p, 4)} for p, i in zip(tp.tolist(), ti.tolist())]})
    return {"steps": steps, "text": tok.decode(new, skip_special_tokens=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--preset", default="sft-alpaca")
    ap.add_argument("--name")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = dict(FT_PRESETS.get(a.preset, {}))
    if a.config:
        cfg.update(json.loads(Path(a.config).read_text()))
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        cfg[k] = v
    if a.name:
        cfg["name"] = a.name
    try:
        run(cfg)
    except Exception as e:
        n = cfg.get("name")
        if n and (RUNS / n).exists():
            Recorder(RUNS / n).status(state="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
        raise


if __name__ == "__main__":
    main()
