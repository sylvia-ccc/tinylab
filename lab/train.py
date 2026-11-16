"""训练循环 + 全量观测日志。

用法：python -m lab.train configs/xxx.json   或   python -m lab.train --preset pycode-tiny --name my-run
输出（runs/<name>/）：
  config.json      本次实验的全部配置、假设、参数分布
  status.json      运行状态（running / done / stopped / failed）
  metrics.jsonl    每 log_every 步：loss、学习率、梯度范数（总 + 分层）、更新幅度比、吞吐
  eval.jsonl       每 eval_every 步：训练集 / 验证集损失（任务型还有准确率）
  snapshots/*.json 关键时刻的"模型体检"：注意力图、逐 token 损失、残差流、神经元死亡率、嵌入分布、生成样本
  ckpt.pt          最新权重
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .data import get_dataset
from .model import GPT

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

DEFAULTS = dict(
    dataset="pycode", tokenizer="bpe", vocab_size=1024,
    n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.0,
    batch_size=32, max_steps=2000, lr=1e-3, min_lr=1e-4, warmup=100, weight_decay=0.1,
    beta1=0.9, beta2=0.99, grad_clip=1.0, seed=1337,
    log_every=10, eval_every=100, eval_batches=60, n_snapshots=16,
    hypothesis="", notes="",
)

PRESETS = {
    "pycode-tiny": dict(dataset="pycode", n_layer=4, n_head=4, n_embd=128, max_steps=2000,
                        hypothesis="基线：0.9M 参数的模型在 2000 步内能学会 Python 的缩进、关键字和括号配对。"),
    "pycode-small": dict(dataset="pycode", n_layer=6, n_head=6, n_embd=192, block_size=256, max_steps=4000,
                         hypothesis="模型变大 3 倍后，验证集损失应明显低于 tiny 基线。"),
    "pycode-char": dict(dataset="pycode", tokenizer="char", max_steps=2000,
                        hypothesis="字符级分词：每个 token 信息更少，同样步数下每字节的损失会更差。"),
    "pycode-lr-high": dict(dataset="pycode", lr=1e-2, min_lr=1e-3, max_steps=2000,
                           hypothesis="学习率提高 10 倍：梯度范数会出现尖峰，训练可能不稳定。"),
    "large-small": dict(dataset="pycode-large", n_layer=6, n_head=6, n_embd=192, block_size=256, max_steps=4000,
                        hypothesis="同一个模型换成 10 倍大的语料：训练和验证 loss 应保持贴近，不再过拟合（对照 pycode-small）。"),
    "grokking": dict(dataset="modadd", p=97, train_frac=0.3, n_layer=1, n_head=4, n_embd=128, block_size=4,
                     batch_size=512, max_steps=6000, lr=1e-3, min_lr=1e-3, warmup=10, weight_decay=1.0,
                     beta2=0.98, log_every=20, eval_every=100, n_snapshots=20,
                     hypothesis="强权重衰减下，模型先背下训练集（训练准确率 100%），很久以后验证准确率才突然跃升（顿悟）。"),
    "grokking-nowd": dict(dataset="modadd", p=97, train_frac=0.3, n_layer=1, n_head=4, n_embd=128, block_size=4,
                          batch_size=512, max_steps=6000, lr=1e-3, min_lr=1e-3, warmup=10, weight_decay=0.0,
                          beta2=0.98, log_every=20, eval_every=100, n_snapshots=20,
                          hypothesis="对照组：去掉权重衰减后，模型只会死记硬背，验证准确率长期停在低位。"),
}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step, c):
    if step < c["warmup"]:
        return c["lr"] * (step + 1) / c["warmup"]
    t = (step - c["warmup"]) / max(1, c["max_steps"] - c["warmup"])
    return c["min_lr"] + 0.5 * (c["lr"] - c["min_lr"]) * (1 + math.cos(math.pi * min(1.0, t)))


def layer_of(name):
    if name.startswith("blocks."):
        return "block " + name.split(".")[1]
    if name.startswith(("tok_emb", "pos_emb")):
        return "embedding"
    return "final LN"


def snapshot_steps(max_steps, n):
    """前期变化快，所以快照按对数间隔取，再补均匀间隔。"""
    geo = np.unique(np.geomspace(1, max_steps, n // 2).astype(int))
    lin = np.linspace(0, max_steps, n // 2 + 1).astype(int)
    return sorted(set([0, max_steps] + geo.tolist() + lin.tolist()))


class Recorder:
    def __init__(self, run_dir):
        self.d = run_dir
        (self.d / "snapshots").mkdir(parents=True, exist_ok=True)
        self.files = {k: open(self.d / f"{k}.jsonl", "a") for k in ("metrics", "eval")}

    def log(self, kind, row):
        self.files[kind].write(json.dumps(row) + "\n")
        self.files[kind].flush()

    def status(self, **kw):
        p = self.d / "status.json"
        s = json.loads(p.read_text()) if p.exists() else {}
        s.update(kw, updated=time.time())
        p.write_text(json.dumps(s))


@torch.no_grad()
def evaluate(model, ds, c, device, gen):
    model.eval()
    out = {}
    gen = torch.Generator().manual_seed(12345)  # 每次评估用同一组 batch：曲线上的变化只来自模型，不来自抽样
    for split in ("train", "val"):
        if ds.task == "modadd":
            seq = ds.train_seq if split == "train" else ds.val_seq
            seq = seq.to(device)
            logits, loss = model(seq[:, :-1], torch.cat([torch.full_like(seq[:, 1:-1], -1), seq[:, -1:]], 1))
            out[f"{split}_loss"] = loss.item()
            out[f"{split}_acc"] = (logits[:, -1].argmax(-1) == seq[:, -1]).float().mean().item()
        else:
            losses = []
            for _ in range(c["eval_batches"]):
                x, y = ds.get_batch(split, c["batch_size"], c["block_size"], gen)
                losses.append(model(x.to(device), y.to(device))[1].item())
            out[f"{split}_loss"] = float(np.mean(losses))
    model.train()
    return out


@torch.no_grad()
def take_snapshot(model, ds, c, device, step):
    """一次"模型体检"：用固定探针文本跑一遍前向，把内部状态都存下来。"""
    model.eval()
    probe = ds.probe_text()[: c["block_size"]]
    x = torch.tensor([probe], device=device)
    logits, _, obs = model(x, observe=True)
    T = x.size(1)
    # 逐 token 损失：模型在探针文本每个位置上，对"真实下一个 token"有多意外
    lp = logits[0, :-1].log_softmax(-1)
    tgt = x[0, 1:]
    token_loss = (-lp.gather(1, tgt[:, None])[:, 0]).tolist()
    top1 = lp.argmax(-1).tolist()
    snap = {
        "step": step,
        "probe_tokens": [ds.tok.token_str(i) for i in probe],
        "token_loss": [round(v, 3) for v in token_loss],
        "pred_tokens": [ds.tok.token_str(i) for i in top1],
        "attn": [], "attn_entropy": [], "mlp_dead_frac": [], "mlp_act_hist": [],
        "resid_norm": [round(r[0].norm(dim=-1).mean().item(), 3) for r in obs["resid"]],
    }
    for o in obs["layers"]:
        a = o["attn"][0].float().cpu()  # [H,T,T]
        snap["attn"].append(np.round(a.numpy(), 3).tolist())
        ent = -(a.clamp_min(1e-9).log() * a).sum(-1).mean(-1)  # 每个头的平均注意力熵
        snap["attn_entropy"].append([round(v, 3) for v in ent.tolist()])
        m = o["mlp_act"][0].float().cpu()
        # 死亡神经元：在整段探针上最大激活都 < 0.01（GELU 下等于"从不正向激活"）
        snap["mlp_dead_frac"].append(round((m.max(0).values < 0.01).float().mean().item(), 4))
        # 稀疏度：激活值 ≤ 0 的比例，越高说明每个神经元越"挑剔"，只对特定模式响应
        snap.setdefault("mlp_sparsity", []).append(round((m <= 0).float().mean().item(), 4))
        h, edges = np.histogram(m.numpy().ravel(), bins=40, range=(-0.5, 3.0))
        snap["mlp_act_hist"].append(h.tolist())
    snap["mlp_hist_edges"] = [-0.5, 3.0]
    # 词嵌入二维投影（PCA）：看语义相近的 token 是否聚到一起
    W = model.tok_emb.weight.float().cpu()
    ids = list(range(min(W.size(0), 300))) if ds.task == "modadd" else \
        [i for i in range(256, min(W.size(0), 556))] or list(range(min(W.size(0), 300)))
    E = W[ids] - W[ids].mean(0)
    U, S, V = torch.pca_lowrank(E, q=2, center=False)
    xy = (E @ V[:, :2]).numpy()
    snap["emb_pca"] = [{"t": ds.tok.token_str(i), "x": round(float(a), 4), "y": round(float(b), 4)} for i, (a, b) in zip(ids, xy)]
    # 权重分布
    snap["weight_std"] = {n: round(p.float().std().item(), 4) for n, p in model.named_parameters() if p.dim() == 2}
    # 生成样本
    if ds.task == "lm":
        g = torch.Generator(device="cpu").manual_seed(0)
        torch.manual_seed(0)
        start = torch.tensor([probe[:8]], device=device)
        out = model.generate(start, 120, temperature=0.8, top_k=50)
        snap["sample"] = ds.tok.decode(out[0].tolist())
        snap["sample_prompt"] = ds.tok.decode(probe[:8])
    else:
        snap["sample"] = "\n".join(
            f"{ds.decode(s[:-1].tolist())} → 模型答 {ds.token_str(int(p))}（正确 {ds.token_str(int(s[-1]))}）"
            for s, p in zip(ds.val_seq[:8], model(ds.val_seq[:8, :-1].to(device))[0][:, -1].argmax(-1).cpu()))
    model.train()
    return snap


def run(cfg):
    c = {**DEFAULTS, **cfg}
    name = c.get("name") or time.strftime("run-%m%d-%H%M%S")
    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = Recorder(run_dir)
    rec.status(state="preparing", step=0, max_steps=c["max_steps"], pid=os.getpid(), started=time.time())
    torch.manual_seed(c["seed"])
    gen = torch.Generator().manual_seed(c["seed"])
    device = pick_device()

    ds = get_dataset(c)
    model = GPT(ds.vocab_size, c["block_size"], c["n_layer"], c["n_head"], c["n_embd"], c["dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": c["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=c["lr"], betas=(c["beta1"], c["beta2"]))
    names = [n for n, _ in model.named_parameters()]
    groups = sorted(set(map(layer_of, names)), key=lambda g: (g != "embedding", g == "output", g))
    snaps = set(snapshot_steps(c["max_steps"], c["n_snapshots"]))

    (run_dir / "config.json").write_text(json.dumps({
        **c, "name": name, "device": device, "n_params": n_params, "layer_groups": groups,
        "param_breakdown": model.param_breakdown(), "data_report": ds.report, "vocab_size": ds.vocab_size,
        "tokens_per_step": c["batch_size"] * (c["block_size"] if ds.task == "lm" else 4),
        "created": time.time(),
    }, ensure_ascii=False, indent=1))
    rec.status(state="running", n_params=n_params, device=device)

    t0 = time.time()
    t_last, tokens_seen = t0, 0
    for step in range(c["max_steps"] + 1):
        if (run_dir / "STOP").exists():
            rec.status(state="stopped", step=step)
            break
        if step % c["eval_every"] == 0 or step == c["max_steps"]:
            ev = evaluate(model, ds, c, device, gen)
            rec.log("eval", {"step": step, "tokens": tokens_seen, **ev})
        if step in snaps:
            snap = take_snapshot(model, ds, c, device, step)
            (run_dir / "snapshots" / f"{step:07d}.json").write_text(json.dumps(snap, ensure_ascii=False))
        if step == c["max_steps"]:
            break

        lr = lr_at(step, c)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = ds.get_batch("train", c["batch_size"], c["block_size"], gen)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        logging = step % c["log_every"] == 0
        if logging:
            gn = {g: 0.0 for g in groups}
            pn = {g: 0.0 for g in groups}
            for n, p in model.named_parameters():
                if p.grad is not None:
                    gn[layer_of(n)] += p.grad.float().pow(2).sum().item()
                pn[layer_of(n)] += p.float().pow(2).sum().item()
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
        total_gn = torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"]).item()
        opt.step()
        tokens_seen += x.numel()

        if logging:
            upd = {g: 0.0 for g in groups}
            for n, p in model.named_parameters():
                upd[layer_of(n)] += (p.detach() - before[n]).float().pow(2).sum().item()
            now = time.time()
            dt = now - t_last
            t_last = now
            rec.log("metrics", {
                "step": step, "tokens": tokens_seen, "loss": round(loss.item(), 5), "lr": lr,
                "grad_norm": round(total_gn, 5),
                "layer_grad": {g: round(math.sqrt(gn[g]), 6) for g in groups},
                # 更新幅度比 |Δθ|/|θ|：经验上 1e-3 左右健康，过大易发散，过小学不动
                "update_ratio": {g: round(math.sqrt(upd[g]) / (math.sqrt(pn[g]) + 1e-12), 7) for g in groups},
                "param_norm": {g: round(math.sqrt(pn[g]), 4) for g in groups},
                "tok_per_s": round((c["log_every"] * x.numel()) / max(dt, 1e-6)) if step else 0,
                "elapsed": round(now - t0, 2),
            })
            if not math.isfinite(loss.item()):
                rec.status(state="failed", step=step, error="loss 变成 NaN/Inf：训练发散")
                return
            rec.status(step=step, elapsed=round(now - t0, 1))
    else:
        pass
    torch.save({"model": model.state_dict(), "config": c, "vocab_size": ds.vocab_size}, run_dir / "ckpt.pt")
    s = json.loads((run_dir / "status.json").read_text())
    if s.get("state") != "stopped":
        rec.status(state="done", step=c["max_steps"], elapsed=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--preset")
    ap.add_argument("--name")
    ap.add_argument("--set", nargs="*", default=[], help="覆盖配置，如 lr=3e-4 n_layer=6")
    a = ap.parse_args()
    cfg = {}
    if a.preset:
        cfg.update(PRESETS[a.preset])
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
        name = cfg.get("name")
        if name and (RUNS / name).exists():
            Recorder(RUNS / name).status(state="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
        raise


if __name__ == "__main__":
    main()
