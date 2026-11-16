"""7B 单变量实验：用 mlx_lm 在 Qwen2.5-7B-Instruct 上训练 LoRA，每 200 步早筛一次，结果写进 TinyLab 看板。

一次只改一个变量（相对原项目 v8 的配方），数据固定为 v8 noGate 训练集。
每个 checkpoint 测（都不依赖 LLM 判官）：
  · 留出回答损失：v8 验证集 48 条回答在各自人设卡下的平均每 token 负对数概率（越低越好；基座 = 第 0 步）
  · 型贴近度：同一条回答放在 16 张卡后面，哪张卡概率最高（随机 6.25%）
  · 生成 12 个没见过的问题（温度 0.7 / top-p 0.9 / 重复惩罚 1.1）：
      切题度（基座打分：配自己的问题 − 配别的问题）、人设卡指令词泄漏、"先别急"口头禅、复读
需要装有 mlx_lm 0.32.x 的环境（苹果芯片），并设好两个环境变量：
  MBTI_V8_RUN=<v8 训练目录> QWEN25_7B_BF16=<bf16 基座目录> python scripts/run_7b_variant.py --variant V1
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RES = ROOT / "results" / "7b"
_LOCAL = json.loads((Path(__file__).resolve().parent.parent / "local_paths.json").read_text()) \
    if (Path(__file__).resolve().parent.parent / "local_paths.json").exists() else {}  # 本机路径，不进 git
V8 = Path(os.environ.get("MBTI_V8_RUN") or _LOCAL.get("MBTI_V8_RUN", "")).expanduser()  # 原项目 v8 训练目录（含 base/、mlx_data/、adapters/）
HF_BF16 = Path(os.environ.get("QWEN25_7B_BF16") or _LOCAL.get("QWEN25_7B_BF16", "")).expanduser()  # Qwen2.5-7B-Instruct 原版 bf16 目录（只有 V4 用）
TYPES = [a + b + c + d for a in "EI" for b in "SN" for c in "TF" for d in "JP"]
SEP = "\n\n请按以上说话风格回复好友。\n"
CARD_WORDS = ("接住", "顶嘴", "和稀泥", "陪伴方式", "口吻")

BASE_RECIPE = dict(num_layers=16, batch_size=2, learning_rate=5e-5, lr_schedule=None, rank=16, scale=2.0,
                   max_seq_length=512, fmt="user", base="q8", adam_bias_correction=False, iters=400, save_every=200, seed=42)

# 每个实验：改了什么、为什么改、预期、怎么读、有什么价值 —— 看板「实验说明」直接展示
VARIANTS = {
    "V0": dict(change={}, title="复现原配方",
               what="完全照搬原项目 v8 的配方（8 位基座、16 层、batch 2、学习率 5e-5 恒定、人设卡放在用户消息里），只训到第 400 步。",
               why="后面每个实验都和它比。先证明这套新流水线能复现旧结果（第 200/400 步退化），否则后面的差异可能只是流水线不同造成的。",
               hypothesis="第 200 步的留出回答损失高于基座（原日志 3.996 对 3.898），第 400 步生成已经答非所问、出现人设卡指令词。",
               value="这是对照组。复现成功，就说明退化是配方造成的，不是偶然；复现失败，就说明原训练当时还有别的问题（比如环境、数据版本）。"),
    "V5": dict(change={"adam_bias_correction": True}, title="打开 Adam 的偏差修正",
               what="只改优化器的一个开关：mlx 的 Adam 默认不做偏差修正（PyTorch 默认是做的），这里把它打开。其它和 V0 完全一样。",
               why="Adam 用梯度的滑动平均来决定每一步走多远。刚开始时滑动平均还没「攒满」，不修正的话，实际步长会被放大：第 1 步约 3.2 倍，第 10–20 步约 6.5 倍，到第 100 步仍有 3.2 倍。原训练日志里损失恰好在第 20 步冲到最高（5.97），冒烟测试里训练 10 步留出损失就从 3.91 跳到 7.70。",
               hypothesis="打开修正后，开头不再冲高；第 200 步留出损失低于基座；第 400 步生成仍然切题、没有人设卡指令词。",
               value="如果成立，就找到了 v6/v8 两代 7B 模型退化的机制性根因：不是数据、不是超参，而是训练框架一个默认值和 PyTorch 不一样。修复只要一行配置，以后所有 mlx 训练都受益。"),
    "V1": dict(change={"lr_schedule": {"name": "cosine_decay", "warmup": 50, "warmup_init": 1e-7, "arguments": [5e-5, 350, 5e-6]}},
               title="加学习率预热 + 余弦衰减",
               what="只改学习率的走法：前 50 步从接近 0 慢慢升到 5e-5，之后按余弦曲线降到 5e-6。其它和 V0 完全一样。",
               why="原训练日志里，前 20 步训练损失从 4.04 冲到 5.97，说明一开始就被大步更新「打坏」了；原配方没有预热，学习率从第 1 步起就是满的。0.6B 实验有预热，就没出现这个现象。",
               hypothesis="开头不再冲高；第 200 步的留出损失就低于基座；第 400 步生成仍然切题，人设卡指令词明显减少。",
               value="如果成立，修复成本几乎为零（改一行配置），而且能解释 v6/v8 两代模型共同的退化。这是「改配置前先看训练曲线」的典型案例。"),
    "V2": dict(change={"num_layers": 28}, title="LoRA 插满全部 28 层",
               what="只改插 LoRA 的层数：从最后 16 层改成全部 28 层。可训练参数从约 2300 万增加到约 4000 万。",
               why="0.6B 实验是全部层都插的。只改后 16 层时，前 12 层完全冻结，「说话风格」可能只能靠后面几层硬扭，更容易扭坏。",
               hypothesis="同样步数下留出损失降得更快，生成更自然；但如果问题主要出在学习率，这一项的帮助有限。",
               value="回答「改动应该放在哪」。和 0.6B 的「④ 改了哪里」对照，能看出风格学习主要发生在哪些层。"),
    "V3": dict(change={"fmt": "system"}, title="人设卡移到 system 消息",
               what="只改提示格式：人设卡放进 system 消息，用户消息只保留「好友说：……」。训练和评估都用这个格式；它的第 0 步基座也用这个格式测。",
               why="7B 退化时会把卡里的指令词（「接住」「顶嘴」）当成自己的话说出来。卡放在用户消息里，模型可能把它当成「对方说的话」去模仿；system 才是 Instruct 模型理解「设定」的位置。",
               hypothesis="人设卡指令词泄漏明显减少；型贴近度不下降。",
               value="提示工程和微调的交界：同一份数据，只换格式就可能决定成败。也是产品上线时人设卡该放在哪的直接证据。"),
    "V4": dict(change={"base": "bf16"}, title="用 bf16 原版基座训练（不用 8 位量化）",
               what="只改基座精度：用原版 bf16 权重（约 15GB）训练，而不是 8 位量化版。内存占用更高，所以排在最后。",
               why="在量化后的基座上训 LoRA，梯度是对「被量化过的权重」算的，有可能带来误差。原项目只因为省内存才选 8 位，没有验证过影响。",
               hypothesis="如果量化是主因，bf16 版第 400 步明显健康；如果不是，结果和 V0 接近。",
               value="决定以后能不能放心在 24GB 的机器上用 8 位基座训练。也是「省资源的选择有没有代价」的直接证据。"),
}


def recipe_of(v):
    return {**BASE_RECIPE, **VARIANTS[v]["change"]}


def msgs(fmt, card, question, answer=None):
    m = ([{"role": "system", "content": card}, {"role": "user", "content": question}] if fmt == "system"
         else [{"role": "user", "content": card + SEP + question}])
    return m + ([{"role": "assistant", "content": answer}] if answer is not None else [])


def prompt_ids(tok, m):
    return tok.encode(tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False), add_special_tokens=False)


def answer_lp(model, tok, prompts, answers, bs=4):
    """prompts: 消息列表；返回回答部分平均每 token 对数概率。"""
    out = []
    for i in range(0, len(prompts), bs):
        P = [prompt_ids(tok, m) for m in prompts[i:i + bs]]
        A = [tok.encode(a + "<|im_end|>", add_special_tokens=False) for a in answers[i:i + bs]]
        L = max(len(p) + len(a) for p, a in zip(P, A))
        ids = np.full((len(P), L), tok.pad_token_id or 0, dtype=np.int32)
        for j, (p, a) in enumerate(zip(P, A)):
            ids[j, :len(p) + len(a)] = p + a
        lp = model(mx.array(ids))[:, :-1].astype(mx.float32)
        lp = np.array(lp - mx.logsumexp(lp, axis=-1, keepdims=True))
        for j, (p, a) in enumerate(zip(P, A)):
            pos = np.arange(len(p) - 1, len(p) + len(a) - 1)
            out.append(float(lp[j, pos, np.array(a)].mean()))
        mx.clear_cache()
    return out


def evaluate(model, tok, fmt, ex, cards):
    t0 = time.time()
    rows = ex["disc"]
    # 1) 留出回答损失：每条回答在它自己的卡下
    nll = -float(np.mean(answer_lp(model, tok, [msgs(fmt, cards[r["type"]], r["question"]) for r in rows],
                                   [r["output"] for r in rows])))
    # 2) 型贴近度
    P, A = [], []
    for r in rows:
        for T in TYPES:
            P.append(msgs(fmt, cards[T], r["question"])); A.append(r["output"])
    lp = np.array(answer_lp(model, tok, P, A)).reshape(len(rows), 16)
    pred = [TYPES[int(i)] for i in lp.argmax(1)]
    truth = [r["type"] for r in rows]
    disc = {"top1": round(float(np.mean([p == t for p, t in zip(pred, truth)])), 4),
            "axes": [round(float(np.mean([p[k] == t[k] for p, t in zip(pred, truth)])), 4) for k in range(4)],
            "pred_counts": {T: pred.count(T) for T in TYPES}}
    # 3) 生成
    mx.random.seed(42)
    sampler = make_sampler(temp=0.7, top_p=0.9)
    procs = make_logits_processors(repetition_penalty=1.1)
    gens = []
    for r in ex["gen"]:
        p = tok.apply_chat_template(msgs(fmt, cards[r["gen_type"]], r["question"]), add_generation_prompt=True, tokenize=False)
        txt = generate(model, tok, prompt=p, max_tokens=160, sampler=sampler, logits_processors=procs, verbose=False)
        gens.append({"kind": r["gen_type"], "body": r["question"], "prompt": r["question"], "output": txt,
                     "card_words": [w for w in CARD_WORDS if w in txt], "tic": "先别急" in txt,
                     "repeat": any(txt.count(txt[i:i + 4]) >= 4 for i in range(len(txt) - 3))})
    return {"val_nll": round(nll, 4), "disc": disc, "gens": gens, "eval_secs": round(time.time() - t0, 1)}


def relevance(base_model, tok, gens):
    k = len(gens)
    q = lambda g: [{"role": "user", "content": g["body"]}]
    own = answer_lp(base_model, tok, [q(g) for g in gens], [g["output"] for g in gens])
    oth = answer_lp(base_model, tok, [q(gens[(i + k // 2) % k]) for i in range(k)], [g["output"] for g in gens])
    for g, a, b in zip(gens, own, oth):
        g["rel"] = round(a - b, 4)
    return round(float(np.mean([a - b for a, b in zip(own, oth)])), 4)


def summarize(ev, rel):
    g = ev["gens"]
    return {"val_nll": ev["val_nll"], "val_loss": ev["val_nll"], "disc": ev["disc"], "relevance": rel,
            "card_word_hits": sum(bool(x["card_words"]) for x in g), "tic_hits": sum(x["tic"] for x in g),
            "repeat_hits": sum(x["repeat"] for x in g), "n_gen": len(g)}


def write_status(d, **kw):
    p = d / "status.json"
    s = json.loads(p.read_text()) if p.exists() else {}
    s.update(kw, updated=time.time())
    p.write_text(json.dumps(s, ensure_ascii=False))


def prep_data(d, fmt):
    out = d / "mlx_data"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("train.jsonl", "valid.jsonl"):
        lines = []
        for l in (V8 / "mlx_data" / f).read_text().splitlines():
            m = json.loads(l)["messages"]
            card, q = m[0]["content"].split(SEP, 1)
            lines.append(json.dumps({"messages": msgs(fmt, card, q, m[1]["content"])}, ensure_ascii=False))
        (out / f).write_text("\n".join(lines) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--save-every", type=int)
    ap.add_argument("--name")
    a = ap.parse_args()
    v = a.variant
    rc = recipe_of(v)
    if a.iters:
        rc["iters"] = a.iters
    if a.save_every:
        rc["save_every"] = a.save_every
    name = a.name or f"7b-{v}"
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    ex = json.loads((RES / "export_v8valid.json").read_text())
    cards = ex["cards"]
    base_path = str(HF_BF16 if rc["base"] == "bf16" else V8 / "base")
    info = VARIANTS[v]
    (d / "config.json").write_text(json.dumps({
        "name": name, "kind": "mlx7b", "variant": v, "title": info["title"], "explain": {k: info[k] for k in ("what", "why", "hypothesis", "value")},
        "change": info["change"], "recipe": rc, "base_model": "Qwen2.5-7B-Instruct（" + ("bf16 原版" if rc["base"] == "bf16" else "8 位量化 MLX") + "）",
        "data": "v8 noGate：训练 2,637 条 / 验证 229 条（教师 MiniMax-M3 + 人设卡 v1）",
        "eval_desc": {"disc_n": len(ex["disc"]), "gen_n": len(ex["gen"]), "decode": "温度 0.7 / top-p 0.9 / 重复惩罚 1.1",
                      "ref": "v8 验证集 48 条（每型 3 条），训练中没见过的回答"},
        "max_steps": rc["iters"], "hypothesis": info["hypothesis"], "notes": "", "created": time.time()}, ensure_ascii=False, indent=1))
    write_status(d, state="preparing", phase="准备数据", step=0, max_steps=rc["iters"], started=time.time(), pid=os.getpid())
    import threading

    def heartbeat():  # 评估一次要好几分钟：定时刷新状态，看板才不会误判为"中断"
        while True:
            time.sleep(30)
            try:
                s = json.loads((d / "status.json").read_text())
                if s.get("state") in ("done", "failed"):
                    return
                write_status(d)
            except Exception:
                pass
    threading.Thread(target=heartbeat, daemon=True).start()
    data = prep_data(d, rc["fmt"])
    for f in ("eval.jsonl", "metrics.jsonl"):
        (d / f).write_text("")
    (d / "snapshots").mkdir(exist_ok=True)

    # 第 0 步：基座（同一提示格式），按格式缓存
    base_cache = RES / f"base_eval_{rc['fmt']}_{rc['base']}.json"
    if base_cache.exists():
        b = json.loads(base_cache.read_text())
    else:
        write_status(d, state="running", phase="评估基座（第 0 步）")
        model, tok = load(base_path)
        ev = evaluate(model, tok, rc["fmt"], ex, cards)
        rel = relevance(model, tok, ev["gens"])
        b = {"summary": summarize(ev, rel), "gens": ev["gens"]}
        base_cache.write_text(json.dumps(b, ensure_ascii=False))
        del model; mx.clear_cache()
    with open(d / "eval.jsonl", "a") as f:
        f.write(json.dumps({"step": 0, **b["summary"]}, ensure_ascii=False) + "\n")
    (d / "snapshots" / "0000000.json").write_text(json.dumps({"step": 0, "probes": b["gens"]}, ensure_ascii=False))

    # 训练（mlx_lm 官方 LoRA 训练器）
    ad = d / "adapters"
    cfg = {"model": base_path, "train": True, "data": str(data), "fine_tune_type": "lora", "num_layers": rc["num_layers"],
           "batch_size": rc["batch_size"], "iters": rc["iters"], "learning_rate": rc["learning_rate"],
           "lora_parameters": {"rank": rc["rank"], "scale": rc["scale"], "dropout": 0.0}, "mask_prompt": True,
           "max_seq_length": rc["max_seq_length"], "grad_checkpoint": True, "save_every": rc["save_every"],
           "steps_per_eval": 100, "steps_per_report": 10, "val_batches": 25, "seed": rc["seed"], "adapter_path": str(ad)}
    if rc["lr_schedule"]:
        cfg["lr_schedule"] = rc["lr_schedule"]
    cfg["optimizer"] = "adam"
    cfg["optimizer_config"] = {"adam": {"bias_correction": bool(rc["adam_bias_correction"])}}
    import yaml
    (d / "lora_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    write_status(d, state="running", phase="训练")
    t0 = time.time()
    log = open(d / "train_output.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "mlx_lm", "lora", "-c", str(d / "lora_config.yaml")],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    for line in proc.stdout:
        log.write(line); log.flush()
        s = ansi.sub("", line)
        m = re.match(r"\s*(\d+)\s+val\s+([\d.]+)", s)
        if m:
            with open(d / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": int(m.group(1)), "val_loss_mlx": float(m.group(2)), "elapsed": round(time.time() - t0, 1)}) + "\n")
            continue
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s+[▲▼]?\s*(\d+)\s+([\d.]+k?)", s)
        if m:
            step = int(m.group(1))
            with open(d / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "loss": float(m.group(2)), "tok_per_s": int(m.group(3)), "elapsed": round(time.time() - t0, 1)}) + "\n")
            write_status(d, step=step, elapsed=round(time.time() - t0, 1))
        if "Insufficient Memory" in s or "Traceback" in s:
            write_status(d, phase="训练出错，见 train_output.log")
    rc_code = proc.wait()
    if rc_code != 0:
        write_status(d, state="failed", error=f"mlx_lm lora 退出码 {rc_code}（见 train_output.log）")
        return
    train_secs = round(time.time() - t0, 1)

    # 早筛：每个 checkpoint 评估
    steps = list(range(rc["save_every"], rc["iters"] + 1, rc["save_every"]))
    all_gens = {}
    for st in steps:
        write_status(d, phase=f"评估第 {st} 步")
        cd = d / f"ckpt{st}"
        cd.mkdir(exist_ok=True)
        (cd / "adapter_config.json").write_text((ad / "adapter_config.json").read_text())
        src = ad / f"{st:07d}_adapters.safetensors"
        if not src.exists():
            src = ad / "adapters.safetensors"
        dst = cd / "adapters.safetensors"
        if not dst.exists():
            dst.symlink_to(src)
        model, tok = load(base_path, adapter_path=str(cd))
        all_gens[st] = evaluate(model, tok, rc["fmt"], ex, cards)
        del model; mx.clear_cache()
    write_status(d, phase="切题度打分（用基座）")
    base_model, tok = load(base_path)
    for st in steps:
        ev = all_gens[st]
        rel = relevance(base_model, tok, ev["gens"])
        with open(d / "eval.jsonl", "a") as f:
            f.write(json.dumps({"step": st, **summarize(ev, rel)}, ensure_ascii=False) + "\n")
        (d / "snapshots" / f"{st:07d}.json").write_text(json.dumps({"step": st, "probes": ev["gens"]}, ensure_ascii=False))
    write_status(d, state="done", phase="完成", step=rc["iters"], train_secs=train_secs, elapsed=round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
