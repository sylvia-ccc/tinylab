#!/usr/bin/env python3
"""最终考试·答题：让一个或多个"答题者"对题集逐题生成回答，输出 JSONL。

答题者写法（可写多个，名字自取，会出现在输出和盲评里；盲评页面不会显示名字）：
  名字=adapter:<实验名>      TinyLab 微调结果 runs/<实验名>/adapter.pt（读取后在内存里合并到基座，不落盘）
  名字=hf:<目录>             已合并的 HF 模型目录（例如 scripts/export_hf.py 的输出），system = "你的人格类型是 XXXX。……"
  名字=card:<基座目录>       对照臂"基座 + 人设卡"：不训练，人设卡 data/hub/mbti-v6/cards/<类型>.txt 放在 system

题集（格式自动识别，可给多个文件）：
  测试题 data/mbti-test/*.jsonl      {"t_id","scene","messages"}      每题按 t_id 固定分配 1 个人格（16 型均衡、可复现）
  红队题 data/mbti-redteam/*.jsonl   {"r_id","category","messages","expected"}
                                     每题分配 4 个人格（四个维度两端各 2 次），用来查底线在所有人格下是否一致

生成参数固定：温度 0.7、top_p 0.9、重复惩罚 1.1、最多 200 个新 token、固定随机种子；
停止符 <|im_end|> 和 <|endoftext|>，记录每条是否正常停下（没停下 = 撞到 200 token 上限）。
重复惩罚不作用于 <|im_start|> <|im_end|> <|endoftext|>：它们在提示里反复出现，若被惩罚，模型会更难说出"结束"，
对没学过停止的基座对照臂尤其不公平。

输出（每行一条，可直接喂给 scripts/make_blind_review.py）：
  {"id","type","question","answers":{答题者: 回答}, ...附加字段}
  测试题：id = t_id；红队题：id = "<r_id>-<类型>"，另带 r_id / category / expected / messages
  附加字段 meta：{答题者: {"stopped": 是否正常停下, "n_tokens": 生成的 token 数}}
  同名文件 <输出>.cache/<答题者>.jsonl 逐批保存进度，中断后重跑同一条命令会接着做。

用法：
  .venv/bin/python scripts/answer_set.py data/mbti-test/test_v1.jsonl --out results/final/test_answers.jsonl \\
      --answerer 微调=adapter:mbti-new-pilot-0.6b --answerer 基座人设卡=card:data/hub/Qwen3-0.6B-Base
  .venv/bin/python scripts/answer_set.py data/mbti-redteam/redteam_v1.jsonl --out results/final/redteam_answers.jsonl \\
      --answerer 微调=adapter:mbti-new-pilot-0.6b --answerer 对照=hf:out/某合并模型
默认用 cpu，不占 GPU（训练可能正在用）；确认 GPU 空闲后可加 --device mps。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HUB = ROOT / "data" / "hub"
CARDS = HUB / "mbti-v6" / "cards"
TYPES = [a + b + c + d for a in "EI" for b in "SN" for c in "TF" for d in "JP"]
FLIP = {"E": "I", "I": "E", "S": "N", "N": "S", "T": "F", "F": "T", "J": "P", "P": "J"}
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]
SPECIAL_EXEMPT = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]


# ------------------------------------------------------------ 题集与人格分配
def load_questions(paths: list[Path]) -> tuple[str, list[dict]]:
    """返回 (kind, 题目列表)。kind 为 test / redteam；一次只能混一种格式。"""
    rows, kinds = [], set()
    for p in paths:
        for no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{p} 第 {no} 行不是合法 JSON：{e}")
            if "t_id" in r:
                kinds.add("test")
            elif "r_id" in r:
                kinds.add("redteam")
            else:
                sys.exit(f"{p} 第 {no} 行既没有 t_id 也没有 r_id")
            m = r.get("messages")
            if not isinstance(m, list) or not m or m[-1].get("role") != "user":
                sys.exit(f"{p} 第 {no} 行：messages 必须非空且最后一条是 user")
            rows.append(r)
    if len(kinds) != 1:
        sys.exit("题集里要么全是测试题（t_id），要么全是红队题（r_id），不能混用")
    kind = kinds.pop()
    key = "t_id" if kind == "test" else "r_id"
    ids = [r[key] for r in rows]
    if len(set(ids)) != len(ids):
        sys.exit(f"{key} 有重复")
    return kind, sorted(rows, key=lambda r: r[key])


def _h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def assign_test_types(rows: list[dict], seed: int) -> dict[str, str]:
    """按 t_id 的哈希排序后，循环发 16 型：每型数量相差不超过 1，且只由 (seed, 题号集合) 决定，可复现。"""
    order = sorted((r["t_id"] for r in rows), key=lambda t: _h(f"{seed}:{t}"))
    return {t: TYPES[i % 16] for i, t in enumerate(order)}


def assign_redteam_types(r_id: str, seed: int) -> list[str]:
    """4 个人格 = A、A 的完全反面、B、B 的完全反面：每个维度两端各 2 次。"""
    rng = random.Random(f"{seed}:{r_id}")
    a = rng.choice(TYPES)
    a2 = "".join(FLIP[c] for c in a)
    b = rng.choice([t for t in TYPES if t not in (a, a2)])
    b2 = "".join(FLIP[c] for c in b)
    return [a, a2, b, b2]


def question_text(msgs: list[dict]) -> str:
    """单轮：原文；多轮：把前几轮拼成可读文本。"""
    if len(msgs) == 1:
        return msgs[0]["content"].strip()
    who = {"user": "用户", "assistant": "助手", "system": "系统"}
    return "\n".join(f"{who.get(m['role'], m['role'])}：{m['content'].strip()}" for m in msgs)


def build_units(kind: str, rows: list[dict], seed: int) -> list[dict]:
    units = []
    if kind == "test":
        tm = assign_test_types(rows, seed)
        for r in rows:
            units.append({"id": r["t_id"], "type": tm[r["t_id"]], "messages": r["messages"],
                          "extra": {"scene": r.get("scene", "")}})
    else:
        for r in rows:
            for T in assign_redteam_types(r["r_id"], seed):
                units.append({"id": f"{r['r_id']}-{T}", "type": T, "messages": r["messages"],
                              "extra": {"r_id": r["r_id"], "category": r.get("category", ""),
                                        "expected": r.get("expected", "")}})
    return units


# ------------------------------------------------------------ 提示格式（与 lab/mbti.py 保持一致）
def sys_text(T: str) -> str:
    return f"你的人格类型是 {T}。用这种人格的方式，像朋友一样和对方聊天。"


def render_chat(system: str, msgs: list[dict]) -> str:
    s = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    for m in msgs:
        s += f"<|im_start|>{m['role']}\n{m['content'].strip()}<|im_end|>\n"
    return s + "<|im_start|>assistant\n"


# ------------------------------------------------------------ 答题者
def parse_answerer(s: str) -> dict:
    if "=" not in s or ":" not in s.split("=", 1)[1]:
        sys.exit(f"答题者写法不对：{s}\n应为 名字=adapter:<实验名> | 名字=hf:<目录> | 名字=card:<基座目录>")
    name, spec = s.split("=", 1)
    kind, path = spec.split(":", 1)
    if kind not in ("adapter", "hf", "card"):
        sys.exit(f"答题者类型只能是 adapter / hf / card：{s}")
    if not name.strip() or "/" in name:
        sys.exit(f"答题者名字不能为空或含 /：{s}")
    if kind == "adapter":
        p = Path(path)
        if p.suffix != ".pt":  # 实验名
            p = ROOT / "runs" / path / "adapter.pt"
        elif not p.is_absolute():
            p = ROOT / p
    else:
        p = Path(path)
        if not p.is_absolute():
            p = (Path.cwd() / p) if (Path.cwd() / p).exists() else ROOT / p
    if not p.exists():
        sys.exit(f"答题者 {name} 的路径不存在：{p}")
    return {"name": name.strip(), "kind": kind, "path": p}


def load_model(a: dict, device: str, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if a["kind"] == "adapter":
        ad = torch.load(a["path"], map_location="cpu", weights_only=True)
        c = ad["config"]
        base = HUB / c["base"]
        tok = AutoTokenizer.from_pretrained(base)
        big = "4B" in c["base"] or "8B" in c["base"]  # 大模型 fp32 加载会占满 24GB 内存，改为 bf16 加载、fp32 计算增量后写回
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16 if big else torch.float32)
        with torch.no_grad():  # 与 scripts/export_hf.py 相同的合并：W += (alpha/r)·B·A；特殊标记嵌入行 += D
            scale = c["lora_alpha"] / c["lora_r"]
            for key, w in ad["lora"].items():
                li, t = key.split(".")
                layer = model.model.layers[int(li)]
                mod = getattr(layer.self_attn, t, None) or getattr(layer.mlp, t)
                mod.weight.copy_((mod.weight.float() + scale * (w["B"] @ w["A"])).to(mod.weight.dtype))
            if ad.get("special") is not None:
                assert model.config.tie_word_embeddings, "这里假设嵌入与输出层共享"
                ids = torch.tensor([tok.convert_tokens_to_ids(t) for t in c["special_tokens"]])
                model.model.embed_tokens.weight[ids] += ad["special"]
        del ad
    else:
        tok = AutoTokenizer.from_pretrained(a["path"])
        model = AutoModelForCausalLM.from_pretrained(a["path"], dtype=dtype)
    model.to(device=device, dtype=dtype).eval()
    tok.padding_side = "left"
    return model, tok


def make_rep_penalty(penalty: float, exempt_ids: list[int]):
    from transformers import LogitsProcessor
    import torch

    class RepPenalty(LogitsProcessor):
        """同 HF 的 repetition_penalty，但 exempt_ids（特殊标记）不被惩罚。"""
        def __init__(self):
            self.p = penalty
            self.exempt = torch.tensor(exempt_ids)

        def __call__(self, input_ids, scores):
            orig = torch.gather(scores, 1, input_ids)
            pen = torch.where(orig < 0, orig * self.p, orig / self.p)
            pen = torch.where(torch.isin(input_ids, self.exempt.to(input_ids.device)), orig, pen)
            return scores.scatter(1, input_ids, pen)
    return RepPenalty()


def generate_batch(model, tok, prompts: list[str], args, device: str, seed: int) -> list[dict]:
    import torch
    from transformers import LogitsProcessorList
    stop_ids = [tok.convert_tokens_to_ids(t) for t in STOP_TOKENS]
    exempt = [tok.convert_tokens_to_ids(t) for t in SPECIAL_EXEMPT]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=args.temperature > 0, temperature=args.temperature or None, top_p=args.top_p if args.temperature > 0 else None, top_k=0,
                             max_new_tokens=args.max_new, eos_token_id=stop_ids,
                             pad_token_id=stop_ids[1], logits_processor=LogitsProcessorList([make_rep_penalty(1.1, exempt)]))
    res = []
    for row in out[:, enc["input_ids"].shape[1]:].tolist():
        n, stopped = len(row), False
        for i, t in enumerate(row):
            if t in stop_ids:
                n, stopped = i, True
                break
        res.append({"answer": tok.decode(row[:n], skip_special_tokens=True).strip(), "stopped": stopped, "n_tokens": n})
    return res


def run_answerer(a: dict, units: list[dict], args, cache_dir: Path) -> dict[str, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cf = cache_dir / f"{a['name']}.jsonl"
    done: dict[str, dict] = {}
    if cf.exists():
        for l in cf.read_text(encoding="utf-8").splitlines():
            if l.strip():
                d = json.loads(l)
                done[d["id"]] = d
    todo = [u for u in units if u["id"] not in done]
    print(f"[{a['name']}] 共 {len(units)} 条，已有缓存 {len(done)}，待生成 {len(todo)}", flush=True)
    if not todo:
        return done
    import torch
    torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    t0 = time.time()
    model, tok = load_model(a, args.device, dtype)
    print(f"[{a['name']}] 模型已加载（{time.time() - t0:.0f}s，{args.device}）", flush=True)
    # 长度相近的放一批，减少补齐浪费
    def prompt_of(u):
        system = (CARDS / f"{u['type']}.txt").read_text(encoding="utf-8").strip() if a["kind"] == "card" else sys_text(u["type"])
        pr = render_chat(system, u["messages"])
        if a["kind"] == "card" and "Base" not in a["path"].name:  # Qwen3 对话版：空思考块 = 关闭思考模式
            pr += "<think>\n\n</think>\n\n"
        return pr
    todo.sort(key=lambda u: len(prompt_of(u)))
    for bi in range(0, len(todo), args.batch_size):
        batch = todo[bi:bi + args.batch_size]
        t1 = time.time()
        res = generate_batch(model, tok, [prompt_of(u) for u in batch], args, args.device, args.seed + bi)
        with cf.open("a", encoding="utf-8") as f:
            for u, r in zip(batch, res):
                d = {"id": u["id"], **r}
                done[u["id"]] = d
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[{a['name']}] {min(bi + args.batch_size, len(todo))}/{len(todo)}  本批 {time.time() - t1:.0f}s", flush=True)
    del model, tok
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", nargs="+", type=Path, help="题集 JSONL（测试题或红队题，可多个文件）")
    ap.add_argument("--answerer", action="append", required=True, help="答题者，可重复：名字=adapter:/hf:/card:路径")
    ap.add_argument("--out", type=Path, required=True, help="输出 JSONL")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"], help="默认 cpu，不和训练抢 GPU")
    ap.add_argument("--seed", type=int, default=20261116, help="人格分配和生成共用的随机种子")
    ap.add_argument("--temperature", type=float, default=0.7, help="采样温度；0 = 贪心")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=200, help="最多生成的新 token 数")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4, help="cpu 线程数")
    ap.add_argument("--limit", type=int, default=0, help="只做前 N 道题（调试用）")
    args = ap.parse_args()

    answerers = [parse_answerer(s) for s in args.answerer]
    if len({a["name"] for a in answerers}) != len(answerers):
        sys.exit("答题者名字有重复")
    kind, rows = load_questions(args.questions)
    if args.limit:
        rows = rows[:args.limit]
    units = build_units(kind, rows, args.seed)
    print(f"题集类型：{'测试题' if kind == 'test' else '红队题'}；{len(rows)} 道题 -> {len(units)} 条待答", flush=True)

    cache_dir = args.out.with_name(args.out.name + ".cache")
    results = {a["name"]: run_answerer(a, units, args, cache_dir) for a in answerers}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for u in units:
            row = {"id": u["id"], "type": u["type"], "question": question_text(u["messages"]),
                   "answers": {n: results[n][u["id"]]["answer"] or "（空回答）" for n in results},
                   "meta": {n: {"stopped": results[n][u["id"]]["stopped"], "n_tokens": results[n][u["id"]]["n_tokens"]} for n in results},
                   "messages": u["messages"], **u["extra"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for n, d in results.items():
        vs = [d[u["id"]] for u in units]
        print(f"{n}: 正常停下 {sum(v['stopped'] for v in vs)}/{len(vs)}，平均 {sum(v['n_tokens'] for v in vs) / len(vs):.0f} token")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
