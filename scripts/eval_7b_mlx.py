"""在 7B（Qwen2.5-7B-Instruct q8，mlx）上复测：v8 noGate adapter（1 epoch） vs 基座 + 同一张人设卡。

用 TinyLab 同一套不依赖 LLM 判官的指标、同一批"没见过的问题"：
  · 型可分度：同一条参考回答放在 16 张人设卡后面，哪张卡让模型觉得它最可能
  · 生成：12 个没见过的问题，贪心解码，算背诵率（对 v8 自己的训练集）和切题度（用基座打分）

必须用原 MBTI 项目的虚拟环境运行（里面有 mlx_lm 0.32.2）：
  python scripts/eval_7b_mlx.py <export.json> <run_dir> <out.json>
export.json 由 TinyLab 导出（同一批 disc / gen 样本、16 张人设卡）。
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler

TYPES = [a + b + c + d for a in "EI" for b in "SN" for c in "TF" for d in "JP"]


def v8_prompt(card, question):
    # v8 训练时的用户消息格式：人设卡 + 固定引导语 + 好友的话（mlx_data/train.jsonl 实测）
    return f"{card}\n\n请按以上说话风格回复好友。\n{question}"


def chat_ids(tok, user, answer=None):
    msgs = [{"role": "user", "content": user}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pi = tok.encode(p, add_special_tokens=False)
    if answer is None:
        return pi
    return pi, tok.encode(answer + "<|im_end|>", add_special_tokens=False)


def answer_lp(model, tok, users, answers, bs=8):
    out = []
    for i in range(0, len(users), bs):
        P, A = [], []
        for u, a in zip(users[i:i + bs], answers[i:i + bs]):
            p, an = chat_ids(tok, u, a)
            P.append(p); A.append(an)
        L = max(len(p) + len(a) for p, a in zip(P, A))
        ids = np.full((len(P), L), tok.pad_token_id or 0, dtype=np.int32)
        for j, (p, a) in enumerate(zip(P, A)):
            ids[j, :len(p) + len(a)] = p + a
        logits = model(mx.array(ids))  # 右侧补齐 + 因果注意力：补齐的位置不影响前面的 token
        lp = logits[:, :-1].astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        lp = np.array(lp)
        for j, (p, a) in enumerate(zip(P, A)):
            pos = np.arange(len(p) - 1, len(p) + len(a) - 1)
            out.append(float(lp[j, pos, np.array(a)].mean()))
        mx.clear_cache()
    return out


def disc(model, tok, rows, cards):
    users, answers = [], []
    for r in rows:
        for T in TYPES:
            users.append(v8_prompt(cards[T], r["question"])); answers.append(r["output"])
    lp = np.array(answer_lp(model, tok, users, answers)).reshape(len(rows), 16)
    pred = [TYPES[int(i)] for i in lp.argmax(1)]
    truth = [r["type"] for r in rows]
    return {"top1": round(float(np.mean([p == t for p, t in zip(pred, truth)])), 4),
            "axes": [round(float(np.mean([p[k] == t[k] for p, t in zip(pred, truth)])), 4) for k in range(4)],
            "mean_rank": round(float(np.mean([(lp[i] > lp[i, TYPES.index(t)]).sum() + 1 for i, t in enumerate(truth)])), 3),
            "pred_counts": {T: pred.count(T) for T in TYPES}}


def main():
    ex = json.loads(Path(sys.argv[1]).read_text())
    run = Path(sys.argv[2])
    outp = Path(sys.argv[3])
    cards = ex["cards"]
    # v8 训练集回答的 8 字片段（背诵率的参照）
    grams = set()
    for f in ("train.jsonl",):
        for l in (run / "mlx_data" / f).read_text().splitlines():
            a = json.loads(l)["messages"][-1]["content"]
            grams.update(a[i:i + 8] for i in range(len(a) - 7))
    res = {"rows": len(ex["disc"]), "gen_n": len(ex["gen"]), "arms": {}}
    # 可选：第 4 个参数 "sample" = 温度 0.7 + top-p 0.9 + 重复惩罚 1.1（更接近真实使用）；默认贪心
    sample = len(sys.argv) > 4 and sys.argv[4] == "sample"
    mx.random.seed(42)
    sampler = make_sampler(temp=0.7, top_p=0.9) if sample else make_sampler(temp=0.0)
    procs = make_logits_processors(repetition_penalty=1.1) if sample else None
    arms = [("base_card", None), ("v8_1ep_card", str(run / "adapters"))]
    if len(sys.argv) > 5:  # 额外的中间 checkpoint：名字=目录
        arms += [tuple(a.split("=", 1)) for a in sys.argv[5:]]
    res["decode"] = "温度 0.7 / top-p 0.9 / 重复惩罚 1.1" if sample else "贪心"
    gens = {}
    for arm, adapter in arms:
        t0 = time.time()
        model, tok = load(str(run / "base"), adapter_path=adapter)
        d = disc(model, tok, ex["disc"], cards)
        g = []
        for r in ex["gen"]:
            prompt = tok.apply_chat_template([{"role": "user", "content": v8_prompt(cards[r["gen_type"]], r["question"])}],
                                             add_generation_prompt=True, tokenize=False)
            txt = generate(model, tok, prompt=prompt, max_tokens=160, sampler=sampler, logits_processors=procs, verbose=False)
            n = len(txt) - 7
            memo = sum(txt[i:i + 8] in grams for i in range(n)) / n if n > 0 else 0.0
            rep = any(txt.count(txt[i:i + 4]) >= 4 for i in range(len(txt) - 3))  # 同一个 4 字片段出现 ≥4 次 = 复读
            g.append({"type": r["gen_type"], "question": r["question"], "output": txt, "memo": round(memo, 4),
                      "repeat": rep, "card_words": sum(w in txt for w in ("接住", "顶嘴", "和稀泥", "陪伴方式", "口吻"))})
        gens[arm] = g
        res["arms"][arm] = {"disc": d, "memo_rate": round(float(np.mean([x["memo"] for x in g])), 4),
                            "repeat": sum(x["repeat"] for x in g), "card_word_hits": sum(x["card_words"] > 0 for x in g),
                            "secs": round(time.time() - t0, 1)}
        del model
        mx.clear_cache()
        print(arm, res["arms"][arm], flush=True)
    # 切题度：都用基座（不带 adapter、不带人设卡）打分，配自己的问题 vs 配另一个问题
    model, tok = load(str(run / "base"))
    for arm, g in gens.items():
        k = len(g)
        own = answer_lp(model, tok, [x["question"] for x in g], [x["output"] for x in g])
        oth = answer_lp(model, tok, [g[(i + k // 2) % k]["question"] for i in range(k)], [x["output"] for x in g])
        for x, a, b in zip(g, own, oth):
            x["rel"] = round(a - b, 4)
        res["arms"][arm]["relevance"] = round(float(np.mean([a - b for a, b in zip(own, oth)])), 4)
    res["gens"] = gens
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({a: {k: v for k, v in r.items() if k != "disc"} | {"top1": r["disc"]["top1"], "axes": r["disc"]["axes"]}
                      for a, r in res["arms"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
