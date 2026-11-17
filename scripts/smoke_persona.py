#!/usr/bin/env python3
"""MBTI 人格模型 ollama 冒烟：4 种人格 × 2 句话，通过 API 的 system 参数换人格，记录全部原文。

用法：python scripts/smoke_persona.py -o out/xxx/smoke.json  标签=ollama模型名 [...]
一次只加载一个模型，测完 keep_alive=0 卸载。
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_ollama import post  # noqa: E402

TYPES = ["ESTJ", "INFP", "ENTP", "ISFJ"]
QS = ["今天被领导当众批评了，好丢人", "周末不知道干嘛，好无聊"]
SYS = "你的人格类型是 {t}。用这种人格的方式，像朋友一样和对方聊天。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    res = {}
    for it in a.items:
        label, model = it.split("=", 1)
        rows = []
        for t in TYPES:
            for q in QS:
                r = post("/api/generate", {"model": model, "prompt": q, "system": SYS.format(t=t), "stream": False,
                                           "keep_alive": "5m", "options": {"num_predict": 400, "temperature": 0.7, "seed": 42}})
                rows.append({"type": t, "q": q, "a": r["response"].strip(), "done_reason": r.get("done_reason"),
                             "eval_count": r["eval_count"]})
        post("/api/generate", {"model": model, "prompt": "", "keep_alive": 0, "stream": False})
        time.sleep(2)
        res[label] = {"ollama_model": model, "smoke": rows}
    Path(a.output).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for l, v in res.items():
        print(l, "stop率", sum(x["done_reason"] == "stop" for x in v["smoke"]), "/", len(v["smoke"]),
              "平均token", sum(x["eval_count"] for x in v["smoke"]) / len(v["smoke"]))


if __name__ == "__main__":
    main()
