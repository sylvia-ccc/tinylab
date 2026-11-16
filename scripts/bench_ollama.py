#!/usr/bin/env python3
"""ollama 测速：对若干 ollama 模型各跑 N 次 /api/generate(stream=false)，取中位数。

用法：
  python scripts/bench_ollama.py -o out/xxx/gguf/benchmark.json \
      q4=tinylab-alpaca-0.6b-q4@out/xxx/gguf/xxx-Q4_K_M.gguf  q8=tinylab-alpaca-0.6b-q8@...

每个参数格式：标签=ollama模型名@gguf路径。一次只加载一个模型，测完 keep_alive=0 卸载。
"""
import argparse, json, os, statistics, subprocess, time, urllib.request

HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
QUESTIONS = ["用三句话解释什么是机器学习。", "中国的首都是哪里？", "写一首关于秋天的四行短诗。"]
BENCH_PROMPT = "请用中文详细介绍一下人工智能的发展历史，至少写三百字。"


def post(path, body, timeout=300):
    req = urllib.request.Request(HOST + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def gen(model, prompt, keep_alive="5m", num_predict=256):
    return post("/api/generate", {"model": model, "prompt": prompt, "stream": False, "keep_alive": keep_alive,
                                  "options": {"num_predict": num_predict, "temperature": 0.7}})


def ps_mem(model):
    out = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    for line in out.splitlines()[1:]:
        if line.startswith(model):
            return line
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    res = {"note": a.note, "prompt": BENCH_PROMPT, "runs": a.runs, "models": {}}
    for it in a.items:
        label, rest = it.split("=", 1)
        model, _, gguf = rest.partition("@")
        entry = {"ollama_model": model, "gguf_size_mb": round(os.path.getsize(gguf) / 1e6, 1) if gguf else None}
        # 功能性问答：确认能回答且会停（done_reason == stop）
        qa = []
        for q in QUESTIONS:
            r = gen(model, q, num_predict=512)
            qa.append({"q": q, "a": r["response"].strip(), "done_reason": r.get("done_reason"),
                       "eval_count": r["eval_count"]})
        entry["qa"] = qa
        # 测速：先预热（已加载），再跑 N 次
        runs = []
        for _ in range(a.runs):
            r = gen(model, BENCH_PROMPT, num_predict=256)
            runs.append({"tok_s": r["eval_count"] / (r["eval_duration"] / 1e9), "eval_count": r["eval_count"],
                         "prompt_eval_count": r.get("prompt_eval_count"),
                         "prompt_eval_ms": r.get("prompt_eval_duration", 0) / 1e6,
                         "load_ms": r.get("load_duration", 0) / 1e6})
        entry["runs"] = runs
        entry["median_tok_s"] = round(statistics.median(x["tok_s"] for x in runs), 1)
        entry["median_first_token_ms"] = round(statistics.median(x["prompt_eval_ms"] for x in runs), 1)
        entry["ollama_ps"] = ps_mem(model)
        # 冷启动加载时间
        gen(model, "hi", keep_alive=0, num_predict=1); time.sleep(2)
        r = gen(model, "hi", keep_alive=0, num_predict=1)
        entry["cold_load_ms"] = round(r.get("load_duration", 0) / 1e6, 1)
        res["models"][label] = entry
        time.sleep(2)
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: {x: v[x] for x in ("gguf_size_mb", "median_tok_s", "median_first_token_ms", "ollama_ps", "cold_load_ms")} for k, v in res["models"].items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
