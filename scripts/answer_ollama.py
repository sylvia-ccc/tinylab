"""用本机 ollama 里的通用模型（如 qwen3:8b）+ 人设卡作答，追加到 answer_set.py 的输出里，作为"通用模型 + 提示词"对照臂。

  python scripts/answer_ollama.py results/x/answers.jsonl --model qwen3:8b --name 通用8B加人设卡
人设卡放在 system；关闭 qwen3 的思考模式；温度 0.7、top_p 0.9、重复惩罚 1.1，与 answer_set.py 一致。
"""
import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def chat(model, system, msgs):
    body = {"model": model, "stream": False, "think": False, "keep_alive": "5m",
            "messages": [{"role": "system", "content": system}] + msgs,
            "options": {"temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.1, "num_predict": 300, "seed": 42}}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return r["message"]["content"].strip(), r.get("done_reason") == "stop", r.get("eval_count", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--name", default="通用8B加人设卡")
    a = ap.parse_args()
    p = Path(a.answers)
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        card = (ROOT / "data" / "hub" / "mbti-v6" / "cards" / f"{r['type']}.txt").read_text(encoding="utf-8").strip()
        msgs = r.get("messages") or [{"role": "user", "content": r["question"]}]
        txt, stopped, n = chat(a.model, card, msgs)
        r["answers"][a.name] = txt or "（空回答）"
        r.setdefault("meta", {})[a.name] = {"stopped": stopped, "n_tokens": n}
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    st = [r["meta"][a.name]["stopped"] for r in rows]
    print(f"{a.name}: 正常停下 {sum(st)}/{len(st)}，平均 {sum(r['meta'][a.name]['n_tokens'] for r in rows) / len(rows):.0f} token")
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=json.dumps({"model": a.model, "keep_alive": 0}).encode(), headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=60).read()


if __name__ == "__main__":
    main()
