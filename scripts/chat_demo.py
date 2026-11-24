#!/usr/bin/env python3
"""离线对话演示：在命令行里和本机 ollama 里的 MBTI 小模型聊天（产品原型）。

每轮用户输入先过应用层安全网关 lab/safety_gate.py：
  - crisis + high：不调用模型，直接输出固定安全话术，并用一行灰字标注"[安全拦截：原因]"
  - identity（问是不是真人）：把 system_hint 拼进 system 再调用模型
  - 其他：正常调用模型

用法：
  python scripts/chat_demo.py                                  # 交互；默认 tinylab-mbti-4b-v5-q4 + INTJ
  python scripts/chat_demo.py --type ENFP --hotline "心理援助热线 xxx-xxxx"
  python scripts/chat_demo.py --script cases.txt               # 无人值守：每行一句用户消息，空行 = /reset
聊天中的命令：/type ENFP 切换人格（同时清空上下文）· /reset 清空上下文 · /quit 退出
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lab.safety_gate import SafetyGate  # noqa: E402

TYPES = [a + b + c + d for a in "EI" for b in "NS" for c in "TF" for d in "JP"]
MAX_TURNS = 10                       # 最多保留最近 10 轮（一问一答 = 1 轮）
OPTIONS = {"temperature": 0.3, "top_p": 0.9, "repeat_penalty": 1.1}
GRAY, RESET = "\033[90m", "\033[0m"


def system_prompt(typ: str) -> str:
    return f"你的人格类型是 {typ}。用这种人格的方式，像朋友一样和对方聊天。"   # 与训练时的格式一致


def stream_chat(host: str, model: str, messages: list[dict], out=sys.stdout) -> str:
    """调用 ollama /api/chat（stream=True），逐字输出并返回完整回复。"""
    body = json.dumps({"model": model, "messages": messages, "stream": True, "options": OPTIONS}).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    full = []
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if not line.strip():
                continue
            j = json.loads(line)
            if j.get("error"):
                raise RuntimeError(j["error"])
            piece = (j.get("message") or {}).get("content", "")
            if piece:
                full.append(piece)
                out.write(piece)
                out.flush()
            if j.get("done"):
                break
    return "".join(full)


def stream_prefill(host: str, model: str, messages: list[dict], prefill: str, out=sys.stdout) -> str:
    """身份问题：用 raw 模式拼 ChatML，让回答强制以 prefill 开头，再由模型接着说。"""
    prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    prompt += "<|im_start|>assistant\n" + prefill
    out.write(prefill); out.flush()
    body = json.dumps({"model": model, "prompt": prompt, "raw": True, "stream": True, "options": OPTIONS}).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body, headers={"Content-Type": "application/json"})
    full = [prefill]
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if not line.strip():
                continue
            j = json.loads(line)
            if j.get("error"):
                raise RuntimeError(j["error"])
            piece = j.get("response", "")
            if piece:
                full.append(piece); out.write(piece); out.flush()
            if j.get("done"):
                break
    return "".join(full)


class Chat:
    def __init__(self, a):
        self.a, self.typ = a, a.type
        self.gate = SafetyGate(hotline_text=a.hotline)
        self.history: list[dict] = []

    def reset(self):
        self.history = []

    def say(self, text: str) -> None:
        """处理一条用户输入（含命令）。返回前已把回复打印完。"""
        t = text.strip()
        if t == "/reset":
            self.reset(); print(f"{GRAY}[已清空上下文]{RESET}"); return
        if t.startswith("/type"):
            new = t[5:].strip().upper()
            if new not in TYPES:
                print(f"{GRAY}[无效类型 {new!r}，可选：{' '.join(TYPES)}]{RESET}"); return
            self.typ = new; self.reset(); print(f"{GRAY}[已切换为 {new}，上下文已清空]{RESET}"); return
        if not t:
            return
        msgs = self.history + [{"role": "user", "content": t}]
        g = self.gate.check(msgs)
        if g["kind"] == "crisis" and g["level"] == "high":
            print(f"{GRAY}[安全拦截：{g['reason']}]{RESET}")
            print("AI> " + g["reply"])
            reply = g["reply"]
        else:
            system = system_prompt(self.typ)
            if g["kind"] == "identity" and g["system_hint"]:
                system += "\n" + g["system_hint"]
                print(f"{GRAY}[身份提示已加入 system：{g['reason']}]{RESET}")
            print("AI> ", end="", flush=True)
            full = [{"role": "system", "content": system}] + msgs
            if g["kind"] == "identity" and g.get("assistant_prefill"):
                reply = stream_prefill(self.a.host, self.a.model, full, g["assistant_prefill"])
            else:
                reply = stream_chat(self.a.host, self.a.model, full)
            print()
        self.history = (msgs + [{"role": "assistant", "content": reply}])[-2 * MAX_TURNS:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="tinylab-mbti-4b-v5-q4")
    ap.add_argument("--type", default="INTJ", type=str.upper, choices=TYPES, metavar="TYPE", help="16 型之一，默认 INTJ")
    ap.add_argument("--hotline", default="当地的心理援助热线或急救电话", help="危机话术里的热线文案")
    ap.add_argument("--host", default="http://127.0.0.1:11434", help="ollama 地址")
    ap.add_argument("--script", type=Path, help="从文件读对话自动跑：每行一句，空行 = /reset")
    a = ap.parse_args()
    c = Chat(a)

    print(f"{GRAY}模型 {a.model} · 人格 {c.typ} · /type XXXX 切换 · /reset 清空 · /quit 退出{RESET}")
    if a.script:
        for raw in a.script.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line == "/quit":
                break
            if not line:
                c.reset(); print(f"\n{GRAY}--- /reset ---{RESET}"); continue
            print(f"\n你[{c.typ}]> {line}")
            c.say(line)
        return
    while True:
        try:
            line = input(f"\n你[{c.typ}]> ")
        except (EOFError, KeyboardInterrupt):
            print(); break
        if line.strip() == "/quit":
            break
        c.say(line)


if __name__ == "__main__":
    main()
