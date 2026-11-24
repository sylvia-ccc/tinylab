#!/usr/bin/env python3
"""生成 ollama 的 Modelfile（Qwen3 ChatML 格式，支持可选 system）。

用法：
  python scripts/make_modelfile.py <gguf路径> [-o Modelfile路径] [--system "默认system文本"]

不传 -o 则打印到标准输出。FROM 使用 gguf 的绝对路径。
模板：有 system 时渲染 <|im_start|>system\\n...<|im_end|>\\n；无 system 则只有 user/assistant。
"""
import argparse
import os
import sys

# Go template：ollama 的 .System / .Prompt / .Response
TEMPLATE = (
    '{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}'
    '{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}'
    '<|im_start|>assistant\n{{ .Response }}<|im_end|>\n'
)


def build(gguf: str, system: str | None = None) -> str:
    lines = [f"FROM {os.path.abspath(gguf)}", "", f'TEMPLATE """{TEMPLATE}"""', ""]
    if system:
        if '"""' in system:
            raise ValueError("system 文本不能包含三个连续双引号")
        lines += [f'SYSTEM """{system}"""', ""]
    lines += [
        'PARAMETER stop "<|im_end|>"',
        'PARAMETER stop "<|endoftext|>"',
        "PARAMETER temperature 0.3",  # AI 盲评：0.3 比 0.7 均分 2.60 对 2.18（results/final_q200/ai_review_decode.md）
        "PARAMETER top_p 0.9",
        "PARAMETER repeat_penalty 1.1",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 ollama Modelfile")
    ap.add_argument("gguf", help="GGUF 文件路径")
    ap.add_argument("-o", "--output", help="输出 Modelfile 路径（默认打印到 stdout）")
    ap.add_argument("--system", default=None, help="默认 system 消息（可选）")
    a = ap.parse_args()
    if not os.path.isfile(a.gguf):
        print(f"错误：找不到 {a.gguf}", file=sys.stderr)
        return 1
    text = build(a.gguf, a.system)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"已写入 {a.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
