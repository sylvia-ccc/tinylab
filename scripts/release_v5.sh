#!/usr/bin/env bash
# v5（4B）发布导出：合并 → GGUF(Q8/Q4) → ollama 注册
set -euo pipefail
cd "$(dirname "$0")/.."
N=mbti-persona-4b-v5
[ -f out/$N/model.safetensors ] || .venv/bin/python scripts/export_hf.py mbti-4b-v5 out/$N
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:?请设置 llama.cpp 目录} QUANTIZE_BIN=${QUANTIZE_BIN:-$LLAMA_CPP_DIR/build/bin/llama-quantize} PYTHON=${PYTHON:-.venv/bin/python} \
  scripts/export_gguf.sh out/$N $N
G=out/$N/gguf
.venv/bin/python scripts/make_modelfile.py $G/$N-Q4_K_M.gguf -o $G/Modelfile.q4
.venv/bin/python scripts/make_modelfile.py $G/$N-Q8_0.gguf -o $G/Modelfile.q8
ollama create tinylab-mbti-4b-v5-q4 -f $G/Modelfile.q4
ollama create tinylab-mbti-4b-v5-q8 -f $G/Modelfile.q8
rm -f $G/$N-f16.gguf
ls -lh $G; cat out/$N/export_check.json
echo "== 导出完成 $(date)"
