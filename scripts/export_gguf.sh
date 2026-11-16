#!/usr/bin/env bash
# =============================================================================
# export_gguf.sh —— HF 模型目录 → f16 GGUF → Q8_0 / Q4_K_M 两个量化版本
#
# 用法：
#   LLAMA_CPP_DIR=/path/to/llama.cpp \
#   [CONVERT_PY=/path/to/convert_hf_to_gguf.py] \
#   [PYTHON=/path/to/python] \
#   scripts/export_gguf.sh <HF模型目录> <输出名>
#
# 环境变量：
#   LLAMA_CPP_DIR  llama.cpp 根目录（必填；需已编译，含 build/bin/llama-quantize）
#   CONVERT_PY     转换脚本路径（可选，默认 $LLAMA_CPP_DIR/convert_hf_to_gguf.py）
#   QUANTIZE_BIN   量化工具路径（可选，默认 $LLAMA_CPP_DIR/build/bin/llama-quantize）
#   PYTHON         运行转换脚本的 Python（可选，默认 python3；需装 gguf/transformers/
#                  sentencepiece/torch，建议用 llama.cpp 要求的环境）
#   OUT_ROOT       输出根目录（可选，默认 <项目根>/out）
#
# 产物：$OUT_ROOT/<输出名>/gguf/{<输出名>-f16,-Q8_0,-Q4_K_M}.gguf
# 示例：
#   LLAMA_CPP_DIR=~/llama.cpp PYTHON=~/llama.cpp/.venv/bin/python \
#     scripts/export_gguf.sh out/qwen3-0.6b-alpaca-zh-sft qwen3-0.6b-alpaca-zh-sft
# =============================================================================
set -euo pipefail

if [ $# -ne 2 ]; then
  sed -n '2,23p' "$0"; exit 1
fi
HF_DIR="$1"; NAME="$2"

: "${LLAMA_CPP_DIR:?请设置环境变量 LLAMA_CPP_DIR（llama.cpp 根目录）}"
CONVERT_PY="${CONVERT_PY:-$LLAMA_CPP_DIR/convert_hf_to_gguf.py}"
QUANTIZE_BIN="${QUANTIZE_BIN:-$LLAMA_CPP_DIR/build/bin/llama-quantize}"
PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$ROOT/out}"

[ -f "$HF_DIR/config.json" ] || { echo "错误：$HF_DIR 不是 HF 模型目录（缺 config.json）" >&2; exit 1; }
[ -f "$CONVERT_PY" ]   || { echo "错误：找不到转换脚本 $CONVERT_PY" >&2; exit 1; }
[ -x "$QUANTIZE_BIN" ] || { echo "错误：找不到量化工具 $QUANTIZE_BIN" >&2; exit 1; }

OUT_DIR="$OUT_ROOT/$NAME/gguf"
mkdir -p "$OUT_DIR"
F16="$OUT_DIR/$NAME-f16.gguf"

echo "[1/3] HF → f16 GGUF"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$CONVERT_PY" "$HF_DIR" --outtype f16 --outfile "$F16"

echo "[2/3] f16 → Q8_0"
"$QUANTIZE_BIN" "$F16" "$OUT_DIR/$NAME-Q8_0.gguf" Q8_0

echo "[3/3] f16 → Q4_K_M"
"$QUANTIZE_BIN" "$F16" "$OUT_DIR/$NAME-Q4_K_M.gguf" Q4_K_M

echo "完成，产物："
ls -lh "$OUT_DIR"/*.gguf
