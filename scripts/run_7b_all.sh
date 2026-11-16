#!/bin/bash
# 依次跑 7B 单变量实验（每个约 45 分钟）。一个失败不影响下一个。
cd "$(dirname "$0")/.."
PY="${MLX_PY:-python3}"   # 装有 mlx_lm 的 Python
export MBTI_V8_RUN QWEN25_7B_BF16
for v in ${@:-V0 V5 V1 V3 V2 V4}; do
  echo "== $v $(date +%H:%M)"
  "$PY" -I scripts/run_7b_variant.py --variant "$v" > "/tmp/7b-$v.log" 2>&1
  echo "   exit $? $(date +%H:%M) $(cat runs/7b-$v/status.json 2>/dev/null | cut -c1-120)"
done
