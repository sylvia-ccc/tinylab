#!/bin/sh
# v4：基座换成 Qwen3-1.7B 对话版；v5：Qwen3-4B-Base。其余与 v2 相同（同一份数据、同一超参）。单变量：起点模型。
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
# 等下载时先做：v1 在温度 0.3 下的红队（和 v4/v5 同口径）
[ -f results/redteam_v1_t03/report.md ] || { $PY scripts/answer_set.py data/mbti-redteam/redteam_v1.jsonl --answerer 正式版v1=adapter:runs/mbti-1.7b-v1/ckpts/adapter_000690.pt --out results/redteam_v1_t03/answers.jsonl --device mps --temperature 0.3 && $PY scripts/eval_redteam.py results/redteam_v1_t03/answers.jsonl --out results/redteam_v1_t03/report.md; }
until [ -f data/hub/Qwen3-1.7B/model-00001-of-00002.safetensors ] && [ -f data/hub/Qwen3-1.7B/model-00002-of-00002.safetensors ] && [ -z "$(find data/hub/Qwen3-1.7B/.cache -name '*.incomplete' 2>/dev/null)" ]; do sleep 30; done
COMMON="dataset=mbti-new dataset_file=data/mbti-new max_len=384 batch_size=8 grad_accum=1 lr=0.0001 warmup=60 max_steps=940 eval_every=235 disc_per_type=4 n_gen=16 save_every_eval=true"
if [ ! -f results/final_q200/answers_v4.jsonl ]; then
echo "== 训练 v4 $(date)"
$PY -m lab.finetune --preset mbti-new-1.7b --name mbti-1.7b-v4-inst --set base=Qwen3-1.7B base_repo=Qwen/Qwen3-1.7B $COMMON
echo "== v4 答题 $(date)"
$PY scripts/answer_set.py data/mbti-redteam/redteam_v1.jsonl --answerer 正式版v4=adapter:mbti-1.7b-v4-inst --out results/redteam_v4/answers.jsonl --device mps --temperature 0.3
$PY scripts/eval_redteam.py results/redteam_v4/answers.jsonl --out results/redteam_v4/report.md
$PY scripts/answer_set.py data/mbti-test/test_q200.jsonl --answerer 正式版v4=adapter:mbti-1.7b-v4-inst --answerer 对话版1.7B加人设卡=card:data/hub/Qwen3-1.7B --out results/final_q200/answers_v4.jsonl --device mps --temperature 0.3
fi
# 4B 下载完再训
while pgrep -f "download.py --extra" >/dev/null; do sleep 30; done
echo "== 训练 v5 $(date)"
rm -rf runs/mbti-4b-v5
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 $PY -m lab.finetune --preset mbti-new-1.7b --name mbti-4b-v5 --set base=Qwen3-4B-Base base_repo=Qwen/Qwen3-4B-Base $COMMON batch_size=2 grad_accum=4
echo "== v5 答题 $(date)"
$PY scripts/answer_set.py data/mbti-redteam/redteam_v1.jsonl --answerer 正式版v5=adapter:mbti-4b-v5 --out results/redteam_v5/answers.jsonl --device mps --temperature 0.3
$PY scripts/eval_redteam.py results/redteam_v5/answers.jsonl --out results/redteam_v5/report.md
$PY scripts/answer_set.py data/mbti-test/test_q200.jsonl --answerer 正式版v5=adapter:mbti-4b-v5 --out results/final_q200/answers_v5.jsonl --device mps --temperature 0.3
echo "== 全部完成 $(date)"
