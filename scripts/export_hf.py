"""把 TinyLab 的微调结果（基座 + runs/<名字>/adapter.pt）合并成标准的 Hugging Face 模型目录。

合并后任何人都能直接用 transformers 加载，不依赖 TinyLab 的代码：
  · LoRA：W ← W + (alpha/r)·B·A
  · 特殊标记修正 D：嵌入矩阵（与输出层共享）对应的行加上 D
合并前后在同一批提示上比较 logits，误差写进 export_check.json。

  python scripts/export_hf.py sft-alpaca out/qwen3-0.6b-alpaca-zh-sft
  python scripts/export_hf.py mbti-1.7b-v1 out/xxx --adapter runs/mbti-1.7b-v1/ckpts/adapter_000690.pt   # 指定存档
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lab.finetune import HUB, SpecialTokens, inject_lora  # noqa: E402

CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--adapter", default=None, help="可选：指定 adapter 文件（默认 runs/<run>/adapter.pt）")
    a = ap.parse_args()
    run, out = a.run, Path(a.out)
    adapter = Path(a.adapter).resolve() if a.adapter else ROOT / "runs" / run / "adapter.pt"
    ad = torch.load(adapter, map_location="cpu", weights_only=True)
    c = ad["config"]
    tok = AutoTokenizer.from_pretrained(HUB / c["base"])
    probe = [tok.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n") for q in
             ["用三句话解释什么是机器学习。", "好友说：今天被领导批评了，好郁闷"]]

    # 参照：TinyLab 训练时的写法（基座 + LoRA 模块 + 特殊标记修正）
    ref = AutoModelForCausalLM.from_pretrained(HUB / c["base"], dtype=torch.float32)
    hits = inject_lora(ref, {**c, "lora_dropout": 0.0})
    for li, t, m in hits:
        w = ad["lora"][f"{li}.{t}"]
        m.A.copy_(w["A"]); m.B.copy_(w["B"])
    if ad.get("special") is not None:
        sp = SpecialTokens(ref, [tok.convert_tokens_to_ids(t) for t in c["special_tokens"]])
        sp.D.copy_(ad["special"])
    ref.eval()
    ref_logits = [ref(torch.tensor([p])).logits[0, -1] for p in probe]
    del ref

    # 合并
    model = AutoModelForCausalLM.from_pretrained(HUB / c["base"], dtype=torch.float32)
    scale = c["lora_alpha"] / c["lora_r"]
    for key, w in ad["lora"].items():
        li, t = key.split(".")
        layer = model.model.layers[int(li)]
        mod = getattr(layer.self_attn, t, None) or getattr(layer.mlp, t)
        mod.weight += scale * (w["B"] @ w["A"])
    if ad.get("special") is not None:
        assert model.config.tie_word_embeddings, "这里假设嵌入与输出层共享"
        ids = torch.tensor([tok.convert_tokens_to_ids(t) for t in c["special_tokens"]])
        model.model.embed_tokens.weight[ids] += ad["special"]
    model.eval()
    diffs = [float((model(torch.tensor([p])).logits[0, -1] - r).abs().max()) for p, r in zip(probe, ref_logits)]

    out.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    tok.chat_template = CHATML
    tok.save_pretrained(out)
    gc = model.generation_config
    gc.eos_token_id = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    gc.do_sample = False
    gc.save_pretrained(out)
    (out / "export_check.json").write_text(json.dumps({
        "run": run, "adapter": str(adapter.relative_to(ROOT)) if adapter.is_relative_to(ROOT) else str(adapter), "base": c["base"], "lora_r": c["lora_r"], "lora_alpha": c["lora_alpha"], "targets": c["targets"],
        "special_tokens": c.get("special_tokens") if ad.get("special") is not None else [],
        "max_abs_logit_diff_fp32": diffs}, ensure_ascii=False, indent=1))
    print("合并前后最大 logit 误差（fp32）：", diffs)


if __name__ == "__main__":
    main()
