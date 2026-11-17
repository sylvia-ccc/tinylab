---
license: apache-2.0
language:
- zh
base_model: Qwen/Qwen3-1.7B-Base
library_name: transformers
pipeline_tag: text-generation
tags:
- persona
- chinese
- lora
- on-device
- companion
- npc
- mbti
---

# TinyLab MBTI 人格对话模型（1.7B）

> **草稿：正式版发布前数字会更新。** 当前数字来自单个随机种子、48 条判型样本，尚未做人工盲评和红队测试。

给定一种 MBTI 人格类型，用这种人格的口吻像朋友一样简短聊天的中文小模型。基于 Qwen3-1.7B-Base，用 LoRA 微调，面向离线、端侧运行。训练过程和诊断见实验报告 [05](experiments/05-persona-small-models.md)。

## 用途
- **适合**：成人潮玩 / 桌面陪伴摆件的离线聊天；游戏 NPC 的人格口吻（同一个模型，给不同 NPC 设不同类型）。
- **不适合**：
  - 面向未成年人的产品。
  - 扮演恋人、亲属或其他亲密关系。
  - 心理咨询、医疗、法律、财务等需要可靠事实或专业判断的场景。
  - 作为"人格测试"的依据：模型只模仿教师数据里的写法，不代表任何人格理论。
- 知识问答（如"INTJ 和 INTP 的区别"）不是训练目标，应由应用层处理。

## 怎么用
人格类型写在 **system 消息**里，格式固定（16 种之一，全大写）：

```
你的人格类型是 {TYPE}。用这种人格的方式，像朋友一样和对方聊天。
```

训练和推理必须用同一格式；把人格放到用户消息里会明显变差（见报告 [04](experiments/04-7b-single-variable.md)）。

### transformers
下面的路径是示意，权重发布前请替换为实际仓库名。LoRA 适配器需合并到基座，或用 `peft` 加载。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "<发布后的仓库名>"  # 合并后的完整模型
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16)

messages = [
    {"role": "system", "content": "你的人格类型是 ISTJ。用这种人格的方式，像朋友一样和对方聊天。"},
    {"role": "user", "content": "今天太阳好大，晒得我睁不开眼"},
]
# 基座没有自带聊天模板时，按下面的 ChatML 格式手动拼接：
prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"
ids = tok(prompt, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9,
                     repetition_penalty=1.1, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

### ollama
已实测（试训版，2026-11-16，苹果 M 系列笔记本，GPU 空闲，3 次取中位数）：

| 版本 | 文件大小 | 运行内存 | 生成速度 | 冷启动加载 |
|---|---|---|---|---|
| 1.7B 4 位（Q4_K_M） | 1.1 GB | 1.6 GB | 161 token/秒 | 0.5 秒 |
| 1.7B 8 位（Q8_0） | 1.8 GB | 2.4 GB | 112 token/秒 | 0.5 秒 |
| 0.6B 4 位（Q4_K_M） | 0.4 GB | 0.9 GB | 297 token/秒 | 0.3 秒 |

端侧芯片上的速度会低得多：按公开基准估算，瑞芯微 RK3576 跑 2B 级别约 11 token/秒，手机跑 1.7B 约 18–36 token/秒【未实测】。

```
# Modelfile
FROM ./tinylab-mbti-1.7b.gguf
SYSTEM """你的人格类型是 ENFP。用这种人格的方式，像朋友一样和对方聊天。"""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
```

```
ollama create tinylab-mbti -f Modelfile
ollama run tinylab-mbti "领导临时加需求，明天就要"
```

## 训练数据
- 由 DeepSeek（`deepseek-v4.1-flash@2026-11`）按[规格书](mbti-data-spec.md)生成；其平台条款允许把输出用于训练其他模型。有 504 条的 `generator` 字段未填写，按口头确认记为 DeepSeek。
- 本次训练：1,804 条对话、227 个核心问题，16 型各约 113 条；多轮对话拆开后 2,904 个样本，其中 2,407 条用于训练。
- 回答 40–150 字，口语短句；规格书禁止在回答里出现人格字母和设定描述词，并限制口头禅。
- **数据暂不公开。** 数据中不应含个人信息，但尚未做面向公开发布的最终审查。
- 训练配方：LoRA r=16、alpha=32，学习率 1e-4，400 步（约 1.3 轮），批大小 8，最大长度 384，单个随机种子（42）；额外训练 `<|im_start|>`、`<|im_end|>` 的嵌入，使模型答完会停。

## 评估
评估集为按核心问题整组留出的、训练中没见过的 288 条样本（评估时用其中的 48 条判型、12 条生成）；留出题与训练数据同源于 DeepSeek。

| 指标 | 1.7B 微调（本模型） | 0.6B 微调 | 基座 + 人设卡（对照，第 0 步） |
|---|---|---|---|
| 型贴近度 top-1（随机 6.25%，48 条） | **56.3%**（区间约 42–69%） | 27.1% | 16.7%（1.7B）/ 20.8%（0.6B） |
| 没见过的问题损失 | 4.366 → 3.117 | 4.884 → 3.590 | — |
| 切题度 | 0.712 | 0.557 | — |
| 答完会停（12 条生成） | 12/12 | 12/12 | 留出题 21 道中 15/21 停下 |
| 背诵率（8 字片段） | 1.1% | 0.8% | — |
| 中文文档损失变化（遗忘） | −0.6% | −0.8% | — |
| Python 代码损失变化 | +0.8% | +1.0% | — |

说明：型贴近度衡量模型回答是否贴近教师按型写的风格，**不等于人觉得像**。48 条样本下，±13 个百分点以内看不出差别。

## 局限与安全
- **不面向未成年人。** 模型没有年龄识别能力，产品方需要自行做年龄限制。
- **不扮演恋人或亲属。** 训练数据里有相关拒绝样本，但**行为尚未验证**；产品方应在应用层加规则。
- **会编造事实。** 例如在借钱问题里编出"房贷"这个题目中没有的前提。不要依赖它提供事实、建议或专业意见。
- **措辞有时奇怪或生硬**，对情绪化问题偏务实，不总能先接住情绪（报告 05 有原文例子）。
- **还没完成红队测试**：自伤、轻生、被欺负、性相关、危险行为、诱导依赖、身份披露（是否会承认自己是 AI）等，均未系统评估。不要把它用作危机场景的第一道防线。
- 单个随机种子，判型只有 48 条，没有人工盲评，没有独立测试集（留出题与训练数据同源）。
- 数据由大模型生成，可能带有教师模型的偏见和风格；模型可能输出不当内容。
- 只测试了简体中文。
- 合规：产品方需自行满足所在地对 AI 陪伴 / 拟人化服务的监管要求（如中国《人工智能拟人化互动服务管理暂行办法》）。

## 引用
基座模型：Qwen3-1.7B-Base（Apache-2.0）。训练代码和实验报告：TinyLab 仓库 `docs/experiments/`。
