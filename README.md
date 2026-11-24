# TinyLab · 看得见的模型训练实验室

> **English summary.** TinyLab is a single-machine lab for *seeing* what happens while a language model trains. It covers (1) training small GPTs from scratch with full observability (per-layer gradients, update ratios, attention maps, residual norms, dead neurons, embedding PCA), (2) real LoRA fine-tuning of open base models (Qwen3) with data lineage, a per-layer change map, forgetting/KL tracking and before/after chat, and (3) a judge-free evaluation toolkit for persona fine-tuning, used to diagnose why a 7B persona model degraded. Everything runs on an Apple-silicon laptop. All experiments, including the failed ones, are documented in [`docs/experiments/`](docs/experiments/).

在一台苹果芯片笔记本上训练语言模型，并且**看见训练过程中的每一件事**：数据从哪来、变成了什么、模型哪里被改了、改完有什么影响。所有实验（包括失败的）都写成了带数据的报告。

## 三部分

| 部分 | 做什么 | 看板上能看到 |
|---|---|---|
| **从零训练 GPT** | 0.9M–2.9M 参数的小 GPT，Python 源码 / 模加法 | 分词、参数分布、损失、分层梯度、更新幅度比、注意力图、逐 token 损失、残差流、激活稀疏度、词嵌入投影、各阶段生成 |
| **微调开源基座**（LoRA SFT） | Qwen3-0.6B-Base + 手写 LoRA（约 40 行） | 数据血缘（出处、SHA-256、每步清洗删了谁）、模型真正看到的 token、28 层 × 7 模块改动地图、遗忘与 KL 偏离、答完会停的比例、基座 vs 微调对话 |
| **人格微调诊断** | 不依赖 LLM 判官的评估：整组留出问题、型贴近度、背诵率、切题度；0.6B 与 7B 对照、7B 单变量实验 | 见过 vs 没见过的问题、判型命中、生成对比、自动解读、单变量总表 |

## 主要发现（每条都有报告和原始数据）

| # | 发现 | 报告 |
|---|---|---|
| 1 | 字符级的 loss（1.29）看起来比 BPE（2.75）好得多，换算成每字节比特后反而更差（1.85 对 1.58）：不同分词器的 loss 不能直接比 | [01](docs/experiments/01-from-scratch.md) |
| 2 | 2.9M 模型在 4MB 语料上第 1800 步开始过拟合；语料扩大 10 倍后不再过拟合，bpb 从 1.46 降到 1.18：瓶颈在数据量 | [01](docs/experiments/01-from-scratch.md) |
| 3 | 模加法上，有权重衰减时训练集满分后约 900 步才泛化（轻度"顿悟"）；没有权重衰减，验证准确率最高 75% 后回落 | [01](docs/experiments/01-from-scratch.md) |
| 4 | Qwen3-0.6B-Base 做 ChatML 指令微调后**学会回答却停不下来**：基座里 `<|im_end|>` 等约 290 个保留标记的嵌入完全相同（从未训练过的占位向量），又和输出层共享、被冻结。额外训练这 2 个标记的嵌入（2,048 个参数）后，答完会停从 0/8 升到 6/8，遗忘不增加 | [02](docs/experiments/02-lora-sft-qwen3.md) |
| 5 | 一份 4,214 条的人格对话数据只有 **149 个不同的核心问题**；按"条"随机切的验证集里问题全都见过，测不出泛化 | [03](docs/experiments/03-persona-diagnosis.md) |
| 6 | 0.6B 微调后，对措辞没见过的问题，判型命中 **58%**；同一基座只给人设卡是 21%（随机 6.25%） | [03](docs/experiments/03-persona-diagnosis.md) |
| 7 | 7B 人格模型退化的原因：单变量实验排除了"步长过大"（Adam 偏差修正、学习率预热都无效），定位到**人设卡放在用户消息里**；改放 system 后，留出损失低于基座，人设卡指令词从 4/12 降到 0 | [04](docs/experiments/04-7b-single-variable.md) |
| 8 | 用新数据（1,804 条、227 个核心问题、人格放 system）训练人格小模型：1.7B 对没见过的问题判型 **56%**，0.6B 27%，基座加人设卡 17–21%（随机 6.25%）；前两批 1,004 条的试训只到随机水平。1.7B 作主力；单种子、48 条判型，未做人工盲评和红队测试 | [05](docs/experiments/05-persona-small-models.md) |
| 9 | 1 名评审对 30 道没见过的题盲评（评审把 1 分当成最好，反转后）：1.7B 基座加人设卡 3.21 → 正式版 4.30，仍低于通用 8B 加人设卡 4.64；红队 100 题 × 4 种人格规则初筛通过 82%，但隐晦自伤信号只有 **18%**，上线前必须配合应用层拦截并补安全数据。单评审、单种子、红队未经人工复核 | [06](docs/experiments/06-human-eval-and-redteam.md) |
| 10 | 换起点还是换规模：1.7B 口语卡在约 2.5 分（AI 评审，5 分制）；换数据来源、解码温度、对话版起点都无效，换 4B 基座后 2.93、自然度五方最高，红队加安全闸 357/380；8B 加人设卡仍为 3.36。**发布版改为 4B（Q4 2.5GB，本机 86 token/秒）**，温度 0.3，配安全闸与身份强制开头；端侧芯片速度未实测 | [07](docs/experiments/07-scale-vs-start-4b-release.md) |

所有结论都是单个随机种子、小样本，报告里写明了置信区间和局限。

## 快速开始

```bash
git clone https://github.com/sylvia-ccc/tinylab && cd tinylab
./start.sh                      # 首次自动建环境（需要 uv），然后打开 http://localhost:8765
python scripts/download.py      # 微调前：下载 Qwen3-0.6B-Base 和 alpaca-zh（约 1.2GB）
```

命令行：

```bash
.venv/bin/python -m lab.train --preset pycode-tiny --name my-run --set lr=3e-4
.venv/bin/python -m lab.finetune --preset sft-alpaca --name my-sft
```

看板里点「＋ 新实验」可以选预设、改参数，**先写假设**，训完在"总览"写结论。

## 预设

| 预设 | 测什么 |
|---|---|
| `pycode-tiny` / `pycode-small` / `pycode-char` / `pycode-lr-high` / `large-small` | 从零训练：基线、模型规模、分词方式、学习率、数据量 |
| `grokking` / `grokking-nowd` | 模加法的"顿悟"，有无权重衰减 |
| `sft-alpaca` / `sft-no-special` | Qwen3-0.6B LoRA 指令微调；对照组不训练特殊标记 |
| `sft-alpaca-attn-only` / `-r4` / `-lr-high` | LoRA 位置、秩、学习率 |
| `mbti-1ep` / `mbti-4ep` / `mbti-v8-1ep` | 人格微调诊断（需要你自己的人格对话数据，见 [03](docs/experiments/03-persona-diagnosis.md)） |

## 苹果芯片上的注意事项

- 微调已开梯度检查点、长度按 64 的倍数补齐、只在回答位置算输出层，并设了显存上限。不这样做，实测 0.6B 的 LoRA 会锁住 20GB 以上内存把机器拖进交换区。
- 同一时间只跑一个微调；0.6B 一次约占 4GB，300 步约 12 分钟。
- 7B 单变量实验用 [mlx_lm](https://github.com/ml-explore/mlx-lm)，脚本在 `scripts/run_7b_variant.py`。

## 代码结构（Python 约 2,400 行 + 前端单文件约 1,000 行，适合逐行读懂）

```
lab/data.py      语料、BPE / 字符分词器、数据报告、模加法任务
lab/model.py     GPT：注意力、MLP、残差、权重共享；可选返回内部状态
lab/train.py     训练循环、学习率调度、全量日志、"体检"快照、预设
lab/finetune.py  真实基座的 LoRA 微调：数据血缘、手写 LoRA、特殊标记修正、遗忘与偏离评估、对话对比
lab/mbti.py      人格数据流水线（按核心问题整组留出）与不依赖判官的评估
lab/server.py    看板后端（只用标准库）
web/index.html   看板前端（单文件，无外部依赖）
web/review.html + scripts/make_blind_review.py / score_blind_review.py   人工盲评：出题（打乱顺序、隐去模型名）、评分页面、汇总（数据放 review/，key.json 不给评审）
scripts/         下载、7B 单变量实验（mlx）、7B 评估
scripts/answer_set.py + eval_redteam.py   最终考试：多个答题者（微调 / 已合并模型 / 基座+人设卡）答测试题和红队题，红队规则初筛出报告，答案可直接喂盲评
docs/experiments/ 实验报告
```

## 许可

- 代码：MIT。
- Qwen3 / Qwen2.5 基座：Apache-2.0（以官方仓库为准）。
- alpaca-zh 数据集卡片标注 CC BY-NC 4.0：用它训练出来的 adapter 仅限研究、非商用。
- 人格对话数据不随仓库发布。
