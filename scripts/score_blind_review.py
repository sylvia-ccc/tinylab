#!/usr/bin/env python3
"""汇总人工盲评结果，输出 Markdown。

读取 <任务目录>/key.json（模型对应关系）和 answers_*.json（评审答卷），按模型汇总：
  - 三项平均分（像不像人格 / 切不切题 / 自然不自然）
  - "整体最好"胜率
  - 按 16 型、按四个维度（E/I、S/N、T/F、J/P）分组的平均分
  - 样本量提示；两个模型时给出胜率的符号检验 p 值

用法：
  python scripts/score_blind_review.py review/mbti-final
  python scripts/score_blind_review.py review/_demo/demo --out report.md
  python scripts/score_blind_review.py review/mbti-final --answers a1.json a2.json   # 指定答卷
  python scripts/score_blind_review.py review/mbti-v1-final --invert                 # 1-5 分反转（6-分）
同一任务里内容完全相同（除 submitted_at 外）的答卷只算一份；
同一个评审人（reviewer 字段相同）多次提交时，只取最新一份；评审人名字为空时按内容去重（内容不同则各算一人）。
--invert：把三项分数反转（6-分数），用于修正评审人按"1 分最好"的反方向打分的答卷；"整体最好"的选择不变。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

DIMS = [("persona", "像不像人格"), ("relevance", "切不切题"), ("natural", "自然不自然")]
POLES = [("E", "I"), ("S", "N"), ("T", "F"), ("J", "P")]
MIN_N = 30  # 每个模型少于这么多题，结论不稳


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def load_answers(d: Path, files):
    paths = [Path(f) for f in files] if files else sorted(d.glob("answers_*.json"))
    latest = {}  # 评审人 -> (提交时间, 内容)
    seen = set()
    for p in paths:
        j = json.loads(p.read_text(encoding="utf-8"))
        body = {k: v for k, v in j.items() if k != "submitted_at"}
        h = hashlib.md5(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if h in seen:          # 内容完全相同（除提交时间）：只算一份
            continue
        seen.add(h)
        who = (j.get("reviewer") or "").strip() or f"匿名:{h[:6]}"
        t = j.get("submitted_at", "") or p.name
        if who not in latest or t >= latest[who][0]:
            latest[who] = (t, j)
    return {w: j for w, (t, j) in latest.items()}


def sign_test_p(w1, w2):
    """两个模型胜场的双侧精确二项检验（零假设：各 50%）。"""
    n = w1 + w2
    if n == 0:
        return None
    k = min(w1, w2)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dir", type=Path, help="盲评任务目录（含 items.json、key.json）")
    ap.add_argument("--answers", nargs="*", help="指定答卷文件；默认读目录下全部 answers_*.json")
    ap.add_argument("--invert", action="store_true", help="把 1-5 分反转（6-分数），用于修正按反方向打的答卷")
    ap.add_argument("--out", type=Path, help="同时写入 Markdown 文件")
    a = ap.parse_args()

    d = a.task_dir
    key = json.loads((d / "key.json").read_text(encoding="utf-8"))
    items = {it["id"]: it for it in json.loads((d / "items.json").read_text(encoding="utf-8"))["items"]}
    models = key["models"]
    answers = load_answers(d, a.answers)
    if not answers:
        sys.exit(f"{d} 里没有答卷（answers_*.json）")

    # 展开成记录：每条 = 一位评审对一道题里一个模型的回答的打分
    sc = defaultdict(lambda: defaultdict(list))     # 模型 -> 维度 -> [分]
    sc_by = defaultdict(lambda: defaultdict(list))  # (分组类别, 分组值) -> 模型 -> [三项均值]
    wins = defaultdict(int)
    best_n = 0
    items_seen = defaultdict(set)                   # 模型 -> 被评过的题
    for who, j in answers.items():
        for iid, r in j.get("ratings", {}).items():
            if iid not in items:
                continue
            order = key["items"][iid]["order"]
            typ = items[iid]["type"]
            for no, s in (r.get("scores") or {}).items():
                m = order[int(no) - 1]
                if a.invert:
                    s = {k: 6 - v for k, v in s.items() if k in dict(DIMS)}
                vals = [s[k] for k, _ in DIMS if k in s]
                for k, _ in DIMS:
                    if k in s:
                        sc[m][k].append(s[k])
                if vals:
                    items_seen[m].add(iid)
                    sc_by[("type", typ)][m].append(mean(vals))
                    if len(typ) == 4:
                        for i in range(4):
                            sc_by[("pole", typ[i])][m].append(mean(vals))
            if r.get("best"):
                wins[order[r["best"] - 1]] += 1
                best_n += 1

    L = []
    L.append(f"# 盲评汇总：{d.name}{'（已反转：分数 = 6 - 原分）' if a.invert else ''}\n")
    if a.invert:
        L.append("> 注意：本表分数已反转（6 - 原分），用于修正评审人把 1 分当成最好的答卷；\"整体最好\"的选择未改动。\n")
    L.append(f"- 题目数 {len(items)}；评审人 {len(answers)}（{', '.join(answers)}）；参与比较的模型 {len(models)} 个\n")

    L.append("## 一、总体\n")
    rows = []
    for m in models:
        allv = [mean(sc[m][k]) for k, _ in DIMS if sc[m][k]]
        rows.append([m, str(len(items_seen[m]))] + [fmt(mean(sc[m][k])) for k, _ in DIMS] + [fmt(mean(allv)),
                     str(wins[m]), (f"{wins[m] / best_n:.0%}" if best_n else "-")])
    L.append(table(["模型", "评过题数"] + [n for _, n in DIMS] + ["三项均分", "选最好次数", "最好胜率"], rows))
    L.append(f"\n（最好胜率 = 被选为\"整体最好\"的次数 / 有效\"最好\"票数 {best_n}）\n")
    if len(models) == 2:
        w1, w2 = wins[models[0]], wins[models[1]]
        p = sign_test_p(w1, w2)
        if p is not None:
            L.append(f"两模型胜负 {models[0]} {w1} : {w2} {models[1]}，符号检验双侧 p = {p:.3f}"
                     f"（{'差异达到常用显著水平 p<0.05' if p < 0.05 else '差异尚不显著，不能据此下结论'}）。\n")

    L.append("## 二、按 16 型\n")
    types = sorted({k[1] for k in sc_by if k[0] == "type"})
    L.append("单元格 = 三项均分（该型下被评的回答条数）\n")
    L.append(table(["人格"] + models, [[t] + [(f"{fmt(mean(sc_by[('type', t)][m]))} ({len(sc_by[('type', t)][m])})"
                                            if sc_by[("type", t)][m] else "-") for m in models] for t in types]))

    L.append("\n## 三、按四个维度\n")
    L.append("单元格 = 三项均分（回答条数）\n")
    for pair in POLES:
        L.append(f"\n**{pair[0]} / {pair[1]}**\n")
        L.append(table(["模型"] + list(pair), [[m] + [(f"{fmt(mean(sc_by[('pole', c)][m]))} ({len(sc_by[('pole', c)][m])})"
                                                      if sc_by[("pole", c)][m] else "-") for c in pair] for m in models]))

    L.append("\n## 四、样本量提示\n")
    warn = [m for m in models if len(items_seen[m]) < MIN_N]
    if warn:
        L.append(f"- 注意：每个模型评过的题少于 {MIN_N} 题（当前最少 {min(len(items_seen[m]) for m in models)} 题；"
                 f"不足的模型：{'、'.join(warn)}），结论不稳，差距小时不要当真；分型、分维度的表格每格样本更少，只能看趋势。")
    else:
        L.append(f"- 每个模型都评过 {MIN_N} 题以上，总体结论可作参考；分型、分维度的格子样本仍然偏少，只看趋势。")
    thin = [t for t in types if min(len(sc_by[("type", t)][m]) for m in models) < 5]
    if thin:
        L.append(f"- 以下人格每个模型的评分条数不足 5 条，不要单独下结论：{'、'.join(thin)}")
    L.append("- 评审越多越稳；单个评审的个人口味会影响结果，建议至少 2 人独立评、再看一致性。")

    out = "\n".join(L) + "\n"
    print(out)
    if a.out:
        a.out.write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
