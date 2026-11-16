#!/usr/bin/env python3
"""生成人工盲评任务。

输入 JSONL，每行：
  {"id": "q01", "type": "INTJ", "question": "用户的话",
   "answers": {"模型A": "回答", "模型B": "回答", ...}}

输出到 <输出根目录>/<任务名>/：
  items.json  给评审看的题目：模型名已去掉，回答顺序每题随机打乱
  key.json    不给评审看：每题"回答 1/2/3"分别对应哪个模型

用法：
  python scripts/make_blind_review.py 题目.jsonl --task mbti-final
  python scripts/make_blind_review.py 题目.jsonl --task demo --root review/_demo
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME_RE = re.compile(r"^[A-Za-z0-9][\w.-]*$")  # 与 lab/server.py 的名字白名单一致


def load_rows(path: Path) -> list[dict]:
    rows = []
    for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"第 {no} 行不是合法 JSON：{e}")
        for k in ("id", "type", "question", "answers"):
            if k not in r:
                sys.exit(f"第 {no} 行缺少字段 {k}")
        if not isinstance(r["answers"], dict) or len(r["answers"]) < 2:
            sys.exit(f"第 {no} 行（{r['id']}）：answers 需要至少 2 个模型的回答")
        rows.append(r)
    if not rows:
        sys.exit("输入文件里没有题目")
    ids = [str(r["id"]) for r in rows]
    if len(set(ids)) != len(ids):
        sys.exit("题目 id 有重复")
    models = set(rows[0]["answers"])
    for r in rows:
        if set(r["answers"]) != models:
            sys.exit(f"题目 {r['id']} 的模型集合和第一题不一致，每题必须有同样的模型")
        for m, a in r["answers"].items():
            if not isinstance(a, str) or not a.strip():
                sys.exit(f"题目 {r['id']} 里模型 {m} 的回答为空")
    return rows


def build(rows: list[dict], seed: int):
    items, key = [], {}
    for r in rows:
        iid = str(r["id"])
        models = list(r["answers"])
        # 每题独立的随机数发生器：只由 (seed, id) 决定，增删别的题不影响这题的顺序
        rng = random.Random(f"{seed}:{iid}")
        rng.shuffle(models)
        items.append({"id": iid, "type": str(r["type"]).upper(), "question": r["question"],
                      "answers": [r["answers"][m] for m in models]})
        key[iid] = {"order": models}  # order[0] 对应"回答 1"
        # 提醒：回答正文里若出现模型名会泄密
        for m, a in zip(models, [r["answers"][m] for m in models]):
            if len(m) >= 3 and m in a:
                print(f"提醒：题目 {iid} 的回答正文里含有模型名 {m}，会暴露身份，请先处理", file=sys.stderr)
    return items, key


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", type=Path, help="输入题目 JSONL")
    ap.add_argument("--task", required=True, help="任务名（字母数字 - _ .）")
    ap.add_argument("--root", type=Path, default=ROOT / "review", help="输出根目录，默认 review/")
    ap.add_argument("--seed", type=int, default=20261116, help="随机种子，默认固定，保证可复现")
    ap.add_argument("--title", default="", help="页面上显示的任务标题")
    ap.add_argument("--force", action="store_true", help="允许覆盖已有任务（会让已收集的答卷失效，慎用）")
    a = ap.parse_args()

    if not NAME_RE.match(a.task):
        sys.exit("任务名只能用字母、数字、- _ .，且以字母或数字开头")
    out = a.root / a.task
    if (out / "items.json").exists() and not a.force:
        sys.exit(f"{out} 已存在。重新生成会改变顺序、使旧答卷失效；确需覆盖请加 --force")
    rows = load_rows(a.jsonl)
    items, key = build(rows, a.seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps({"task": a.task, "title": a.title or a.task, "items": items},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "key.json").write_text(json.dumps({"task": a.task, "seed": a.seed, "models": sorted(rows[0]["answers"]),
                                              "items": key}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已生成 {len(items)} 题、{len(rows[0]['answers'])} 个模型 -> {out}/items.json（给评审）、key.json（保密，不要发给评审）")


if __name__ == "__main__":
    main()
