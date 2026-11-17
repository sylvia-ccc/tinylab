#!/usr/bin/env python3
"""校验一个 chunk 是否与 brief 对齐，并检查长度、禁用词、轮次、开头重复。

用法：python3 scripts/lint_mbti_chunk.py <brief.json> <chunk.jsonl>
通过则退出码 0。
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BANNED = [
    r"MBTI", r"人格", r"作为一个", r"[EI]人", r"接住", r"被顶嘴",
    r"陪伴方式", r"口吻", r"说话风格", r"[EI][SN][TF][JP]",
]
TICS = ("哈哈", "先别急", "我懂你", "我懂")
PII = [r"(?<!\d)1[3-9]\d{9}(?!\d)", r"[\w.+-]+@[\w-]+\.[\w.]+", r"(?<!\d)\d{17}[\dXx](?!\d)"]
EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\u2600-\u27BF"
    "\uFE0F"
    "]"
)


def bigrams(s):
    s = re.sub(r"\s", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def head4(t):
    return re.sub(r"[，。！？、\s]", "", t)[:4]


def main():
    brief = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    lines = [l for l in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if l.strip()]
    expect = []
    for q in brief["questions"]:
        for s in q["slots"]:
            expect.append((q, s))
    err = []
    if len(lines) != len(expect):
        err.append(f"行数 {len(lines)}，应为 {len(expect)}")
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            err.append(f"第{i+1}行 JSON 失败 {e}")
            rows.append(None)
    openings = defaultdict(list)
    answers_seen = {}
    qend = 0
    n_ans = 0
    by_q_single = defaultdict(list)
    for i, (row, (q, s)) in enumerate(zip(rows, expect)):
        if row is None:
            continue
        loc = row.get("id") or f"line{i+1}"
        if row.get("id") != s["id"]:
            err.append(f"{loc} id 应为 {s['id']}")
        if row.get("type") != s["type"]:
            err.append(f"{loc} type 应为 {s['type']} 实际 {row.get('type')}")
        if row.get("q_id") != q["q_id"]:
            err.append(f"{loc} q_id 应为 {q['q_id']}")
        if row.get("scene") != q["scene"]:
            err.append(f"{loc} scene 应为 {q['scene']}")
        if row.get("turns") != s["turns"]:
            err.append(f"{loc} turns 应为 {s['turns']}")
        if row.get("generator") != "deepseek-v4.1-flash@2026-11":
            err.append(f"{loc} generator 不对")
        if row.get("batch") != "b04":
            err.append(f"{loc} batch 应为 b04")
        ms = row.get("messages") or []
        need = 2 if s["turns"] == 1 else 6
        if len(ms) != need:
            err.append(f"{loc} messages {len(ms)} 条，应为 {need}")
            continue
        if ms[0].get("content") != q["text"]:
            err.append(f"{loc} 首句必须与题干完全一致")
        if any(m.get("role") != ("user" if k % 2 == 0 else "assistant") for k, m in enumerate(ms)):
            err.append(f"{loc} 角色没有交替")
        if ms[-1].get("role") != "assistant":
            err.append(f"{loc} 最后一条不是 assistant")
        assts = [m["content"] for m in ms if m["role"] == "assistant"]
        if len(assts) != s["turns"]:
            err.append(f"{loc} assistant 条数 {len(assts)} ≠ turns")
        for a in assts:
            n_ans += 1
            L = len(a)
            if not 40 <= L <= 150:
                err.append(f"{loc} 回答长度 {L}：{a[:24]}")
            for p in BANNED:
                if re.search(p, a):
                    err.append(f"{loc} 禁用词 {p}：{a[:30]}")
                    break
            if a.startswith(TICS):
                err.append(f"{loc} 口头禅开头：{a[:12]}")
            for p in PII:
                if re.search(p, a):
                    err.append(f"{loc} 隐私：{a[:20]}")
            if a.rstrip().endswith(("？", "?", "吗", "呢")):
                qend += 1
            openings[s["type"]].append((s["id"], head4(a)))
            if a in answers_seen:
                err.append(f"{loc} 回答与 {answers_seen[a]} 完全相同")
            answers_seen[a] = s["id"]
        users = [m["content"] for m in ms if m["role"] == "user"]
        if EMOJI.search(users[0] or ""):
            err.append(f"{loc} 首句不能有表情")
        if s["turns"] == 3:
            flags = [bool(EMOJI.search(u)) for u in users[1:]]
            if sum(flags) != 1:
                err.append(f"{loc} 两句追问里要有且只有一句带表情，现在 {sum(flags)}")
        else:
            by_q_single[q["q_id"]].append((s["type"], assts[0] if assts else ""))
        note = q.get("note") or ""
        if "当地的心理援助热线" in note and "情绪很重时" not in note:
            if not any("当地的心理援助热线" in a for a in assts):
                err.append(f"{loc} 缺少「当地的心理援助热线」")
        blob = "\n".join(assts)
        if re.search(r"1[3-9]\d{9}", blob) or re.search(r"\d{3,4}-\d{7,8}", blob):
            err.append(f"{loc} 像电话号码")

    for t, pairs in openings.items():
        c = defaultdict(list)
        for i_d, h in pairs:
            if h:
                c[h].append(i_d)
        for h, ids in c.items():
            if len(ids) >= 2:
                err.append(f"{t} 开头「{h}」重复：{','.join(ids)}")

    for qid, sing in by_q_single.items():
        for i in range(len(sing)):
            for j in range(i + 1, len(sing)):
                bi, bj = bigrams(sing[i][1]), bigrams(sing[j][1])
                if bi and bj and len(bi & bj) / len(bi | bj) >= 0.5:
                    err.append(f"{qid} {sing[i][0]}≈{sing[j][0]} 回答太像")

    if n_ans and qend / n_ans > 0.45:
        err.append(f"问句结尾 {qend}/{n_ans} = {qend/n_ans:.0%}，请压到 45% 以下")

    if err:
        print(f"FAIL {len(err)}")
        print("\n".join(err[:80]))
        sys.exit(1)
    print(f"OK {len(rows)} 行，回答 {n_ans}，问句结尾 {qend}/{n_ans}")


if __name__ == "__main__":
    main()
