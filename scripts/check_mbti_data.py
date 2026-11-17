"""MBTI 人格对话数据集的自动验收（对应 docs/mbti-data-spec.md）。

  python scripts/check_mbti_data.py data/mbti-new/batch01.jsonl [更多文件…] [--report report.md] [--pilot]

--pilot：试产批次（约 500 条），数量类的门槛按比例放宽，只看质量类指标。
每项给出 通过 / 不通过 / 警告，以及不合格样例。退出码：全部通过 0，否则 1。
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

TYPES = [a + b + c + d for a in "EI" for b in "SN" for c in "TF" for d in "JP"]
SCENES = ["情绪支持", "分享喜悦", "日常闲聊", "求建议做决定", "人际关系", "工作学习压力", "兴趣爱好",
          "无聊找话聊", "被顶嘴或被质疑", "深夜独处", "自我怀疑", "安全类"]
BANNED = [r"\b[EI][SN][TF][JP]\b", r"[EI][SN][TF][JP]", r"MBTI", r"人格", r"作为一个", r"[EI]人", r"接住你的情绪", r"接住",
          r"被顶嘴", r"陪伴方式", r"口吻", r"说话风格"]
TICS = ["哈哈", "先别急", "我懂你"]
PII = [(r"(?<!\d)1[3-9]\d{9}(?!\d)", "手机号"), (r"[\w.+-]+@[\w-]+\.[\w.]+", "邮箱"), (r"(?<!\d)\d{17}[\dXx](?!\d)", "身份证号")]


def bigrams(s):
    s = re.sub(r"\s", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--report")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--npc", action="store_true", help="游戏 NPC 数据：不查场景配额和安全占比，回答长度按 20–100 字")
    a = ap.parse_args()
    rows, bad_json = [], []
    for f in a.files:
        for i, l in enumerate(Path(f).read_text(encoding="utf-8").splitlines()):
            if not l.strip():
                continue
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                bad_json.append(f"{f}:{i + 1}")
    res = []  # (项目, 状态, 说明, 样例)

    def add(name, ok, msg, ex=None, warn=False):
        res.append((name, "通过" if ok else ("警告" if warn else "不通过"), msg, (ex or [])[:3]))

    n = len(rows)
    add("JSON 可解析", not bad_json, f"{n} 条可解析，{len(bad_json)} 行解析失败", bad_json)

    # 格式
    fmt_bad = []
    for r in rows:
        m = r.get("messages") or []
        ok = (r.get("type") in TYPES and r.get("q_id") and r.get("scene") and m
              and m[-1].get("role") == "assistant" and m[0].get("role") == "user"
              and all(x.get("role") == ("user" if k % 2 == 0 else "assistant") for k, x in enumerate(m))
              and all(isinstance(x.get("content"), str) and x["content"].strip() for x in m))
        if not ok:
            fmt_bad.append(json.dumps(r, ensure_ascii=False)[:160])
    add("字段与轮次格式", not fmt_bad, f"{len(fmt_bad)} 条格式不对（缺 type/q_id/scene，或角色不交替，或最后一条不是 assistant）", fmt_bad)
    rows = [r for r in rows if json.dumps(r, ensure_ascii=False)[:160] not in set(fmt_bad)]
    if not rows:
        return finish(res, a)

    # 16 型配比
    tc = collections.Counter(r["type"] for r in rows)
    lo, hi = min(tc.get(t, 0) for t in TYPES), max(tc.values())
    add("16 型配比", lo > 0 and hi / max(lo, 1) <= 1.3, f"最少 {lo}，最多 {hi}，比值 {hi / max(lo, 1):.2f}（要求 ≤ 1.3）",
        [f"{t}:{tc.get(t, 0)}" for t in TYPES if tc.get(t, 0) in (lo, hi)])

    # 核心问题
    qs = collections.defaultdict(list)
    for r in rows:
        qs[r["q_id"]].append(r)
    need_q = 50 if a.pilot else 600
    add("核心问题数", len(qs) >= need_q, f"{len(qs)} 个（要求 ≥ {need_q}）")
    per_q = [len({x['type'] for x in v}) for v in qs.values()]
    add("每个问题的人格数", sum(p >= 6 for p in per_q) / len(per_q) >= 0.8,
        f"{sum(p >= 6 for p in per_q)}/{len(per_q)} 个问题配了 ≥ 6 种人格（建议 8 种；要求 ≥ 80% 的问题达到 6 种）", warn=True)
    firsts = {q: v[0]["messages"][0]["content"] for q, v in qs.items()}
    items = list(firsts.items())
    B = [bigrams(t) for _, t in items]
    dup = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            u = len(B[i] | B[j])
            if u and len(B[i] & B[j]) / u >= 0.6:
                dup.append(f"{items[i][0]}「{items[i][1][:24]}」≈ {items[j][0]}「{items[j][1][:24]}」")
    add("核心问题之间是否换说法重复", len(dup) <= 0.02 * len(items), f"{len(dup)} 对相似度 ≥ 0.6（要求 ≤ 核心问题数的 2%）：会导致测试集泄漏", dup)
    mixed = [q for q, v in qs.items() if len({x["scene"] for x in v}) > 1]
    add("同一 q_id 场景一致", not mixed, f"{len(mixed)} 个 q_id 被标了不同场景", mixed)

    # 场景配额
    if a.npc:
        miss = [r["id"] for r in rows if not (r.get("npc") or {}).get("role")]
        add("NPC 角色字段", not miss, f"{len(miss)} 条缺 npc.role", miss)
        roles = {(r.get("npc") or {}).get("role") for r in rows}
        cover = collections.defaultdict(set)
        for r in rows:
            cover[r["type"]].add((r.get("npc") or {}).get("role"))
        add("各人格覆盖的 NPC 情境数", min(len(cover.get(t, ())) for t in TYPES) >= 3,
            f"{len(roles)} 个情境；每种人格至少出现在 3 个情境里才算通过；最少的：" +
            ", ".join(f"{t}:{len(cover.get(t, ()))}" for t in sorted(TYPES, key=lambda t: len(cover.get(t, ())))[:4]))
    sc = collections.Counter(firsts and qs[q][0]["scene"] for q in qs)
    unknown = [s for s in sc if s not in SCENES]
    need_s = 3 if a.pilot else 40
    short = [f"{s}:{sc.get(s, 0)}" for s in SCENES if sc.get(s, 0) < need_s]
    if not a.npc:
      add("场景配额", not short and not unknown, f"每类核心问题要求 ≥ {need_s}；不足：{', '.join(short) or '无'}；未知场景：{', '.join(unknown) or '无'}")
    safe = sum(r["scene"] == "安全类" for r in rows) / len(rows)
    if not a.npc:
        add("安全类占比", 0.04 <= safe <= 0.09, f"{100 * safe:.1f}%（要求 4–9%）")

    # 多轮
    multi = sum(len(r["messages"]) > 2 for r in rows) / len(rows)
    lo_m, hi_m = (0.05, 0.2) if a.npc else (0.25, 0.35)
    add("多轮对话占比", lo_m <= multi <= hi_m, f"{100 * multi:.1f}%（要求 {100 * lo_m:.0f}–{100 * hi_m:.0f}%）", warn=a.pilot)

    # 回答层面
    answers = [(r, x["content"]) for r in rows for x in r["messages"] if x["role"] == "assistant"]
    lens = [len(t) for _, t in answers]
    lo_, hi_ = (20, 100) if a.npc else (40, 150)
    out_len = [t[:40] for _, t in answers if not lo_ <= len(t) <= hi_]
    add(f"回答长度 {lo_}–{hi_} 字", len(out_len) / len(answers) <= 0.05,
        f"{len(out_len)}/{len(answers)} 条超出范围（要求 ≤ 5%）；中位数 {sorted(lens)[len(lens) // 2]} 字", out_len)
    hits = []
    for r, t in answers:
        for p in BANNED:
            if re.search(p, t):
                hits.append(f"[{r['type']}] {p} → {t[:50]}")
                break
    add("禁用词（人格字母、设定描述词）", not hits, f"{len(hits)} 条回答含禁用词（要求 0）", hits)
    tic = [t for _, t in answers if any(t.startswith(x) for x in TICS)]
    add("口头禅开头（哈哈 / 先别急 / 我懂你）", len(tic) / len(answers) <= 0.03, f"{100 * len(tic) / len(answers):.1f}%（要求 ≤ 3%）", tic)
    conc = []
    for T in TYPES:
        ts = [t for r, t in answers if r["type"] == T]
        if len(ts) < 20:
            continue
        c = collections.Counter(re.sub(r"[，。！？、\s]", "", t)[:4] for t in ts).most_common(1)[0]
        if c[1] / len(ts) > 0.05:
            conc.append(f"{T}：「{c[0]}」开头占 {100 * c[1] / len(ts):.0f}%")
    add("各型开头短语集中度", not conc, f"{len(conc)} 种人格有开头短语占比 > 5%：会被模型学成口头禅", conc)
    qend = sum(t.rstrip().endswith(("？", "?", "吗", "呢")) for _, t in answers) / len(answers)
    add("问句结尾占比", qend <= 0.5, f"{100 * qend:.0f}%（要求 ≤ 50%）", warn=True)
    seen, dupa = {}, []
    for r, t in answers:
        k = re.sub(r"\s", "", t)
        if k in seen:
            dupa.append(t[:50])
        seen[k] = 1
    add("完全重复的回答", len(dupa) <= 0.005 * len(answers), f"{len(dupa)} 条", dupa)
    # 同一问题下不同人格回答太像
    same = []
    for q, v in qs.items():
        sing = [(x["type"], x["messages"][1]["content"]) for x in v if len(x["messages"]) == 2]
        for i in range(len(sing)):
            for j in range(i + 1, len(sing)):
                bi, bj = bigrams(sing[i][1]), bigrams(sing[j][1])
                if bi and bj and len(bi & bj) / len(bi | bj) >= 0.5:
                    same.append(f"{q} {sing[i][0]} ≈ {sing[j][0]}：{sing[i][1][:30]}")
    add("同一问题下不同人格的回答是否拉开差别", len(same) <= 0.02 * len(qs), f"{len(same)} 对相似度 ≥ 0.5（要求 ≤ 核心问题数的 2%）", same)
    pii = [f"{name}：{t[:40]}" for _, t in answers for p, name in PII if re.search(p, t)]
    pii += [f"{name}（用户消息）：{x['content'][:40]}" for r in rows for x in r["messages"] if x["role"] == "user" for p, name in PII if re.search(p, x["content"])]
    add("隐私信息（手机号、邮箱、证件号）", not pii, f"{len(pii)} 处", pii)
    gens = collections.Counter(r.get("generator") or "（未填写）" for r in rows)
    ok_gen = ("deepseek", "qwen", "glm")  # 条款允许蒸馏 / 开源权重本地自跑（2026-11-16 调研）
    badg = [f"{g}：{c} 条" for g, c in gens.items() if not any(k in g.lower() for k in ok_gen)]
    if "（未填写）" in gens:
        badg.insert(0, f"{gens['（未填写）']} 条没有填 generator 字段：无法确认能否用于开源模型")
    add("生成模型是否允许用于训练开源模型", not badg, "；".join(f"{g}：{c} 条" for g, c in gens.most_common())
        + ("。不在白名单（DeepSeek / 本地 Qwen / 本地 GLM）：" + "、".join(badg) if badg else ""), badg)
    return finish(res, a)


def finish(res, a):
    lines = ["# MBTI 数据验收报告", "", "| 项目 | 结果 | 说明 |", "|---|---|---|"]
    for name, st, msg, ex in res:
        lines.append(f"| {name} | {'✅' if st == '通过' else '⚠️' if st == '警告' else '❌'} {st} | {msg} |")
    lines.append("")
    for name, st, msg, ex in res:
        if st != "通过" and ex:
            lines.append(f"**{name}** 样例：")
            lines += [f"- {e}" for e in ex]
            lines.append("")
    fails = sum(st == "不通过" for _, st, _, _ in res)
    lines.append(f"**结论：{'全部通过' if not fails else f'{fails} 项不通过，退回修改'}**")
    txt = "\n".join(lines)
    print(txt)
    if a.report:
        Path(a.report).write_text(txt, encoding="utf-8")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
