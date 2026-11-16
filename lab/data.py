"""数据层：语料构建、分词器（字符级 / BPE）、数据统计报告、批次采样。

每个数据集准备一次，缓存到 data/cache/<name>/，并输出 report.json 供看板展示。
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
import sysconfig
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
RAW = DATA / "raw"  # 用户自己的 .txt / .md 放这里，即可作为 "custom" 语料


# ---------------------------------------------------------------- 语料来源

def corpus_pycode(max_bytes=4_000_000) -> str:
    """Python 标准库源码：本地现成、量足、结构清晰，适合观察模型学语法。"""
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in stdlib.rglob("*.py") if "test" not in p.parts and "site-packages" not in p.parts)
    random.Random(0).shuffle(files)
    out, n = [], 0
    for p in files:
        try:
            t = p.read_text(encoding="utf-8")
        except Exception:
            continue
        out.append(t)
        n += len(t.encode())
        if n >= max_bytes:
            break
    return "\n\n".join(out)


def corpus_pycode_large(max_bytes=40_000_000) -> str:
    """约 10 倍的 Python 源码（标准库 + 已安装的包），用于缩放规律实验，避免小语料过拟合。"""
    roots = [Path(sysconfig.get_paths()["stdlib"]), Path(sysconfig.get_paths()["purelib"])]
    files = sorted(p for r in roots for p in r.rglob("*.py") if "test" not in str(p).lower())
    random.Random(0).shuffle(files)
    out, n = [], 0
    for p in files:
        try:
            t = p.read_text(encoding="utf-8")
        except Exception:
            continue
        out.append(t)
        n += len(t.encode())
        if n >= max_bytes:
            break
    return "\n\n".join(out)


def corpus_custom() -> str:
    RAW.mkdir(parents=True, exist_ok=True)
    files = sorted(list(RAW.rglob("*.txt")) + list(RAW.rglob("*.md")))
    if not files:
        raise FileNotFoundError(f"把 .txt / .md 文件放进 {RAW} 后再选 custom 语料")
    return "\n\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)


TEXT_CORPORA = {"pycode": corpus_pycode, "pycode-large": corpus_pycode_large, "custom": corpus_custom}


# ---------------------------------------------------------------- 分词器

class CharTokenizer:
    kind = "char"

    def __init__(self, chars):
        self.itos = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def train(cls, text, vocab_size=None):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def token_str(self, i):
        return self.itos[i]

    def state(self):
        return {"kind": "char", "itos": self.itos}


PRETOK = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z_]+| ?\d+| ?[^\sA-Za-z_\d]+|\s+(?!\S)|\s+""")


class BPETokenizer:
    """字节级 BPE（GPT-2 思路的极简实现）。训练过程中记录每次合并，看板可回放。"""
    kind = "bpe"

    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache = {}

    @classmethod
    def train(cls, text, vocab_size=1024, sample_bytes=1_500_000):
        sample = text[: sample_bytes]
        words = collections.Counter(PRETOK.findall(sample))
        seqs = {w: list(w.encode()) for w in words}
        merges = []
        for _ in range(vocab_size - 256):
            pairs = collections.Counter()
            for w, c in words.items():
                s = seqs[w]
                for a, b in zip(s, s[1:]):
                    pairs[(a, b)] += c
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new = 256 + len(merges)
            merges.append(best)
            for w in words:
                s = seqs[w]
                if len(s) < 2:
                    continue
                i, out = 0, []
                while i < len(s):
                    if i < len(s) - 1 and (s[i], s[i + 1]) == best:
                        out.append(new); i += 2
                    else:
                        out.append(s[i]); i += 1
                seqs[w] = out
        return cls(merges)

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def _encode_word(self, w):
        if w in self._cache:
            return self._cache[w]
        s = list(w.encode())
        while len(s) > 1:
            pairs = [(self.ranks.get((a, b), 1e18), i) for i, (a, b) in enumerate(zip(s, s[1:]))]
            r, i = min(pairs)
            if r == 1e18:
                break
            s = s[:i] + [256 + int(r)] + s[i + 2:]
        self._cache[w] = s
        return s

    def encode(self, text):
        out = []
        for w in PRETOK.findall(text):
            out.extend(self._encode_word(w))
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_str(self, i):
        return self.vocab[i].decode("utf-8", errors="replace")

    def state(self):
        return {"kind": "bpe", "merges": self.merges}


def load_tokenizer(state):
    return CharTokenizer(state["itos"]) if state["kind"] == "char" else BPETokenizer(state["merges"])


# ---------------------------------------------------------------- 数据集

class TextDataset:
    task = "lm"

    def __init__(self, name, tokenizer="bpe", vocab_size=1024, val_frac=0.1):
        self.name = name
        key = f"{name}-{tokenizer}{vocab_size if tokenizer == 'bpe' else ''}"
        d = CACHE / key
        if not (d / "tokens.npy").exists():
            d.mkdir(parents=True, exist_ok=True)
            text = TEXT_CORPORA[name]()
            tok = (BPETokenizer if tokenizer == "bpe" else CharTokenizer).train(text, vocab_size)
            ids = np.array(tok.encode(text), dtype=np.uint16)
            np.save(d / "tokens.npy", ids)
            (d / "tokenizer.json").write_text(json.dumps(tok.state()))
            (d / "report.json").write_text(json.dumps(build_report(name, text, tok, ids), ensure_ascii=False))
        self.tok = load_tokenizer(json.loads((d / "tokenizer.json").read_text()))
        ids = torch.from_numpy(np.load(d / "tokens.npy").astype(np.int64))
        n = int(len(ids) * (1 - val_frac))
        self.train_ids, self.val_ids = ids[:n], ids[n:]
        self.report = json.loads((d / "report.json").read_text())
        # bpb 要用验证集自己的压缩率（字节/token），全语料的平均值会让 bpb 偏低
        self.report["val_compression"] = round(len(self.tok.decode(self.val_ids.tolist()).encode()) / len(self.val_ids), 4)
        self.vocab_size = self.tok.vocab_size

    def get_batch(self, split, batch_size, block_size, gen=None):
        data = self.train_ids if split == "train" else self.val_ids
        ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
        return x, y

    def probe_text(self):
        """固定探针：每次快照都用同一段验证集文本，便于对比训练前后模型内部的变化。"""
        ids = self.val_ids[1000:1048].tolist()
        return ids


def build_report(name, text, tok, ids):
    counts = collections.Counter(ids.tolist())
    top = counts.most_common(60)
    freqs = np.array(sorted(counts.values(), reverse=True), dtype=np.float64)
    sample = text[5000:5600]
    sample_ids = tok.encode(sample)
    merges_preview = []
    if tok.kind == "bpe":
        for i in [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 400, 700]:
            if i < len(tok.merges):
                merges_preview.append({"rank": i, "token": tok.token_str(256 + i)})
    lines = text.splitlines()
    return {
        "name": name,
        "chars": len(text),
        "bytes": len(text.encode()),
        "lines": len(lines),
        "tokens": int(len(ids)),
        "tokenizer": tok.kind,
        "vocab_size": tok.vocab_size,
        "compression": round(len(text.encode()) / max(1, len(ids)), 3),  # 每个 token 平均多少字节
        "unique_used": len(counts),
        "top_tokens": [{"token": tok.token_str(i), "count": c} for i, c in top],
        "zipf": [float(f) for f in freqs[np.unique(np.geomspace(1, len(freqs), 80).astype(int) - 1)]],
        "zipf_rank": [int(r) for r in np.unique(np.geomspace(1, len(freqs), 80).astype(int))],
        "sample_tokens": [tok.token_str(i) for i in sample_ids[:220]],
        "merges_preview": merges_preview,
        "line_len_hist": np.histogram([min(len(l), 200) for l in lines], bins=20, range=(0, 200))[0].tolist(),
    }


class ModAddDataset:
    """模加法任务 (a + b) mod p —— 经典的 "顿悟"（grokking）研究场景：
    训练集准确率很早到 100%，验证集准确率要很久之后才突然跃升。"""
    task = "modadd"

    def __init__(self, p=97, train_frac=0.3, seed=0):
        self.name, self.p = "modadd", p
        # 词表：0..p-1 是数字，p 是 '+', p+1 是 '='
        self.vocab_size = p + 2
        pairs = [(a, b) for a in range(p) for b in range(p)]
        random.Random(seed).shuffle(pairs)
        n = int(len(pairs) * train_frac)
        mk = lambda ps: torch.tensor([[a, p, b, p + 1, (a + b) % p] for a, b in ps], dtype=torch.long)
        self.train_seq, self.val_seq = mk(pairs[:n]), mk(pairs[n:])
        self.tok = self
        self.report = {
            "name": "modadd", "task": f"(a + b) mod {p}", "tokenizer": "数字即 token",
            "vocab_size": self.vocab_size, "train_examples": n, "val_examples": len(pairs) - n,
            "train_frac": train_frac, "tokens": int(len(pairs) * 5),
            "examples": [self.decode(s.tolist()) for s in self.train_seq[:12]],
        }

    def token_str(self, i):
        return "+" if i == self.p else "=" if i == self.p + 1 else str(i)

    def decode(self, ids):
        return " ".join(self.token_str(i) for i in ids)

    def encode(self, s):
        out = []
        for t in s.split():
            out.append(self.p if t == "+" else self.p + 1 if t == "=" else int(t) % self.p)
        return out

    def get_batch(self, split, batch_size, block_size, gen=None):
        data = self.train_seq if split == "train" else self.val_seq
        ix = torch.randint(len(data), (batch_size,), generator=gen)
        s = data[ix]
        x, y = s[:, :-1], s[:, 1:].clone()
        y[:, :-1] = -1  # 只在 "=" 之后的答案位置计算损失
        return x, y

    def probe_text(self):
        return self.val_seq[0, :-1].tolist()


def get_dataset(cfg):
    if cfg["dataset"] == "modadd":
        return ModAddDataset(p=cfg.get("p", 97), train_frac=cfg.get("train_frac", 0.3))
    return TextDataset(cfg["dataset"], cfg.get("tokenizer", "bpe"), cfg.get("vocab_size", 1024))
