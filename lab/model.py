"""极简 GPT（decoder-only Transformer），带"可观测"开关：
需要时返回每层每头的注意力矩阵、残差流范数、MLP 激活统计。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x, observe=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        if observe is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            att = att.masked_fill(~mask, float("-inf")).softmax(-1)
            observe["attn"] = att.detach()
            y = att @ v
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, h, dropout)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, observe=None):
        a = self.attn(self.ln1(x), observe)
        x = x + self.drop(a)
        hmid = F.gelu(self.fc(self.ln2(x)))
        if observe is not None:
            observe["mlp_act"] = hmid.detach()
            observe["attn_out"] = a.detach()
        x = x + self.drop(self.out(hmid))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=4, n_head=4, n_embd=128, dropout=0.0, **_):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # 权重共享
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("out.weight") or n.endswith("proj.weight"):
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, idx, targets=None, observe=False):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        obs = [] if observe else None
        resid = [x.detach()] if observe else None
        for blk in self.blocks:
            o = {} if observe else None
            x = blk(x, o)
            if observe:
                obs.append(o)
                resid.append(x.detach())
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        if observe:
            return logits, loss, {"layers": obs, "resid": resid}
        return logits, loss

    def param_breakdown(self):
        """按组件统计参数量，供看板画结构图。权重共享的 head 不重复计。"""
        seen, rows = set(), []
        def add(name, mod):
            n = 0
            for p in mod.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); n += p.numel()
            rows.append({"name": name, "params": n})
        add("token embedding (与输出层共享)", self.tok_emb)
        add("position embedding", self.pos_emb)
        for i, b in enumerate(self.blocks):
            add(f"block {i} · attention", b.attn)
            add(f"block {i} · mlp", nn.ModuleList([b.fc, b.out]))
            add(f"block {i} · layernorm", nn.ModuleList([b.ln1, b.ln2]))
        add("final layernorm", self.ln_f)
        return rows

    @torch.no_grad()
    def generate(self, idx, n, temperature=1.0, top_k=None):
        for _ in range(n):
            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1] / max(temperature, 1e-5)
            if top_k:
                v, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(logits.softmax(-1), 1)], 1)
        return idx
