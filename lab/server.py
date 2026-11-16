"""看板后端：只用 Python 标准库 + torch，无需额外依赖。

python -m lab.server  → 打开 http://localhost:8765
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch

from .train import DEFAULTS, PRESETS, RUNS, ROOT, pick_device
from .finetune import FT_DEFAULTS, FT_PRESETS, HUB

WEB = ROOT / "web"
NAME_RE = re.compile(r"^[A-Za-z0-9][\w.-]*$")
_models = {}  # run -> (mtime, model, dataset)

# ---- 人工盲评（review.html）：只读 items.json，绝不读取或返回 key.json ----
REVIEW = ROOT / "review"
REVIEW_DIMS = ("persona", "relevance", "natural")  # 像不像这种人格 / 切不切题 / 自然不自然


def review_dir(name):
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise FileNotFoundError("任务名不合法")
    for base in (REVIEW, REVIEW / "_demo"):  # _demo 下放演示任务
        if (base / name / "items.json").is_file():
            return base / name
    raise FileNotFoundError(f"没有这个盲评任务：{name}")


def review_tasks():
    out = []
    for base in (REVIEW, REVIEW / "_demo"):
        if base.is_dir():
            for d in sorted(base.iterdir()):
                if NAME_RE.match(d.name) and (d / "items.json").is_file():
                    try:
                        it = json.loads((d / "items.json").read_text(encoding="utf-8"))
                        out.append({"task": d.name, "title": it.get("title", d.name), "n": len(it["items"]),
                                    "answers": len(list(d.glob("answers_*.json")))})
                    except (ValueError, KeyError):
                        pass
    return out


def save_review(name, body):
    """校验并保存一份答卷；只接受 items.json 里存在的题、1-5 的分数、合法的"最好"编号。"""
    d = review_dir(name)
    items = json.loads((d / "items.json").read_text(encoding="utf-8"))["items"]
    raw = body.get("ratings")
    if not isinstance(raw, dict):
        raise ValueError("ratings 必须是对象")
    ratings = {}
    for it in items:
        r = raw.get(it["id"])
        if not isinstance(r, dict):
            continue
        n = len(it["answers"])
        sc = {}
        for k, v in (r.get("scores") or {}).items():
            if not (isinstance(k, str) and k.isdigit() and 1 <= int(k) <= n and isinstance(v, dict)):
                continue
            one = {dim: v[dim] for dim in REVIEW_DIMS if type(v.get(dim)) is int and 1 <= v[dim] <= 5}
            if one:
                sc[k] = one
        best = r.get("best")
        best = best if type(best) is int and 1 <= best <= n else None
        if sc or best:
            ratings[it["id"]] = {"scores": sc, "best": best}
    now = time.strftime("%Y%m%d-%H%M%S")
    f = d / f"answers_{now}.json"
    i = 2
    while f.exists():
        f = d / f"answers_{now}-{i}.json"
        i += 1
    f.write_text(json.dumps({"task": name, "reviewer": str(body.get("reviewer", ""))[:40].strip(),
                             "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"), "n_items": len(items),
                             "ratings": ratings}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"ok": True, "file": f.name, "rated": len(ratings), "n_items": len(items)}


def read_jsonl(p):
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # 正在写入的半行
    return rows


def run_summary(d):
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {"name": d.name}
    st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
    if st.get("state") in ("running", "preparing") and time.time() - st.get("updated", 0) > 120:
        st["state"] = "stale"  # 进程可能被杀了
    ev = read_jsonl(d / "eval.jsonl")
    last = ev[-1] if ev else {}
    best = min((e["val_loss"] for e in ev), default=None)
    keep = {k: cfg.get(k) for k in list(DEFAULTS) + list(FT_DEFAULTS) + ["name", "n_params", "n_trainable", "device", "p", "train_frac", "created", "title", "variant"]}
    return {**keep, "status": st, "last_eval": last, "best_val_loss": best}


def load_model(name):
    from .data import get_dataset
    from .model import GPT
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise FileNotFoundError("实验名不合法")
    ck = RUNS / name / "ckpt.pt"
    if not ck.exists():
        raise FileNotFoundError("这个实验还没有保存权重（训练结束后才有）")
    mt = ck.stat().st_mtime
    if name in _models and _models[name][0] == mt:
        return _models[name][1:]
    blob = torch.load(ck, map_location="cpu", weights_only=True)
    c = blob["config"]
    ds = get_dataset(c)
    m = GPT(blob["vocab_size"], c["block_size"], c["n_layer"], c["n_head"], c["n_embd"])
    m.load_state_dict(blob["model"])
    m.eval()
    _models[name] = (mt, m, ds)
    return m, ds


@torch.no_grad()
def generate(name, prompt, n=80, temperature=0.8, top_k=40):
    """逐 token 生成，并返回每一步的概率和备选 token —— 让"模型在想什么"可见。"""
    m, ds = load_model(name)
    ids = ds.tok.encode(prompt) if ds.task == "lm" else ds.encode(prompt)
    if not ids:
        ids = [0]
    steps = []
    idx = torch.tensor([ids])
    n = 1 if ds.task == "modadd" else min(int(n), 300)
    for _ in range(n):
        logits = m(idx[:, -m.block_size:])[0][0, -1]
        probs = (logits / max(temperature, 1e-5)).softmax(-1)
        if top_k:
            v, _ = probs.topk(min(int(top_k), probs.numel()))
            probs = torch.where(probs >= v[-1], probs, torch.zeros_like(probs))
            probs = probs / probs.sum()
        nxt = torch.multinomial(probs, 1).item() if temperature > 0 else probs.argmax().item()
        raw = logits.softmax(-1)
        tp, ti = raw.topk(5)
        steps.append({"token": ds.tok.token_str(nxt), "p": round(raw[nxt].item(), 4),
                      "alts": [{"t": ds.tok.token_str(i), "p": round(p, 4)} for p, i in zip(tp.tolist(), ti.tolist())]})
        idx = torch.cat([idx, torch.tensor([[nxt]])], 1)
    return {"prompt_tokens": [ds.tok.token_str(i) for i in ids], "steps": steps}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj, code=200, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def err(self, msg, code=400):
        self.send({"error": msg}, code)

    def run_dir(self, name):
        if not NAME_RE.match(name) or not (RUNS / name).is_dir():
            raise FileNotFoundError(f"没有这个实验：{name}")
        return RUNS / name

    def do_GET(self):
        if (self.headers.get("Host") or "").split(":")[0] not in ("localhost", "127.0.0.1"):
            return self.err("只接受本机访问", 403)
        p = urlparse(self.path).path.rstrip("/")
        try:
            if p in ("", "/index.html"):
                return self.send((WEB / "index.html").read_bytes(), ctype="text/html; charset=utf-8")
            if p == "/api/meta":
                return self.send({"defaults": DEFAULTS, "presets": PRESETS, "device": pick_device(),
                                  "ft_defaults": FT_DEFAULTS, "ft_presets": FT_PRESETS,
                                  "ft_ready": (HUB / FT_DEFAULTS["base"] / "model.safetensors").exists(),
                                  "raw_dir": str(ROOT / "data" / "raw")})
            if p == "/api/runs":
                RUNS.mkdir(exist_ok=True)
                ds = sorted((d for d in RUNS.iterdir() if d.is_dir() and (d / "status.json").exists()), key=lambda d: d.stat().st_mtime, reverse=True)
                return self.send([run_summary(d) for d in ds])
            if p in ("/review", "/review.html"):
                return self.send((WEB / "review.html").read_bytes(), ctype="text/html; charset=utf-8")
            if p == "/api/review":
                return self.send(review_tasks())
            m = re.match(r"^/api/review/([^/]+)$", p)
            if m:
                return self.send((review_dir(m.group(1)) / "items.json").read_bytes())  # 只给 items.json
            m = re.match(r"^/api/run/([^/]+)(?:/(\w+))?(?:/(\d+))?$", p)
            if m:
                name, what, step = m.groups()
                d = self.run_dir(name)
                if what is None:
                    return self.send({"config": json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {},
                                      "summary": run_summary(d)})
                if what == "lineage":
                    return self.send((d / "lineage.json").read_bytes())
                if what in ("metrics", "eval"):
                    return self.send(read_jsonl(d / f"{what}.jsonl"))
                if what == "snapshots":
                    return self.send(sorted(int(f.stem) for f in (d / "snapshots").glob("*.json")))
                if what == "snapshot" and step is not None:
                    return self.send((d / "snapshots" / f"{int(step):07d}.json").read_bytes())
            self.err("not found", 404)
        except FileNotFoundError as e:
            self.err(str(e), 404)
        except Exception as e:
            self.err(f"{type(e).__name__}: {e}", 500)

    def do_POST(self):
        p = urlparse(self.path).path.rstrip("/")
        # 防跨站调用：只接受本机来源的 JSON 请求
        origin = self.headers.get("Origin")
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("localhost", "127.0.0.1") or (origin and urlparse(origin).hostname not in ("localhost", "127.0.0.1")) \
                or not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self.err("拒绝：只接受本机页面发出的 JSON 请求", 403)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self.err("请求体不是合法的 JSON 对象")
        try:
            if p == "/api/train":
                ft = body.get("kind") == "finetune"
                if ft and any(r.get("kind") == "finetune" and r["status"].get("state") in ("running", "preparing")
                              for r in (run_summary(x) for x in RUNS.iterdir() if x.is_dir())):
                    return self.err("已有一个微调在运行（内存只够同时跑一个），等它结束或先停止")
                cfg = {**(FT_PRESETS if ft else PRESETS).get(body.pop("preset", ""), {}), **body}
                name = cfg.get("name") or time.strftime("run-%m%d-%H%M%S")
                if not NAME_RE.match(name):
                    return self.err("实验名只能用字母、数字、- _ .")
                if (RUNS / name).exists():
                    return self.err(f"实验 {name} 已存在，换个名字")
                cfg["name"] = name
                RUNS.mkdir(exist_ok=True)
                (RUNS / name).mkdir()
                cp = RUNS / name / "request.json"
                cp.write_text(json.dumps(cfg, ensure_ascii=False))
                log = open(RUNS / name / "train.log", "w")
                subprocess.Popen([sys.executable, "-m", "lab.finetune" if ft else "lab.train", str(cp)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
                return self.send({"name": name})
            m = re.match(r"^/api/run/([^/]+)/stop$", p)
            if m:
                (self.run_dir(m.group(1)) / "STOP").touch()
                return self.send({"ok": True})
            m = re.match(r"^/api/run/([^/]+)/notes$", p)
            if m:
                d = self.run_dir(m.group(1))
                c = json.loads((d / "config.json").read_text())
                c["notes"] = str(body.get("notes", ""))[:20000]
                (d / "config.json").write_text(json.dumps(c, ensure_ascii=False, indent=1))
                return self.send({"ok": True})
            m = re.match(r"^/api/review/([^/]+)/answers$", p)
            if m:
                if n > 2_000_000:
                    return self.err("答卷太大", 413)
                try:
                    return self.send(save_review(m.group(1), body))
                except ValueError as e:
                    return self.err(str(e))
            if p == "/api/chat":
                from .finetune import chat
                name = str(body.get("run", ""))
                if not NAME_RE.match(name) or not (RUNS / name / "adapter.pt").exists():
                    return self.err("这个微调还没有保存 adapter（训练结束后才有）", 404)
                if any(r.get("kind") == "finetune" and r["status"].get("state") in ("running", "preparing")
                       for r in (run_summary(x) for x in RUNS.iterdir() if x.is_dir())):
                    return self.err("有微调正在训练，内存不够同时加载第二份模型，请等训练结束")
                return self.send(chat(name, str(body.get("prompt", ""))[:2000], bool(body.get("use_lora", True)),
                                      min(int(body.get("max_new", 160)), 400), float(body.get("temperature", 0))))
            if p == "/api/generate":
                return self.send(generate(body["run"], body.get("prompt", ""), body.get("n", 80),
                                          float(body.get("temperature", 0.8)), int(body.get("top_k", 40))))
            self.err("not found", 404)
        except FileNotFoundError as e:
            self.err(str(e), 404)
        except Exception as e:
            self.err(f"{type(e).__name__}: {e}", 500)


def main():
    port = int(os.environ.get("PORT", 8765))
    print(f"TinyLab 看板：http://localhost:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    main()
