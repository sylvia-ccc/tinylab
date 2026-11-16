"""下载微调用的基座模型和数据集到 data/hub/（只需要一次）。

  python scripts/download.py            # Qwen3-0.6B-Base（约 1.2GB）+ alpaca-zh（约 21MB）
  python scripts/download.py --qwen3-1.7b   # 另外下载 Qwen3-1.7B-Base（约 3.4GB）
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

HUB = Path(__file__).resolve().parent.parent / "data" / "hub"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen3-1.7b", action="store_true", dest="q17")
    a = ap.parse_args()
    models = ["Qwen/Qwen3-0.6B-Base"] + (["Qwen/Qwen3-1.7B-Base"] if a.q17 else [])
    for repo in models:
        snapshot_download(repo, local_dir=HUB / repo.split("/")[1],
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE", "README.md"])
    hf_hub_download("shibing624/alpaca-zh", "default/train/0000.parquet", repo_type="dataset",
                    revision="refs/convert/parquet", local_dir=HUB / "alpaca-zh")
    print("完成：", *(p.name for p in HUB.iterdir()))


if __name__ == "__main__":
    main()
