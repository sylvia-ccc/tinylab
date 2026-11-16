#!/bin/bash
# 一键启动 TinyLab：首次运行自动装环境，然后打开看板
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  uv venv -p 3.12 .venv -q && uv pip install --system-certs -q -p .venv -r requirements.txt
fi
(sleep 1.5; open http://localhost:8765) &
exec .venv/bin/python -m lab.server
