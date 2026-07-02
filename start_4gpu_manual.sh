#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export DEVICE_MAP="manual"
export GPU_MEMORY_RESERVE_GB="1"

# 首次启动会创建/更新管理员账号；请在服务器上改成强密码。
export QWEN_ADMIN_USERNAME="admin"
export QWEN_ADMIN_PASSWORD="rokibot321"

python app.py
