#!/bin/bash

# 确保脚本发生任何错误时直接中断
set -e

echo "========= 1. 检查安全脱敏状态 ========="
if [ -z "$OPENAI_API_KEY" ]; then
    echo "[错误] 未在系统环境中检测到 \$OPENAI_API_KEY 变量。"
    echo "请在终端中运行 'export OPENAI_API_KEY=your_key_here' 后再执行本脚本，以确保密钥脱敏！"
    exit 1
fi
echo "[安全检查通过] 已安全载入环境变量 API 密钥。"

echo "========= 2. 初始化 Conda 科学实验环境 ========="
# 激活本地 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
if ! conda env list | grep -q "agent_memory"; then
    echo "正在根据 environment.yml 创建新环境..."
    conda env create -f environment.yml
fi
conda activate agent_memory

echo "========= 3. 启动知识增强型智能体评估 ========="
# 运行核心决策循环
python agent_memory_core.py

echo "========= 4. 评估完成 ========="