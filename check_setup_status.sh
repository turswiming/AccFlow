#!/bin/bash
# 检查AccFlow环境搭建进度 / Check AccFlow environment setup progress

PROJECT_DIR="/root/autodl-tmp/AccFlow"
LOG_FILE="$PROJECT_DIR/setup_env.log"

echo "=========================================="
echo "检查环境搭建状态..."
echo "Checking environment setup status..."
echo "=========================================="
echo ""

# 检查进程是否在运行
if pgrep -f "conda.*environment.yaml" > /dev/null; then
    echo "✓ Conda环境创建进程正在运行..."
    echo "✓ Conda environment creation process is running..."
else
    echo "✗ Conda环境创建进程未运行"
    echo "✗ Conda environment creation process is not running"
fi

echo ""

# 检查环境是否已创建
if conda env list | grep -q "^opensf "; then
    echo "✓ Conda环境 'opensf' 已创建"
    echo "✓ Conda environment 'opensf' has been created"
else
    echo "✗ Conda环境 'opensf' 尚未创建"
    echo "✗ Conda environment 'opensf' has not been created yet"
fi

echo ""
echo "=========================================="
echo "最近的日志输出："
echo "Recent log output:"
echo "=========================================="
if [ -f "$LOG_FILE" ]; then
    tail -20 "$LOG_FILE"
else
    echo "日志文件尚未生成"
    echo "Log file has not been generated yet"
fi

echo ""
echo "=========================================="
echo "使用以下命令查看完整日志："
echo "Use the following command to view full log:"
echo "  tail -f $LOG_FILE"
echo "=========================================="
