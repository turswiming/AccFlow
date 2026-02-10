#!/bin/bash
# AccFlow环境搭建脚本 / AccFlow Environment Setup Script

set -e

# 加载网络代理配置 / Load network proxy configuration
echo "=========================================="
echo "加载网络代理配置..."
echo "Loading network proxy configuration..."
echo "=========================================="
source /etc/network_turbo

# 显示代理设置 / Display proxy settings
echo "当前代理设置 / Current proxy settings:"
env | grep -i proxy || echo "未检测到代理环境变量 / No proxy environment variables detected"
echo ""

PROJECT_DIR="/root/autodl-tmp/AccFlow"
cd "$PROJECT_DIR"

echo "=========================================="
echo "配置conda使用代理..."
echo "Configuring conda to use proxy..."
echo "=========================================="

# 备份conda配置 / Backup conda config
if [ -f ~/.condarc ]; then
    cp ~/.condarc ~/.condarc.backup
    echo "已备份conda配置到 ~/.condarc.backup"
    echo "Conda config backed up to ~/.condarc.backup"
fi

# 临时修改conda配置，使用conda-forge和pytorch官方源（通过代理访问）
# Temporarily modify conda config to use conda-forge and pytorch official channels (via proxy)
cat > ~/.condarc << 'EOF'
channels:
  - conda-forge
  - pytorch
  - defaults
show_channel_urls: true
ssl_verify: true
EOF

echo "=========================================="
echo "开始创建AccFlow conda环境..."
echo "Creating AccFlow conda environment..."
echo "=========================================="

# 创建conda环境 / Create conda environment
conda env create -f environment.yaml

echo "=========================================="
echo "环境创建完成，激活环境..."
echo "Environment created, activating..."
echo "=========================================="

# 激活环境并编译CUDA扩展 / Activate environment and compile CUDA extensions
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opensf

echo "=========================================="
echo "编译CUDA扩展: mmcv..."
echo "Compiling CUDA extension: mmcv..."
echo "=========================================="
cd "$PROJECT_DIR/assets/cuda/mmcv"
python setup.py install

echo "=========================================="
echo "编译CUDA扩展: chamfer3D..."
echo "Compiling CUDA extension: chamfer3D..."
echo "=========================================="
cd "$PROJECT_DIR/assets/cuda/chamfer3D"
python setup.py install

echo "=========================================="
echo "编译CUDA扩展: histlib..."
echo "Compiling CUDA extension: histlib..."
echo "=========================================="
cd "$PROJECT_DIR/assets/cuda/histlib"
python setup.py install

echo "=========================================="
echo "环境搭建完成！"
echo "Environment setup completed!"
echo "=========================================="
echo ""
echo "使用以下命令激活环境："
echo "Use the following command to activate the environment:"
echo "  conda activate opensf"
echo ""
