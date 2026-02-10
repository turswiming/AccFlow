#!/bin/bash
# AccFlow手动安装脚本 / AccFlow Manual Installation Script
# 分步骤安装，避免依赖解析卡住 / Step-by-step installation to avoid dependency resolution stuck

set -e

PROJECT_DIR="/root/autodl-tmp/AccFlow"
ENV_NAME="opensf"

echo "=========================================="
echo "AccFlow手动安装脚本"
echo "AccFlow Manual Installation Script"
echo "=========================================="

# 加载代理配置 / Load proxy configuration
if [ -f /etc/network_turbo ]; then
    echo ""
    echo "加载网络代理配置..."
    echo "Loading network proxy configuration..."
    source /etc/network_turbo
    echo "代理已加载 / Proxy loaded"
fi

cd "$PROJECT_DIR"

# 检查conda/mamba是否可用 / Check if conda/mamba is available
PKG_MANAGER="conda"
echo "使用 conda 进行安装 / Using conda for installation"


# 配置conda/mamba / Configure conda/mamba
echo ""
echo "配置conda/mamba..."
echo "Configuring conda/mamba..."
$PKG_MANAGER config --set ssl_verify false 2>/dev/null || true

# # 步骤1: 创建基础环境 / Step 1: Create base environment
# echo ""
# echo "=========================================="
# echo "步骤1: 创建基础环境（Python 3.8）"
# echo "Step 1: Create base environment (Python 3.8)"
# echo "=========================================="
# $PKG_MANAGER create -n $ENV_NAME python=3.8 -y

# 激活环境 / Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 步骤2: 安装基础工具包 / Step 2: Install basic tools
echo ""
echo "=========================================="
echo "步骤2: 安装基础工具包"
echo "Step 2: Install basic tools"
echo "=========================================="
$PKG_MANAGER install -y \
    pip \
    setuptools==69.5.1 \
    gxx_linux-64==11.4.0 \
    pathtools \
    -c conda-forge

# 步骤3: 安装MKL（需要先安装，其他包依赖它）/ Step 3: Install MKL (needed first, other packages depend on it)
echo ""
echo "=========================================="
echo "步骤3: 安装MKL"
echo "Step 3: Install MKL"
echo "=========================================="
$PKG_MANAGER install -y mkl==2024.0.0 -c conda-forge

# 步骤4: 安装CUDA工具包 / Step 4: Install CUDA toolkit
echo ""
echo "=========================================="
echo "步骤4: 安装CUDA工具包"
echo "Step 4: Install CUDA toolkit"
echo "=========================================="
$PKG_MANAGER install -y cuda -c nvidia/label/cuda-11.7.0

# 步骤5: 安装PyTorch相关包 / Step 5: Install PyTorch packages
echo ""
echo "=========================================="
echo "步骤5: 安装PyTorch相关包"
echo "Step 5: Install PyTorch packages"
echo "=========================================="
$PKG_MANAGER install -y \
    pytorch=2.0.0 \
    torchvision \
    pytorch-cuda=11.7 \
    -c pytorch

# 步骤6: 安装科学计算基础包 / Step 6: Install scientific computing base packages
echo ""
echo "=========================================="
echo "步骤6: 安装科学计算基础包"
echo "Step 6: Install scientific computing base packages"
echo "=========================================="
$PKG_MANAGER install -y \
    numpy \
    scipy \
    pandas \
    scikit-learn==1.3.2 \
    -c conda-forge

# 步骤7: 安装其他conda包 / Step 7: Install other conda packages
echo ""
echo "=========================================="
echo "步骤7: 安装其他conda包"
echo "Step 7: Install other conda packages"
echo "=========================================="
$PKG_MANAGER install -y \
    lightning==2.0.1 \
    tensorboard \
    numba \
    tqdm \
    h5py \
    wandb \
    omegaconf \
    hydra-core \
    fire \
    tabulate \
    hdbscan \
    rerun-sdk \
    -c conda-forge

# 步骤8: 安装pip包 / Step 8: Install pip packages
echo ""
echo "=========================================="
echo "步骤8: 安装pip包"
echo "Step 8: Install pip packages"
echo "=========================================="
pip install \
    open3d==0.18.0 \
    av2==0.2.1 \
    spconv-cu117==2.3.6 \
    dztimer \
    dufomap==1.1.0 \
    linefit==1.1.0

# 步骤9: 安装本地CUDA扩展 / Step 9: Install local CUDA extensions
echo ""
echo "=========================================="
echo "步骤9: 编译并安装本地CUDA扩展"
echo "Step 9: Compile and install local CUDA extensions"
echo "=========================================="

# 编译mmcv
echo ""
echo "编译CUDA扩展: mmcv..."
echo "Compiling CUDA extension: mmcv..."
cd "$PROJECT_DIR/assets/cuda/mmcv"
pip install .

# 编译chamfer3D
echo ""
echo "编译CUDA扩展: chamfer3D..."
echo "Compiling CUDA extension: chamfer3D..."
cd "$PROJECT_DIR/assets/cuda/chamfer3D"
pip install .

# 编译histlib
echo ""
echo "编译CUDA扩展: histlib..."
echo "Compiling CUDA extension: histlib..."
cd "$PROJECT_DIR/assets/cuda/histlib"
pip install .

# 返回项目目录 / Return to project directory
cd "$PROJECT_DIR"

# 步骤10: 验证安装 / Step 10: Verify installation
echo ""
echo "=========================================="
echo "步骤10: 验证安装"
echo "Step 10: Verify installation"
echo "=========================================="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || echo "PyTorch验证失败 / PyTorch verification failed"

python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" || echo "NumPy验证失败 / NumPy verification failed"

echo ""
echo "=========================================="
echo "安装完成！"
echo "Installation completed!"
echo "=========================================="
echo ""
echo "使用以下命令激活环境："
echo "Use the following command to activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
