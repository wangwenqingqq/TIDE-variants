#!/bin/bash
# ================================================================
# C3 优化版本编译脚本
# 创建时间：2026-03-13 08:17 UTC
# 用途：在有 CUDA 编译环境的服务器上重新编译 GTS
# ================================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "GTS C3 Optimization Build Script"
echo "=========================================="

# 检查 CUDA 环境
echo "[1/5] 检查 CUDA 环境..."
if ! command -v nvcc &> /dev/null; then
    echo "❌ 错误：未找到 nvcc 编译器"
    echo "请确保 CUDA 已安装并添加到 PATH"
    echo "示例：export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

echo "✅ CUDA 版本："
nvcc --version | head -n 4

# 检查编译工具
echo ""
echo "[2/5] 检查编译工具..."
if ! command -v cmake &> /dev/null; then
    echo "❌ 错误：未找到 cmake"
    exit 1
fi

if ! command -v make &> /dev/null; then
    echo "❌ 错误：未找到 make"
    exit 1
fi

echo "✅ cmake: $(cmake --version | head -n 1)"
echo "✅ make: $(make --version | head -n 1)"

# 进入 build 目录
echo ""
echo "[3/5] 配置构建环境..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "build" ]; then
    mkdir -p build
fi

cd build

# 运行 CMake
echo "运行 CMake..."
cmake -DCMAKE_BUILD_TYPE=Release ..

# 编译
echo ""
echo "[4/5] 开始编译（使用 $(nproc) 个核心）..."
make -j$(nproc)

# 检查结果
echo ""
echo "[5/5] 检查编译结果..."
if [ -f "bin/GTS" ]; then
    echo "✅ 编译成功！"
    echo "可执行文件位置：$SCRIPT_DIR/bin/GTS"
    ls -lh bin/GTS
else
    echo "❌ 编译失败：未找到可执行文件"
    exit 1
fi

echo ""
echo "=========================================="
echo "编译完成！"
echo "=========================================="
echo ""
echo "下一步：运行测试"
echo "  cd $SCRIPT_DIR"
echo "  bin/GTS ../Datasets/sift.txt ../Datasets/sift1m/sift_query.txt 0 8 cost_q_8_sift_c3.txt vec ../Datasets/sift1m/sift_groundtruth.txt"
