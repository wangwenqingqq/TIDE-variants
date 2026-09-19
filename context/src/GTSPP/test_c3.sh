#!/bin/bash
# ================================================================
# C3 优化性能测试脚本
# 创建时间：2026-03-13 08:17 UTC
# 用途：对比 C3 优化前后的性能差异
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "GTS C3 Optimization Performance Test"
echo "=========================================="
echo ""

# 测试 SIFT
echo "[1/2] 测试 SIFT 数据集..."
echo "运行命令："
echo "  bin/GTS ../Datasets/sift.txt ../Datasets/sift1m/sift_query.txt 0 8 cost_q_8_sift_c3.txt vec ../Datasets/sift1m/sift_groundtruth.txt"
echo ""

bin/GTS ../Datasets/sift.txt ../Datasets/sift1m/sift_query.txt 0 8 cost_q_8_sift_c3.txt vec ../Datasets/sift1m/sift_groundtruth.txt 2>&1 | tee test_sift_c3.log

echo ""
echo "[2/2] 测试 GIST 数据集..."
echo "运行命令："
echo "  bin/GTS ../Datasets/gist1m/gist_learn.txt ../Datasets/gist1m/gist_query.txt 0 8 cost_q_8_gist_c3.txt vec ../Datasets/gist1m/gist_groundtruth.txt"
echo ""

bin/GTS ../Datasets/gist1m/gist_learn.txt ../Datasets/gist1m/gist_query.txt 0 8 cost_q_8_gist_c3.txt vec ../Datasets/gist1m/gist_groundtruth.txt 2>&1 | tee test_gist_c3.log

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "结果文件："
echo "  - SIFT: test_sift_c3.log"
echo "  - GIST: test_gist_c3.log"
echo ""
echo "关键指标："
echo "  - Time of index construction"
echo "  - Average search time"
echo "  - Recall@8"
echo ""
echo "对比方式："
echo "  对比 test_sift_c3.log 与之前的 baseline 结果"
echo "  计算 search time 加速比 = baseline_search_time / c3_search_time"
