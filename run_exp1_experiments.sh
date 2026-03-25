#!/bin/bash

# 運行 Experiment 1 的 4 個實驗
# 1%-99% Percentile + 4 種補值策略

echo "=========================================="
echo "開始運行 Experiment 1 系列"
echo "=========================================="

# 方法 1：全局 Mean
echo ""
echo "▶️ 運行 Exp 1-1: 方法1 - 全局 Mean"
python main_train.py --config configs/exp1_p99_method1_global_mean.yaml
echo "✅ Exp 1-1 完成"

# 方法 2：論文範圍中點
echo ""
echo "▶️ 運行 Exp 1-2: 方法2 - 論文範圍中點（按性別）"
python main_train.py --config configs/exp1_p99_method2_paper_range.yaml
echo "✅ Exp 1-2 完成"

# 方法 3：分群平均值
echo ""
echo "▶️ 運行 Exp 1-3: 方法3 - 分群平均值（按 star_sign）"
python main_train.py --config configs/exp1_p99_method3_grouped_mean.yaml
echo "✅ Exp 1-3 完成"

echo ""
echo "=========================================="
echo "🎉 所有 Experiment 1 實驗完成！"
echo "=========================================="
echo ""
echo "查看結果："
echo "cat experiments/experiment_log.csv"
