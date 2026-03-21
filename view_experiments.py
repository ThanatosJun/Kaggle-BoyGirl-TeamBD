#!/usr/bin/env python3
"""
查看實驗結果的簡單腳本
使用方法: python view_experiments.py
"""

import pandas as pd
import os
import sys

def main():
    log_file = 'experiments/experiment_log.csv'

    if not os.path.exists(log_file):
        print("❌ 找不到實驗記錄檔: experiments/experiment_log.csv")
        print("請先執行 python main_train.py 進行訓練")
        return

    # 讀取實驗記錄
    df = pd.read_csv(log_file)

    if len(df) == 0:
        print("📝 尚無實驗記錄")
        return

    print(f"\n{'='*80}")
    print(f"🔬 實驗總覽（共 {len(df)} 個實驗）")
    print(f"{'='*80}\n")

    # 按 F1 分數排序
    df_sorted = df.sort_values('mean_f1', ascending=False)

    # 顯示關鍵欄位
    display_cols = ['exp_id', 'name', 'use_smote', 'mean_accuracy', 'mean_f1', 'mean_precision', 'mean_recall']

    print(df_sorted[display_cols].to_string(index=False))

    # 顯示最佳實驗
    print(f"\n{'='*80}")
    best_exp = df_sorted.iloc[0]
    print(f"🏆 最佳實驗: {best_exp['exp_id']}")
    print(f"{'='*80}")
    print(f"名稱: {best_exp['name']}")
    print(f"描述: {best_exp['description']}")
    print(f"時間: {best_exp['timestamp']}")
    print(f"\n📊 指標:")
    print(f"  - Accuracy:  {best_exp['mean_accuracy']:.4f} (± {best_exp['std_accuracy']:.4f})")
    print(f"  - F1-Score:  {best_exp['mean_f1']:.4f} (± {best_exp['std_f1']:.4f})")
    print(f"  - Precision: {best_exp['mean_precision']:.4f} (± {best_exp['std_precision']:.4f})")
    print(f"  - Recall:    {best_exp['mean_recall']:.4f} (± {best_exp['std_recall']:.4f})")
    print(f"\n⚙️  參數:")
    print(f"  - SMOTE: {best_exp['use_smote']}")
    print(f"  - Learning Rate: {best_exp['learning_rate']}")
    print(f"  - Max Depth: {best_exp['max_depth']}")
    print(f"\n💡 使用此模型預測:")
    exp_num = best_exp['exp_id'].split('_')[1]
    print(f"   python main_predict.py {int(exp_num)}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
