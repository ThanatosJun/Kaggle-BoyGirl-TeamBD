# Experiments Directory

此目錄用於自動保存所有實驗的結果。

## 目錄結構

```
experiments/
├── experiment_log.csv          # 所有實驗的總記錄表（提交到 Git）
├── exp_001_baseline/           # 實驗 1
│   ├── config.yaml             # 實驗配置（完整保存）
│   ├── cv_results.json         # 交叉驗證詳細結果
│   ├── preprocessor.pkl        # Full Train 特徵處理器（不提交到 Git）
│   ├── model.pkl               # Full Train 模型（不提交到 Git）
│   ├── fold_0_model.pkl        # Fold 0 模型（不提交到 Git）
│   ├── fold_0_preprocessor.pkl # Fold 0 特徵處理器（不提交到 Git）
│   └── ...
├── exp_002_no_smote/           # 實驗 2
│   └── ...
└── ...
```

## 自動編號

每次執行 `python main_train.py` 時，系統會：
1. 自動分配下一個實驗編號（exp_001, exp_002, ...）
2. 創建對應的實驗資料夾
3. 保存配置、模型和結果
4. 更新 `experiment_log.csv`

## 實驗記錄表

`experiment_log.csv` 包含所有實驗的關鍵指標：

| 欄位 | 說明 |
|------|------|
| exp_id | 實驗編號（exp_001_baseline）|
| timestamp | 實驗時間 |
| name | 實驗名稱 |
| description | 實驗描述 |
| use_smote | 是否使用 SMOTE |
| learning_rate | 配置中的 learning_rate（目前取自 xgb_params） |
| max_depth | 配置中的 max_depth（目前取自 xgb_params） |
| mean_accuracy | 平均準確率 |
| std_accuracy | 準確率標準差 |
| mean_f1 | 平均 F1 分數 |
| std_f1 | F1 標準差 |
| mean_precision | 平均精確率 |
| std_precision | 精確率標準差 |
| mean_recall | 平均召回率 |
| std_recall | 召回率標準差 |
| full_train_accuracy | Full Train 準確率（訓練集） |
| full_train_f1 | Full Train F1（訓練集） |
| full_train_precision | Full Train 精確率（訓練集） |
| full_train_recall | Full Train 召回率（訓練集） |
| full_train_metric_scope | 指標範圍註記（train_set_only_no_validation） |

若有啟用搜尋網格，對應實驗資料夾也會產生 `best_params.json`。

## 查看實驗結果

### 方法 1：使用命令列
```bash
# 查看實驗記錄
cat experiments/experiment_log.csv

# 查看特定實驗的詳細結果
cat experiments/exp_001_baseline/cv_results.json
```

### 方法 2：使用 Python
```python
import pandas as pd

# 讀取並排序實驗結果
df = pd.read_csv('experiments/experiment_log.csv')
df_sorted = df.sort_values('mean_f1', ascending=False)
print(df_sorted[['exp_id', 'name', 'mean_accuracy', 'mean_f1']])

# 找出最佳實驗
best_exp = df_sorted.iloc[0]
print(f"\n最佳實驗: {best_exp['exp_id']}")
print(f"F1-Score: {best_exp['mean_f1']:.4f}")
```

## 使用指定實驗的模型預測

```bash
# 使用最新實驗
python main_predict.py

# 使用實驗 3 的模型
python main_predict.py 3

# 使用實驗 3 的 fold ensemble 模式
python main_predict.py 3 fold
```

## 管理實驗

### 刪除表現不佳的實驗
```bash
# 刪除實驗 5
rm -rf experiments/exp_005_*/
```

### 清理所有實驗（保留記錄）
```bash
# 備份實驗記錄
cp experiments/experiment_log.csv experiment_log_backup.csv

# 刪除所有實驗資料夾
rm -rf experiments/exp_*/
```

## Git 管理

- ✅ **提交**: `experiment_log.csv` （追蹤所有實驗結果）
- ✅ **提交**: `experiments/README.md` （說明文檔）
- ❌ **不提交**: `exp_*/` 資料夾（模型檔案太大）

實驗資料夾已在 `.gitignore` 中排除。
