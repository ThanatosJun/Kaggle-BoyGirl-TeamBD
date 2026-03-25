# Boy or Girl 2026 - Kaggle Competition

性別預測二元分類競賽專案（TeamBD）。

## 📁 專案結構

```
Kaggle-BoyGirl-TeamBD/
├── configs/                  # 配置檔案
│   └── default_config.yaml   # 預設訓練配置
├── dataset/                  # 資料集
│   ├── train.csv            # 訓練資料
│   └── test.csv             # 測試資料
├── src/                      # 原始碼模組
│   ├── __init__.py
│   ├── data_loader.py       # 資料載入與清理
│   ├── features.py          # 特徵工程 Pipeline
│   ├── models.py            # 模型建立
│   └── evaluate.py          # 評估與交叉驗證
├── notebooks/                # Jupyter Notebooks (探索用)
├── experiments/              # 實驗結果目錄（自動產生）
│   ├── experiment_log.csv   # 總實驗記錄表
│   ├── exp_001_baseline/    # 實驗 1
│   │   ├── config.yaml
│   │   ├── cv_results.json
│   │   ├── preprocessor.pkl
│   │   ├── model.pkl
│   │   ├── fold_0_model.pkl
│   │   ├── fold_0_preprocessor.pkl
│   │   └── ...
│   ├── exp_002_no_smote/    # 實驗 2
│   │   └── ...
│   └── ...
├── result/                   # 預測輸出目錄（自動產生）
│   ├── submission_full.csv
│   ├── submission_fold.csv
│   └── submission_exp_00X_name_{mode}.csv
├── main_train.py            # 主訓練腳本
├── main_predict.py          # 主預測腳本
├── training_workflow_main.md # 訓練流程文檔
├── start_train.md           # 訓練啟動指南
└── requirements.txt         # Python 依賴套件
```

---

## 🚀 快速開始

詳細步驟請參考 **[start_train.md](start_train.md)**

### 1. 創建環境

```bash
conda create -n boygirl python=3.13 -y
conda activate boygirl
```

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

### 3. 執行訓練

```bash
python main_train.py
```

每次執行會自動創建新的實驗資料夾（exp_001, exp_002, ...）

### 4. 執行預測

```bash
# 使用最新的實驗模型預測
python main_predict.py

# 使用指定實驗編號的模型預測（例如實驗 2）
python main_predict.py 2

# 指定預測模式：full 或 fold
python main_predict.py 2 full
python main_predict.py 2 fold
```

---

## 📋 訓練流程

完整的訓練流程定義在 [`training_workflow_main.md`](training_workflow_main.md) 中：

### Training 階段（train.csv）：
1. **EDA** - 探索數據特性
2. **Cleaning** - 移除無用欄位（id, yt, self_intro）
3. **Data Split** - 5-Fold CV（在 evaluate.py 中執行）
4. **Imputation** - 填補空值
   - 數值型：中位數（height, weight, iq, fb_friends）
   - 類別型：**-1 常數填充**（star_sign, phone_os, sleepiness）- 保留缺失資訊作為獨立類別
5. **Feature Transformation**
   - height, weight, iq：Clipping (1%-99%) + StandardScaler
   - **fb_friends：移除負值 + log(1+x) + StandardScaler** ⚠️
   - star_sign, phone_os：One-Hot Encoding
   - sleepiness：保持數值（有序特徵）
6. **Model Training** - 可切換 xgboost / lightgbm / random_forest + 5-Fold CV（SMOTE 在每個 fold 內部執行）
7. **Model Evaluation** - Accuracy, F1-Score, Precision, Recall

### Prediction 階段（test.csv）：
1. **Cleaning** - 同 train（但不移除 outliers）
2. **Imputation** - 使用 train 的統計值
3. **Feature Transformation** - 使用 train 的參數（scaler, encoder）
4. **Prediction** - 使用訓練好的模型預測

---

## ⚙️ 配置說明

所有訓練參數都定義在 `configs/default_config.yaml` 中：

### 配置結構

```yaml
experiment:
  name: "baseline"                # 實驗名稱
  description: "初始 baseline 實驗，使用 SMOTE 和預設參數"

data:
  train_path: "dataset/train.csv"
  test_path: "dataset/test.csv"
  drop_cols: ["id", "yt", "self_intro"]  # 無預測能力的欄位
  target_col: "gender"

features:
  numeric_cols: ["height", "weight", "iq"]           # 一般數值特徵
  numeric_log_cols: ["fb_friends"]                   # 長尾數值特徵（需 log 轉換）
  categorical_cols: ["star_sign", "phone_os"]        # 無序類別特徵
  ordinal_cols: ["sleepiness"]                       # 有序類別特徵

model:
  type: "xgboost"            # xgboost | lightgbm | random_forest
  xgb_params:
    objective: "binary:logistic"
    eval_metric: "logloss"
    learning_rate: 0.1
    max_depth: 6
    random_state: 42

training:
  n_splits: 5            # 5-Fold Cross Validation
  use_smote: true        # 是否使用 SMOTE 平衡數據
  random_state: 42
  save_dir: "experiments"  # 實驗結果保存目錄
```

### 特徵處理細節

#### 數值特徵處理

**一般數值型（height, weight, iq）**
```
中位數補值 → Clipping (1%-99%) → StandardScaler
```

**長尾數值型（fb_friends）** ⚠️ **重要**
```
中位數補值 → 移除負值 (clip to 0) → log(1+x) → StandardScaler
```
> 注意：數據中存在 -1000 這樣的負值，必須在 log1p 前處理，否則會產生 NaN

#### 類別特徵處理

**無序類別（star_sign, phone_os）**
```
"-1" 字串補值 → One-Hot Encoding
```
> 使用 "-1" (字串) 表示缺失，One-Hot 後會產生 `star_sign_-1`, `phone_os_-1` 等特徵
> ⚠️ 必須是字串，因為原始資料是字串型（雙魚座、Android 等）

**有序類別（sleepiness: 1-5）**
```
-1 常數補值 → 保持數值型（維持順序關係）
```
> 使用 -1 表示缺失（原始值 1-5，-1 可明確區分），保留缺失資訊供模型學習

#### Target 處理

**gender** - 二元分類
- 原始數據: `1` = 男, `2` = 女
- 訓練時轉換: `1` = 男, `0` = 女（模型內部使用）
- 預測輸出轉換: `1` = 男, `2` = 女（還原為原始格式）

---

## 📊 評估指標

Cross-Validation 會輸出以下指標（每個 fold）：
- **Accuracy**: 準確率
- **F1-Score**: F1 分數
- **Precision**: 精確率
- **Recall**: 召回率

最終輸出：
- **Mean ± Std**: 5 個 fold 的平均值和標準差

---

## 📁 輸出檔案

### 訓練後產生（在 experiments/ 目錄下）：
每次訓練會自動創建新的實驗資料夾：
```
experiments/
├── experiment_log.csv          # 所有實驗的總記錄表
└── exp_001_baseline/           # 實驗 1
    ├── config.yaml             # 實驗配置
    ├── cv_results.json         # 交叉驗證詳細結果
  ├── preprocessor.pkl        # Full Train 特徵處理器
  ├── model.pkl               # Full Train 模型
  ├── fold_0_model.pkl        # Fold 0 模型
  ├── fold_0_preprocessor.pkl # Fold 0 特徵處理器
  └── ...
```

**experiment_log.csv** 包含所有實驗的關鍵指標：
- exp_id, timestamp, name, description
- use_smote, learning_rate, max_depth
- mean_accuracy, std_accuracy, mean_f1, std_f1
- mean_precision, std_precision, mean_recall, std_recall
- full_train_accuracy, full_train_f1, full_train_precision, full_train_recall
- full_train_metric_scope（`train_set_only_no_validation`）

### 預測後產生：
- `result/submission_exp_00X_name_full.csv` - 對應實驗的 full 模式結果
- `result/submission_exp_00X_name_fold.csv` - 對應實驗的 fold 模式結果
- `result/submission_full.csv` - 最新 full 模式結果（通用版本）
- `result/submission_fold.csv` - 最新 fold 模式結果（通用版本）
  ```csv
  id,gender
  1,1
  2,2
  ...
  ```
  > 注意：gender 值為 `1`（男）或 `2`（女），與原始數據格式一致

---

## 🔧 實驗流程建議

### 1. Baseline 實驗（exp_001）
```bash
# 使用預設配置執行第一次訓練
python main_train.py
```
系統會自動創建 `experiments/exp_001_baseline/`

### 2. 調整配置進行迭代實驗
編輯 `configs/default_config.yaml`：

```yaml
experiment:
  name: "no_smote"             # 改變實驗名稱
  description: "測試不使用 SMOTE 的效果"

training:
  use_smote: false             # 關閉 SMOTE
```

```bash
# 執行實驗 2
python main_train.py
```
系統會自動創建 `experiments/exp_002_no_smote/`

### 3. 調整模型參數實驗
```yaml
experiment:
  name: "tuned_xgb"
  description: "調整 XGBoost learning_rate 和 max_depth"

model:
  xgb_params:
    learning_rate: 0.05         # 降低學習率
    max_depth: 8                # 增加樹深度
```

```bash
# 執行實驗 3
python main_train.py
```

### 4. 比較實驗結果

查看實驗總記錄：
```bash
# 查看所有實驗結果
cat experiments/experiment_log.csv
```

或使用 Python：
```python
import pandas as pd
df = pd.read_csv('experiments/experiment_log.csv')
print(df[['exp_id', 'name', 'mean_accuracy', 'mean_f1']].sort_values('mean_f1', ascending=False))
```

### 5. 使用最佳模型預測

```bash
# 使用最新實驗（假設是最佳的）
python main_predict.py

# 或指定特定實驗編號（例如實驗 2 表現最好）
python main_predict.py 2

# 使用 fold ensemble 預測
python main_predict.py 2 fold
```

---

## 🛠️ 進階使用

### 修改特徵工程

編輯 `src/features.py` 來調整特徵處理策略：
- 修改 Clipping 百分位數（預設 1%-99%）
- 改變補值策略（median/mean/mode）
- 調整 Pipeline 順序

### 更換模型

編輯 `configs/default_config.yaml` 的 `model.type`：
```yaml
model:
  type: "lightgbm"  # 或 "xgboost" / "random_forest"
```

### 自定義評估指標

編輯 `src/evaluate.py` 來新增評估指標：
```python
from sklearn.metrics import roc_auc_score

# 在 cross_validate_with_smote 中加入
auc = roc_auc_score(y_val_fold, preds)
```

---

## 🐛 常見問題

### 1. RuntimeWarning: invalid value encountered in log1p
**原因**: `fb_friends` 包含負值
**解決**: 已在 `src/features.py` 的 `log_pipeline` 中加入 `clip_negative` 步驟

### 2. ValueError: Input X contains NaN (SMOTE)
**原因**: 特徵轉換後仍有 NaN 殘留
**解決**: 確認所有 Pipeline 都有 Imputer，且順序正確（Imputation → Transformation）

### 3. KeyError: 'gender'
**原因**: Target 欄位名稱錯誤
**解決**: 檢查 `configs/default_config.yaml` 中 `target_col` 設定

### 4. 模型預測錯誤
**原因**: 未訓練模型或 `experiments/exp_xxx/` 缺少模型檔案
**解決**: 先執行 `python main_train.py` 再執行 `python main_predict.py`

---

## 📝 重要注意事項

### ⚠️ Data Leakage 防範

1. **補值（Imputation）必須在 Data Split 後執行**
   - ✅ 正確：在每個 CV fold 內部分別 fit Imputer
   - ❌ 錯誤：在整個 dataset 上先 fit Imputer

2. **SMOTE 必須在 CV 內部執行**
   - ✅ 正確：僅在 train fold 執行 SMOTE
   - ❌ 錯誤：在切分前對整個數據執行 SMOTE

3. **特徵轉換參數必須由 train 決定**
   - ✅ 正確：`preprocessor.fit_transform(X_train)` + `preprocessor.transform(X_val)`
   - ❌ 錯誤：分別 fit_transform train 和 validation

### 📌 最佳實踐

- **實驗命名規範**：使用有意義的名稱（baseline, no_smote, tuned_xgb, add_features 等）
- **記錄實驗筆記**：在 config 的 description 中詳細說明實驗目的
- **追蹤變更**：每次實驗只改變一個變量，方便分析影響
- **保留實驗記錄**：`experiment_log.csv` 應提交到 Git，方便團隊協作
- **定期清理**：實驗資料夾會佔用空間，表現不佳的實驗可以刪除（但保留 log）

---

## 📞 相關資源

- **完整訓練流程**: [training_workflow_main.md](training_workflow_main.md)
- **訓練啟動指南**: [start_train.md](start_train.md)
- **競賽**: Boy or Girl 2026 NEW | Kaggle
- **團隊**: TeamBD

---

**祝訓練順利！🚀**

