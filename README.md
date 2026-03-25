# Boy or Girl 2026 - Kaggle Competition

性別預測二元分類競賽專案（TeamBD）。本專案提供完整的機器學習 Pipeline，支援多種模型、自動化實驗管理、以及靈活的特徵工程策略。

## 🎯 專案特色

- **多模型支援**: XGBoost、LightGBM、Random Forest、CatBoost
- **自動實驗管理**: 每次訓練自動創建實驗資料夾，記錄所有配置與結果
- **防止 Data Leakage**: 嚴格的 CV 流程，SMOTE 與特徵轉換都在 fold 內執行
- **靈活配置**: 單一 YAML 檔案控制所有訓練參數
- **完整評估**: 提供 Accuracy、F1、Precision、Recall 等多項指標

---

## 📁 專案結構

```
Kaggle-BoyGirl-TeamBD/
├── configs/
│   └── default_config.yaml      # 🔧 訓練配置檔案（修改參數從這裡開始）
├── dataset/
│   ├── train.csv                # 訓練資料
│   └── test.csv                 # 測試資料
├── src/                         # 📦 核心模組
│   ├── data_loader.py           # 資料載入與清理
│   ├── features.py              # 特徵工程 Pipeline
│   ├── models.py                # 模型建立
│   └── evaluate.py              # 交叉驗證與評估
├── experiments/                 # 📊 實驗結果（自動產生）
│   ├── experiment_log.csv       # 所有實驗的總記錄
│   └── exp_XXX_name/            # 個別實驗資料夾
├── result/                      # 📤 預測結果（自動產生）
├── notebooks/                   # 📓 探索性分析 Notebooks
├── main_train.py                # ▶️ 訓練主程式
├── main_predict.py              # ▶️ 預測主程式
└── docs/
    ├── experiment_guide.md      # 📖 實驗參數調整指南
    ├── training_workflow_main.md # 訓練流程說明
    └── start_train.md           # 快速開始指南
```

---

## 🚀 快速開始

### 1. 環境設置

```bash
# 創建並啟動虛擬環境
conda create -n boygirl python=3.13 -y
conda activate boygirl

# 安裝依賴套件
pip install -r requirements.txt
```

### 2. 執行訓練

```bash
python main_train.py
```

訓練完成後會在 `experiments/` 目錄下自動創建實驗資料夾（如 `exp_001_baseline/`），包含：
- 模型檔案（.pkl）
- 配置檔案（config.yaml）
- 交叉驗證結果（cv_results.json）
- 實驗記錄會自動寫入 `experiments/experiment_log.csv`

### 3. 執行預測

```bash
# 使用最新實驗的模型預測
python main_predict.py

# 使用指定實驗編號（例如實驗 2）
python main_predict.py 2

# 指定預測模式：full（整個訓練集訓練的單一模型）或 fold（5個折疊模型集成）
python main_predict.py 2 full   # 使用 full 模式
python main_predict.py 2 fold   # 使用 fold ensemble 模式
```

預測結果會儲存在 `result/` 目錄下。

> 📖 **詳細說明**:
> - 完整訓練流程: [training_workflow_main.md](docs/training_workflow_main.md)
> - 參數調整指南: [experiment_guide.md](docs/experiment_guide.md)
> - 啟動教學: [start_train.md](docs/start_train.md)

---

## 📦 核心模組說明

### `src/` 模組架構

專案採用模組化設計，每個檔案負責特定功能：

| 檔案 | 功能 | 主要職責 |
|------|------|----------|
| **data_loader.py** | 資料載入與清理 | <ul><li>讀取 CSV 檔案</li><li>移除無用欄位（id, yt, self_intro）</li><li>標籤映射轉換（男=1, 女=0）</li><li>基礎資料型態轉換</li></ul> |
| **features.py** | 特徵工程 Pipeline | <ul><li>數值特徵處理（補值、剪裁、標準化）</li><li>長尾特徵處理（log 轉換）</li><li>類別特徵處理（One-Hot Encoding）</li><li>有序特徵處理（保持數值順序）</li></ul> |
| **models.py** | 模型建立 | <ul><li>支援 XGBoost / LightGBM / Random Forest / CatBoost</li><li>根據 config 自動初始化模型</li><li>參數覆寫與靈活配置</li></ul> |
| **evaluate.py** | 交叉驗證與評估 | <ul><li>5-Fold Stratified Cross-Validation</li><li>防止 Data Leakage（SMOTE 在 fold 內執行）</li><li>計算評估指標（Accuracy, F1, Precision, Recall）</li><li>保存每個 fold 的模型與 preprocessor</li></ul> |

### 資料處理流程

```
原始資料 (train.csv / test.csv)
    ↓
[data_loader.py] 資料載入與清理
    ├── 移除無用欄位
    ├── 標籤映射
    └── 型態轉換
    ↓
[evaluate.py] 5-Fold 資料分割（僅訓練時）
    ↓
[features.py] 特徵工程 Pipeline（在每個 fold 內 fit）
    ├── 數值特徵: 補值 → 剪裁 → 標準化
    ├── 長尾特徵: 補值 → 移除負值 → log1p → 標準化
    ├── 類別特徵: 補值 → One-Hot Encoding
    └── 有序特徵: 補值 → 保留數值
    ↓
[evaluate.py] SMOTE 過採樣（可選，僅在訓練集上）
    ↓
[models.py] 模型訓練
    ↓
[evaluate.py] 模型評估與保存
```

---

## ⚙️ 配置檔案簡介

所有訓練參數都定義在 `configs/default_config.yaml` 中。主要配置區塊：

### 配置結構概覽

```yaml
experiment:
  name: "baseline"                      # 實驗名稱（會加上編號）
  description: "實驗描述"               # 建議詳細記錄改動內容

data:
  train_path: "dataset/train.csv"
  test_path: "dataset/test.csv"
  drop_cols: ["id", "yt", "self_intro"]
  target_col: "gender"

features:
  numeric_cols: ["height", "weight", "iq"]              # 一般數值特徵
  numeric_log_cols: ["fb_friends"]                      # 長尾分佈特徵（需 log 轉換）
  categorical_cols: ["star_sign", "phone_os"]           # 無序類別特徵
  ordinal_cols: ["sleepiness"]                          # 有序類別特徵（1-5）

preprocessing:
  imputation_mode: "new"                # new（類別用 -1）| old（類別用 most_frequent）
  numeric_imputer_strategy: "median"    # 數值補值策略
  clipping_lower_percentile: 1          # 數值剪裁下界（%）
  clipping_upper_percentile: 99         # 數值剪裁上界（%）
  scaler: "standard"                    # standard | minmax | robust | none

model:
  type: "xgboost"                       # xgboost | lightgbm | random_forest | catboost

  # 模型參數（依據 type 選用）
  xgb_params: {...}
  lgbm_params: {...}
  random_forest_params: {...}
  catboost_params: {...}

search:
  enabled: false                        # 是否啟用自動網格搜尋
  param_grid_mode: "quick"              # quick | full
  metric: "f1"                          # 調參目標指標

training:
  n_splits: 5                           # 交叉驗證折數
  use_smote: false                      # 是否使用 SMOTE 過採樣
  class_weight: "balanced"              # null | balanced | {0: 1.5, 1: 1.0}
  random_state: 42

prediction:
  default_mode: "full"                  # full | fold
  output_dir: "result"
```

> 📖 **詳細參數調整指南**: [experiment_guide.md](docs/experiment_guide.md)

---

## 📊 實驗管理

### 自動實驗追蹤

每次執行 `python main_train.py` 時，系統會自動：

1. **創建新的實驗資料夾**: `experiments/exp_XXX_name/`
2. **保存完整配置**: 將當前的 `default_config.yaml` 複製到實驗資料夾
3. **記錄所有模型**: 保存 full train 模型與每個 fold 的模型
4. **寫入實驗日誌**: 自動更新 `experiments/experiment_log.csv`

### 實驗資料夾結構

```
experiments/
├── experiment_log.csv          # 所有實驗的總記錄表
└── exp_001_baseline/
    ├── config.yaml             # 該次實驗的配置
    ├── cv_results.json         # 交叉驗證詳細結果
    ├── model.pkl               # Full train 模型
    ├── preprocessor.pkl        # Full train 特徵處理器
    ├── fold_0_model.pkl        # Fold 0 模型
    ├── fold_0_preprocessor.pkl
    ├── fold_1_model.pkl
    └── ...
```

### 比較實驗結果

```bash
# 查看所有實驗結果
cat experiments/experiment_log.csv
```

或使用 Python:
```python
import pandas as pd
df = pd.read_csv('experiments/experiment_log.csv')
print(df[['exp_id', 'name', 'mean_accuracy', 'mean_f1']].sort_values('mean_f1', ascending=False))
```

---

## 🔍 評估指標

Cross-Validation 會輸出以下指標：
- **Accuracy**: 整體準確率
- **F1-Score**: 精確率與召回率的調和平均
- **Precision**: 預測為正類中實際為正類的比例
- **Recall**: 實際為正類中被正確預測的比例

最終輸出為 **Mean ± Std**（5 個 fold 的平均值與標準差）

---

## ⚠️ 重要注意事項

### Data Leakage 防範

本專案嚴格遵守以下原則，防止資料洩漏：

1. **補值（Imputation）在 CV fold 內執行**
   - ✅ 正確：在每個 fold 的訓練集上 fit，驗證集上 transform
   - ❌ 錯誤：在整個 dataset 上先 fit imputer

2. **SMOTE 過採樣在 CV fold 內執行**
   - ✅ 正確：僅在訓練 fold 執行 SMOTE，驗證 fold 保持原始分佈
   - ❌ 錯誤：在切分前對整個數據執行 SMOTE

3. **特徵轉換參數由訓練集決定**
   - ✅ 正確：`preprocessor.fit_transform(X_train)` + `preprocessor.transform(X_val)`
   - ❌ 錯誤：分別在 train 和 validation 上 fit_transform

### 最佳實踐建議

- **實驗命名規範**: 使用有意義的名稱（例如：baseline, no_smote, tuned_lgbm, add_interaction_features）
- **記錄實驗筆記**: 在 config 的 `description` 中詳細說明實驗目的與改動內容
- **控制變因**: 每次實驗只改變一個變量，方便分析影響
- **保留實驗記錄**: `experiment_log.csv` 應提交到 Git，方便團隊協作與追蹤
- **定期清理**: 實驗資料夾會佔用空間，表現不佳的可刪除（但保留在 log 中的記錄）

---

## 🐛 常見問題

### RuntimeWarning: invalid value encountered in log1p
**原因**: `fb_friends` 包含負值（如 -1000）
**解決**: 已在 `src/features.py` 的 `log_pipeline` 中加入 `clip_min` 步驟

### ValueError: Input X contains NaN (SMOTE)
**原因**: 特徵轉換後仍有 NaN 殘留
**解決**: 確認所有 Pipeline 都有 Imputer，且順序正確（Imputation → Transformation）

### 預測結果與預期不符
**原因**: 未訓練模型或實驗資料夾缺少模型檔案
**解決**: 先執行 `python main_train.py` 再執行預測

---

## 📚 相關文件

| 文件 | 說明 |
|------|------|
| [experiment_guide.md](docs/experiment_guide.md) | 📖 **參數調整與實驗指南**（如何調整 default_config.yaml） |
| [training_workflow_main.md](docs/training_workflow_main.md) | 完整訓練流程說明 |
| [start_train.md](docs/start_train.md) | 快速開始教學 |

---

## 📞 競賽資訊

- **競賽**: Boy or Girl 2026 NEW | Kaggle
- **團隊**: TeamBD
- **Repository**: [Kaggle-BoyGirl-TeamBD](https://github.com/your-repo-link)

---

**祝訓練順利！🚀**

