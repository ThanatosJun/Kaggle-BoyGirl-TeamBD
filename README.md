# Boy or Girl 2026 - Kaggle Competition

性別預測二元分類競賽專案。

## 📁 專案結構

```
Kaggle-BoyGirl-TeamBD/
├── config/                   # 配置檔案
│   └── train_config.yaml     # 訓練配置
├── dataset/                  # 資料集
│   ├── train.csv            # 訓練資料
│   └── test.csv             # 測試資料
├── src/                      # 原始碼模組
│   ├── data.py              # 資料載入、清理、切分
│   ├── features.py          # 特徵工程
│   ├── models.py            # 模型訓練與評估
│   └── utils.py             # 工具函數
├── notebooks/                # Jupyter Notebooks (探索用)
│   └── 01_quick_eda.ipynb   # 快速 EDA
├── results/                  # 實驗結果 (不納入版本控制)
│   └── {exp_name}_{timestamp}/
│       ├── config.yaml      # 該次實驗的配置
│       ├── metrics.json     # CV 評估結果
│       ├── models/          # 訓練好的模型
│       └── submission.csv   # Kaggle 提交檔案
├── logs/                     # 訓練日誌 (不納入版本控制)
├── train.py                  # 主訓練腳本
├── workflow.md               # 完整訓練流程說明
└── requirements.txt          # Python 依賴套件
```

---

## 🚀 快速開始

### 方法一：使用 Docker（推薦）

#### 1. 準備資料
將 Kaggle 競賽資料下載後放入 `dataset/` 資料夾：
- `train.csv` - 訓練資料
- `test.csv` - 測試資料（之後會有）

#### 2. 建立 Docker Image
```bash
docker-compose build
```

#### 3. 執行訓練
```bash
# 使用預設配置
docker-compose up

# 或使用自訂配置
docker-compose run boygirl-train python train.py --exp_name my_experiment
```

#### 4. 查看結果
結果會自動儲存到本機的 `results/` 和 `logs/` 資料夾

---

### 方法二：本機 Python 環境

#### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

#### 2. 準備資料
將 Kaggle 競賽資料下載後放入 `dataset/` 資料夾：
- `train.csv` - 訓練資料
- `test.csv` - 測試資料

#### 3. 執行訓練
```bash
# 使用預設配置
python train.py

# 指定配置檔案和實驗名稱
python train.py --config config/train_config.yaml --exp_name baseline_v1
```

#### 4. 查看結果
訓練完成後，結果會儲存在：
```
results/{exp_name}_{timestamp}/
├── config.yaml          # 該次實驗配置
├── metrics.json         # CV 評估指標
├── models/              # 訓練好的模型
│   ├── fold_0.pkl
│   ├── fold_1.pkl
│   └── ...
└── submission.csv       # Kaggle 提交檔案
```

---

## 📋 訓練流程

完整的訓練流程定義在 [`workflow.md`](workflow.md) 中，包含 4 個階段：

### Phase 1: 資料探索與準備
- EDA (Exploratory Data Analysis)
- Data Cleaning & Imputation
- Baseline Model

### Phase 2: 特徵工程迭代
- Feature Transformation (編碼、標準化)
- Feature Selection (選擇重要特徵)
- Feature Extraction (PCA、多項式特徵)

### Phase 3: 模型訓練與驗證
- 5-Fold Cross-Validation
- Data Balance (class_weight / SMOTE)
- Model Training & Evaluation

### Phase 4: 結果分析與決策
- CV Results Analysis
- Iteration Decision
- Final Training & Prediction

---

## ⚙️ 配置說明

所有訓練參數都定義在 `config/train_config.yaml` 中：

### 關鍵配置項

**實驗設定**
```yaml
experiment:
  name: baseline_5fold_cv
  seed: 42
  description: "實驗描述"
```

**資料路徑**
```yaml
data:
  train_path: dataset/train.csv
  test_path: dataset/test.csv
  target_column: Gender
```

**前處理**
```yaml
preprocessing:
  numeric_imputer: median        # mean, median, most_frequent
  categorical_imputer: most_frequent
  scaling: standard              # standard, minmax, robust, none
```

**模型配置**
```yaml
model:
  type: random_forest            # random_forest, xgboost, logistic_regression

  random_forest:
    n_estimators: 200
    max_depth: 10
    class_weight: balanced
```

**訓練策略**
```yaml
training:
  validation_strategy: cross_validation
  n_folds: 5
  stratified: true
  balance_method: class_weight   # class_weight, smote, undersample
```

---

## 🔧 工具使用規範

### 1. 實驗命名規範

建議使用清晰的實驗名稱，例如：
```
baseline_v1              # 第一次 baseline
rf_tuned_v1              # Random Forest 調參版本 1
xgb_feature_eng_v2       # XGBoost + 特徵工程版本 2
ensemble_final           # 最終 ensemble 模型
```

### 2. 配置管理原則

**每次實驗都會自動保存完整配置**
- 實驗配置會自動複製到 `results/{exp_name}/config.yaml`
- 可以追溯每次實驗的完整設定
- 方便複現實驗結果

**修改配置流程**
1. 複製 `config/train_config.yaml` 為新檔案 (可選)
2. 修改參數
3. 執行訓練: `python train.py --config config/new_config.yaml --exp_name new_exp`

### 3. 實驗記錄最佳實踐

**建議維護一個實驗記錄表 (可用 Excel 或 Markdown)**

| Exp Name | Date | Model | Features | CV Score | Notes |
|----------|------|-------|----------|----------|-------|
| baseline_v1 | 2026-03-20 | RF | Raw | 0.8234 | 初始 baseline |
| rf_tuned_v1 | 2026-03-21 | RF | Raw | 0.8456 | 調整樹深度 |
| ... | ... | ... | ... | ... | ... |

### 4. Git 版本控制建議

**應該提交的檔案**
- 所有原始碼 (`src/`, `train.py`)
- 配置檔案範本 (`config/train_config.yaml`)
- 文件 (`README.md`, `workflow.md`)

**不應該提交的檔案** (已在 `.gitignore`)
- 實驗結果 (`results/`)
- 訓練日誌 (`logs/`)
- 訓練好的模型 (`*.pkl`)
- 資料集檔案 (`dataset/*.csv`)

### 5. 程式碼擴充指南

**新增資料處理邏輯**
- 編輯 `src/data.py`

**新增特徵工程方法**
- 編輯 `src/features.py`

**新增模型類型**
- 在 `src/models.py` 的 `create_model()` 函數中新增
- 在 `config/train_config.yaml` 中新增對應配置

**新增評估指標**
- 編輯 `src/models.py` 的 `evaluate_model()` 函數

---

## 📊 評估指標

預設計算以下指標（在每個 fold 上）：
- **Accuracy**: 準確率
- **Precision**: 精確率
- **Recall**: 召回率
- **F1 Score**: F1 分數
- **ROC-AUC**: ROC 曲線下面積

Cross-Validation 會輸出每個指標的：
- **Mean**: 5 個 fold 的平均值
- **Std**: 5 個 fold 的標準差

---

## 🐳 Docker 使用詳解

### Docker 架構說明

專案使用 Docker Compose 管理容器，主要特點：

**Volume 掛載**（本機 ↔️ 容器）
```yaml
./dataset     → /workspace/dataset (唯讀)
./config      → /workspace/config
./results     → /workspace/results (輸出)
./logs        → /workspace/logs (輸出)
./src         → /workspace/src (即時同步)
```

**優點**：
- ✅ 環境一致性（不同機器相同結果）
- ✅ 隔離性（不影響本機環境）
- ✅ 可攜性（輕鬆部署到雲端）
- ✅ 即時開發（src/ 修改立即生效）

### 常用 Docker 指令

```bash
# 建立 image
docker-compose build

# 執行訓練（預設配置）
docker-compose up

# 背景執行
docker-compose up -d

# 查看日誌
docker-compose logs -f

# 停止容器
docker-compose down

# 進入容器互動
docker-compose run boygirl-train bash

# 在容器內執行自訂指令
docker-compose run boygirl-train python train.py --exp_name test_v1

# 清理所有（包含 volumes）
docker-compose down -v
```

### Jupyter Notebook in Docker

如果想在 Docker 中使用 Jupyter：

1. 修改 `docker-compose.yml` 加入 port mapping：
```yaml
ports:
  - "8888:8888"
command: jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

2. 執行：
```bash
docker-compose up
# 從日誌中複製 token，在瀏覽器開啟
```

---

## 🐛 除錯指南

### 常見問題

**1. 找不到資料檔案**
```
FileNotFoundError: [Errno 2] No such file or directory: 'dataset/train.csv'
```
→ 確認 `dataset/train.csv` 存在

**2. Target column 不存在**
```
ValueError: Target column 'Gender' not found
```
→ 檢查 `config/train_config.yaml` 中的 `target_column` 設定是否正確

**3. 模組匯入失敗**
```
ModuleNotFoundError: No module named 'xgboost'
```
→ 安裝對應套件: `pip install xgboost`

**4. Docker 權限問題（Linux/Mac）**
```
Permission denied: 'results/'
```
→ 檢查資料夾權限，或在 Dockerfile 中設定 USER

**5. Docker build 失敗**
```
ERROR: failed to solve: failed to compute cache key
```
→ 使用 `docker-compose build --no-cache` 清除快取重建

---

## 📝 後續工作

- [ ] 完成初步 EDA (notebooks/01_quick_eda.ipynb)
- [ ] 執行 baseline 訓練
- [ ] 特徵工程實驗
- [ ] 超參數調優
- [ ] 模型 ensemble

---

## 📞 聯絡資訊

**競賽**: Boy or Girl 2026 NEW | Kaggle
**團隊**: TeamBD

---

**祝訓練順利！🚀**
