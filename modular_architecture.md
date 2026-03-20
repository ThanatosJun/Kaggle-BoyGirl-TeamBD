# 工程化訓練流程架構 (Modular Training Pipeline)

為了確保實驗的重現性、避免 Data Leakage，並讓團隊協作更順暢，我們將基於 `training_workflow_main.md` 規劃的流程，設計以下模組化的 Python 專案架構。

## 📁 專案目錄結構設計

```text
Kaggle-BoyGirl-TeamBD/
├── configs/                  # 存放所有的超參數與實驗設定
│   ├── default_config.yaml   # 模型參數、特徵欄位定義、實驗設定
│   └── ...
├── data/                     # 存放原始與處理後的資料 (建議加入 .gitignore)
│   ├── raw/                  # train.csv, test.csv
│   └── processed/            # 存放拆分或轉換後的中繼資料 (可選)
├── src/                      # 核心 Pipeline 模組程式碼
│   ├── __init__.py
│   ├── data_loader.py        # 負責讀取與初步清理 (Cleaning, Drop columns)
│   ├── features.py           # 核心模組：定義各種 Transformer 與 Pipeline
│   ├── models.py             # 模型封裝 (XGBoost)
│   ├── evaluate.py           # CV 迴圈、Metrics 計算與紀錄
│   └── predict.py            # 推論模組 (負責讀取測試集並產生 submission.csv)
├── notebooks/                # EDA 與實驗 (如 01_quick_eda.ipynb)
├── main_train.py             # 程式進入點：執行完整訓練流程 (讀資料 -> 組裝 Pipeline -> 訓練 -> 存檔)
├── main_predict.py           # 程式進入點：執行推論流程
└── requirements.txt          # 套件版本依賴
```

---

## 🧩 各模組功能與對應的 Workflow 步驟

以下詳細說明 `src/` 中各個檔案各自負責您規劃的哪一個環節。

### 1. `src/data_loader.py`
**負責步驟**：Cleaning、Data Split
- **函式 `load_and_clean_data(file_path)`**：
  - 讀取 CSV。
  - **Drop 特徵**：丟棄 `id`, `yt`, `self_intro`。
  - **Label Encoding**：將 Target (`gender`) 轉為 `0` 和 `1`。
  - 移除「絕對不合理」的異常行（如有人身高寫 -5 cm）。
- **函式 `split_data(df)`**：
  - 將資料切分為 `X_train`, `X_val`, `y_train`, `y_val`。
  - *確保切分在所有特徵工程之前進行！*

### 2. `src/features.py` 
**負責步驟**：Imputation、Feature Transformation (Scaling, One-Hot, Ordinal)、Feature Selection
這是整個架構的**大腦**。我們強烈建議使用 `sklearn.compose.ColumnTransformer` 與 `sklearn.pipeline.Pipeline` 來封裝。
- **類別 `FeaturePipelineFactory`**：
  - **數值特徵 Pipeline (`height`, `weight`, `iq`, `fb_friends`)**：
    - `SimpleImputer(strategy='median')`
    - (自訂 Transformer) `Log1pTransformer()` 針對 `fb_friends`。
    - (自訂 Transformer) `ClippingTransformer()` 針對其他數值。
    - `StandardScaler()`
  - **無序類別特徵 Pipeline (`star_sign`, `phone_os`)**：
    - `SimpleImputer(strategy='most_frequent')`
    - `OneHotEncoder(handle_unknown='ignore')`
  - **有序類別特徵 Pipeline (`sleepiness`)**：
    - `SimpleImputer(strategy='most_frequent')`
    - `OrdinalEncoder()` / 手動替換成 `float`。
  - **最後使用 `ColumnTransformer` 將以上三組打包成一個完整的 `preprocessor`**。

### 3. `src/models.py`
**負責步驟**：Model Selection
- **類別 `XGBoostTrainer`**：
  - 封裝 XGBClassifier。
  - 統一吃 `configs/` 的參數（如 `learning_rate`, `max_depth` 等）。

### 4. `src/evaluate.py`
**負責步驟**：Data Balance (SMOTE)、Model Training (5-fold CV)、Model Evaluation
- **函式 `cross_validate_with_smote(X, y, preprocessor, model)`**：
  - 建立 5-Fold 切割 (`StratifiedKFold`)。
  - **⚠️ 核心防呆邏輯**：在 `for train_idx, val_idx in fold.split(X, y)` 的**迴圈內部**：
    1. 取出該次 fold 的 `X_train_fold`, `y_train_fold`。
    2. 使用 `preprocessor.fit_transform()`。
    3. 對其進行 **SMOTE** 取樣。
    4. 丟入模型訓練 (`model.fit`)。
    5. 使用 `preprocessor.transform()` 轉換 `X_val_fold`。
    6. 預測並計算 Accuracy, F1-Score。
  - 確保 SMOTE 絕對不會污染到 Validation Data。

### 5. `main_train.py`
**負責步驟**：串接所有的流程 (The Orchestrator)
1. 呼叫 `data_loader.py` 讀取並切分 `train.csv`。
2. 呼叫 `features.py` 取得定義好的 `preprocessor`。
3. 呼叫 `models.py` 實例化 XGBoost。
4. 將 `preprocessor` 和 `model` 組合在一起 (或統稱為 `model_pipeline`)。
5. 傳入 `evaluate.py` 跑完 5-Fold，印出成績。
6. (最後) 將 `preprocessor` 和 `model` 使用 `joblib` 或 `pickle` 儲存為檔案（給預測時使用）。

### 6. `main_predict.py`
**負責步驟**：Prediction 規劃順序
1. 載入 `test.csv` 並通過 `data_loader` 進行基本清理。
2. **載入訓練好的** `preprocessor.pkl` 與 `model.pkl`。
3. **關鍵：** 執行 `preprocessor.transform(X_test)` (*絕對不要呼叫 fit*)。
4. 執行模型預測，並輸出 `submission.csv`。
