# Training 真實規劃順序 train.csv (5-Fold CV 流程)

- **EDA**：找出 dataset 的 feature 特性與極端值分佈

- **Data Loading & Basic Cleaning**：丟棄無用欄位 (id)，過濾掉 Target (gender) 空缺的紀錄，並轉換 gender (1=男, 0=女)。

- **K-Fold Splitting**：將 train.csv 切分成 5 個 Train / Validation 組合 **(⚠️ 核心原則：後續所有 Imputation / Scaling / SMOTE 皆在 CV 迴圈內部嚴格防範 Data Leakage)**。

- **Pre-Imputation Clipping**：針對極端值特徵 (如 height, weight, iq) 以 **Train Fold** 計算 1%~99% 分位數並截斷 Train/Val (註：此步驟在補值前進行)。

- **Custom Imputation (Method 0~3)**：利用 `Gender` 或 `Global` 計算統計量進行空缺值填補 (Fit in Train Fold, Transform in Train & Val Folds)。

- **Derived Feature Engineering**：在補值後，計算特徵工程 (如 `height_weight_ratio`, `BMI`, `ponderal_index`)。

- **Feature Transformation (ColumnTransformer)**: 

  - 數值特徵：StandardScaler & 長尾變數 log1p

  - 類別特徵：One-Hot Encoding

  - 文本特徵 (Text)：TF-IDF 或 MiniLM 分詞鑲嵌 (可選附帶 PCA 降維)

- **Data Balance (SMOTE)**: **(⚠️ 僅針對轉換後的 Train Fold 內部執行，且不在 Val 執行)**

- **Model Selection & Grid Search**: 依 config 切換 CatBoost / XGBoost / LightGBM / Random_Forest，並透過內部循環挑選最佳參數。

- **Model Training**：5-fold 迴圈各自訓練並保存 `fold_i_model.pkl`、`fold_i_preprocessor.pkl`、`fold_i_imputer.pkl`

- **Feature Importance Evaluation**：統整各 Fold 歸一化後的 Feature Importance，並輸出結果日誌。



# Prediction 真實規劃順序 test.csv

- **Data Loading & Basic Cleaning**: 丟掉無用特徵並將對應欄位強轉數值，保留對照 ID。

- **K-Fold Model Loading**: 從實驗目錄中載入 `fold_i_model`, `fold_i_preprocessor`, `fold_i_imputer`。

- **Pre-Imputation Clipping**: 載入並使用訓練階段留下來的 Clipping Bounds 進行極端值截斷。

- **Inference Pipeline (迴圈處理每個 fold)**:

  - **Imputation**: 使用該 Fold 專屬的 `imputer` 進行 Transform 填補。

  - **Derived Feature Engineering**: 計算 BMI, Ratio 等衍生特徵。

  - **Feature Transformation**: 使用該 Fold 專屬的 `preprocessor` 轉換資料。

  - **Model Prediction**: 支援 `full` (整體模型預測) 以及 `fold` (5個 fold 模型進行投票集成)。

- **Output Mapping**: 將預測結果轉換回原始格式（0→2 for 女生，1→1 for 男生）。

- **Output File**: 輸出到 `result/`（`submission_*_fold.csv` 等）。

## 預測目標： gender (binary classification)
## 特徵需要處理的項目：
- id: 無預測能力的識別碼，直接丟棄
- 數值型特徵（5個）：['height', 'weight', 'sleepiness', 'iq', 'fb_friends']
- 類別型特徵（3個）：['gender','star_sign', 'phone_os']
- text特徵（2個）：['yt', 'self_intro']

### 1. 先不納入計算：
- **id**：無預測能力的識別碼，直接丟棄
- **yt, self_intro**：純文字欄位，且在最簡易版中暫時忽略

### 2. 數值型特徵處理 outliers 與 scaling：
- **height, weight, iq**: 處理 outliers：剪裁 (**Clipping**) 處理極端值，保持常態分佈。
  - Pipeline: 中位數補值 → Clipping (1%-99%) → StandardScaler

- **fb_friends**: 由於有極端值和長尾分佈，需要特殊處理：
  - Pipeline: 中位數補值 → **移除負值 (clip to 0)** → **log(1+x)** 轉換 → StandardScaler
  - ⚠️ **重要**: 數據中存在負值（如 -1000），必須先 clip 到 0，否則 log1p 會產生 NaN

> **Imputation**： height, weight, iq, fb_friends 使用 **中位數** 填補

> **Scaling**： height, weight, iq, fb_friends: 數值特徵使用 **StandardScaler** 進行標準化

### gender: 1 (男), 0 (女) label encoding
> **注意**:
> - 原始數據中 gender: `1`=男, `2`=女
> - 訓練時轉換為: `1`=男, `0`=女（模型內部使用）
> - 預測輸出會轉換回: `1`=男, `2`=女（還原為原始格式）

### 3. 無序類別特徵處理：
- **start_sign**: 雙魚座、牡羊座、金牛座、雙子座、巨蟹座、獅子座、處女座、天秤座、天蠍座、射手座、摩羯座、水瓶座、None
- **phone_os**: Android、iOS、Others
> **Imputation**： start_sign, phone_os 使用 **"-1"** (字串) 填補（保留缺失資訊作為獨立類別）
> - ⚠️ 必須是字串 "-1"，不能是數字 -1，因為原始資料是字串型
> - 原方法（已停用）：~~使用眾數填補~~
> - One-Hot 編碼後會產生 `star_sign_-1`, `phone_os_-1` 等特徵，讓模型學習缺失模式

> **Encoding**：進行 **One-Hot Encoding**

### 4. 有序類別特徵處理：
- **sleepiness**: 1,2,3,4,5,None
> **Imputation**：sleepiness 使用 **-1** 填補（保留缺失資訊，原始值 1-5，-1 可明確區分）
> - 原方法（已停用）：~~使用眾數填補~~
> - 缺失值與性別關聯強（缺失時男性比例 80.2% vs 總體 74.7%），保留此資訊有助於預測

> **Encoding**：**不做 One-Hot**，直接轉換為數值型特徵 (`float` / `int`)，維持其大小關係。

![alt text](image.png)