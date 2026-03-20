# Boy or Girl 2026 NEW - 簡易版訓練流程 (Simple Baseline Workflow)

> **目標**：以最少、最清晰的程式碼建立一個「絕對沒有資料洩漏（Data Leakage）」且「可運作」的基準模型（Baseline）。
> **核心概念**：使用 `sklearn.pipeline` 將資料處理與模型綁定，一次解決所有防漏水與步驟繁瑣的問題。

---

## 📋 流程概覽

為了將流程**最單純化**，我們暫時捨棄複雜的特徵選擇（Feature Selection）、特徵提取（PCA）和進階處理，專注於建立一個穩定的 Baseline。

```mermaid
graph TD
    A[讀取資料 Train/Test] --> B[基本清理 Basic Clean]
    B --> C[建立 Scikit-learn Pipeline]
    
    subgraph Pipeline [強大的 Pipeline (自動防止資料洩漏)]
        D[數值/類別 Transformer] --> E[模型分類器 Classifier]
    end
    
    C --> Pipeline
    Pipeline --> F[Cross-Validation 評估]
    F --> G[訓練全資料集並預測 Test]
```

---

## 階段 1: 基礎清理 (Basic Cleaning)
這是唯一在 Pipeline 之外做的處理，僅處理對「整體結構」有影響的基礎清洗。

- **處理項目**：
  1. 移除無預測能力的欄位（如 `id` 識別碼，以及在最簡易版中暫時忽略的純文字 `self_intro`）。
  2. 移除完全重複的資料列 (`drop_duplicates()`)。
  3. （選擇性）手動修正絕對不可能的極端錯誤值（例如 `height`, `weight`, `iq` 欄位出現負數）。
- **原則**：**絕不在這裡進行任何統計處理**（如計算平均值、Z-score等），以確保不發生資料洩漏。

---

## 階段 2: 建立處理流水線 (Pipeline)
利用 `sklearn.compose.ColumnTransformer` 與 `sklearn.pipeline.Pipeline`，讓缺失值填補、特徵轉換與模型訓練一氣呵成。

### 2.1 數值特徵處理 (Numeric Features)
- **填補缺失值 (Imputation)**：使用中位數 (`SimpleImputer(strategy='median')`)。
- **特徵縮放 (Scaling)**：使用標準化 (`StandardScaler()`)。

### 2.2 類別特徵處理 (Categorical Features)
- **填補缺失值 (Imputation)**：使用眾數 (`SimpleImputer(strategy='most_frequent')`)。
- **類別編碼 (Encoding)**：使用獨熱編碼 (`OneHotEncoder(handle_unknown='ignore')`)。

### 2.3 綁定模型 (Model)
- **選擇模型**：從最穩定的 RandomForest 或是 Logistic Regression 開始。
- 將前面的特徵處理與模型串接成單一的 `Pipeline` 物件。

---

## 階段 3: 交叉驗證與評估 (Cross-Validation)
不需手動寫 for-loop 切分資料與 fillna。將整包 `Pipeline` 丟給 `sklearn.model_selection.cross_validate`。

- **折數**：5-Fold Stratified CV（確保各折 Boy/Girl 比例一致）。
- **指標**：Accuracy, F1-Score。
- **Pipeline 的好處**：CV 過程中，每一折訓練時，Imputer 和 Scaler 都**只會**使用該折的 Train data 去 fit，然後 transform Validation data，100% 避免洩漏。

---

## 階段 4: 最終訓練與預測
1. 將剛才設計好的 `Pipeline` 在**包含所有 423 筆的完整 train data** 上進行 `.fit(X, y)`。
2. 直接將 test data 丟入 `.predict(X_test)` 產出結果。
3. 建立 submission 檔案。

---

## 🎯 接下來的實作步驟 (Action Items)

要實現這個極簡架構，我們只需要創建或修改以下兩個核心檔案：

1. **`src/data_pipeline.py`** (或直接寫在 train.py 裡)
   - 負責定義 `ColumnTransformer` 和 `Pipeline`。
2. **`train_simple.py`**
   - 簡潔的主程式碼：讀檔 -> 建 Pipeline -> CV -> Fit 全樣本 -> 預測。
