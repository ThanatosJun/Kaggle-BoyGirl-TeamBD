# Boy or Girl 2026 Competition - Training Workflow

## 資料集概況
- **來源**: Kaggle - Boy or Girl 2026 NEW
- **特徵數量**: 11 個 (含 ID)
- **訓練樣本**: 423 筆
- **任務類型**: 二元分類 (性別預測)

---

## 完整訓練流程

### Phase 1: 資料探索與準備
**1. EDA (Exploratory Data Analysis)**
   - 資料分佈分析（數值型、類別型特徵）
   - 缺失值統計與分析
   - 異常值檢測（outliers）
   - 類別不平衡檢查
   - 特徵相關性分析

**2. Data Cleaning & Imputation**
   - 刪除重複資料
   - 刪除或修正不合理的數值
   - 缺失值填補（median for numeric, most_frequent for categorical）
   - 記錄清理前後的資料變化

**3. Baseline Model**
   - 目的：在特徵工程前建立基準性能
   - 使用簡單模型（Logistic Regression 或 Decision Tree）
   - 快速評估：資料質量是否足夠、有無明顯可分性
   - 記錄 baseline 分數作為改進參考

---

### Phase 2: 特徵工程迭代 (Loop)

**4. Feature Transformation**
   - One-hot encoding（類別型 → 數值型）
   - Normalization/Standardization（數值型標準化）
   - Log/Sqrt transformation（偏態分佈處理）
   - Binning/Discretization（連續變數分段）

**5. Feature Selection**
   - 方法選擇：
     - Correlation analysis（相關係數）
     - Recursive Feature Elimination (RFE)
     - Feature importance from tree models
   - 移除冗餘或低貢獻特徵

**6. Feature Extraction**
   - PCA（主成分分析）- 降維
   - Polynomial features（多項式特徵）- 創造交互項
   - Domain-specific features（領域知識特徵工程）

---

### Phase 3: 模型訓練與驗證

**7. Data Split (5-Fold Cross-Validation)**
   - 將資料分成 5 等份
   - 每次使用 4 份訓練 (80%)，1 份驗證 (20%)
   - Stratified split（保持類別比例）
   - **目的**: 小資料集需要 CV 來確保穩定評估，避免過擬合

**8. Data Balance (在每個 fold 的訓練集)**
   - 使用 class_weight='balanced' 自動調整
   - 或使用 SMOTE/undersampling（依需求）
   - ⚠️ **重要**: 必須在 CV split 之後才 balance

**9. Model Training**
   - 在每個 fold 上訓練模型
   - 使用固定的超參數配置（從 config 讀取）
   - ⚠️ **防止 Data Leakage**:
     - Scaler/Imputer 在訓練集 fit
     - 在驗證集 transform（不重新 fit）
   - 記錄每個 fold 的訓練時間和內存使用

**10. Model Evaluation (每個 fold)**
   - 計算多個指標：Accuracy, Precision, Recall, F1, ROC-AUC
   - 記錄每個 fold 的分數

---

### Phase 4: 結果分析與決策

**11. Cross-Validation Results**
   - 計算 5 個 fold 的平均分數和標準差
   - 分析穩定性（std 太高表示不穩定）
   - 比較與 baseline 的改進幅度

**12. Iteration Decision**
   - 若性能未達標：回到 Phase 2 調整特徵工程
   - 若性能滿意：進入最終訓練

**13. Final Training (Optional)**
   - 使用相同超參數在全部 423 筆資料上重新訓練
   - 得到最終模型用於測試集預測
   - 或：保留 5 個 CV 模型做 ensemble

**14. Test Set Prediction**
   - 載入測試集
   - 使用最終模型或 ensemble 進行預測
   - 產生 submission 檔案

---

## 訓練策略說明

### 為什麼用 5-Fold CV？
- ✓ 資料量小（423筆）→ 單次切分不穩定
- ✓ 每筆資料都會當過驗證集 → 更可靠的評估
- ✓ 可以觀察模型在不同資料分佈下的穩定性

### CV 訓練次數
- **單一模型配置**: 5 次訓練（每個 fold 一次）
- **如果做超參數搜索**: N_configs × 5 次

### 為什麼在 Loop 內？
- 每次改變特徵工程後，都要重新評估模型性能
- CV 能告訴我們：這次的特徵改進是否有幫助

---

## 實驗記錄

### 第一輪訓練
- [ ] 完成 EDA
- [ ] Baseline model 分數: ___
- [ ] 使用的特徵工程: ___
- [ ] 最終 CV 平均分數: ___
- [ ] 標準差: ___