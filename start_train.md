# 🚀 訓練啟動指南

本文檔說明如何從零開始設置環境並執行訓練。

---

## 📦 步驟 1: 創建 Conda 環境

```bash
# 創建 Python 3.13 環境
conda create -n boygirl python=3.13 -y
```

---

## ⚙️ 步驟 2: 啟動環境

```bash
# 啟動 boygirl 環境
conda activate boygirl
```

> **重要**: 每次執行訓練或預測前，都需要先啟動環境！

---

## 📥 步驟 3: 安裝依賴套件

```bash
# 安裝所有需要的套件
pip install -r requirements.txt
```

---

## ✅ 步驟 4: 驗證安裝

```bash
# 驗證所有核心套件是否安裝成功
python -c "import xgboost, sklearn, imblearn, pandas, numpy; print('✅ 所有套件安裝成功！')"
```

如果執行成功，會顯示：`✅ 所有套件安裝成功！`

---

## 🏋️ 步驟 5: 執行訓練

```bash
# 執行訓練腳本（自動創建實驗資料夾）
python main_train.py
```

### 實驗管理系統

每次執行訓練，系統會：
1. **自動分配實驗編號**（exp_001, exp_002, exp_003, ...）
2. **創建實驗資料夾** `experiments/exp_00X_name/`
3. **保存完整配置** `config.yaml`
4. **保存 CV 結果** `cv_results.json`
5. **保存模型和處理器** `preprocessor.pkl`, `model.pkl`
6. **更新實驗記錄** `experiments/experiment_log.csv`

### 訓練流程會執行：
1. 載入 `dataset/train.csv`
2. 清理數據（移除 id, yt, self_intro）
3. 特徵工程（完整 Pipeline）：
   - **Imputation**: 數值型用中位數，類別型用眾數
   - **Negative Value Handling**: fb_friends 負值 clip 到 0（防止 log1p 產生 NaN）⚠️
   - **Transformation**: Clipping, log(1+x), One-Hot Encoding
   - **Scaling**: StandardScaler
4. 執行 5-Fold Cross Validation（使用 SMOTE 平衡數據）
5. 訓練最終模型
6. 儲存 preprocessor 和 model 到 `models_saved/` 目錄

### 預期輸出：
```
🔖 讀取實驗設定檔: configs/default_config.yaml
📦 正在載入訓練資料: dataset/train.csv
🛠️ 正在建立 Preprocessor 特徵工程管線 與 模型...
開始 5-Fold 交叉驗證 (SMOTE=開啟)...
Fold 1: Accuracy=0.xxxx, F1-Score=0.xxxx
Fold 2: Accuracy=0.xxxx, F1-Score=0.xxxx
...
--- 🏁 CV 最終平均結果 ---
Mean Accuracy: 0.xxxx (± 0.xxxx)
Mean F1: 0.xxxx (± 0.xxxx)
...
🚀 正在使用「所有訓練集」訓練最終對外預測模型...
💾 儲存成功！已將前處理器存至 models_saved/preprocessor.pkl，模型存至 models_saved/model.pkl
```

---

## 🔮 步驟 6: 執行預測（訓練完成後）

```bash
# 執行預測腳本
python main_predict.py
```

### 預測流程會執行：
1. 載入 `dataset/test.csv`
2. 載入訓練好的 preprocessor 和 model
3. 應用相同的特徵轉換
4. 進行預測
5. 將預測結果轉換回原始格式（0→2 for 女生，1保持不變）
6. 輸出 `submission.csv`（格式：id, gender | 1=男, 2=女）

---

## 🛠️ 常見問題排除

### 1. 如果套件安裝失敗
```bash
# 降級到 Python 3.11
conda deactivate
conda remove -n boygirl --all -y
conda create -n boygirl python=3.11 -y
conda activate boygirl
pip install -r requirements.txt
```

### 2. 如果訓練過程出錯
- 檢查 `dataset/train.csv` 是否存在
- 檢查 `configs/default_config.yaml` 配置是否正確
- 確認環境已啟動（`conda activate boygirl`）

### 3. 如果預測過程出錯
- 確認訓練已完成（`models_saved/` 目錄下有 `preprocessor.pkl` 和 `model.pkl`）
- 檢查 `dataset/test.csv` 是否存在

---

## 🔄 後續使用

### 每次重新開始工作時：
```bash
# 1. 啟動環境
conda activate boygirl

# 2. 執行訓練或預測
python main_train.py    # 訓練
python main_predict.py  # 預測
```

### 結束工作時：
```bash
# 退出環境
conda deactivate
```

---

## 📁 輸出檔案

### 訓練後產生：
- `models_saved/preprocessor.pkl` - 特徵處理器
- `models_saved/model.pkl` - 訓練好的 XGBoost 模型

### 預測後產生：
- `submission.csv` - 最終預測結果（可提交至 Kaggle）
  - 格式：`id,gender`
  - gender 值：`1`（男生）或 `2`（女生）

---

## 🎯 下一步

訓練成功後，可以：
1. 調整 `configs/default_config.yaml` 中的參數進行實驗
2. 修改特徵工程策略（`src/features.py`）
3. 嘗試不同的模型參數或演算法
4. 提交 `submission.csv` 到 Kaggle 競賽
