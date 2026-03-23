# 🚀 訓練啟動指南

本文檔說明如何從零開始設置環境、訓練模型與產生 submission。

---

## 📦 步驟 1: 創建 Conda 環境

```bash
conda create -n boygirl python=3.13 -y
```

## ⚙️ 步驟 2: 啟動環境

```bash
conda activate boygirl
```

## 📥 步驟 3: 安裝依賴套件

```bash
pip install -r requirements.txt
```

## ✅ 步驟 4: 驗證安裝

```bash
python -c "import xgboost, lightgbm, sklearn, imblearn, pandas, numpy, joblib; print('OK')"
```

模型切換在 `configs/default_config.yaml` 設定 `model.type`（`xgboost` / `lightgbm` / `random_forest`）。

---

## 🏋️ 步驟 5: 執行訓練

```bash
python main_train.py
```

每次訓練會自動建立新實驗資料夾：
- `experiments/exp_00X_<name>/config.yaml`
- `experiments/exp_00X_<name>/cv_results.json`
- `experiments/exp_00X_<name>/preprocessor.pkl`（full train 用）
- `experiments/exp_00X_<name>/model.pkl`（full train 模型）
- `experiments/exp_00X_<name>/fold_i_model.pkl`（i=0..n_splits-1）
- `experiments/exp_00X_<name>/fold_i_preprocessor.pkl`（i=0..n_splits-1）
- `experiments/experiment_log.csv`（總實驗記錄）

`experiment_log.csv` 會同時記錄：
- CV 指標（`mean_*`, `std_*`）
- Full Train 指標（`full_train_*`）
- `full_train_metric_scope=train_set_only_no_validation`（標明僅訓練集表現）

---

## 🔮 步驟 6: 執行預測

```bash
# 最新實驗 + full 模式（預設）
python main_predict.py

# 指定實驗 + full 模式
python main_predict.py 2 full

# 指定實驗 + fold ensemble 模式
python main_predict.py 2 fold
```

輸出會放在 `result/`：
- `result/submission_exp_00X_name_full.csv`
- `result/submission_exp_00X_name_fold.csv`
- `result/submission_full.csv`（最新 full）
- `result/submission_fold.csv`（最新 fold）

---

## 🛠️ 常見問題排除

### 1. 訓練或預測找不到套件
- 確認已 `conda activate boygirl`
- 重新安裝 `pip install -r requirements.txt`

### 2. 預測找不到模型
- 確認先成功執行過 `python main_train.py`
- 確認實驗資料夾下存在 `preprocessor.pkl`、`model.pkl` 以及 fold 檔案

### 3. 預測模式參數錯誤
- 合法模式只有：`full`、`fold`

---

## 🔄 日常使用

```bash
conda activate boygirl
python main_train.py
python main_predict.py
```
