# Boy or Girl 2026 - Kaggle Competition (TeamBD)

本專案為性別預測二元分類競賽實作，提供完整可重現的機器學習流程：

- 多模型訓練：XGBoost、LightGBM、Random Forest、CatBoost
- 模組化特徵工程：數值、類別、文本（TF-IDF / MiniLM / handcrafted）
- 實驗追蹤：每次訓練自動保存 config、CV 指標、模型與特徵處理器
- 批次實驗：Exp1 / Exp2 / Exp3 可獨立批次執行並輸出匯總表

---

## 1. 最新文件入口（建議先看）

### ALL 報告（主要決策依據）
- [docs/EXPERIMENT_1_ALL.md](docs/EXPERIMENT_1_ALL.md)
- [docs/EXPERIMENT_2_ALL.md](docs/EXPERIMENT_2_ALL.md)
- [docs/EXPERIMENT_3_ALL.md](docs/EXPERIMENT_3_ALL.md)

### 補充與比較
- [docs/EXPERIMENT_1_SUMMARY.md](docs/EXPERIMENT_1_SUMMARY.md)
- [docs/EXPERIMENT_2_SUMMARY.md](docs/EXPERIMENT_2_SUMMARY.md)
- [docs/EXPERIMENT_3_COMPARISON.md](docs/EXPERIMENT_3_COMPARISON.md)
- [docs/EXPERIMENT_COMPARIR.md](docs/EXPERIMENT_COMPARIR.md)

---

## 2. Method 對照（與 ALL 文件一致）

### Exp1（補值策略）
| 代碼 | 方法名稱 |
|---|---|
| method0 | Median |
| method1 | Mean |
| method2 | Paper Range Midpoint |
| method3 | Cluster Mean |

### Exp2（衍生特徵策略）
| 代碼 | 方法名稱 |
|---|---|
| method0 | baseline |
| method1 | weight ratio |
| method2 | BMI |
| method3 | PI |

### Exp3（文本方法，報告內統一代碼）
| 代碼 | 方法名稱 |
|---|---|
| method0 | minilm |
| method1 | tfidf |
| method2 | both |
| method3 | handcrafted |

註：Exp3 實際 config 檔名通常直接使用 tfidf/minilm/both/handcrafted，不一定直接以 method0~3 命名。

---

## 3. 專案結構

```text
Kaggle-BoyGirl-TeamBD/
├── configs/
│   ├── default_config_all.yaml
│   ├── default_config_exp3.yaml
│   ├── exp1_*_method*.yaml
│   ├── exp2_*_method*.yaml
│   ├── exp3_*.yaml
│   └── old_config/
├── dataset/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── evaluate.py
│   └── imputation_strategies.py
├── experiments/
│   ├── experiment_log.csv
│   ├── experiment_record.csv
│   ├── experiment_record_exp2.csv
│   ├── experiment_record_exp3.csv
│   └── exp_XXX_.../
├── result/
├── docs/
├── notebooks/
├── main_train.py
├── main_predict.py
├── exp1_batch_train_and_record.py
├── exp2_batch_train_and_record.py
├── exp3_batch_train_and_record.py
├── training_workflow_main.md
└── start_train.md
```

---

## 4. 環境安裝

```bash
conda create -n boygirl311 python=3.11 -y
conda activate boygirl311
pip install -r requirements.txt
```

建議使用 Python 3.11。

---

## 5. 單次訓練

### 5.1 用單一 config 訓練

```bash
python main_train.py --config configs/default_config_all.yaml
```

或直接指定實驗 config：

```bash
python main_train.py --config configs/exp1_CB_method0.yaml
python main_train.py --config configs/exp2_CB_method2.yaml
python main_train.py --config configs/exp3_tfidf_exp1base.yaml
```

訓練後會自動建立 `experiments/exp_XXX_.../`，包含：

- `config.yaml`
- `cv_results.json`
- `model.pkl` / `preprocessor.pkl`
- `fold_*_model.pkl` / `fold_*_preprocessor.pkl`

---

## 6. 預測

### 6.1 指令用法

```bash
# 最新實驗 + default mode
python main_predict.py

# 指定實驗編號
python main_predict.py 81

# 指定模式
python main_predict.py 81 full
python main_predict.py 81 fold
```

輸出會寫到 `result/`，並同步更新 `result/prediction_stats_log.csv`。

### 6.2 重要提醒

`main_predict.py` 會先讀取 `configs/default_config.yaml`。
若你的環境尚未有此檔，請先建立：

```bash
cp configs/default_config_all.yaml configs/default_config.yaml
```

---

## 7. 批次訓練與匯總

### 7.1 Exp1 批次（4 模型 x 4 補值）

```bash
python exp1_batch_train_and_record.py
```

預設輸出：`experiments/experiment_record.csv`

### 7.2 Exp2 批次（4 模型 x 4 特徵法）

```bash
python exp2_batch_train_and_record.py
```

預設輸出：`experiments/experiment_record_exp2.csv`

### 7.3 Exp3 批次（文本方法）

```bash
python exp3_batch_train_and_record.py
```

預設輸出：

- `experiments/experiment_record_exp3.csv`
- `experiments/experiment_record_exp3_latest_comparison.csv`

### 7.4 進階參數（通用）

三個批次腳本都支援：

- `--config-glob`
- `--output-csv`
- `--main-script`
- `--stop-on-error`

範例：

```bash
python exp3_batch_train_and_record.py \
  --config-glob "configs/exp3_*.yaml" \
  --output-csv experiments/experiment_record_exp3.csv \
  --stop-on-error
```

---

## 8. 評估與實驗追蹤

核心評估指標：

- Accuracy
- F1
- Precision
- Recall

報告通常以 5-fold CV 的 `mean ± std` 呈現。

常用檔案：

- `experiments/experiment_log.csv`：主訓練歷史
- `experiments/experiment_record*.csv`：批次匯總
- `experiments/exp_XXX_.../cv_results.json`：單次詳情

---

## 9. Data Leakage 防範原則

本專案流程遵守：

1. 補值在 fold 內 fit/transform（非全資料先 fit）
2. 特徵轉換參數由訓練 fold 決定
3. SMOTE（若啟用）只在訓練 fold 執行

---

## 10. 相關文件

- [docs/experiment_guide.md](docs/experiment_guide.md)
- [docs/Experiment1_Imputation.md](docs/Experiment1_Imputation.md)
- [docs/Experiment2_Features.md](docs/Experiment2_Features.md)
- [docs/Experiment3_TextEmbedding.md](docs/Experiment3_TextEmbedding.md)
- [docs/TF-IDF_Parameters_Guide.md](docs/TF-IDF_Parameters_Guide.md)
- [training_workflow_main.md](training_workflow_main.md)
- [training_workflow_simple.md](training_workflow_simple.md)
- [start_train.md](start_train.md)

---

## 11. 團隊建議工作流

1. 先看對應的 ALL 文件確認 method 定義與目前基準。
2. 每次只改一類變因（補值 / 衍生特徵 / 文本策略）。
3. 執行批次腳本，更新 `experiment_record*.csv`。
4. 以 CV 指標決策，不用 full-train 指標直接下結論。
5. 寫回 docs 並同步更新 README（若 method 定義或流程有變更）。

---

如需我協助把 README 再拆成「快速版（1頁）」與「完整版（本文件）」，可以直接告訴我要保留哪些章節。