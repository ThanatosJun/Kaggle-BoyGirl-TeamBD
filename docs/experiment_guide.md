# 實驗參數調整指南

本文件詳細說明如何透過調整 `configs/default_config.yaml` 進行實驗，以及各個參數的意義與建議值。

---

## 📖 目錄

1. [配置檔案結構](#配置檔案結構)
2. [實驗配置](#實驗配置)
3. [資料配置](#資料配置)
4. [特徵配置](#特徵配置)
5. [預處理配置](#預處理配置)
6. [模型配置](#模型配置)
7. [網格搜尋配置](#網格搜尋配置)
8. [訓練配置](#訓練配置)
9. [預測配置](#預測配置)
10. [實驗範例](#實驗範例)
11. [調參建議](#調參建議)

---

## 配置檔案結構

`configs/default_config.yaml` 分為以下幾個主要區塊：

```yaml
experiment:      # 實驗名稱與描述
data:            # 資料路徑與目標欄位
features:        # 特徵欄位分類
preprocessing:   # 特徵處理策略
model:           # 模型選擇與參數
search:          # 網格搜尋配置
training:        # 訓練流程配置
prediction:      # 預測配置
```

---

## 實驗配置

### `experiment` 區塊

```yaml
experiment:
  name: "baseline"                                          # 實驗名稱
  description: "使用 catboost + class_weight 類別缺失值用 old 方式補值"  # 實驗描述
```

**參數說明**:

| 參數 | 說明 | 建議 |
|------|------|------|
| `name` | 實驗名稱，系統會自動加上編號（exp_001_baseline） | 使用有意義的名稱，例如：baseline, no_smote, tuned_lgbm, add_features |
| `description` | 實驗描述，記錄在 experiment_log.csv 中 | 詳細記錄改動內容、實驗目的與假設 |

**範例**:
```yaml
# 實驗 1: Baseline
experiment:
  name: "baseline"
  description: "初始實驗，使用 XGBoost 預設參數，啟用 class_weight"

# 實驗 2: 測試 SMOTE
experiment:
  name: "with_smote"
  description: "測試 SMOTE 過採樣，關閉 class_weight，比較與 baseline 差異"

# 實驗 3: 調整學習率
experiment:
  name: "tuned_lr"
  description: "降低 learning_rate 至 0.03，增加 iterations 至 800，觀察是否過擬合"
```

---

## 資料配置

### `data` 區塊

```yaml
data:
  train_path: "dataset/train.csv"
  test_path: "dataset/test.csv"
  drop_cols: ["id", "yt", "self_intro"]
  target_col: "gender"

  # 訓練映射：原始標籤 -> 模型內部標籤
  target_train_mapping:
    "男": 1
    "女": 0
    1: 1
    2: 0

  # 提交映射：模型內部標籤 -> 輸出標籤
  target_output_mapping:
    0: 2
    1: 1
```

**參數說明**:

| 參數 | 說明 | 通常不需要修改 |
|------|------|----------------|
| `train_path` | 訓練資料路徑 | ✅ |
| `test_path` | 測試資料路徑 | ✅ |
| `drop_cols` | 要移除的無用欄位 | ⚠️ 通常保持預設 |
| `target_col` | 目標欄位名稱 | ✅ |
| `target_train_mapping` | 訓練時的標籤映射 | ✅ |
| `target_output_mapping` | 預測輸出的標籤映射 | ✅ |

---

## 特徵配置

### `features` 區塊

```yaml
features:
  numeric_cols: ["height", "weight", "iq"]           # 一般數值特徵
  numeric_log_cols: ["fb_friends"]                   # 長尾分佈特徵（需 log 轉換）
  categorical_cols: ["star_sign", "phone_os"]        # 無序類別特徵
  ordinal_cols: ["sleepiness"]                       # 有序類別特徵（1-5）
```

**參數說明**:

| 參數 | 說明 | 處理方式 |
|------|------|----------|
| `numeric_cols` | 一般數值特徵 | 補值 → 剪裁（1%-99%）→ 標準化 |
| `numeric_log_cols` | 長尾分佈數值特徵 | 補值 → 移除負值 → log1p → 標準化 |
| `categorical_cols` | 無序類別特徵 | 補值 → One-Hot Encoding |
| `ordinal_cols` | 有序類別特徵 | 補值 → 保持數值型 |

**調整建議**:

如果你想嘗試不同的特徵處理方式：

```yaml
# 範例：將 iq 也當作長尾特徵處理
features:
  numeric_cols: ["height", "weight"]
  numeric_log_cols: ["fb_friends", "iq"]          # 加入 iq
  categorical_cols: ["star_sign", "phone_os"]
  ordinal_cols: ["sleepiness"]
```

> ⚠️ **注意**: 修改特徵分類後，需要檢查資料分佈是否適合該處理方式

---

## 預處理配置

### `preprocessing` 區塊

```yaml
preprocessing:
  # 缺失值補值模式：new | old
  imputation_mode: "new"

  # 數值補值策略
  numeric_imputer_strategy: "median"

  # 數值剪裁百分位數
  clipping_lower_percentile: 1
  clipping_upper_percentile: 99

  # log 特徵下界（避免負值）
  log_clip_min: 0

  # 標準化方法
  scaler: "standard"

  # OneHotEncoder 設定
  onehot_handle_unknown: "ignore"
  onehot_sparse_output: false
```

**參數說明**:

| 參數 | 選項 | 說明 | 建議範圍 |
|------|------|------|----------|
| `imputation_mode` | `new` / `old` | new: 類別缺失用 -1 常數<br>old: 類別缺失用 most_frequent | new（保留缺失資訊） |
| `numeric_imputer_strategy` | `median` / `mean` | 數值補值策略 | median（對 outlier 較穩健） |
| `clipping_lower_percentile` | 0-10 | 剪裁下界百分位數 | 1-5 |
| `clipping_upper_percentile` | 90-100 | 剪裁上界百分位數 | 95-99 |
| `log_clip_min` | 0 以上 | log1p 前的最小值 | 0 |
| `scaler` | `standard` / `minmax` / `robust` / `none` | 標準化方法 | standard |

**`scaler` 選項比較**:

| Scaler | 適用情況 | 優點 | 缺點 |
|--------|----------|------|------|
| `standard` | **一般情況（推薦）** | 標準常態分佈，適用大多數模型 | 受極端值影響 |
| `minmax` | 需要固定範圍（0-1） | 保持原始分佈形狀 | 對 outlier 敏感 |
| `robust` | 資料有極端值 | 使用中位數與 IQR，對 outlier 穩健 | 可能損失部分資訊 |
| `none` | 樹模型（XGBoost/LightGBM/Random Forest） | 保留原始數值 | 不適用線性模型 |

**實驗範例**:

```yaml
# 實驗 A: 使用 robust scaler（對 outlier 更穩健）
preprocessing:
  imputation_mode: "new"
  scaler: "robust"
  clipping_lower_percentile: 5      # 更激進的剪裁
  clipping_upper_percentile: 95

# 實驗 B: 不使用標準化（樹模型可嘗試）
preprocessing:
  imputation_mode: "new"
  scaler: "none"
  clipping_lower_percentile: 1
  clipping_upper_percentile: 99
```

---

## 模型配置

### `model` 區塊

```yaml
model:
  type: "catboost"    # xgboost | lightgbm | random_forest | catboost

  # 各模型參數（依據 type 選用）
  xgb_params: {...}
  lgbm_params: {...}
  random_forest_params: {...}
  catboost_params: {...}
```

### 6.1 XGBoost 參數

```yaml
model:
  type: "xgboost"
  xgb_params:
    objective: "binary:logistic"
    eval_metric: "logloss"
    learning_rate: 0.1              # 🔧 學習率
    max_depth: 6                    # 🔧 樹深度
    random_state: 42
    enable_categorical: false
```

**重要參數**:

| 參數 | 說明 | 建議範圍 | 調整方向 |
|------|------|----------|----------|
| `learning_rate` | 學習率，控制每步更新幅度 | 0.01-0.3 | 降低 → 更穩定但訓練慢<br>提高 → 訓練快但易過擬合 |
| `max_depth` | 樹的最大深度 | 3-10 | 增加 → 模型複雜度高<br>減少 → 防止過擬合 |
| `n_estimators` | 樹的數量（若有設定） | 100-1000 | 通常搭配 early_stopping 使用 |
| `subsample` | 樣本採樣比例 | 0.5-1.0 | 降低可防止過擬合 |
| `colsample_bytree` | 特徵採樣比例 | 0.5-1.0 | 降低可防止過擬合 |

**調參範例**:

```yaml
# 保守設定（防止過擬合）
model:
  type: "xgboost"
  xgb_params:
    learning_rate: 0.03             # 較低學習率
    max_depth: 4                    # 較淺樹
    subsample: 0.8
    colsample_bytree: 0.8

# 激進設定（提高模型複雜度）
model:
  type: "xgboost"
  xgb_params:
    learning_rate: 0.1
    max_depth: 8                    # 較深樹
    subsample: 1.0
    colsample_bytree: 1.0
```

### 6.2 LightGBM 參數

```yaml
model:
  type: "lightgbm"
  lgbm_params:
    objective: "binary"
    learning_rate: 0.03             # 🔧 學習率
    n_estimators: 500               # 🔧 樹的數量
    num_leaves: 15                  # 🔧 葉子節點數（LightGBM 特有）
    max_depth: 5                    # 🔧 樹深度
    min_child_samples: 30           # 🔧 葉節點最小樣本數
    feature_fraction: 0.8           # 🔧 特徵採樣比例
    bagging_fraction: 0.8           # 🔧 樣本採樣比例
    bagging_freq: 1
    reg_lambda: 2.0                 # 🔧 L2 正則化
    reg_alpha: 0.5                  # 🔧 L1 正則化
    verbosity: -1
    random_state: 42
```

**重要參數**:

| 參數 | 說明 | 建議範圍 | 調整方向 |
|------|------|----------|----------|
| `num_leaves` | 葉子節點最大數量（LightGBM 核心參數） | 15-63 | 增加 → 模型複雜度高<br>減少 → 防止過擬合 |
| `max_depth` | 樹的最大深度 | 3-8 | 控制過擬合 |
| `min_child_samples` | 葉節點最小樣本數 | 10-50 | 增加 → 防止過擬合 |
| `feature_fraction` | 每棵樹隨機選擇的特徵比例 | 0.5-1.0 | 降低可防止過擬合 |
| `bagging_fraction` | 每棵樹隨機選擇的樣本比例 | 0.5-1.0 | 降低可防止過擬合 |
| `reg_lambda` | L2 正則化強度 | 0-5 | 增加 → 防止過擬合 |
| `reg_alpha` | L1 正則化強度 | 0-5 | 增加 → 特徵稀疏化 |

**調參範例**:

```yaml
# 保守設定
model:
  type: "lightgbm"
  lgbm_params:
    learning_rate: 0.01             # 很低的學習率
    n_estimators: 800               # 增加樹的數量補償低學習率
    num_leaves: 10                  # 較少葉子
    min_child_samples: 50           # 較大最小樣本數
    reg_lambda: 5.0                 # 較強 L2 正則化

# 激進設定
model:
  type: "lightgbm"
  lgbm_params:
    learning_rate: 0.05
    n_estimators: 300
    num_leaves: 31                  # 較多葉子
    min_child_samples: 10
    reg_lambda: 0                   # 無正則化
```

### 6.3 Random Forest 參數

```yaml
model:
  type: "random_forest"
  random_forest_params:
    n_estimators: 400               # 🔧 樹的數量
    max_depth: 10                   # 🔧 樹深度
    min_samples_split: 5            # 🔧 分裂節點最小樣本數
    min_samples_leaf: 3             # 🔧 葉節點最小樣本數
    max_features: "sqrt"            # 🔧 特徵選擇策略
    class_weight: null              # 由 training.class_weight 統一控制
    n_jobs: -1
    random_state: 42
```

**重要參數**:

| 參數 | 說明 | 建議範圍 | 調整方向 |
|------|------|----------|----------|
| `n_estimators` | 樹的數量 | 200-800 | 增加 → 更穩定但訓練慢 |
| `max_depth` | 樹的最大深度 | 6-15 | 增加 → 模型複雜度高 |
| `min_samples_split` | 節點分裂所需最小樣本數 | 2-10 | 增加 → 防止過擬合 |
| `min_samples_leaf` | 葉節點最小樣本數 | 1-5 | 增加 → 防止過擬合 |
| `max_features` | 每次分裂考慮的特徵數 | `sqrt`, `log2`, `None` | sqrt 通常最佳 |

**`max_features` 選項**:

| 選項 | 說明 | 適用情況 |
|------|------|----------|
| `sqrt` | 使用 √n 個特徵 | **一般推薦，平衡效果** |
| `log2` | 使用 log₂(n) 個特徵 | 特徵數量很多時 |
| `None` | 使用所有特徵 | 特徵數量少或每個都重要 |

**調參範例**:

```yaml
# 高方差低偏差設定（深樹）
model:
  type: "random_forest"
  random_forest_params:
    n_estimators: 600               # 多樹
    max_depth: 15                   # 深樹
    min_samples_split: 2            # 容易分裂
    min_samples_leaf: 1

# 低方差高偏差設定（淺樹）
model:
  type: "random_forest"
  random_forest_params:
    n_estimators: 300
    max_depth: 6                    # 淺樹
    min_samples_split: 10           # 較難分裂
    min_samples_leaf: 5
```

### 6.4 CatBoost 參數

```yaml
model:
  type: "catboost"
  catboost_params:
    iterations: 500                 # 🔧 訓練輪數
    learning_rate: 0.05             # 🔧 學習率
    depth: 6                        # 🔧 樹深度
    l2_leaf_reg: 3                  # 🔧 L2 正則化
    border_count: 128               # 🔧 特徵分箱數量
    random_seed: 42
    verbose: False
    auto_class_weights: "Balanced"  # 🔧 自動類別權重
```

**重要參數**:

| 參數 | 說明 | 建議範圍 | 調整方向 |
|------|------|----------|----------|
| `iterations` | 訓練輪數 | 100-1000 | 增加 → 更好擬合但易過擬合 |
| `learning_rate` | 學習率 | 0.01-0.3 | 降低 → 更穩定 |
| `depth` | 樹的深度 | 4-10 | 增加 → 模型複雜度高 |
| `l2_leaf_reg` | L2 正則化 | 1-10 | 增加 → 防止過擬合 |
| `border_count` | 數值特徵的分箱數量 | 32-255 | 增加 → 更精細但訓練慢 |
| `auto_class_weights` | 類別權重策略 | `Balanced` / `None` | Balanced 自動處理不平衡 |

**調參範例**:

```yaml
# 保守設定
model:
  type: "catboost"
  catboost_params:
    iterations: 800
    learning_rate: 0.03             # 較低學習率
    depth: 4                        # 較淺樹
    l2_leaf_reg: 5                  # 較強正則化

# 激進設定
model:
  type: "catboost"
  catboost_params:
    iterations: 300
    learning_rate: 0.1              # 較高學習率
    depth: 8                        # 較深樹
    l2_leaf_reg: 1                  # 較弱正則化
```

---

## 網格搜尋配置

### `search` 區塊

```yaml
search:
  enabled: false                    # 🔧 是否啟用網格搜尋
  param_grid_mode: "quick"          # 🔧 quick | full
  metric: "f1"                      # 🔧 評估指標
```

**參數說明**:

| 參數 | 說明 | 選項 |
|------|------|------|
| `enabled` | 是否啟用自動網格搜尋 | true（慢但找最佳參數）/ false（快速使用固定參數） |
| `param_grid_mode` | 搜尋範圍 | `quick`（少量組合）/ `full`（完整搜尋） |
| `metric` | 調參目標指標 | `accuracy` / `f1` / `precision` / `recall` |

**網格搜尋範例**:

在 `model.param_grid` 或 `model.param_grid_quick` 中定義搜尋空間：

```yaml
model:
  type: "xgboost"
  # ... xgb_params ...

  param_grid:
    xgboost:
      learning_rate: [0.03, 0.05, 0.1]
      max_depth: [4, 6, 8]
    # ... 其他模型的網格 ...

search:
  enabled: true
  param_grid_mode: "full"
  metric: "f1"
```

**使用時機**:

- ✅ **啟用網格搜尋**: 新模型初次調參、有充足時間、希望找到最佳組合
- ❌ **不啟用網格搜尋**: 快速實驗、固定參數對比、時間有限

---

## 訓練配置

### `training` 區塊

```yaml
training:
  n_splits: 5                       # 🔧 交叉驗證折數
  use_smote: false                  # 🔧 是否使用 SMOTE 過採樣
  class_weight: "balanced"          # 🔧 類別權重
  random_state: 42
  save_dir: "experiments"

  # SMOTE 參數（use_smote=true 時生效）
  smote_params:
    random_state: 42
```

**參數說明**:

| 參數 | 說明 | 建議 |
|------|------|------|
| `n_splits` | K-Fold CV 折數 | 3-10（5 為常見值） |
| `use_smote` | 是否使用 SMOTE 過採樣 | **優先使用 class_weight，SMOTE 可能造成過擬合** |
| `class_weight` | 類別權重策略 | `balanced`（推薦）/ `null` / `{0: 1.5, 1: 1.0}` |
| `random_state` | 隨機種子 | 保持固定以利結果重現 |

**`class_weight` vs `use_smote`**:

| 方法 | 優點 | 缺點 | 建議 |
|------|------|------|------|
| **class_weight** | 不改變資料分佈，訓練快，泛化好 | 需模型支援 | ✅ **優先使用** |
| **SMOTE** | 增加少數類樣本，資料更平衡 | 可能造成過擬合，訓練慢 | 僅在 class_weight 無效時使用 |

**實驗範例**:

```yaml
# 實驗 A: 使用 class_weight（推薦）
training:
  n_splits: 5
  use_smote: false
  class_weight: "balanced"

# 實驗 B: 使用 SMOTE
training:
  n_splits: 5
  use_smote: true
  class_weight: null              # 關閉 class_weight

# 實驗 C: 手動設定類別權重（假設女性為少數類）
training:
  n_splits: 5
  use_smote: false
  class_weight:
    0: 2.0                        # 女性權重 2.0
    1: 1.0                        # 男性權重 1.0
```

---

## 預測配置

### `prediction` 區塊

```yaml
prediction:
  default_mode: "full"              # full | fold
  output_dir: "result"
```

**參數說明**:

| 參數 | 說明 | 選項 |
|------|------|------|
| `default_mode` | 預設預測模式 | `full`（整個訓練集訓練的模型）/ `fold`（5 個模型集成） |
| `output_dir` | 預測結果保存目錄 | 通常不需要修改 |

**預測模式比較**:

| 模式 | 說明 | 優點 | 缺點 |
|------|------|------|------|
| **full** | 使用整個訓練集訓練的單一模型 | 訓練資料更多，模型更穩定 | 無法評估泛化性能 |
| **fold** | 使用 5 個 fold 模型的平均預測 | 模型集成，泛化性能可能更好 | 複雜度高，可能效果相近 |

---

## 實驗範例

### 範例 1: Baseline 實驗

**目標**: 建立初始 baseline，使用 XGBoost + class_weight

```yaml
experiment:
  name: "baseline"
  description: "初始 baseline，XGBoost 預設參數 + class_weight"

model:
  type: "xgboost"
  xgb_params:
    learning_rate: 0.1
    max_depth: 6

training:
  n_splits: 5
  use_smote: false
  class_weight: "balanced"

search:
  enabled: false
```

執行：
```bash
python main_train.py
```

---

### 範例 2: 測試不同模型

**目標**: 比較 XGBoost, LightGBM, Random Forest, CatBoost 效果

**步驟**:

1. **實驗 2: LightGBM**
```yaml
experiment:
  name: "lightgbm_baseline"
  description: "測試 LightGBM 效果"

model:
  type: "lightgbm"
```

2. **實驗 3: Random Forest**
```yaml
experiment:
  name: "rf_baseline"
  description: "測試 Random Forest 效果"

model:
  type: "random_forest"
```

3. **實驗 4: CatBoost**
```yaml
experiment:
  name: "catboost_baseline"
  description: "測試 CatBoost 效果"

model:
  type: "catboost"
```

執行每個實驗後比較結果：
```python
import pandas as pd
df = pd.read_csv('experiments/experiment_log.csv')
print(df[['exp_id', 'name', 'mean_f1', 'mean_accuracy']].sort_values('mean_f1', ascending=False))
```

---

### 範例 3: 調整學習率與樹深度

**目標**: 找到最佳的 learning_rate 和 max_depth 組合（以 LightGBM 為例）

**方法 1: 手動調整**

```yaml
# 實驗 5: 降低學習率
experiment:
  name: "lgbm_lr_003"
  description: "降低學習率至 0.03，增加 iterations 補償"

model:
  type: "lightgbm"
  lgbm_params:
    learning_rate: 0.03
    n_estimators: 800

# 實驗 6: 增加樹深度
experiment:
  name: "lgbm_depth_7"
  description: "增加樹深度至 7"

model:
  type: "lightgbm"
  lgbm_params:
    max_depth: 7
    num_leaves: 31
```

**方法 2: 網格搜尋**

```yaml
experiment:
  name: "lgbm_grid_search"
  description: "自動搜尋最佳 learning_rate 和 num_leaves"

model:
  type: "lightgbm"
  param_grid:
    lightgbm:
      learning_rate: [0.03, 0.05]
      num_leaves: [15, 31]
      max_depth: [4, 5, 6]

search:
  enabled: true
  param_grid_mode: "full"
  metric: "f1"
```

---

### 範例 4: 測試 SMOTE vs class_weight

**目標**: 比較 SMOTE 與 class_weight 對模型效果的影響

```yaml
# 實驗 7: class_weight（推薦）
experiment:
  name: "class_weight"
  description: "使用 class_weight 處理不平衡"

training:
  use_smote: false
  class_weight: "balanced"

# 實驗 8: SMOTE
experiment:
  name: "smote"
  description: "使用 SMOTE 處理不平衡"

training:
  use_smote: true
  class_weight: null

# 實驗 9: 兩者都不使用（對照組）
experiment:
  name: "no_balance"
  description: "不處理不平衡，作為對照組"

training:
  use_smote: false
  class_weight: null
```

---

### 範例 5: 調整特徵處理方式

**目標**: 測試不同的缺失值處理策略

```yaml
# 實驗 10: 新版補值（保留缺失資訊）
experiment:
  name: "imputation_new"
  description: "類別缺失用 -1 常數，保留缺失資訊"

preprocessing:
  imputation_mode: "new"

# 實驗 11: 舊版補值（用眾數填補）
experiment:
  name: "imputation_old"
  description: "類別缺失用 most_frequent"

preprocessing:
  imputation_mode: "old"
```

---

### 範例 6: 調整數值剪裁範圍

**目標**: 測試更激進的 outlier 剪裁

```yaml
# 實驗 12: 激進剪裁
experiment:
  name: "aggressive_clipping"
  description: "使用 5%-95% 剪裁，移除更多極端值"

preprocessing:
  clipping_lower_percentile: 5
  clipping_upper_percentile: 95

# 實驗 13: 無剪裁
experiment:
  name: "no_clipping"
  description: "不剪裁極端值，保留原始分佈"

preprocessing:
  clipping_lower_percentile: 0
  clipping_upper_percentile: 100
```

---

## 調參建議

### 10.1 模型選擇建議

| 模型 | 適用情況 | 優點 | 缺點 |
|------|----------|------|------|
| **XGBoost** | 通用性強，表格資料首選 | 效果穩定，調參容易 | 訓練速度中等 |
| **LightGBM** | 資料量大、特徵多 | 訓練速度快，記憶體效率高 | 小資料集易過擬合 |
| **Random Forest** | 需要穩定基線 | 不易過擬合，魯棒性高 | 效果可能不如 Boosting |
| **CatBoost** | 有類別特徵、不平衡資料 | 自動處理類別特徵，內建不平衡處理 | 訓練速度較慢 |

### 10.2 調參優先級

**階段 1: 選擇模型**
1. 依序嘗試 XGBoost, LightGBM, CatBoost, Random Forest
2. 選擇 F1-Score 最高的模型作為基準

**階段 2: 調整資料處理**（通常提升較大）
1. 測試 `class_weight` vs `SMOTE` vs 無平衡處理
2. 測試 `imputation_mode: new` vs `old`
3. 調整 `clipping_percentile`（影響 outlier 處理）

**階段 3: 調整模型參數**（精細調優）
1. **學習率與樹數量**: 降低 `learning_rate`，增加 `n_estimators/iterations`
2. **樹深度**: 調整 `max_depth` / `num_leaves`
3. **正則化**: 調整 `reg_lambda`, `reg_alpha`, `l2_leaf_reg`
4. **採樣比例**: 調整 `subsample`, `feature_fraction`, `bagging_fraction`

**階段 4: 網格搜尋**
1. 使用 `search.enabled: true` 自動搜尋最佳組合
2. 先用 `param_grid_mode: quick` 快速測試
3. 再用 `param_grid_mode: full` 完整搜尋

### 10.3 避免過擬合的技巧

1. **降低模型複雜度**
   - 減少 `max_depth` / `num_leaves`
   - 增加 `min_samples_leaf` / `min_child_samples`

2. **增加正則化**
   - 提高 `reg_lambda`, `reg_alpha`, `l2_leaf_reg`

3. **使用集成方法**
   - 增加 `n_estimators` / `iterations`
   - 使用 `fold` 模式預測（5 個模型集成）

4. **降低學習率**
   - 降低 `learning_rate`，增加訓練輪數

5. **資料增強**
   - 使用更激進的 clipping（5%-95%）
   - 使用 `robust` scaler

### 10.4 提升 F1-Score 的技巧

1. **處理類別不平衡**
   - 優先使用 `class_weight: "balanced"`
   - 若無效再嘗試 `use_smote: true`
   - 手動調整類別權重（如 `{0: 2.0, 1: 1.0}`）

2. **優化閾值**（需要修改程式碼）
   - 預測機率後，調整分類閾值（預設 0.5）
   - 尋找使 F1 最大的閾值

3. **特徵工程**
   - 測試不同的缺失值處理策略
   - 嘗試特徵交互作用（需修改 `src/features.py`）

4. **模型集成**
   - 使用 `fold` 模式（5 個模型平均）
   - 或訓練多個不同模型後手動集成

---

## 🎯 快速實驗 Checklist

開始新實驗前，檢查以下項目：

- [ ] 修改 `experiment.name` 為有意義的名稱
- [ ] 填寫 `experiment.description` 詳細說明改動
- [ ] 確認只改變一個變量（方便分析影響）
- [ ] 確認 `random_state` 保持固定（結果可重現）
- [ ] 儲存修改後的 `default_config.yaml`
- [ ] 執行 `python main_train.py`
- [ ] 查看 `experiments/experiment_log.csv` 比較結果

---

## 📞 相關資源

- **專案 README**: [../README.md](../README.md)
- **訓練流程說明**: [training_workflow_main.md](training_workflow_main.md)
- **快速開始指南**: [start_train.md](start_train.md)

---

**祝實驗順利！🚀**
