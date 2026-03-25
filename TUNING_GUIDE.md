# 🔧 參數調整指南

本文件說明 `configs/default_config.yaml` 中標註為 **🔧 [可調整]** 和 **🔧 [重要]** 的參數如何影響模型表現。

---

## 📊 參數優先級

### ⭐⭐⭐ 高優先級（對模型影響最大）

#### 1. **模型選擇** (`model.type`)
```yaml
type: "random_forest"  # 或 "xgboost" | "lightgbm"
```
- **Random Forest**: 目前表現最好 (F1: 0.9196)，適合小樣本
- **LightGBM**: 訓練快，處理類別特徵好
- **XGBoost**: 穩定性好，但需調參

#### 2. **類別不平衡處理** (`training.class_weight`)
```yaml
class_weight: balanced  # 推薦
# class_weight: null    # 不進行平衡
# class_weight: {0: 2.0, 1: 1.0}  # 自訂權重（女生權重加倍）
```
**建議**: 保持 `balanced`，效果已經很好

#### 3. **缺失值處理策略** (`preprocessing`)
```yaml
# 目前使用 "-1" 字串補值（保留缺失資訊）
categorical_imputer_strategy: "constant"
categorical_fill_value: "-1"  # ⚠️ 必須是字串，因為原始資料是字串型

# 切換回眾數補值（對比實驗）
# categorical_imputer_strategy: "most_frequent"
```
**建議**: 保持 "-1" 補值，實驗對比效果

---

### ⭐⭐ 中優先級（需要調整以避免過擬合）

#### Random Forest 專屬參數

```yaml
random_forest_params:
  n_estimators: 400       # 🔧 樹的數量（200-800）
  max_depth: 10           # 🔧 最大深度（6-15）
  min_samples_split: 5    # 🔧 分裂最小樣本（2-10）
  min_samples_leaf: 3     # 🔧 葉節點最小樣本（1-5）
```

**調整方向**:
- **過擬合** (train 準確度高但 CV 準確度低):
  - ⬇️ 降低 `max_depth` (10 → 8)
  - ⬆️ 增加 `min_samples_leaf` (3 → 5)

- **欠擬合** (train 和 CV 準確度都低):
  - ⬆️ 增加 `max_depth` (10 → 12)
  - ⬆️ 增加 `n_estimators` (400 → 600)

#### LightGBM 專屬參數

```yaml
lgbm_params:
  learning_rate: 0.03     # 🔧 學習率（0.01-0.1）
  n_estimators: 500       # 🔧 樹的數量（100-1000）
  num_leaves: 15          # 🔧 葉子數量（15-63）
  max_depth: 5            # 🔧 最大深度（3-8）
  min_child_samples: 30   # 🔧 葉節點最小樣本（10-50）
```

**調整方向**:
- **過擬合**:
  - ⬇️ 降低 `num_leaves` (15 → 10)
  - ⬆️ 增加 `min_child_samples` (30 → 40)
  - ⬆️ 增加正則化 `reg_lambda` (2.0 → 3.0)

- **欠擬合**:
  - ⬆️ 增加 `num_leaves` (15 → 31)
  - ⬆️ 增加 `n_estimators` (500 → 800)

#### XGBoost 專屬參數

```yaml
xgb_params:
  learning_rate: 0.1      # 🔧 學習率（0.01-0.3）
  max_depth: 6            # 🔧 最大深度（3-10）
```

**調整方向**:
- **過擬合**: ⬇️ 降低 `max_depth` (6 → 4)
- **欠擬合**: ⬆️ 增加 `max_depth` (6 → 8)

#### CatBoost 專屬參數

```yaml
catboost_params:
  iterations: 500         # 🔧 訓練輪數（100-1000）
  learning_rate: 0.05     # 🔧 學習率（0.01-0.3）
  depth: 6                # 🔧 最大深度（4-10）
  l2_leaf_reg: 3          # 🔧 L2 正則化（1-10）
  auto_class_weights: "Balanced"  # 🔧 自動類別平衡
```

**調整方向**:
- **過擬合**:
  - ⬇️ 降低 `depth` (6 → 4)
  - ⬆️ 增加 `l2_leaf_reg` (3 → 5)
  - ⬇️ 降低 `iterations` (500 → 300)

- **欠擬合**:
  - ⬆️ 增加 `depth` (6 → 8)
  - ⬆️ 增加 `iterations` (500 → 800)

**CatBoost 特色**:
- ✅ **原生支援類別特徵**：不需要 One-Hot Encoding
- ✅ **內建缺失值處理**：比其他模型更優
- ✅ **自動處理類別不平衡**：`auto_class_weights="Balanced"`
- ✅ **預設參數就很強**：較少需要調參

---

### ⭐ 低優先級（微調用）

#### 數據預處理

```yaml
clipping_lower_percentile: 1   # 🔧 下界百分位（1-5）
clipping_upper_percentile: 99  # 🔧 上界百分位（95-99）
scaler: "standard"             # 🔧 或 "minmax" | "robust"
```

**建議**: 通常不需要調整

#### 網格搜尋

```yaml
search:
  enabled: false          # 🔧 true=自動調參（慢），false=固定參數（快）
  param_grid_mode: quick  # 🔧 quick 或 full
```

**使用時機**:
- 第一次訓練: `enabled: false` (快速驗證)
- 找到好方向後: `enabled: true` + `param_grid_mode: quick`
- 最終衝榜: `enabled: true` + `param_grid_mode: full`

---

## 🎯 實驗策略建議

### 階段 1: Baseline（已完成）
✅ 目前最佳: Random Forest + class_weight=balanced + -1 補值
- F1-Score: 0.9196

### 階段 2: 微調 Random Forest（推薦下一步）

```yaml
# 實驗 A: 增加樹的數量
n_estimators: 600

# 實驗 B: 調整深度和樣本數
max_depth: 12
min_samples_leaf: 2

# 實驗 C: 啟用網格搜尋
search:
  enabled: true
  param_grid_mode: quick
```

### 階段 3: 嘗試其他模型

```yaml
# 實驗 D: LightGBM
model:
  type: "lightgbm"

# 實驗 E: 對比補值方式
preprocessing:
  categorical_imputer_strategy: "most_frequent"  # 舊方法
```

### 階段 4: Ensemble（進階）
- 組合 Random Forest + LightGBM + XGBoost
- 使用 Voting 或 Stacking

---

## 📝 調參記錄模板

每次實驗建議記錄：

```yaml
experiment:
  name: "RF_tuned_depth12"
  description: "增加 max_depth 到 12，測試是否能提升 F1-score"

# 改動的參數：
random_forest_params:
  max_depth: 12  # 從 10 改為 12

# 結果：
# - F1-Score: 0.9xxx (vs baseline 0.9196)
# - 結論: [提升/下降/持平]
```

---

## 🚨 常見錯誤

### ❌ 錯誤 1: 同時使用 SMOTE + class_weight
```yaml
use_smote: true
class_weight: balanced  # 重複加權！
```
**解決**: 選擇其一，推薦 `class_weight`

### ❌ 錯誤 2: 網格搜尋參數範圍太大
```yaml
# 不好：27 種組合需要 30-50 分鐘
param_grid:
  random_forest:
    n_estimators: [300, 400, 600]
    max_depth: [8, 10, 12]
    min_samples_leaf: [2, 3, 5]
```
**解決**: 先用 `quick` 模式測試

### ❌ 錯誤 3: random_state 不固定
```yaml
random_state: 99  # 不同實驗用不同 seed
```
**解決**: 保持 `random_state: 42` 以便對比

---

## 💡 快速檢查清單

執行新實驗前，確認：
- [ ] 修改了 `experiment.name` 和 `description`
- [ ] 只改變一個變量（方便分析影響）
- [ ] `random_state: 42` 保持不變
- [ ] 記錄了預期效果和實際結果
- [ ] Git commit 保存了設定檔

---

**祝調參順利！🚀**

有問題請參考 [README.md](README.md) 或 [training_workflow_main.md](training_workflow_main.md)
