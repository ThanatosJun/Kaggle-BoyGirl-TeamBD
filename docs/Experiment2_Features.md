# 🎉 Experiment 2 - 衍生特徵實驗 | 完整報告

## 📋 實驗概要

**時間區間**: 2026-03-25 20:49 ~ 2026-03-26 14:28  
**實驗總數**: 4 個（基礎特徵 × 衍生特徵組合）  
**模型**: CatBoost (iterations=500, depth=6, lr=0.05)  
**補值策略**: 全局 Median（Exp1 最優方法）  
**評估方式**: 5-Fold 交叉驗證  
**前提**: 不做超參數調整，純粹比較特徵工程效果

---

## 🏆 最終排名 & 結果

| 排名 | 實驗 | 特徵組合 | F1-Score | Accuracy | Precision | Recall | 評分 |
|------|------|---------|----------|----------|-----------|--------|------|
| 🥇 | Exp 2-3 | 基礎 + BMI | **0.9276 ± 0.0259** | **0.8914 ± 0.0387** | 0.9223 ± 0.0246 | 0.9336 ± 0.0365 | ⭐⭐⭐⭐⭐ |
| 🥈 | Exp 2-2 | 基礎 + Ratio | 0.9259 ± 0.0255 | 0.8890 ± 0.0382 | 0.9218 ± 0.0228 | 0.9305 ± 0.0340 | ⭐⭐⭐⭐ |
| 3️⃣ | Exp 2-4 | 基礎 + PI | 0.9247 ± 0.0231 | 0.8866 ± 0.0343 | 0.9164 ± 0.0197 | **0.9336 ± 0.0336** | ⭐⭐⭐ |
| 4️⃣ | Exp 2-1 | 僅基礎特徵 | 0.9238 ± 0.0274 | 0.8866 ± 0.0403 | **0.9249 ± 0.0285** | 0.9241 ± 0.0441 | ⭐⭐ |

---

## 📐 特徵定義

### 基礎特徵（所有實驗共用）
| 類型 | 特徵 |
|------|------|
| Numeric | height, weight, iq |
| Numeric (log-tail) | fb_friends |
| Categorical | star_sign, phone_os |
| Ordinal | sleepiness |

### 衍生特徵

#### Weight/Height Ratio（Exp 2-2）
$$\text{ratio} = \frac{\text{weight}}{\text{height}}$$

- 單位：kg/cm
- 計算時機：補值之後、Clipping 之前
- 加入 `numeric_cols` 接受標準化處理

#### BMI（Exp 2-3）
$$\text{BMI} = \frac{\text{weight}}{(\text{height} / 100)^2}$$

- 單位：kg/m²
- 標準 BMI 公式：體重(kg) / 身高(m)²
- 計算時機：補值之後、Clipping 之前
- 加入 `numeric_cols` 接受標準化處理

#### Ponderal Index（Exp 2-4）
$$\text{PI} = \frac{\text{weight}}{(\text{height} / 100)^3}$$

- 單位：kg/m³
- 又稱 Rohrer's Index，是 BMI 的三次方版本，對身高差異的修正更線性
- 計算時機：補值之後、Clipping 之前
- 加入 `numeric_cols` 接受標準化處理

---

## 🔍 逐項詳細分析

### 1️⃣ Exp 2-1：僅基礎特徵（對照組）

```
F1-Score:  0.9238 ± 0.0274
Accuracy:  0.8866 ± 0.0403
Precision: 0.9249 ± 0.0285
Recall:    0.9241 ± 0.0441
```

**Per-Fold 結果**:
| Fold | Accuracy | F1-Score |
|------|----------|----------|
| 1 | 0.8471 | 0.8943 |
| 2 | 0.9412 | 0.9612 |
| 3 | 0.8353 | 0.8923 |
| 4 | 0.8929 | 0.9256 |
| 5 | 0.9167 | 0.9457 |

- 這是 Exp 1 最優模型（Method 0）的複現，做為 Exp 2 的基準線
- F1 std = 0.0274（合理穩定性）
- Recall std = 0.0441（最高不穩定性，因無衍生特徵時模型更依賴模糊邊界）

---

### 2️⃣ Exp 2-2：基礎特徵 + Weight/Height Ratio

```
F1-Score:  0.9259 ± 0.0255  (+0.0021 vs 對照)
Accuracy:  0.8890 ± 0.0382  (+0.0024 vs 對照)
Precision: 0.9218 ± 0.0228  (-0.0031 vs 對照)
Recall:    0.9305 ± 0.0340  (+0.0064 vs 對照)
```

**Per-Fold 結果**:
| Fold | Accuracy | F1-Score |
|------|----------|----------|
| 1 | 0.8588 | 0.9032 |
| 2 | 0.9412 | 0.9612 |
| 3 | 0.8353 | 0.8923 |
| 4 | 0.8929 | 0.9280 |
| 5 | 0.9167 | 0.9449 |

**分析**：
- F1 和 Recall 均略有提升，Precision 略降
- Ratio 提供了 weight 和 height 的「交互作用」信號
- std 降低（F1: 0.0274→0.0255）：加入 ratio 後模型更穩定
- 優勢：簡單計算，模型可直接獲得「體態比例」信息而不需組合兩個特徵

---

### 3️⃣ Exp 2-3：基礎特徵 + BMI（**最優 F1 & Accuracy**）

```
F1-Score:  0.9276 ± 0.0259  (+0.0038 vs 對照)  ← 最高 F1
Accuracy:  0.8914 ± 0.0387  (+0.0048 vs 對照)  ← 最高 Accuracy
Precision: 0.9223 ± 0.0246  (-0.0026 vs 對照)
Recall:    0.9336 ± 0.0365  (+0.0095 vs 對照)  ← 並列最高 Recall
```

**Per-Fold 結果**:
| Fold | Accuracy | F1-Score |
|------|----------|----------|
| 1 | 0.8588 | 0.9032 |
| 2 | 0.9176 | 0.9457 |
| 3 | 0.8353 | 0.8923 |
| 4 | 0.9048 | 0.9355 |
| 5 | 0.9405 | 0.9612 |

**分析**：
- 最高 F1、Accuracy 和 Recall
- BMI = weight / (height/100)² 是一個非線性組合，比線性 ratio 更能捕捉體態特征
- BMI 本身具有醫學意義（underweight / normal / overweight / obese），與性別分佈強相關
- Recall 提升最明顯（+0.0095），代表誤診率進一步降低

---

### 4️⃣ Exp 2-4：基礎特徵 + Ponderal Index（**最穩定 std**）

```
F1-Score:  0.9247 ± 0.0231  (+0.0009 vs 對照)  ← 最低 F1 std
Accuracy:  0.8866 ± 0.0343  (=0.0000 vs 對照)  ← 最低 Accuracy std
Precision: 0.9164 ± 0.0197  (-0.0085 vs 對照)  ← 最低 Precision std
Recall:    0.9336 ± 0.0336  (+0.0095 vs 對照)  ← 並列最高 Recall，最低 Recall std
```

**Per-Fold 結果**:
| Fold | Accuracy | F1-Score |
|------|----------|----------|
| 1 | 0.8588 | 0.9032 |
| 2 | 0.9294 | 0.9538 |
| 3 | 0.8353 | 0.8923 |
| 4 | 0.9048 | 0.9365 |
| 5 | 0.9048 | 0.9375 |

**分析**：
- 所有指標的 std 均為四個實驗中最低，代表最穩定的預測行為
- PI = weight / (height/100)³ 是三次方指數，對身高的縮放更激進，產生不同的特徵分佈
- Mean F1 略低於 BMI，但 Precision 也下降最多（-0.0085），呈現「高 Recall 低 Precision」傾向
- 高穩定性使其在 Recall 上與 BMI 並列最高（0.9336）

---

## 📊 特徵效益比較

### 增益方向（vs 對照組 Exp 2-1）

| 指標 | +Ratio 增益 | +BMI 增益 | +PI 增益 |
|------|------------|----------|---------|
| F1-Score | +0.0021 (+0.23%) | **+0.0038 (+0.41%)** | +0.0009 (+0.10%) |
| Accuracy | +0.0024 (+0.27%) | **+0.0048 (+0.54%)** | +0.0000 (+0.00%) |
| Precision | -0.0031 (-0.34%) | -0.0026 (-0.28%) | -0.0085 (-0.92%) |
| Recall | +0.0064 (+0.69%) | **+0.0095 (+1.03%)** | **+0.0095 (+1.03%)** |

### 穩定性（std）

| 指標 | 對照 std | +Ratio std | +BMI std | +PI std |
|------|---------|-----------|---------|--------|
| F1-Score | 0.0274 | 0.0255 | 0.0259 | **0.0231** |
| Accuracy | 0.0403 | 0.0382 | 0.0387 | **0.0343** |
| Precision | 0.0285 | 0.0228 | 0.0246 | **0.0197** |
| Recall | 0.0441 | 0.0340 | 0.0365 | **0.0336** |

- BMI 在平均性能（F1、Accuracy）上領先
- PI 在所有 std 指標上最低，為最穩定的預測器
- Ratio 在性能與穩定性之間取得均衡
- PI 和 BMI 均達到相同的最高 Recall（0.9336），但 PI 的 Precision 代價更高

---

## 💡 結論與建議

### 核心發現

1. **BMI 總體最優**：加入 BMI 帶來最高的 F1（0.9276）與 Accuracy（0.8914）
2. **PI 最穩定**：所有 std 指標最低，但平均 Precision 代價最大（-0.0085 vs 對照）
3. **PI = BMI 的 Recall**：兩者均達到最高 Recall（0.9336），但途徑不同
4. **所有衍生特徵均降低 std**：相較基準線，加入任何體態特徵都能讓預測更一致
5. **增益幅度有限**：最大 F1 增益約 +0.4%，反映 CatBoost 本身已能隱式學習交互

### BMI vs Ponderal Index：關鍵區別

| 維度 | BMI (²) | Ponderal Index (³) |
|------|---------|-------------------|
| 縮放指數 | 二次方 | 三次方 |
| 對高/矮體型的修正 | 對高個子體重略高估 | 對身高縮放更線性 |
| 臨床應用 | 廣泛使用（成人體重評估） | 主要用於嬰兒、運動員 |
| 本實驗性能 | **F1 最高** | **std 最低** |
| Precision | 較高（0.9223） | 較低（0.9164） |

### 推薦方案

| 用途 | 推薦配置 |
|------|---------|
| **純性能優先** | +BMI (Exp 2-3)：F1=0.9276，Accuracy 最高 |
| **穩定性優先** | +PI (Exp 2-4)：所有 std 最低，預測最一致 |
| **Recall 優先（低誤診）** | +BMI 或 +PI（並列）：Recall=0.9336 |
| **簡潔基準** | 基礎特徵 (Exp 2-1)：差異微小（<0.5%），維護成本最低 |

> **結論**：若後續繼續 tuning，建議沿用 **+BMI** 配置（Exp 2-3），因為 BMI 的平均性能最高；若重視跨 fold 的一致性，**+PI** 是更安全的選擇。

---

## 📦 交付文件

### 新增/修改檔案
- ✅ `src/features.py` - 添加 `engineer_features()` 函數（支持 ratio、BMI & PI）
- ✅ `src/evaluate.py` - 在每個 CV fold 補值後調用特徵工程
- ✅ `main_train.py` - 在全量訓練補值後調用特徵工程
- ✅ `configs/exp2_method0_base_features.yaml` - Exp 2-1 配置
- ✅ `configs/exp2_method0_with_ratio.yaml` - Exp 2-2 配置
- ✅ `configs/exp2_method0_with_bmi.yaml` - Exp 2-3 配置
- ✅ `configs/exp2_method0_with_pi.yaml` - Exp 2-4 配置（新增）

### 實驗存檔路徑
- `experiments/exp_002_exp2_method0_base_features/`
- `experiments/exp_003_exp2_method0_with_ratio/`
- `experiments/exp_004_exp2_method0_with_bmi/`
- `experiments/exp_001_exp2_method0_with_pi/`（新增）

---

## 📞 技術細節

### 特徵工程實作位置

```python
# src/features.py
def engineer_features(X: pd.DataFrame, config: dict) -> pd.DataFrame:
    feat_cfg = config.get('features', {})
    X = X.copy()
    if feat_cfg.get('add_weight_height_ratio', False):
        height_safe = X['height'].replace(0, np.nan).fillna(1e-6)
        X['weight_height_ratio'] = X['weight'] / height_safe
    if feat_cfg.get('add_bmi', False):
        height_m = (X['height'] / 100.0).replace(0, np.nan).fillna(1e-6)
        X['bmi'] = X['weight'] / (height_m ** 2)
    if feat_cfg.get('add_ponderal_index', False):
        height_m = (X['height'] / 100.0).replace(0, np.nan).fillna(1e-6)
        X['ponderal_index'] = X['weight'] / (height_m ** 3)
    return X
```

### 呼叫流程（無數據洩漏）

```
[每個 CV Fold]:
  1. 分割 train/val
  2. Imputer.fit_transform(X_train) → Imputer.transform(X_val)
  3. engineer_features(X_train, config) ← 補值後計算           ← 新增
  4. engineer_features(X_val, config)   ← 補值後計算           ← 新增
  5. Preprocessor.fit_transform(X_train_eng)
  6. Preprocessor.transform(X_val_eng)
  7. Model.fit() / Model.predict()
```

### 配置格式

```yaml
# +PI 配置範例（exp2_method0_with_pi.yaml）
features:
  numeric_cols: ["height", "weight", "iq", "ponderal_index"]
  add_weight_height_ratio: false
  add_bmi: false
  add_ponderal_index: true                                     # flag 控制
```

### 實驗設定
```yaml
model: catboost
catboost_params:
  iterations: 500
  depth: 6
  learning_rate: 0.05
  auto_class_weights: Balanced
imputation: method0 (global Median)
clipping: 1%-99% percentile
n_splits: 5-fold CV
search_enabled: false
```

---

## 🚀 後續建議

### 下一步選項
1. **超參數優化**：在 +BMI 特徵基礎上做 grid search（目前所有 Exp2 都是 base model，無 tuning）
2. **組合特徵**：同時加入 BMI + PI，觀察是否有協同效應（風險：兩者高度相關，可能互消）
3. **其他衍生特徵**：探索 IQ × sleepiness 交互、年齡估算等社交行為維度
4. **特徵重要性分析**：使用 CatBoost 內建的 feature_importance 確認 BMI / PI 排名
5. **PI 的適用場景**：若後續需要跨 fold 高一致性（如線上推論穩定性），PI 是更佳選擇

---

*報告更新時間: 2026-03-26 14:28*  
*新增實驗: Exp 2-4（+Ponderal Index）*  
*推薦行動：以 Exp 2-3（+BMI）作為後續優化的基準；若重視穩定性則選 Exp 2-4（+PI）*
