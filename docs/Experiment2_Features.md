# 🎉 Experiment 2 - 衍生特徵實驗 | 完整報告

## 📋 實驗概要

**時間區間**: 2026-03-25 20:49 ~ 20:50  
**實驗總數**: 3 個（基礎特徵 × 衍生特徵組合）  
**模型**: CatBoost (iterations=500, depth=6, lr=0.05)  
**補值策略**: 全局 Median（Exp1 最優方法）  
**評估方式**: 5-Fold 交叉驗證  
**前提**: 不做超參數調整，純粹比較特徵工程效果

---

## 🏆 最終排名 & 結果

| 排名 | 實驗 | 特徵組合 | F1-Score | Accuracy | Precision | Recall | 評分 |
|------|------|---------|----------|----------|-----------|--------|------|
| 🥇 | Exp 2-3 | 基礎 + BMI | **0.9276 ± 0.0259** | **0.8914 ± 0.0387** | 0.9223 ± 0.0246 | **0.9336 ± 0.0365** | ⭐⭐⭐⭐⭐ |
| 🥈 | Exp 2-2 | 基礎 + Ratio | 0.9259 ± 0.0255 | 0.8890 ± 0.0382 | 0.9218 ± 0.0228 | 0.9305 ± 0.0340 | ⭐⭐⭐⭐ |
| 3️⃣ | Exp 2-1 | 僅基礎特徵 | 0.9238 ± 0.0274 | 0.8866 ± 0.0403 | **0.9249 ± 0.0285** | 0.9241 ± 0.0441 | ⭐⭐⭐ |

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

### 3️⃣ Exp 2-3：基礎特徵 + BMI（最優）

```
F1-Score:  0.9276 ± 0.0259  (+0.0038 vs 對照)  ← 最高
Accuracy:  0.8914 ± 0.0387  (+0.0048 vs 對照)  ← 最高
Precision: 0.9223 ± 0.0246  (-0.0026 vs 對照)
Recall:    0.9336 ± 0.0365  (+0.0095 vs 對照)  ← 最高
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
- std 介於 ratio 和 baseline 之間（合理）

---

## 📊 特徵效益比較

### 增益方向（vs 對照組 Exp 2-1）

| 指標 | +Ratio 增益 | +BMI 增益 |
|------|------------|----------|
| F1-Score | +0.0021 (+0.23%) | **+0.0038 (+0.41%)** |
| Accuracy | +0.0024 (+0.27%) | **+0.0048 (+0.54%)** |
| Precision | -0.0031 (-0.34%) | -0.0026 (-0.28%) |
| Recall | +0.0064 (+0.69%) | **+0.0095 (+1.03%)** |

### 穩定性（std）

| 指標 | 對照 std | +Ratio std | +BMI std |
|------|---------|-----------|---------|
| F1-Score | 0.0274 | **0.0255** | 0.0259 |
| Accuracy | 0.0403 | **0.0382** | 0.0387 |
| Recall | 0.0441 | **0.0340** | 0.0365 |

- 衍生特徵降低了所有 std，說明特徵本身包含更清晰的決策信號
- Ratio 的穩定性略優於 BMI，但 BMI 的性能更高

---

## 💡 結論與建議

### 核心發現

1. **BMI 優於 Ratio 優於 Base**：加入 BMI 和 Ratio 均有微幅提升，BMI 效果更好
2. **增益幅度有限**：最大 F1 增益約 +0.4%，統計顯著性不高（需更多 CV 確認）
3. **Precision 略降**：兩個衍生特徵均讓 Precision 微降，換取了更高的 Recall
4. **穩定性提升**：衍生特徵降低了 std，意味著模型預測更可靠

### 為什麼 BMI 更好？

- **非線性優勢**：BMI = w/(h/100)² 是二次關係，能捕捉 ratio 無法表達的曲率
- **醫學有效性**：BMI 是臨床認可的體態指標，性別差異在 BMI 分佈上清晰可見
- **信息密度**：BMI 將 weight 和 height 壓縮為「體型」概念，提供了更有意義的維度

### 為什麼增益有限？

- CatBoost 本身已能隱式學習 weight × height 的交互（depth=6 允許高階交互）
- 衍生特徵與原始特徵高度相關（多重共線性），邊際信息增益自然有限

### 推薦方案

| 用途 | 推薦配置 |
|------|---------|
| **純性能優先** | +BMI (Exp 2-3)：F1=0.9276，Recall 最高（誤診最低） |
| **穩定性優先** | +Ratio (Exp 2-2)：std 最低，預測最穩定 |
| **簡潔基準** | 基礎特徵 (Exp 2-1)：差異微小（<0.5%），維護成本最低 |

> **結論**：若後續繼續 tuning，建議沿用 **+BMI** 配置（Exp 2-3），因為 BMI 有更高的天花板且具醫學可解釋性。

---

## 📦 交付文件

### 新增/修改檔案
- ✅ `src/features.py` - 添加 `engineer_features()` 函數（支持 ratio & BMI）
- ✅ `src/evaluate.py` - 在每個 CV fold 補值後調用特徵工程
- ✅ `main_train.py` - 在全量訓練補值後調用特徵工程
- ✅ `configs/exp2_method0_base_features.yaml` - Exp 2-1 配置
- ✅ `configs/exp2_method0_with_ratio.yaml` - Exp 2-2 配置
- ✅ `configs/exp2_method0_with_bmi.yaml` - Exp 2-3 配置

### 實驗存檔路徑
- `experiments/exp_002_exp2_method0_base_features/`
- `experiments/exp_003_exp2_method0_with_ratio/`
- `experiments/exp_004_exp2_method0_with_bmi/`

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
features:
  numeric_cols: ["height", "weight", "iq", "bmi"]  # 需包含衍生特徵名
  add_weight_height_ratio: false
  add_bmi: true                                      # flag 控制
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
1. **超參數優化**：在 +BMI 特徵基礎上做 grid search（之前 Exp2 都是 base model，無 tuning）
2. **兩者組合**：同時加入 ratio + BMI，觀察是否有協同效應（可能因共線性互消）
3. **其他衍生特徵**：探索 IQ × sleepiness 交互、年齡估算等
4. **特徵重要性分析**：使用 CatBoost 內建的 feature_importance 確認 BMI 排名

---

*報告生成時間: 2026-03-25 20:50*  
*實驗持續時間: ~1 分鐘（3個實驗）*  
*推薦行動：以 Exp 2-3（+BMI）作為後續優化的基準*
