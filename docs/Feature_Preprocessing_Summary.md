# 特徵前處理完整清單

## 📋 一覽表

| 特徵 | 類型 | Outlier 處理 | 補值方法 | 備註 |
|------|------|-----------|--------|------|
| **height** | 數值 | **1%-99% Percentile** ✅ | 4 種方法 (可切換) | 主實驗焦點 |
| **weight** | 數值 | **1%-99% Percentile** ✅ | 4 種方法 (可切換) | 主實驗焦點 |
| **iq** | 數值 | **1%-99% Percentile** ✅ | Median (固定) | 按性別無強相關性 |
| **fb_friends** | 數值 (長尾) | **1%-99% Percentile** ✅ | Median → log1p | 負值處理後才能 log |
| **star_sign** | 類別 | — | most_frequent / '-1' | 依 imputation_mode |
| **phone_os** | 類別 | — | most_frequent / '-1' | 依 imputation_mode |
| **sleepiness** | 有序類別 | — | most_frequent / -1 | 依 imputation_mode |

---

## 🔢 詳細特徵說明

### A. 數值特徵（一般）
#### 1⃣ height（身高）
- **型態**：數值型，連續
- **缺失率**：7.78%
- **異常值**：
  - 最小值: -187（明顯異常）
  - 最大值: 216（可能合理，極端高身材）
  - **推薦剪裁設定**：1%-99% Percentile
- **補值方法**（*實驗變數*）：
  - 方法 0️⃣：**全局 Median**（Baseline）
  - 方法 1️⃣：全局 Mean（易受異常值影響）
  - 方法 2️⃣：論文範圍中點（按性別）
    - 男: 171.2 cm（範圍 164.4-178.0）
    - 女: 158.4 cm（範圍 152.6-164.2）
  - 方法 3️⃣：分群平均（按 star_sign + gender）
- **處理流程**：
  ```
  缺失值補值 → 1%-99% Percentile 剪裁 → StandardScaler
  ```

#### 2⃣ weight（體重）
- **型態**：數值型，連續
- **缺失率**：8.31%
- **異常值**：
  - 最小值: -1000（明顯異常，數據洩漏佔位符？）
  - 最大值: 156（可能合理）
  - **推薦剪裁設定**：1%-99% Percentile
- **補值方法**（*實驗變數*）：
  - 方法 0️⃣：**全局 Median**（Baseline）
  - 方法 1️⃣：全局 Mean（易受 -1000 影響）
  - 方法 2️⃣：論文範圍中點（按性別）
    - 男: 72.2 kg（範圍 61.7-82.7）
    - 女: 57.4 kg（範圍 48.6-66.2）
  - 方法 3️⃣：分群平均（按 star_sign + gender）
- **處理流程**：
  ```
  缺失值補值 → 1%-99% Percentile 剪裁 → StandardScaler
  ```

#### 3⃣ iq（智商）
- **型態**：數值型，連續
- **缺失率**：0.47%
- **異常值**：存在但數量少
- **補值方法**：**固定 Median**（不在實驗變數中）
- **Outlier 處理**：1%-99% Percentile
- **處理流程**：
  ```
  Median 補值 → 1%-99% Percentile 剪裁 → StandardScaler
  ```
- **為何不實驗**：
  - IQ 與性別無強相關性
  - 不涉及 missing value 補值策略的主要比較

---

### B. 數值特徵（長尾分佈）
#### 4⃣ fb_friends（Facebook 朋友數）
- **型態**：數值型，長尾分佈（0 到數千都有）
- **缺失率**：1.19%
- **異常值**：
  - 負值存在（-1000 等），需先移除
  - 存在超大數值（長尾）
- **補值方法**：**固定 Median**
- **Outlier 處理**：1%-99% Percentile
- **處理流程**（⚠️ 順序重要）：
  ```
  1. Median 補值
  2. Clip 負值到 0（log1p 前置條件）
  3. log(1+x) 轉換（處理長尾分佈）
  4. StandardScaler 正規化
  ```
- **為何需要 log1p**：
  - 原始分佈高度傾斜（正偏態）
  - 大數值會主導模型學習
  - log 轉換將其壓縮到可管理範圍

---

### C. 類別特徵
#### 5⃣ star_sign（星座）
- **型態**：類別型，無序
- **缺失率**：0%(無缺失值)
- **獨特值**：12 個星座
- **補值方法**：
  - **舊版** (`imputation_mode: old`)：most_frequent
  - **新版** (`imputation_mode: new`)：constant '-1'（缺失標記）
- **處理流程**：
  ```
  補值 → One-Hot Encoding (handle_unknown='ignore')
  ```
- **用途**：
  - 方法 3（分群平均補值）中用作分群依據
  - 一般特徵工程或交叉特徵

#### 6⃣ phone_os（手機系統）
- **型態**：類別型，無序
- **缺失率**：22.33%（高缺失）
- **獨特值**：2-3 個（iOS, Android 等）
- **補值方法**：
  - **舊版** (`imputation_mode: old`)：most_frequent
  - **新版** (`imputation_mode: new`)：constant '-1'（缺失標記）
- **處理流程**：
  ```
  補值 → One-Hot Encoding (handle_unknown='ignore')
  ```
- **注意事項**：
  - 缺失率高，可能隱含年齡/收入特徵
  - 補值策略選擇對模型有影響

---

### D. 有序類別特徵
#### 7⃣ sleepiness（睡眠狀況）
- **型態**：有序類別，轉換為數值 (1-5 或類似)
- **缺失率**：21.51%（高缺失）
- **獨特值**：5 級（通常 1-5）
- **補值方法**：
  - **舊版** (`imputation_mode: old`)：most_frequent
  - **新版** (`imputation_mode: new`)：constant -1（缺失標記）
- **處理流程**：
  ```
  to_numeric() 轉換 → 補值（不做 One-Hot，保持順序）
  ```
- **特點**：
  - 維持原序（不做 One-Hot），保留順序信息
  - 對 Gradient Boosting 模型友善

---

## 📊 Experiment 1 補值方法對比

### 實驗設定

所有實驗固定使用 **1%-99% Percentile** Outlier 處理，變動補值方法：

| 方法編號 | 補值策略 | height / weight | iq / fb_friends | 備註 |
|---------|--------|-----------------|-----------------|------|
| **方法 0** | 全局 Median | ✅ Median | Median | **Baseline** |
| **方法 1** | 全局 Mean | ✅ Mean | Mean | 易受異常值影響 |
| **方法 2** | 論文範圍中點（按性別） | ✅ 中點值 | Median | 醫學標準 |
| **方法 3** | 分群平均（按 star_sign） | ✅ 分群 Mean | Median | 數據驅動 |

### 實驗結果

（根據 Experiment1_Imputation.md 第 270 行後的實驗結果）

```
🥇 方法 0 (全局 Median - Baseline)
   F1-Score: 0.9315 ✅
   Accuracy: 0.8961 ✅
   Recall: 0.9495 (最低誤診率)

🥈 方法 2 (論文範圍中點 - 按性別)
   F1-Score: 0.9213 (僅低 1.1%)
   Accuracy: 0.8842
   Precision: 0.9320 (最低誤報率)

❌ 方法 1 (全局 Mean)
   F1-Score: 0.8295 (下降 12.3%)
   
❌ 方法 3 (分群平均 - star_sign)
   F1-Score: 0.8108 (下降 12.9%)
```

---

## ⚙️ 配置檔案對應

### 檔案位置
```
configs/
├── exp1_p99_method0_baseline.yaml          ← 方法 0 (Median)
├── exp1_p99_method1_global_mean.yaml       ← 方法 1 (Mean)
├── exp1_p99_method2_paper_range.yaml       ← 方法 2 (論文中點)
├── exp1_p99_method3_grouped_mean.yaml      ← 方法 3 (分群)
└── default_config.yaml                    ← 預設配置
```

### 關鍵配置參數

```yaml
preprocessing:
  # 補值方法選擇 (method0 | method1 | method2 | method3)
  imputation_method: "method0"
  
  # 補值模式 (old | new)
  imputation_mode: "old"  # 舊版: most_frequent | 新版: constant
  
  # 數值特徵補值策略（只有 method0 使用，其他自訂）
  numeric_imputer_strategy: "median"
  
  # Outlier 剪裁設定（固定 1%-99%）
  clipping_lower_percentile: 1
  clipping_upper_percentile: 99
  
  # Log 轉換前的最小值 clip（用於 fb_friends）
  log_clip_min: 0
  
  # 正規化方式
  scaler: "standard"
```

---

## 🔍 特徵處理管道（Pipeline）

### 1️⃣ 數值特徵標準處理 (height, weight, iq)

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # 或 'mean' (方法 1) 等
    ('clipper', ClippingTransformer(lower_percentile=1, upper_percentile=99)),
    ('scaler', StandardScaler())
])
```

**處理順序**：
```
缺失值 → 補值 → 剪裁極值 → 正規化
```

### 2️⃣ 長尾數值特徵 (fb_friends)

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clip_min', FunctionTransformer(clip_min_value, min_value=0)),  # 負值 → 0
    ('log1p', FunctionTransformer(np.log1p)),  # log(1+x)
    ('scaler', StandardScaler())
])
```

**處理順序**（⚠️ 順序關鍵）：
```
缺失值 → Median 補值 → 移除負值 → log(1+x) → 正規化
```

**為何要這個順序**：
- ❌ 錯誤：補值 → log1p，如果補值為負則產生 NaN
- ✅ 正確：補值（Median > 0）→ clip 到 0 → log1p

### 3️⃣ 無序類別特徵 (star_sign, phone_os)

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 舊版
    # 或
    ('imputer', SimpleImputer(strategy='constant', fill_value='-1')),  # 新版
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
```

**特點**：
- `handle_unknown='ignore'` 避免測試集新類別導致錯誤
- `sparse_output=False` 產出密集矩陣（易於模型處理）

### 4️⃣ 有序類別特徵 (sleepiness)

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # 不做 One-Hot，保持原數值
])
```

**特點**：
- 保留順序資訊（1 < 2 < 3 < ... 有意義）
- 適合 Gradient Boosting（可自動分割閾值）

---

## 💡 關鍵決策理由

### 為什麼選擇 Median 而不是 Mean？

| 指標 | Median | Mean |
|------|--------|------|
| 異常值敏感度 | 低 ✅ | 高 ❌ |
| weight=-1000 影響 | 無 | 很大 |
| 實驗結果 | F1=0.9315 | F1=0.8295 |
| 計算複雜度 | 低 | 低 |

**結論**：Median 是更安全、更穩健的選擇

### 為什麼選擇 1%-99% 而不是 5%-95%？

| 指標 | 1%-99% | 5%-95% |
|------|--------|--------|
| 保留樣本率 | 97.67% ✅ | 90.68% ❌ |
| 裁剪比例 | 2.33% | 9.32% |
| 數據損失 | 少 | 多 |
| 模型穩定性 | 高 | 中等 |

**結論**：1%-99% 更適合小數據集（423 筆）

### 為什麼分群平均（star_sign）表現差？

**問題**：
- star_sign（星座）與 height/weight 無因果關係
- 分群反而引入噪聲
- 模型過度擬合到無關特徵

**結論**：避免無根據的分群補值

---

## 📋 配置文檔使用指南

### 如何切換補值方法？

**修改檔案**：`configs/exp1_p99_method0_baseline.yaml`

```yaml
preprocessing:
  imputation_method: "method2"  # 改为 method0/1/2/3
```

然後執行：
```bash
python main_train.py --config configs/exp1_p99_method0_baseline.yaml
```

### 如何切換 imputation_mode？

```yaml
preprocessing:
  imputation_mode: "new"  # old | new
```

**預期變化**：
- `old` → most_frequent（類別用最常出現的值）
- `new` → constant '-1'（類別用缺失標記）

### 如何使用測試集？

⚠️ **重要**：測試集補值需要特殊處理
- 如果使用 gender 信息（方法 2 & 3），測試集不能用 gender（因為 gender 是目標變數）
- **解決方案**：在測試集上使用方法 0（全局 Median，無性別依賴）

詳見 `src/imputation_strategies.py` 中的優先級邏輯

---

## 🎯 最終建議

### ✅ 使用方法：方法 0（全局 Median）

**原因**：
1. 性能最優（F1=0.9315）
2. 最穩健（對異常值不敏感）
3. 最簡單（易於實現和維護）
4. 無數據洩漏風險（訓練集和測試集一致）
5. Recall 最高（臨床應用中最低誤診率）

### 配置檔：
```bash
configs/exp1_p99_method0_baseline.yaml
```

### 執行命令：
```bash
python main_train.py --config configs/exp1_p99_method0_baseline.yaml
```

---

## 📚 參考文檔

- [Experiment1_Imputation.md](Experiment1_Imputation.md) - 完整實驗記錄
- [src/features.py](../src/features.py) - 特徵處理管道實現
- [src/imputation_strategies.py](../src/imputation_strategies.py) - 4 種補值方法實現
- [src/data_loader.py](../src/data_loader.py) - 數據加載邏輯

