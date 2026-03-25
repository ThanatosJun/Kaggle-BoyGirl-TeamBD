# Experiment 1: Imputation Strategy Comparison

## 實驗概覽

**實驗規模**：8 個配置檔，對比 2 種 Outlier 處理 × 4 種補值策略

**實驗目標**：
1. 比較不同 Outlier 處理強度（1%-99% vs 5%-95%）的影響
2. 驗證不同 Missing Value 補值策略的效果：
   - 中位數 vs 平均值（全局）
   - 資料集平均值 vs 論文平均值（性別區分）
   - 論文平均值 vs 分群平均值
3. 找出針對 weight 和 height 的最佳處理組合

**預期耗時**：約 1-1.5 小時（取決於模型訓練速度）

**主要針對特徵**：`weight`, `height`（與性別強相關的數值特徵）

---

## 背景與理論依據

### EDA 發現（2026-03-25）

**數據質量問題**：
- 存在嚴重異常值：`weight=-1000`, `height=-187`, 以及天文數字級別的極大值
- 這些異常值可能是**缺失值佔位符**或**數據輸入錯誤**

**論文範圍分析**：
根據 `notebooks/EDA_Outlier_Analysis.ipynb` 的分析結果：
- 📊 論文範圍（23-64歲）會**裁剪過多樣本**（>10%）
- ✅ **結論**：使用 **1%-99% Percentile 裁剪**作為 Outlier 處理策略
- ❌ 論文範圍不適用於本競賽數據（可能包含更年輕/年長者，或數據分布差異）

### 論文參考範圍（僅供參考）
來源：[PMC8306797](https://pmc.ncbi.nlm.nih.gov/articles/PMC8306797/)（23-64歲年齡區間）

| 特徵 | 性別 | 下限 | 上限 | 平均值 (待補充) |
|------|------|------|------|----------------|
| Weight | 男 (Boy) | 61.7 kg | 82.7 kg | 72.2 kg |
| Weight | 女 (Girl) | 48.6 kg | 66.2 kg | 57.4 kg |
| Height | 男 (Boy) | 164.4 cm | 177.0 cm | 170.2 cm |
| Height | 女 (Girl) | 152.6 cm | 164.2 cm | 158.4 cm |

**TODO**: 補充論文中的平均值數據（方法二需要）

---

## 實驗設計

### 實驗策略

**基於 EDA 的決策**：
- 🔬 **實驗維度 1**：比較不同的 **Outlier 處理方式**（1%-99% vs 5%-95% 百分位數裁剪）
- 🔬 **實驗維度 2**：比較不同的 **Missing Value 補值策略**（4 種方法）

**實驗組合**：2 種 Outlier 處理 × 4 種補值策略 = **8 個實驗**

---

### 完整實驗矩陣

| 實驗編號 | Outlier 處理 | Missing Value 補值策略 | 配置檔名稱 |
|---------|-------------|----------------------|----------|
| **Exp 1-0** | 1%-99% Percentile | **全局 Median** | `exp1_p99_method0_baseline.yaml` |
| **Exp 1-1** | 1%-99% Percentile | 全局 Mean（資料集平均） | `exp1_p99_method1_global_mean.yaml` |
| **Exp 1-2** | 1%-99% Percentile | 論文範圍中點（按性別） | `exp1_p99_method2_paper_range.yaml` |
| **Exp 1-3** | 1%-99% Percentile | 分群 Mean（其他特徵） | `exp1_p99_method3_grouped_mean.yaml` |
| **Exp 1-4** | 5%-95% Percentile | **全局 Median** | `exp1_p95_method0_baseline.yaml` |
| **Exp 1-5** | 5%-95% Percentile | 全局 Mean（資料集平均） | `exp1_p95_method1_global_mean.yaml` |
| **Exp 1-6** | 5%-95% Percentile | 論文範圍中點（按性別） | `exp1_p95_method2_paper_range.yaml` |
| **Exp 1-7** | 5%-95% Percentile | 分群 Mean（其他特徵） | `exp1_p95_method3_grouped_mean.yaml` |

**備註**：
- `p99` = 1%-99% Percentile（保留更多數據，裁剪更少）
- `p95` = 5%-95% Percentile（保守裁剪，移除更多極端值）
- **Baseline** = 全局 Median（系統預設）

---

### 補值方法說明

| 方法編號 | Missing Value 補值策略 | 說明 | 數據來源 |
|---------|----------------------|------|---------|
| **方法 0** | 全局 Median | **Baseline**（系統預設），穩健性高 | 資料集 |
| **方法 1** | 全局 Mean | 使用資料集的平均值 | 資料集 |
| **方法 2** | 論文範圍中點（按性別） | 使用論文報告的上下限範圍的中點值 | 論文 [PMC8306797] |
| **方法 3** | 分群 Mean | 按其他特徵（如 star_sign）分群計算平均值 | 資料集 |

---

#### 方法細節

#### 方法細節

### A. Outlier 處理方式（兩種）

#### 方式 A: 1%-99% Percentile Clipping
```yaml
# 配置檔中設定（configs/*.yaml）
preprocessing:
  clipping_lower_percentile: 1
  clipping_upper_percentile: 99
```
- **保留更多數據**：只裁剪最極端的 1% 和 99%
- **適用場景**：數據量較小時，希望保留更多資訊
- **風險**：可能保留部分異常值（如 EDA 所見）

#### 方式 B: 5%-95% Percentile Clipping
```yaml
# 配置檔中設定（configs/*.yaml）
preprocessing:
  clipping_lower_percentile: 5
  clipping_upper_percentile: 95
```
- **更保守的裁剪**：移除兩端各 5% 的極端值
- **適用場景**：數據質量較差，需要更嚴格的清洗
- **風險**：可能裁剪掉真實的極端值（如特別高/矮的人）

---

### B. Missing Value 補值方法（四種）

##### 方法 0: Baseline（全局 Median）
```python
# 系統預設行為
# Outlier: 百分位數 clipping（1%-99% 或 5%-95%）
# Missing Value: 全局 median (不分性別)
SimpleImputer(strategy='median')
```
**特點**：穩健性高，不受異常值影響

##### 方法 1: 全局 Mean（資料集平均值）
```python
# Outlier: 百分位數 clipping
# Missing Value: 全局 mean（計算自資料集）
SimpleImputer(strategy='mean')
```
**特點**：使用資料集的統計資訊，但可能受異常值影響

##### 方法 2: 論文範圍中點（按性別）
```python
# Outlier: 百分位數 clipping
# Missing Value: 使用論文報告的上下限範圍的中點值，按性別補值

# 論文範圍（來自 PMC8306797）
PAPER_RANGES = {
    1: {  # 男
        'weight': (61.7, 82.7),   # kg
        'height': (164.4, 177.0)  # cm
    },
    2: {  # 女
        'weight': (48.6, 66.2),   # kg
        'height': (152.6, 164.2)  # cm
    }
}

# 計算範圍中點作為補值
PAPER_MIDPOINT = {
    1: {  # 男
        'weight': (61.7 + 82.7) / 2,   # = 72.2 kg
        'height': (164.4 + 177.0) / 2  # = 170.7 cm
    },
    2: {  # 女
        'weight': (48.6 + 66.2) / 2,   # = 57.4 kg
        'height': (152.6 + 164.2) / 2  # = 158.4 cm
    }
}

# 補值邏輯（需在 data_loader.py 或自訂 preprocessing 中實作）
for gender in [1, 2]:  # 1=男, 2=女
    mask = df['gender'] == gender
    # 只補 missing value，不改變已有的值
    df.loc[mask, 'weight'] = df.loc[mask, 'weight'].fillna(PAPER_MIDPOINT[gender]['weight'])
    df.loc[mask, 'height'] = df.loc[mask, 'height'].fillna(PAPER_MIDPOINT[gender]['height'])
```
**特點**：
- 基於醫學研究的標準範圍，理論基礎強
- 使用中點值（mean of range）作為合理的估計值
- 保守穩健，避免極端值

**替代方案**（可選）：
- 使用範圍內的隨機值：`np.random.uniform(lower, upper)`
- 使用下限或上限（不建議，過於保守）

##### 方法 3: 分群平均值（按其他特徵）
```python
# Outlier: 百分位數 clipping
# Missing Value: 按性別+其他特徵分群計算平均值

# 選項 A: 按星座分群
for gender in [1, 2]:
    mask_gender = df['gender'] == gender

    # 按星座分組計算平均值
    group_means = df[mask_gender].groupby('star_sign')[['weight', 'height']].transform('mean')

    # Fallback: 如果分組內全為缺失，使用性別平均值
    fallback = df[mask_gender][['weight', 'height']].mean()

    df.loc[mask_gender, 'weight'] = df.loc[mask_gender, 'weight'].fillna(
        group_means['weight']
    ).fillna(fallback['weight'])

    df.loc[mask_gender, 'height'] = df.loc[mask_gender, 'height'].fillna(
        group_means['height']
    ).fillna(fallback['height'])

# 選項 B: 按手機系統分群（同理，改用 'phone_os'）
```
**特點**：考慮更多特徵的影響，可能捕捉隱藏模式
**風險**：分群過細可能導致小樣本問題

---

### 分群特徵選擇（方法 3）

方法 3 需要選擇用來分群的特徵，有以下選項：

| 分群特徵 | 類別數 | 優點 | 缺點 | 建議 |
|---------|-------|------|------|-----|
| `star_sign` | 12 | 類別數適中 | 可能無直接相關性 | ✅ 優先嘗試 |
| `phone_os` | 2-3 | 簡單，可能隱含年齡/收入 | 類別數少，有缺失值 | 備選 |

**決策**：
- **主實驗**：使用 `star_sign` 作為分群特徵（配置檔：`exp1_p99_method3_grouped_mean.yaml`）
- **可選**：如果想進一步探索，可以創建 `phone_os` 版本作為額外實驗

**不採用的選項**：
- ❌ `sleepiness`：有較多缺失值（21.51%），不適合作為分群依據
- ❌ KMeans 聚類：過於複雜，暫不考慮

---

## 實驗流程

### 步驟 1: 確認論文範圍數據
從論文 [PMC8306797](https://pmc.ncbi.nlm.nih.gov/articles/PMC8306797/) 確認上下限：

| 特徵 | 性別 | 下限 | 上限 | 中點值（補值用） |
|------|------|------|------|----------------|
| Weight | 男 | 61.7 kg | 82.7 kg | **72.2 kg** |
| Weight | 女 | 48.6 kg | 66.2 kg | **57.4 kg** |
| Height | 男 | 164.4 cm | 178.0 cm | **171.2 cm** |
| Height | 女 | 152.6 cm | 164.2 cm | **158.4 cm** |

✅ 數據已確認，可以直接使用

### 步驟 2: 實作補值邏輯
創建 `src/imputation_strategies.py`，實作：
- 方法 2: 論文範圍中點補值（按性別）
- 方法 3: 分群平均值補值（按 star_sign）

### 步驟 3: 創建配置文件
在 `configs/` 下創建 **8 個配置檔**：

**1%-99% Percentile 組（4 個）**：
```
exp1_p99_method0_baseline.yaml        # 全局 Median
exp1_p99_method1_global_mean.yaml     # 全局 Mean
exp1_p99_method2_paper_range.yaml     # 論文範圍中點
exp1_p99_method3_grouped_mean.yaml    # 分群 Mean (star_sign)
```

**5%-95% Percentile 組（4 個）**：
```
exp1_p95_method0_baseline.yaml        # 全局 Median
exp1_p95_method1_global_mean.yaml     # 全局 Mean
exp1_p95_method2_paper_range.yaml     # 論文範圍中點
exp1_p95_method3_grouped_mean.yaml    # 分群 Mean (star_sign)
```

### 步驟 4: 執行實驗
```bash
# ========================================
# 1%-99% Percentile 組
# ========================================
python main_train.py --config configs/exp1_p99_method0_baseline.yaml
python main_train.py --config configs/exp1_p99_method1_global_mean.yaml
python main_train.py --config configs/exp1_p99_method2_paper_range.yaml
python main_train.py --config configs/exp1_p99_method3_grouped_mean.yaml

# ========================================
# 5%-95% Percentile 組
# ========================================
python main_train.py --config configs/exp1_p95_method0_baseline.yaml
python main_train.py --config configs/exp1_p95_method1_global_mean.yaml
python main_train.py --config configs/exp1_p95_method2_paper_range.yaml
python main_train.py --config configs/exp1_p95_method3_grouped_mean.yaml
```

### 步驟 5: 結果比較與分析

#### 5.1 比較維度一：補值策略（固定 Outlier 處理）
**在 1%-99% 組內比較**：
- Baseline (方法 0) vs 全局 Mean (方法 1)：Median vs Mean 的穩健性
- 全局 Mean (方法 1) vs 論文範圍中點 (方法 2)：資料集統計 vs 醫學標準範圍
- 論文範圍中點 (方法 2) vs 分群平均 (方法 3)：固定值 vs 數據驅動

**在 5%-95% 組內比較**：同上

#### 5.2 比較維度二：Outlier 處理（固定補值策略）
- **Baseline 對比**：Exp 1-0 (1%-99%) vs Exp 1-4 (5%-95%)
- **最佳方法對比**：選擇表現最好的補值策略，比較兩種 Percentile 的差異

#### 5.3 綜合分析
從 `experiments/experiment_log.csv` 中提取：
- **主要指標**：F1 Score, Accuracy
- **次要指標**：Precision, Recall, CV Std
- **訓練時間**

找出整體最佳組合（Outlier 處理 + 補值策略）

---

## 評估指標

主要指標：
- **F1 Score** （主要優化目標）
- **Accuracy**
- **Precision / Recall**

次要分析：
- 訓練時間
- 穩定性（CV 標準差）

---

## 預期結果與假設

### 假設 A: Outlier 處理方式（1%-99% vs 5%-95%）
1. **5%-95% 可能表現更好**：
   - 理由：EDA 顯示數據有嚴重異常值（-1000, 9e107, 500, 900 等），更嚴格的裁剪可以清除這些垃圾數據
   - 風險：可能裁掉真實的極端值（樣本數較少時更明顯）

2. **1%-99% 的優勢**：
   - 保留更多真實數據（樣本數 423 筆，每筆都珍貴）
   - 風險：保留異常值可能干擾模型學習

### 假設 B: 補值策略（方法 0-3）
1. **方法 0 (Median) vs 方法 1 (Mean)**：
   - Median 應該比 Mean 更穩健（因為有異常值）
   - 預期：方法 0 > 方法 1

2. **方法 1 (資料集平均) vs 方法 2 (論文範圍中點)**：
   - 論文範圍中點基於醫學研究（23-64歲），但可能與競賽數據分布不同
   - 資料集平均更貼近實際數據分布
   - 論文中點值更保守穩健（避免極端值）
   - 預期：**不確定**，需要實驗驗證

3. **方法 2 (論文範圍中點) vs 方法 3 (分群平均)**：
   - 分群平均考慮更多特徵，可能捕捉隱藏模式
   - 論文中點基於醫學標準，理論基礎強
   - 風險：分群特徵（star_sign）可能與 weight/height 無關
   - 預期：**方法 2 較穩定**，方法 3 可能有驚喜或表現更差

### 預期最佳組合

🥇 **第一名候選**：`Exp 1-6` (5%-95% + 論文範圍中點)
- 理由：嚴格清除異常值 + 基於醫學標準的穩健補值

🥈 **第二名候選**：`Exp 1-2` (1%-99% + 論文範圍中點)
- 理由：保留更多數據 + 醫學標準補值

🥉 **第三名候選**：`Exp 1-4` (5%-95% + 全局 Median)
- 理由：保守穩健的組合（Baseline 升級版）

**意外黑馬**：`Exp 1-7` (5%-95% + 分群平均)
- 如果 star_sign 與 weight/height 有意外相關性

### 關鍵比較
1. **Median vs Mean**：Exp 1-0 vs Exp 1-1（或 Exp 1-4 vs Exp 1-5）
2. **資料集 vs 論文範圍**：Exp 1-1 vs Exp 1-2（或 Exp 1-5 vs Exp 1-6）
3. **論文範圍 vs 分群**：Exp 1-2 vs Exp 1-3（或 Exp 1-6 vs Exp 1-7）
4. **1%-99% vs 5%-95%**：Exp 1-0 vs Exp 1-4（Baseline 對比）

### 風險與注意事項
- ⚠️ **數據洩漏風險（方法 2 & 3）**：使用 gender 來補值，但 gender 是 target
  - **緩解方案**：
    - 訓練時：在 CV 的每個 fold 內部使用性別資訊補值（合法）
    - 測試時：**不能使用 gender**，需改用全局平均或其他策略
    - **重要**：記錄這個限制，測試集補值需要修改方法
- ⚠️ **論文數據適用性（方法 2）**：
  - 論文範圍是 23-64 歲，競賽數據可能包含不同年齡層
  - 使用範圍中點可能不完全符合實際數據分布
- ⚠️ **過度擬合（方法 3）**：star_sign 分群可能僅對訓練集有效
- ⚠️ **缺失不隨機（MNAR）**：如果缺失本身是重要特徵，補值可能抹除信號
- ⚠️ **樣本數減少（5%-95%）**：更嚴格的裁剪會減少有效訓練樣本數

---

## 實驗結果（待填寫）

### 結果表格

| 實驗編號 | Outlier | 補值策略 | F1 Score | Accuracy | Precision | Recall | CV Std | 訓練時間 | 備註 |
|---------|--------|---------|----------|----------|-----------|--------|--------|---------|------|
| Exp 1-0 | 1%-99% | 全局 Median | - | - | - | - | - | - | Baseline |
| Exp 1-1 | 1%-99% | 全局 Mean | - | - | - | - | - | - | |
| Exp 1-2 | 1%-99% | 論文範圍中點 | - | - | - | - | - | - | |
| Exp 1-3 | 1%-99% | 分群平均 | - | - | - | - | - | - | |
| Exp 1-4 | 5%-95% | 全局 Median | - | - | - | - | - | - | |
| Exp 1-5 | 5%-95% | 全局 Mean | - | - | - | - | - | - | |
| Exp 1-6 | 5%-95% | 論文範圍中點 | - | - | - | - | - | - | 預期候選 |
| Exp 1-7 | 5%-95% | 分群平均 | - | - | - | - | - | - | |

### 分析發現（待填寫）

#### 1. Outlier 處理影響
- **1%-99% vs 5%-95%**：
  - Baseline 對比 (Exp 1-0 vs Exp 1-4)：？
  - 最佳方法對比：？
  - **結論**：？

#### 2. 補值策略影響
- **Median vs Mean** (Exp 1-0 vs Exp 1-1)：？
- **資料集平均 vs 論文範圍中點** (Exp 1-1 vs Exp 1-2)：？
- **論文範圍中點 vs 分群平均** (Exp 1-2 vs Exp 1-3)：？
- **結論**：？

#### 3. 最佳組合
- **實驗編號**：？
- **配置**：？% Percentile + ？補值策略
- **性能**：F1 = ？, Accuracy = ？
- **相比 Baseline (Exp 1-0) 提升**：？%

#### 4. 意外發現
- （記錄任何意外的實驗結果）

#### 5. 測試集策略
- ⚠️ **重要**：如果最佳方法使用了性別資訊（方法 2 或 3），測試集預測時需要：
  - 選項 A：改用全局平均（犧牲部分性能）
  - 選項 B：先用模型預測性別，再補值（可能有風險）
  - 選項 C：保留缺失值，讓模型自行處理（如 CatBoost）

---

## TODO
- [x] EDA 確認當前數據的 weight/height 分布（與論文範圍比較）
- [x] 確定實驗設計（2 種 Outlier × 4 種補值 = 8 個實驗）
- [x] 確認論文範圍數據並計算中點值
- [ ] 實作 `src/imputation_strategies.py`：
  - [ ] 方法 2: 論文範圍中點補值邏輯（按性別）
  - [ ] 方法 3: 分群平均值補值邏輯 (star_sign)
- [ ] 創建 8 個配置檔（4 個 p99 + 4 個 p95）
- [ ] 執行 8 個實驗並記錄結果
- [ ] 分析結果：
  - [ ] Median vs Mean 的比較
  - [ ] 資料集平均 vs 論文範圍中點的比較
  - [ ] 論文範圍中點 vs 分群平均的比較
  - [ ] 1%-99% vs 5%-95% 的比較
  - [ ] 找出最佳組合
- [ ] 制定測試集補值策略（如果最佳方法使用性別資訊）
- [ ] 撰寫結論並更新 `MEMORY.md`

---

## 參考資料
1. [PMC8306797 - Anthropometric measurements](https://pmc.ncbi.nlm.nih.gov/articles/PMC8306797/)
2. 當前專案文檔：`training_workflow_main.md`, `TUNING_GUIDE.md`
3. EDA 分析：`notebooks/EDA_Outlier_Analysis.ipynb`
4. 現有實作：`src/features.py` (ClippingTransformer), `src/data_loader.py`
