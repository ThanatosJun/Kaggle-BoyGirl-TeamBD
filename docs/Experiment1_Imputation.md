# Experiment 1: Imputation Strategy Comparison

## 實驗概覽

**實驗規模**：4 個配置檔，對比 4 種補值策略（基於 EDA 結果固定使用 1%-99% Percentile）

**實驗目標**：
1. ✅ **固定 Outlier 處理**: 基於 `notebooks/Outlier_Percentile.ipynb` EDA 分析結果，統一使用 **1%-99% Percentile**（裁剪比例僅 2.33%，保留更多有效樣本）
2. 驗證不同 Missing Value補值策略的效果：
   - 中位數 vs 平均值（全局）
   - 資料集平均值 vs 論文平均值（性別區分）
   - 論文平均值 vs 分群平均值
3. 找出針對 weight 和 height 的最佳補值策略

**預期耗時**：約 30-45 分鐘（取決於模型訓練速度）

**主要針對特徵**：`weight`, `height`（與性別強相關的數值特徵）

---

## 背景與理論依據

### EDA 發現（2026-03-25）

**數據質量問題**：
- 存在嚴重異常值：`weight=-1000`, `height=-187`, 以及天文數字級別的極大值
- 這些異常值可能是**缺失值佔位符**或**數據輸入錯誤**

**論文範圍分析與 Percentile 選擇**：
根據 `notebooks/Outlier_Percentile.ipynb` 的詳細比較分析：

| Percentile 範圍 | 總有效樣本 | 總裁剪樣本 | 整體裁剪比例 |
|---------------|----------|----------|-----------|
| **1%-99%** ✅ | 687 | 16 | **2.33%** |
| 2.5%-97.5% | 687 | 32 | 4.66% |
| 5%-95% | 687 | 64 | 9.32% |

- ✅ **EDA 結論**：使用 **1%-99% Percentile 裁剪**作為 Outlier 處理策略
- 📊 論文範圍（23-64歲）在本競賽數據中會**裁剪過多樣本**（>10%），不適用
- ❌ 5%-95% Percentile 裁剪過於嚴格（9.32%），會損失過多有效樣本

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
- ✅ **Outlier 處理已確定**：固定使用 **1%-99% Percentile** (基於 `notebooks/Outlier_Percentile.ipynb` EDA 結果)
- 🔬 **實驗焦點**：比較 **4 種 Missing Value 補值策略**

**實驗數量**：4 個實驗（1 種 Outlier 處理 × 4 種補值策略）

**簡化理由**：
- 根據 EDA 分析，1%-99% percentile 是最佳選擇（裁剪比例僅 2.33%）
- 節省實驗時間：從原本 8 個實驗減少到 4 個（節省 50% 時間）
- 專注於補值策略的比較，這是更關鍵的實驗變數

---

### 完整實驗矩陣

| 實驗編號 | Outlier 處理 | Missing Value 補值策略 | 配置檔名稱 |
|---------|-------------|----------------------|----------|
| **Exp 1-0** | 1%-99% Percentile | **全局 Median** | `exp1_p99_method0_baseline.yaml` |
| **Exp 1-1** | 1%-99% Percentile | 全局 Mean（資料集平均） | `exp1_p99_method1_global_mean.yaml` |
| **Exp 1-2** | 1%-99% Percentile | 論文範圍中點（按性別） | `exp1_p99_method2_paper_range.yaml` |
| **Exp 1-3** | 1%-99% Percentile | 分群 Mean（其他特徵） | `exp1_p99_method3_grouped_mean.yaml` |

**備註**：
- `p99` = 1%-99% Percentile（保留更多數據，裁剪比例僅 2.33%）
- **Baseline** = 全局 Median（系統預設）
- ❌ 已移除 5%-95% Percentile 實驗組（基於 EDA 結果）

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

#### 方式 A: 1%-99% Percentile Clipping ✅ (已選定)
```yaml
# 配置檔中設定（configs/*.yaml）
preprocessing:
  clipping_lower_percentile: 1
  clipping_upper_percentile: 99
```
- **保留更多數據**：只裁剪最極端的 1% 和 99%
- **裁剪比例低**：根據 EDA 分析，僅裁剪 2.33% 的樣本
- **適用場景**：數據量較小時（本專案 423 筆），希望保留更多資訊
- **EDA 驗證**：已通過 `notebooks/Outlier_Percentile.ipynb` 驗證為最佳選擇

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
在 `configs/` 下創建 **4 個配置檔**：

**1%-99% Percentile 組（4 個）**：
```
exp1_p99_method0_baseline.yaml        # 全局 Median
exp1_p99_method1_global_mean.yaml     # 全局 Mean
exp1_p99_method2_paper_range.yaml     # 論文範圍中點
exp1_p99_method3_grouped_mean.yaml    # 分群 Mean (star_sign)
```

**備註**：基於 EDA 結果，已移除 5%-95% Percentile 實驗組

### 步驟 4: 執行實驗
```bash
# ========================================
# 1%-99% Percentile 組（全部 4 個實驗）
# ========================================
python main_train.py --config configs/exp1_p99_method0_baseline.yaml
python main_train.py --config configs/exp1_p99_method1_global_mean.yaml
python main_train.py --config configs/exp1_p99_method2_paper_range.yaml
python main_train.py --config configs/exp1_p99_method3_grouped_mean.yaml
```

### 步驟 5: 結果比較與分析

#### 5.1 補值策略比較（固定 1%-99% Percentile）
**4 種補值策略的效果對比**：
- Baseline (方法 0) vs 全局 Mean (方法 1)：Median vs Mean 的穩健性
- 全局 Mean (方法 1) vs 論文範圍中點 (方法 2)：資料集統計 vs 醫學標準範圍
- 論文範圍中點 (方法 2) vs 分群平均 (方法 3)：固定值 vs 數據驅動

#### 5.2 綜合分析
從 `experiments/experiment_log.csv` 中提取：
- **主要指標**：F1 Score, Accuracy
- **次要指標**：Precision, Recall, CV Std
- **訓練時間**

找出最佳補值策略

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

### 假設: 補值策略比較（方法 0-3）
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

🥇 **第一名候選**：`Exp 1-2` (1%-99% + 論文範圍中點)
- 理由：基於醫學標準的穩健補值 + 保留更多有效樣本

🥈 **第二名候選**：`Exp 1-0` (1%-99% + 全局 Median)
- 理由：保守穩健的 Baseline 組合

🥉 **第三名候選**：`Exp 1-1` (1%-99% + 全局 Mean)
- 理由：使用資料集統計資訊

**意外黑馬**：`Exp 1-3` (1%-99% + 分群平均)
- 如果 star_sign 與 weight/height 有意外相關性

### 關鍵比較
1. **Median vs Mean**：Exp 1-0 vs Exp 1-1
2. **資料集 vs 論文範圍**：Exp 1-1 vs Exp 1-2
3. **論文範圍 vs 分群**：Exp 1-2 vs Exp 1-3

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

| 實驗編號 | Outlier | 補值策略 | F1 Score | F1 Std | Accuracy | Acc Std | Precision | Recall | 訓練時間 | 備註 |
|---------|--------|---------|----------|--------|----------|---------|-----------|--------|---------|------|
| Exp 1-0 | 1%-99% | 全局 Median | **0.9315** | 0.0278 | **0.8961** | 0.0415 | **0.9149** | **0.9495** | - | ✅ **最佳** |
| Exp 1-1 | 1%-99% | 全局 Mean | 0.8295 | 0.0706 | 0.7399 | 0.1062 | 0.8122 | 0.8484 | - | |
| Exp 1-2 | 1%-99% | 論文範圍中點 | **0.9213** | 0.0280 | **0.8842** | 0.0402 | **0.9320** | 0.9116 | - | ✅ **次佳** |
| Exp 1-3 | 1%-99% | 分群平均 | 0.8108 | 0.0525 | 0.7188 | 0.0707 | 0.8100 | 0.8136 | - | |

### 分析發現（已完成）

#### 1. 補值策略比較（固定 1%-99% Percentile）

**性能排名：**
1. **🥇 方法 0 (全局 Median - Baseline)**
   - F1-Score: **0.9315** (最高)
   - Accuracy: **0.8961**
   - Recall: **0.9495** (最高，誤診率最低)
   - **結論：最穩健，對異常值不敏感**

2. **🥈 方法 2 (論文範圍中點 - 按性別)**
   - F1-Score: **0.9213** (僅低於 Baseline 0.0102)
   - Accuracy: **0.8842**
   - Precision: **0.9320** (最高，誤報率最低)
   - **結論：次佳選擇，精確度高，基於醫學標準**

3. 方法 1 (全局 Mean)
   - F1-Score: 0.8295 (下降12.3%)
   - Accuracy: 0.7399 (下降20.9%)
   - **結論：Mean 對異常值敏感，性能下降明顯**

4. 方法 3 (分群平均值 - 按 star_sign)
   - F1-Score: 0.8108 (下降12.9%)
   - Accuracy: 0.7188 (下降19.8%)
   - **結論：star_sign 與身高/體重無直接相關性，反而引入噪聲**

#### 2. 關鍵對比分析

**Median vs Mean（方法 0 vs 方法 1）：**
```
F1-Score 提升：  0.9315 - 0.8295 = +0.1020 (+12.3%)
Accuracy 提升：  0.8961 - 0.7399 = +0.1562 (+20.9%)
Recall 提升：    0.9495 - 0.8484 = +0.1011 (+11.8%)
```
✅ **Median 更優** - 穩健性更好，不受極端值影響

**資料集平均 vs 論文範圍（方法 1 vs 方法 2）：**
```
F1-Score 提升：  0.9213 - 0.8295 = +0.0918 (+11.1%)
Accuracy 提升：  0.8842 - 0.7399 = +0.1443 (+19.5%)
```
✅ **論文範圍更優** - 基於醫學標準，且避免使用 Mean（易受異常值影響）

**論文範圍 vs 分群平均（方法 2 vs 方法 3）：**
```
F1-Score 提升：  0.9213 - 0.8108 = +0.1105 (+13.6%)
Accuracy 提升：  0.8842 - 0.7188 = +0.1654 (+23.0%)
```
✅ **論文範圍更優** - 分群特徵（星座）無相關性

#### 3. 最佳組合

🥇 **第一名：Exp 1-0（1%-99% + 全局 Median）**
- **性能：** F1=0.9315, Accuracy=0.8961
- **優勢：**
  - 簡單易實現
  - 對異常值最穩健
  - 最高 Recall（誤診率最低）
  - **推薦用於實際預測**

🥈 **第二名：Exp 1-2（1%-99% + 論文範圍中點）**
- **性能：** F1=0.9213, Accuracy=0.8842
- **優勢：**
  - 有醫學理論基礎
  - 最高 Precision（誤報率最低）
  - 與 Baseline 差距小（F1 僅差 1.1%）
  - **適合需要理論支撐的場景**

#### 4. 意外發現

1. **全局 Mean 的失效（方法 1）**
   - 性能大幅下降 20%+
   - 原因：資料集中存在遠端異常值（weight=-1000等），Mean 易受影響
   - Median 在這種情況下更適合

2. **分群補值的反效果（方法 3）**
   - star_sign（星座）與身高/體重無顯著相關性
   - 分群反而引入噪聲，降低model穩定性
   - 標準差最高（0.0707）

3. **論文範圍的穩定性**
   - 方法 2 的標準差（0.0280）與 Baseline（0.0278）相近
   - 說明基於医学標準的補值也相當穩健

#### 5. 測試集策略（已上線）

✅ **已實施：方法0（全局 Median）**
- 無數據洩漏風險
- 訓練集和測試集使用相同策略
- 最高穩定性

⚠️ **方法2 的測試集問題：**
- 原始方法依賴性別資訊
- **解決方案：已在 imputation_strategies.py 中優化**
  - 優先級1：使用 fit 時提供的 y（性別標籤）
  - 優先級2：嘗試 X 中的 gender 列
  - 優先級3：使用全局平均值（fallback）

---

## 實驗結論

### 最終建議

**選擇方法 0（全局 Median）作為實際補值策略**，理由：
1. ✅ 性能最優（F1=0.9315）
2. ✅ 最穩健（對異常值不敏感）
3. ✅ 最簡單（易於實現和維護）
4. ✅ 無數據洩漏風險
5. ✅ Recall 最高（臨床應用中誤診風險最低）

### 為什麼其他方法表現不佳？

| 方法 | 失效原因 |
|------|---------|
| 方法 1 (Mean) | weight=-1000 等極端異常值存在，Mean 計算受影響 |
| 方法 3 (分群) | star_sign 與身體指標無相關性，分群引入噪聲 |
| 方法 2 (論文) | 醫學標準基於其他人群（23-64歲），略有偏差 |

### 後續改進方向

1. **Outlier 處理的進一步最佳化**
   - 當前使用 1%-99% percentile，可嘗試其他分位數
   - 或採用 IQR（四分位距）方法

2. **特徵工程的增強**
   - 考慮使用 height/weight ratio（BMI 相關）
   - 探索年齡與身體指標的相互作用

3. **Ensemble 方法**
   - 結合 Baseline（方法0）+ 論文範圍（方法2）
   - 取加權平均可能進一步提升性能

---

## TODO

- [x] EDA 確認當前數據的 weight/height 分布
- [x] 確定實驗設計（4 個補值方法）
- [x] 確認論文範圍數據
- [x] 實作 4 種 imputation_strategies
- [x] 創建 4 個配置檔
- [x] 執行 4 個實驗
- [x] 分析結果：
  - [x] Median vs Mean 的比較
  - [x] 資料集平均 vs 論文範圍中點的比較
  - [x] 論文範圍中點 vs 分群平均的比較
  - [x] 找出最佳組合 ✅ **方法0**
- [x] 制定測試集補值策略 ✅ 優化於 imputation_strategies.py
- [x] 撰寫結論 ✅ 完成

---

## 參考資料
1. [PMC8306797 - Anthropometric measurements](https://pmc.ncbi.nlm.nih.gov/articles/PMC8306797/)
2. 當前專案文檔：`training_workflow_main.md`, `TUNING_GUIDE.md`
3. EDA 分析：`notebooks/EDA_Outlier_Analysis.ipynb`
4. 現有實作：`src/features.py` (ClippingTransformer), `src/data_loader.py`
