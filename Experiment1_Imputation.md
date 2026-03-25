# Experiment 1: Imputation Strategy Comparison

## 實驗目標
驗證不同的 missing value 和 outlier 處理方式對模型性能的影響，主要針對 **weight** 和 **height** 兩個關鍵數值特徵。

---

## 背景與理論依據

### 論文參考範圍
根據論文 [PMC8306797](https://pmc.ncbi.nlm.nih.gov/articles/PMC8306797/)（23-64歲年齡區間）：

| 特徵 | 性別 | 下限 | 上限 | 平均值 (待補充) |
|------|------|------|------|----------------|
| Weight | 男 (Boy) | 61.7 kg | 82.7 kg | ??? |
| Weight | 女 (Girl) | 48.6 kg | 66.2 kg | ??? |
| Height | 男 (Boy) | 164.4 cm | 178.0 cm | ??? |
| Height | 女 (Girl) | 152.6 cm | 164.2 cm | ??? |

**TODO**: 補充論文中的平均值數據（方法二需要）

---

## 實驗設計

### 階段一：Outlier 處理策略

#### 方法對比

| 方法編號 | Outlier 處理 | Missing Value 補值策略 | 說明 |
|---------|-------------|----------------------|------|
| **方法 0** | 不處理 outlier | 全局平均值 | Baseline（目前系統預設） |
| **方法 1** | 裁剪至論文上下限 | 全局平均值 | 簡單粗暴，忽略性別差異 |
| **方法 2** | 裁剪至論文上下限 | 論文平均值（按性別） | 使用論文提供的統計數據 |
| **方法 3** | 裁剪至論文上下限 | 分群後計算平均值 | 考慮其他特徵的影響 |

#### 方法細節

##### 方法 0: Baseline（不處理 outlier）
```python
# Outlier: 不處理
# Missing Value: 全局 median
df['weight'].fillna(df['weight'].median(), inplace=True)
df['height'].fillna(df['height'].median(), inplace=True)
```

##### 方法 1: 論文裁剪 + 全局平均值
```python
# Step 1: 根據論文範圍裁剪 outlier
# 注意：需要先推斷性別或使用全局上下限
df.loc[df['weight'] < 48.6, 'weight'] = 48.6  # 女性下限（保守）
df.loc[df['weight'] > 82.7, 'weight'] = 82.7  # 男性上限（保守）
df.loc[df['height'] < 152.6, 'height'] = 152.6
df.loc[df['height'] > 178.0, 'height'] = 178.0

# Step 2: 補值使用裁剪後的全局平均值
df['weight'].fillna(df['weight'].mean(), inplace=True)
df['height'].fillna(df['height'].mean(), inplace=True)
```

##### 方法 2: 論文裁剪 + 論文平均值（按性別）
```python
# Step 1: 根據性別裁剪 outlier
for gender in ['男', '女']:
    mask_gender = df['gender'] == gender
    if gender == '男':
        df.loc[mask_gender & (df['weight'] < 61.7), 'weight'] = 61.7
        df.loc[mask_gender & (df['weight'] > 82.7), 'weight'] = 82.7
        df.loc[mask_gender & (df['height'] < 164.4), 'height'] = 164.4
        df.loc[mask_gender & (df['height'] > 178.0), 'height'] = 178.0
    else:  # 女
        df.loc[mask_gender & (df['weight'] < 48.6), 'weight'] = 48.6
        df.loc[mask_gender & (df['weight'] > 66.2), 'weight'] = 66.2
        df.loc[mask_gender & (df['height'] < 152.6), 'height'] = 152.6
        df.loc[mask_gender & (df['height'] > 164.2), 'height'] = 164.2

# Step 2: 使用論文平均值補缺失值（TODO: 從論文查詢平均值）
PAPER_MEAN = {
    '男': {'weight': ???, 'height': ???},
    '女': {'weight': ???, 'height': ???}
}
for gender in ['男', '女']:
    mask_gender = df['gender'] == gender
    df.loc[mask_gender, 'weight'].fillna(PAPER_MEAN[gender]['weight'], inplace=True)
    df.loc[mask_gender, 'height'].fillna(PAPER_MEAN[gender]['height'], inplace=True)
```

##### 方法 3: 論文裁剪 + 分群補值
```python
# Step 1: 同方法 2 裁剪 outlier

# Step 2: 根據其他特徵分群（例如：star_sign, phone_os, sleepiness 等）
# 使用 GroupBy 計算分組平均值
for gender in ['男', '女']:
    mask_gender = df['gender'] == gender

    # 例如：按星座分組補值
    group_means = df[mask_gender].groupby('star_sign')[['weight', 'height']].transform('mean')

    # 如果分組後仍有 NaN（該分組內全為缺失），則用性別平均值補
    fallback_mean = df[mask_gender][['weight', 'height']].mean()

    df.loc[mask_gender, 'weight'] = df.loc[mask_gender, 'weight'].fillna(
        group_means['weight']
    ).fillna(fallback_mean['weight'])

    df.loc[mask_gender, 'height'] = df.loc[mask_gender, 'height'].fillna(
        group_means['height']
    ).fillna(fallback_mean['height'])
```

---

### 階段二：分群策略選擇（僅方法 3）

如果採用方法 3，需要決定用哪些特徵進行分群：

| 分群特徵組合 | 說明 | 優點 | 缺點 |
|------------|------|------|------|
| `star_sign` | 星座 | 簡單，類別數適中 | 可能無直接相關性 |
| `phone_os` | 手機系統 | 可能隱含年齡/收入 | 類別數少 |
| `sleepiness` | 睡眠程度 | 生活習慣相關 | 有缺失值 |
| `star_sign + phone_os` | 組合分群 | 更細緻 | 可能有空群組 |
| `KMeans(iq, fb_friends)` | 基於數值特徵聚類 | 數據驅動 | 較複雜，需調參 |

**建議優先級**：
1. `star_sign`（簡單且穩定）
2. `phone_os`（作為對照）
3. KMeans 聚類（如果前兩者效果不佳）

---

## 實驗流程

### 步驟 1: 準備實驗腳本
創建 `src/imputation_strategies.py`，實作各方法的補值邏輯。

### 步驟 2: 修改配置文件
在 `configs/` 下創建各方法的配置檔：
- `imputation_method0.yaml` (baseline)
- `imputation_method1.yaml` (論文裁剪 + 全局平均)
- `imputation_method2.yaml` (論文裁剪 + 論文平均)
- `imputation_method3_starSign.yaml` (論文裁剪 + 星座分群)
- `imputation_method3_phoneOS.yaml` (論文裁剪 + OS 分群)

### 步驟 3: 執行實驗
```bash
# 依序執行各配置
python main_train.py --config configs/imputation_method0.yaml
python main_train.py --config configs/imputation_method1.yaml
python main_train.py --config configs/imputation_method2.yaml
python main_train.py --config configs/imputation_method3_starSign.yaml
python main_train.py --config configs/imputation_method3_phoneOS.yaml
```

### 步驟 4: 結果比較
從 `experiments/experiment_log.csv` 中比較：
- Accuracy
- F1 Score
- Precision
- Recall
- CV 平均分數

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

### 假設
1. **方法 1** 可能因忽略性別差異而表現較差
2. **方法 2** 理論上最符合人體統計學，應有較好表現（前提是論文數據與競賽數據分布接近）
3. **方法 3** 如果分群有效，可能捕捉到更多隱藏模式

### 風險與注意事項
- ⚠️ **訓練集 vs 測試集分布差異**：論文範圍可能不適用於此競賽數據
- ⚠️ **過度擬合**：過於複雜的分群策略可能僅對訓練集有效
- ⚠️ **缺失不隨機（MNAR）**：如果缺失本身是重要特徵，補值可能抹除信號

---

## TODO
- [ ] 從論文中查詢並填入平均值數據
- [ ] EDA 確認當前數據的 weight/height 分布（與論文範圍比較）
- [ ] 實作 `src/imputation_strategies.py`
- [ ] 創建各方法的配置檔
- [ ] 執行實驗並記錄結果
- [ ] 分析最佳方法並撰寫結論

---

## 參考資料
1. [PMC8306797 - Anthropometric measurements](https://pmc.ncbi.nlm.nih.gov/articles/PMC8306797/)
2. 當前專案文檔：`training_workflow_main.md`, `TUNING_GUIDE.md`
