# Experiment 3 快速對比表

## 🎯 文本嵌入方法對比

### A 組：基於 Exp2 設定 (depth=6, 簡化 param_grid)

| 維度 | Exp 3-1: MiniLM | Exp 3-2: TF-IDF | Exp 3-3: 兩者併用 | Exp 3-4: MiniLM+PCA30 | Exp 3-5: Both+PCA30 | Exp 3-4b: MiniLM+PCA64 | Exp 3-5b: Both+PCA64 |
|------|----------------|----------------|-----------------|---------------------|-------------------|---------------------|-------------------|
| **配置檔** | `exp3_exp2_with_bmi_minilm.yaml` | `exp3_exp2_with_bmi_tfidf.yaml` | `exp3_exp2_with_bmi_both.yaml` | `exp3_minilm_pca30.yaml` | `exp3_both_pca30.yaml` | `exp3_minilm_pca64.yaml` | `exp3_both_pca64.yaml` |
| **text_embedding_method** | `"minilm"` | `"tfidf"` | `"both"` | `"minilm"` | `"both"` | `"minilm"` | `"both"` |
| **PCA** | ❌ | ❌ | ❌ | ✅ 30維 | ✅ 30維 | ✅ 64維 | ✅ 64維 |
| **特徵維度** | ~402 | ~68 | ~452 | ~48 | ~98 | ~82 | ~132 |
| **CatBoost depth** | 6 | 6 | 6 | 6 | 6 | 6 | 6 |

### B 組：基於 Exp1 設定 (depth=8, 完整 param_grid)

| 維度 | Exp 3-6: MiniLM | Exp 3-7: TF-IDF | Exp 3-8: 兩者併用 | Exp 3-9: MiniLM+PCA30 | Exp 3-10: Both+PCA30 | Exp 3-9b: MiniLM+PCA64 | Exp 3-10b: Both+PCA64 |
|------|----------------|----------------|-----------------|---------------------|-------------------|---------------------|-------------------|
| **配置檔** | `exp3_minilm_exp1base.yaml` | `exp3_tfidf_exp1base.yaml` | `exp3_both_exp1base.yaml` | `exp3_minilm_pca30_exp1base.yaml` | `exp3_both_pca30_exp1base.yaml` | `exp3_minilm_pca64_exp1base.yaml` | `exp3_both_pca64_exp1base.yaml` |
| **text_embedding_method** | `"minilm"` | `"tfidf"` | `"both"` | `"minilm"` | `"both"` | `"minilm"` | `"both"` |
| **PCA** | ❌ | ❌ | ❌ | ✅ 30維 | ✅ 30維 | ✅ 64維 | ✅ 64維 |
| **特徵維度** | ~402 | ~68 | ~452 | ~48 | ~98 | ~82 | ~132 |
| **CatBoost depth** | 8 | 8 | 8 | 8 | 8 | 8 | 8 |

### A 組 vs B 組差異

| 設定 | A 組 (Exp2 base) | B 組 (Exp1 base) |
|------|-----------------|-----------------|
| `train_path` | `train.csv` | `train.csv` |
| `test_path` | `test.csv` | `test.csv` |
| `catboost depth` | 6 | **8** |
| `param_grid` | 僅 catboost | 完整四模型 (xgb/lgbm/rf/catboost) |
| `param_grid_quick` | 無 | 有 |

### C 組：手工文本特徵 (基於 Exp1 設定, depth=8)

> 不使用嵌入模型 (MiniLM / TF-IDF)，改用從 `self_intro` 萃取的低維度手工特徵。
> 透過 `TextHandcraftedTransformer` 實現，位於 `src/features.py`。

| 維度 | Exp 3-11: HC(12) | Exp 3-12: TF-IDF+HC(12) | Exp 3-13: HC(12)+Search | Exp 3-14: HC(15) | Exp 3-15: HC(15)+Search |
|------|-----------------|------------------------|------------------------|-----------------|------------------------|
| **配置檔** | `exp3_handcrafted_exp1base.yaml` | `exp3_tfidf_handcrafted_exp1base.yaml` | `exp3_handcrafted_exp1base_enablesearch.yaml` | `exp3_handcrafted15_exp1base.yaml` | `exp3_handcrafted15_exp1base.yaml` (search=true) |
| **text_embedding_method** | `"handcrafted"` | `"tfidf"` | `"handcrafted"` | `"handcrafted"` | `"handcrafted"` |
| **use_text_handcrafted** | (自動啟用) | `true` | (自動啟用) | (自動啟用) | (自動啟用) |
| **tfidf_max_features** | — | 20 | — | — | — |
| **手工特徵維度** | 12 維 | 12 維 | 12 維 | 15 維 | 15 維 |
| **總文本特徵維度** | 12 | 32 (20+12) | 12 | 15 | 15 |
| **Grid Search** | ❌ | ❌ | ✅ full | ❌ | ✅ full |
| **CatBoost depth** | 8 | 8 | (search) | 8 | (search) |

#### 手工特徵說明 (15 維)

`TextHandcraftedTransformer` 從 `self_intro` 欄位萃取以下特徵：

| # | 特徵名稱 | 說明 | 性別區分度 |
|---|---------|------|-----------|
| 1 | `text_len` | 文字總長度 | p=0.10 |
| 2 | `word_count` | 詞數 (空格分割) | p=0.18 |
| 3 | `avg_word_len` | 平均每個詞的長度 | p=0.34 |
| 4 | `is_empty` | 是否為空文本 (1/0) | p=0.73 |
| 5 | `chinese_ratio` | 中文字元佔比 | p=0.025 * |
| 6 | `upper_ratio` | 大寫字母佔比 | p=0.99 |
| 7 | `punctuation_count` | 標點符號數量 `.,!?;:~-` | p=0.06 |
| 8 | `has_self_ref` | 是否包含第一人稱 (I / I'm / I am) | p=0.30 |
| 9 | **`has_male_keyword`** | 是否包含男性傾向關鍵詞 | **p<0.0001 \*\*\*** |
| 10 | **`has_female_keyword`** | 是否包含女性傾向關鍵詞 | **p<0.0001 \*\*\*** |
| 11 | `sentiment_positive` | 正向形容詞出現次數 | p=0.82 |
| 12 | `ends_with_period` | 是否以句號 `.` 結尾 | p=0.07 |
| 13 | **`is_all_lower`** | 全小寫文本 (M=0.152 vs F=0.065) | **p=0.006 \*\*** |
| 14 | **`has_number`** | 含數字字元 (100% 男性) | **p=0.002 \*\*** |
| 15 | **`is_short_simple`** | ≤8字元且無空格 (M=0.354 vs F=0.234) | **p=0.015 \*** |

**關鍵詞清單：**
- 男性傾向：`handsome`, `man`, `boy`, `cool`, `nerd`, `hard`, `hungry`, `super`, `foolish`
- 女性傾向：`beautiful`, `girl`, `brilliant`, `cute`, `introvert`, `never`
- 正向形容詞：`happy`, `good`, `nice`, `smart`, `positive`, `love`, `great`, `wonderful`, `amazing`

#### 啟用方式

**方式 A：只用手工特徵 (15 維)**
```yaml
features:
  text_cols: ["self_intro"]
  text_embedding_method: "handcrafted"
```

**方式 B：TF-IDF + 手工特徵併用 (20+15 = 35 維)**
```yaml
features:
  text_cols: ["self_intro"]
  text_embedding_method: "tfidf"
  tfidf_max_features: 20
  use_text_handcrafted: true
```

#### 設計理念

- **低維度優先**：423 筆樣本配 15 維 (28:1 樣本特徵比)，避免維度災難
- **有監督信號**：`has_male_keyword` (t=+7.6) 和 `has_female_keyword` (t=-4.3) 具有極顯著性別區分度
- **新增 3 特徵**：`is_all_lower` (p=0.006)、`has_number` (p=0.002, 100% 男性)、`is_short_simple` (p=0.015) 均通過 t-test 顯著性檢定
- **可與嵌入併用**：透過 `use_text_handcrafted: true` 可在任何嵌入方法上額外疊加手工特徵

---

## 🚀 訓練指令

### A 組（基於 Exp2 設定）

```powershell
# Exp 3-1: MiniLM (實驗編號 1)
python main_train.py --config configs/exp3_exp2_with_bmi_minilm.yaml

# Exp 3-2: TF-IDF (實驗編號 2)
python main_train.py --config configs/exp3_exp2_with_bmi_tfidf.yaml

# Exp 3-3: 兩者併用 (實驗編號 3)
python main_train.py --config configs/exp3_exp2_with_bmi_both.yaml

# Exp 3-4: MiniLM + PCA30 (實驗編號 4)
python main_train.py --config configs/exp3_minilm_pca30.yaml

# Exp 3-5: Both + PCA30 (實驗編號 5)
python main_train.py --config configs/exp3_both_pca30.yaml

# Exp 3-4b: MiniLM + PCA64
python main_train.py --config configs/exp3_minilm_pca64.yaml

# Exp 3-5b: Both + PCA64
python main_train.py --config configs/exp3_both_pca64.yaml
```

### B 組（基於 Exp1 設定）

```powershell
# Exp 3-6: MiniLM (Exp1 base)
python main_train.py --config configs/exp3_minilm_exp1base.yaml

# Exp 3-7: TF-IDF (Exp1 base)
python main_train.py --config configs/exp3_tfidf_exp1base.yaml

# Exp 3-8: 兩者併用 (Exp1 base)
python main_train.py --config configs/exp3_both_exp1base.yaml

# Exp 3-9: MiniLM + PCA30 (Exp1 base)
python main_train.py --config configs/exp3_minilm_pca30_exp1base.yaml

# Exp 3-10: Both + PCA30 (Exp1 base)
python main_train.py --config configs/exp3_both_pca30_exp1base.yaml

# Exp 3-9b: MiniLM + PCA64 (Exp1 base)
python main_train.py --config configs/exp3_minilm_pca64_exp1base.yaml

# Exp 3-10b: Both + PCA64 (Exp1 base)
python main_train.py --config configs/exp3_both_pca64_exp1base.yaml
```

### C 組（手工文本特徵）

```powershell
# Exp 3-11: 手工文本特徵 (12維)      → exp_016
python main_train.py --config configs/exp3_handcrafted_exp1base.yaml

# Exp 3-12: TF-IDF(20維) + 手工(12維) → exp_017
python main_train.py --config configs/exp3_tfidf_handcrafted_exp1base.yaml

# Exp 3-13: 手工文本特徵 (12維) + Grid Search → exp_018
python main_train.py --config configs/exp3_handcrafted_exp1base_enablesearch.yaml

# Exp 3-14: 手工文本特徵 (15維)      → exp_019
python main_train.py --config configs/exp3_handcrafted15_exp1base.yaml

# Exp 3-15: 手工文本特徵 (15維) + Grid Search → exp_020
python main_train.py --config configs/exp3_handcrafted15_exp1base.yaml  # (search.enabled: true)
```

---

## 🔮 預測指令

> 預測前需確認 test.csv 存在，且該實驗已完成訓練。
> 實驗編號 N 對應 `experiments/exp_00N_*` 資料夾。

### A 組（已完成訓練：exp 1-5）

```powershell
# Exp 3-1: MiniLM
python main_predict.py 1

# Exp 3-2: TF-IDF
python main_predict.py 2

# Exp 3-3: 兩者併用
python main_predict.py 3

# Exp 3-4: MiniLM + PCA30
python main_predict.py 4

# Exp 3-5: Both + PCA30
python main_predict.py 5
```

### B 組（實際實驗編號 7-11）

```powershell
# Exp 3-6: MiniLM (Exp1 base)       → exp_007
python main_predict.py 7

# Exp 3-7: TF-IDF (Exp1 base)       → exp_008
python main_predict.py 8

# Exp 3-8: 兩者併用 (Exp1 base)      → exp_009
python main_predict.py 9

# Exp 3-9: MiniLM + PCA30 (Exp1 base) → exp_010
python main_predict.py 10

# Exp 3-10: Both + PCA30 (Exp1 base)  → exp_011
python main_predict.py 11
```

> ⚠️ exp_006 為路徑修正前的失敗訓練（無模型），跳過不用。

### PCA64 實驗（A 組 + B 組，訓練後執行）

```powershell
# Exp 3-4b: MiniLM + PCA64 (depth=6)   → 訓練後查看 experiments/ 取得編號
python main_predict.py 12

# Exp 3-5b: Both + PCA64 (depth=6)
python main_predict.py 13

# Exp 3-9b: MiniLM + PCA64 (depth=8)
python main_predict.py 14

# Exp 3-10b: Both + PCA64 (depth=8)
python main_predict.py 15
```

> 💡 PCA64 為 PCA30 和無 PCA 之間的中間方案，預期在效能和穩定性間取得平衡。

### C 組（手工文本特徵，實際實驗編號 16-20）

```powershell
# Exp 3-11: 手工文本特徵 (12維)       → exp_016
python main_predict.py 16

# Exp 3-12: TF-IDF(20維) + 手工(12維) → exp_017
python main_predict.py 17

# Exp 3-13: 手工(12維) + Grid Search  → exp_018
python main_predict.py 18

# Exp 3-14: 手工文本特徵 (15維)       → exp_019 (無預測，未啟用 search)
# python main_predict.py 19

# Exp 3-15: 手工(15維) + Grid Search  → exp_020
python main_predict.py 20
```

---

## 📈 特徵拼接示意圖

### Exp 3-1: MiniLM Only
```
基礎特徵 (height, weight, iq, fb_friends, bmi, star_sign, phone_os, sleepiness)
    ↓
ColumnTransformer 拼接
    ├─ 數值特徵: ~7-10 維
    ├─ 類別特徵 (OneHot): ~15-20 維
    └─ 文本特徵 (MiniLM): 384 維
    ↓
總特徵: ~406-414 維
```

### Exp 3-2: TF-IDF Only
```
基礎特徵
    ↓
ColumnTransformer 拼接
    ├─ 數值特徵: ~7-10 維
    ├─ 類別特徵 (OneHot): ~15-20 維
    └─ 文本特徵 (TF-IDF): 50 維
    ↓
總特徵: ~72-80 維
```

### Exp 3-3: Both (兩者併用)
```
基礎特徵
    ↓
ColumnTransformer 拼接
    ├─ 數值特徵: ~7-10 維
    ├─ 類別特徵 (OneHot): ~15-20 維
    └─ 文本特徵:
        ├─ TF-IDF: 50 維
        └─ MiniLM: 384 維
    ↓
總特徵: ~456-464 維 ⚠️（最高維度）
```

---

## 🔍 如何選擇？

### 推薦策略

1. **先跑 Exp 3-2 (TF-IDF)**
   - ✅ 速度最快（5-10分鐘）
   - ✅ 建立 Baseline
   - ✅ 檢驗文本特徵是否有效

2. **再跑 Exp 3-1 (MiniLM)**
   - ⏰ 需要 20-30 分鐘（首次下載模型）
   - 📊 對比是否優於 TF-IDF
   - 🎯 確認語意信息的價值

3. **最後跑 Exp 3-3 (Both)**
   - ⏰ 最慢（25-35 分鐘）
   - 🎯 只在前兩者都有提升時嘗試
   - ⚠️ 注意 CV Std（過擬合指標）

### 判斷標準

**如果 Exp 3-2 vs Exp 2-3**:
- F1 提升 < 0.2% → 文本特徵無效，放棄 ❌
- F1 提升 0.2-0.5% → 有小幅幫助，可嘗試 MiniLM
- F1 提升 > 0.5% → 文本很重要，必試 MiniLM + Both

**如果 Exp 3-1 vs Exp 3-2**:
- MiniLM 更優 → 語意信息重要，優先用 3-3
- TF-IDF 更優 → 關鍵詞主導，保持 3-2
- 兩者接近 → 用 TF-IDF（更快）

**如果 Exp 3-3 出現過擬合**:
```yaml
# 增加正則化
catboost_params:
  l2_leaf_reg: 5  # 原 3，增加到 5
  depth: 4  # 原 6，降低到 4
```

---

## 📊 結果記錄表格

### A 組結果（基於 Exp2 設定, depth=6）

| 實驗 | F1-Score | Accuracy | CV Std (F1) | 總特徵數 | 備註 |
|------|----------|----------|-------------|---------|------|
| Exp 2-3 (Baseline) | 0.9276 | 0.8914 | 0.0259 | ~30-40 | 無文本 |
| Exp 3-1 (MiniLM) | 0.9098 | 0.8630 | 0.0184 | ~402 | 語意嵌入 |
| Exp 3-2 (TF-IDF) | 0.9126 | 0.8701 | 0.0210 | ~68 | 詞頻嵌入 |
| Exp 3-3 (Both) | 0.9129 | 0.8677 | 0.0162 | ~452 | 兩者併用 |
| Exp 3-4 (MiniLM+PCA30) | 0.9173 | 0.8748 | 0.0173 | ~48 | PCA降維 |
| Exp 3-5 (Both+PCA30) | 0.9158 | 0.8724 | 0.0175 | ~98 | Both+PCA |
| Exp 3-4b (MiniLM+PCA64) | 0.9128 | 0.8676 | 0.0112 | ~82 | PCA64 |
| Exp 3-5b (Both+PCA64) | 0.9062 | 0.8582 | 0.0162 | ~132 | PCA64 |

### B 組結果（基於 Exp1 設定, depth=8）

| 實驗 | F1-Score | Accuracy | CV Std (F1) | 總特徵數 | 備註 |
|------|----------|----------|-------------|---------|------|
| Exp 3-6 (MiniLM) | 0.9153 | 0.8700 | 0.0110 | ~402 | depth=8 |
| Exp 3-7 (TF-IDF) | **0.9274** | **0.8913** | 0.0183 | ~68 | depth=8 |
| Exp 3-8 (Both) | 0.9162 | 0.8724 | 0.0170 | ~452 | depth=8 |
| Exp 3-9 (MiniLM+PCA30) | 0.9177 | 0.8748 | 0.0191 | ~48 | depth=8 |
| Exp 3-10 (Both+PCA30) | 0.9194 | 0.8771 | 0.0133 | ~98 | depth=8 |
| Exp 3-9b (MiniLM+PCA64) | 0.9114 | 0.8653 | 0.0096 | ~82 | PCA64 |
| Exp 3-10b (Both+PCA64) | 0.9132 | 0.8677 | 0.0193 | ~132 | PCA64 |

### C 組結果（手工文本特徵, depth=8）

| 實驗 | F1-Score | Accuracy | CV Std (F1) | 總特徵數 | Grid Search | 備註 |
|------|----------|----------|-------------|---------|-------------|------|
| Exp 3-11 (HC 12維) | 0.9194 | 0.8795 | 0.0250 | ~30 | ❌ | 手工特徵12維 |
| Exp 3-12 (TF-IDF+HC 12維) | 0.9192 | 0.8795 | 0.0285 | ~50 | ❌ | TF-IDF20維+手工12維 |
| Exp 3-13 (HC 12維+Search) | 0.9229 | 0.8866 | 0.0167 | ~30 | ✅ full | **CV Std 最低** |
| Exp 3-14 (HC 15維) | 0.9173 | 0.8771 | 0.0269 | ~33 | ❌ | 15維(+all_lower/has_number/short_simple) |
| Exp 3-15 (HC 15維+Search) | **0.9241** | **0.8866** | 0.0241 | ~33 | ✅ full | **C組最佳 F1** |

### 總結

- **B 組全面優於 A 組**（depth=8 > depth=6）
- **最佳 Exp3**: Exp 3-7 (TF-IDF + depth=8) F1=0.9274，幾乎追平 Exp2 Baseline (0.9276)
- **最穩定**: Exp 3-6 (MiniLM + depth=8) CV Std=0.0110 最低
- **PCA 有效**: PCA30 的 B 組均優於無 PCA 的 A 組對應實驗
- **Grid Search 顯著提升**: Exp 3-13/3-15 啟用 search 後 F1 從 ~0.919 提升至 ~0.923-0.924
- **15維 + Search 最佳**: Exp 3-15 (HC 15維+Search) F1=0.9241 為 C 組最高，且 full_train_acc=1.0
- **手工特徵穩定超越嵌入**: 所有 HC 實驗 (12/15維) 均優於大多數 MiniLM/Both 嵌入方法
- **新增 3 特徵有效**: 15 維在搭配 Grid Search 時 F1=0.9241 > 12 維的 0.9229

---

## 🔮 預測結果

### 預測分布統計

| 實驗 | 實驗編號 | 預測檔案 | 男 (1) | 女 (2) | 總數 |
|------|---------|---------|--------|--------|------|
| Exp 3-1 (MiniLM) | exp_001 | `submission_exp_001_exp3_minilm_full.csv` | 316 | 110 | 426 |
| Exp 3-2 (TF-IDF) | exp_002 | `submission_exp_002_exp3_tfidf_full.csv` | 293 | 133 | 426 |
| Exp 3-3 (Both) | exp_003 | `submission_exp_003_exp3_both_full.csv` | 312 | 114 | 426 |
| Exp 3-4 (MiniLM+PCA30) | exp_004 | `submission_exp_004_exp3_minilm_pca30_full.csv` | 311 | 115 | 426 |
| Exp 3-5 (Both+PCA30) | exp_005 | `submission_exp_005_exp3_both_pca30_full.csv` | 305 | 121 | 426 |
| Exp 3-6 (MiniLM, d=8) | exp_007 | `submission_exp_007_exp3_minilm_exp1base_full.csv` | 315 | 111 | 426 |
| Exp 3-7 (TF-IDF, d=8) | exp_008 | `submission_exp_008_exp3_tfidf_exp1base_full.csv` | 297 | 129 | 426 |
| Exp 3-8 (Both, d=8) | exp_009 | `submission_exp_009_exp3_both_exp1base_full.csv` | 316 | 110 | 426 |
| Exp 3-9 (MiniLM+PCA30, d=8) | exp_010 | `submission_exp_010_exp3_minilm_pca30_exp1base_full.csv` | 309 | 117 | 426 |
| Exp 3-10 (Both+PCA30, d=8) | exp_011 | `submission_exp_011_exp3_both_pca30_exp1base_full.csv` | 315 | 111 | 426 |
| Exp 3-4b (MiniLM+PCA64) | exp_012 | `submission_exp_012_exp3_minilm_pca64_full.csv` | 314 | 112 | 426 |
| Exp 3-5b (Both+PCA64) | exp_013 | `submission_exp_013_exp3_both_pca64_full.csv` | 316 | 110 | 426 |
| Exp 3-9b (MiniLM+PCA64, d=8) | exp_014 | `submission_exp_014_exp3_minilm_pca64_exp1base_full.csv` | 315 | 111 | 426 |
| Exp 3-10b (Both+PCA64, d=8) | exp_015 | `submission_exp_015_exp3_both_pca64_exp1base_full.csv` | 314 | 112 | 426 |
| Exp 3-11 (HC 12維, d=8) | exp_016 | `submission_exp_016_exp3_handcrafted_exp1base_full.csv` | 302 | 124 | 426 |
| Exp 3-12 (TF-IDF+HC, d=8) | exp_017 | `submission_exp_017_exp3_tfidf_handcrafted_exp1base_full.csv` | 304 | 122 | 426 |
| Exp 3-13 (HC 12維+Search) | exp_018 | `submission_exp_018_exp3_handcrafted_exp1base_full.csv` | 283 | 143 | 426 |
| Exp 3-15 (HC 15維+Search) | exp_020 | `submission_exp_020_exp3_handcrafted15_exp1base_full.csv` | 299 | 127 | 426 |

### 預測分布觀察

- **TF-IDF 系列** (exp_002, exp_008) 預測男女比較均衡（~293-297 男 / 129-133 女）
- **MiniLM 系列** 偏向預測男性（~309-316 男 / 110-117 女）
- **Both 系列** 行為接近 MiniLM（MiniLM 嵌入維度佔主導地位）
- **PCA 降維** 使預測比例稍微往均衡方向調整
- **手工特徵系列** (exp_016-020) 預測男女比例最均衡 (~283-304 男 / 122-143 女)，優於 MiniLM 系列
- **Grid Search 系列** (exp_018, exp_020) 預測更多女性 (143/127)，模型更均衡

所有預測檔案位於 `result/` 資料夾。

**關鍵指標**:
- F1-Score 越高越好
- CV Std < 0.03 為穩定，> 0.05 可能過擬合
- 總特徵數越高，過擬合風險越大

---

## 💡 進階調整

### 如果 Exp 3-3 過擬合

#### 方案 1: 降低 TF-IDF 特徵數
```yaml
tfidf_max_features: 30  # 從 50 降到 30
```

#### 方案 2: 對文本特徵使用 PCA 降維
```python
# 在 TextEmbeddingTransformer 中添加 PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=50)  # 將 434 維降到 50 維
```

#### 方案 3: 增加模型正則化
```yaml
catboost_params:
  l2_leaf_reg: 7
  depth: 4
  min_data_in_leaf: 10  # 新增
```

---

## ✅ 檢查清單

- [ ] 確認 `transformers` 和 `torch` 已安裝
- [ ] 確認 `self_intro` 未被刪除（檢查 data.drop_cols）
- [ ] 先運行 Exp 3-2（TF-IDF，最快）
- [ ] 再運行 Exp 3-1（MiniLM，中速）
- [ ] 最後運行 Exp 3-3（Both，最慢）
- [ ] 記錄所有實驗的 F1-Score 和 CV Std
- [ ] 對比 `experiments/experiment_log.csv`
- [ ] 檢查是否過擬合（CV Std）
- [ ] 選出最優配置進行超參數調整

---

## 📖 參考文檔

- 完整說明: [docs/Experiment3_TextEmbedding.md](Experiment3_TextEmbedding.md)
- TF-IDF 參數指南: [docs/TF-IDF_Parameters_Guide.md](TF-IDF_Parameters_Guide.md)
- 實驗 2 結果: [docs/Experiment2_Features.md](Experiment2_Features.md)
