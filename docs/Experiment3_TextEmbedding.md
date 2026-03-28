# Experiment 3: 文本嵌入特徵實驗

## 📋 實驗目標

在 Exp 2 最優設定（Median補值 + BMI）的基礎上，加入 `self_intro`（自我介紹）文本特徵的嵌入表示。

---

## 🎯 支援的文本嵌入方法

### 1️⃣ MiniLM（深度學習語意嵌入）

**模型**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

**特點**:
- ✅ 多語言支援（中文、英文皆可）
- ✅ 捕捉語意相似性（不僅是關鍵詞匹配）
- ✅ 固定 384 維向量輸出
- ⚠️ 需要下載模型（約 420MB）
- ⚠️ 推理速度較慢（適合小數據集）

**配置範例**:
```yaml
features:
  text_cols: ["self_intro"]
  text_embedding_method: "minilm"  # 使用 MiniLM
```

**輸出**: `(n_samples, 384)` 的 numpy array

---

### 2️⃣ TF-IDF（統計詞頻嵌入）

**模型**: sklearn TfidfVectorizer

**特點**:
- ✅ 輕量快速，無需下載模型
- ✅ 捕捉關鍵詞重要性
- ✅ 可調整特徵數量（50-200）
- ⚠️ 不理解語意（只懂詞頻）
- ⚠️ 需要足夠的訓練文本

**配置範例**:
```yaml
features:
  text_cols: ["self_intro"]
  text_embedding_method: "tfidf"  # 使用 TF-IDF
  tfidf_max_features: 50  # 最多保留 50 個特徵
```

**TF-IDF 參數**:
- `max_features`: 最大特徵數（建議: 50-80 for 短文本）
- `ngram_range`: (1, 1) 只用 unigram
- `min_df`: 1（保留所有出現的詞）
- `max_df`: 0.95（只過濾極端高頻詞）

**輸出**: `(n_samples, max_features)` 的 numpy array

---

### 3️⃣ 兩者併用（TF-IDF + MiniLM）✨ 新增

**模型**: TF-IDF + MiniLM 特徵拼接

**特點**:
- ✅ 結合兩者優勢：關鍵詞 + 語意
- ✅ TF-IDF 捕捉顯性特徵，MiniLM 捕捉隱性語意
- ✅ 特徵維度更豐富（50 + 384 = 434 維）
- ⚠️ 特徵維度較高，需要防止過擬合
- ⚠️ 訓練和推理時間最長

**配置範例**:
```yaml
features:
  text_cols: ["self_intro"]
  text_embedding_method: "both"  # ✅ 同時使用兩者
  tfidf_max_features: 50  # TF-IDF 部分的特徵數
```

**輸出**: `(n_samples, tfidf_max_features + 384)` 的 numpy array

**特徵維度**: 基礎特徵 + 384 維 MiniLM

---

### Exp 3-2: TF-IDF 嵌入
**配置檔**: `configs/exp3_exp2_with_bmi_tfidf.yaml`

```yaml
features:
  text_cols: ["self_intro"]
  text_embedding_method: "tfidf"
  tfidf_max_features: 50
```

**執行**:
```powershell
python main_train.py --config configs/exp3_exp2_with_bmi_tfidf.yaml
```

**特徵維度**: 基礎特徵 + 50 維 TF-IDF

---

### Exp 3-3: 兩者併用 ✨ 新增
**配置檔**: `configs/exp3_exp2_with_bmi_both.yaml`

```yaml
features:
  text_cols: ["self_intro"]
  text_embedding_method: "both"  # ✅ 同時使用
  tfidf_max_features: 50
```

**執行**:
```powershell
python main_train.py --config configs/exp3_exp2_with_bmi_both.yaml
```

**特徵維度**: 基礎特徵 + 50 維 TF-IDF + 384 維 MiniLM = **434 維文本特徵**

**優勢**:
- TF-IDF 捕捉顯性關鍵詞（如 "籃球"、"化妝"）
- MiniLM 捕捉隱性語意（如描述風格、情感傾向）
- 互補性強，可能帶來最佳性能

**風險**:
- 特徵維度增加，過擬合風險上升
- 需要更強的正則化（如增加 l2_leaf_reg） Exp 3-1: MiniLM 嵌入
**配置檔**: `configs/exp3_exp2_with_bmi_minilm.yaml`

```yaml
data:
  drop_cols: ["id", "yt"]  # ✅ 不刪除 self_intro

features:
  numeric_cols: ["height", "weight", "iq", "bmi"]
  numeric_log_cols: ["fb_friends"]
  text_cols: ["self_intro"]
  text_embedding_method: "minilm"
```

**執行**:
```powershell
python main_train.py --config configs/exp3_exp2_with_bmi_minilm.yaml
```

---

### Exp 3-2: TF-IDF 嵌入
**配置檔**: `configs/exp3_exp2_with_bmi_tfidf.yaml`

```yaml
features:
  text_cols: ["self_intro"]
  text_embedding_method: "tfidf"
  tfidf_max_features: 100
```

**執行**:
```powershell
python main_train.py --config configs/exp3_exp2_with_bmi_tfidf.yaml
```

---

## 🔧 技術實現

### 特徵處理流程

``` 過擬合風險 |
|------|---------|---------|---------|---------|-----------|
| Exp 2-3 | 基礎 + BMI | 無 | 0 | Baseline | 低 |
| Exp 3-1 | 基礎 + BMI + MiniLM | 語意向量 | +384 | ⭐⭐⭐ 中高 | 中 |
| Exp 3-2 | 基礎 + BMI + TF-IDF | 詞頻向量 | +50 | ⭐⭐ 中低 | 低 |
| Exp 3-3 | 基礎 + BMI + Both | 兩者併用 | +434 | ⭐⭐⭐⭐ 最高 | **高** ⚠️ |

**預測**:
- **MiniLM (Exp 3-1)** 可能捕捉到性別相關的語言風格（如描述方式、用詞偏好）
- **TF-IDF (Exp 3-2)** 可能捕捉到性別相關的關鍵詞（如興趣、職業）
- **兩者併用 (Exp 3-3)** 結合兩者優勢，可能達到最高性能，但需要防止過擬合
- 如果 `self_intro` 訊息量豐富，預期：
  - Exp 3-2: **+0.3-0.5% F1**
  - Exp 3-1: **+0.5-1.0% F1**
  - Exp 3-3: **+0.8-1.5% F1**（若無過擬合）
  │   ↓ ColumnTransformer 拼接所有特徵
  │   ↓ 進入 CatBoost 訓練
```

### 核心類別

**`TextEmbeddingTransformer`** ([src/features.py](../src/features.py)):
- 繼承自 `BaseEstimator`, `TransformerMixin`
- 支援 sklearn Pipeline 整合
- `fit()`: 加載模型或訓練 TF-IDF vectorizer
- `transform()`: 將文本轉換為 numpy array
- 返回格式符合 sklearn 規範

**重要**:
- sklearn Pipeline 要求返回 **numpy.ndarray**（不能是 torch.Tensor）
- 使用 `sentence_embeddings.cpu().numpy()` 轉換
- TF-IDF 使用 `.toarray()` 轉為 dense array

---

## 📊 預期結果對比

| 實驗 | 特徵組合 | 文本嵌入 | 新增維度 | 預期增益 |
|------|---------|---------|---------|---------|
| Exp 2-3 | 基礎 + BMI | 無 | 0 | Baseline |
| Exp 3-1 | 基礎 + BMI + MiniLM | 語意向量 | +384 | ⭐⭐⭐ 中高 |
| Exp 3-2 | 基礎 + BMI + TF-IDF | 詞頻向量 | +100 | ⭐⭐ 中低 |

**預測**:
- **MiniLM** 可能捕捉到性別相關的語言風格（如描述方式、用詞偏好）
- **TF-IDF** 可能捕捉到性別相關的關鍵詞（如興趣、職業）
- 如果 `self_intro` 訊息量豐富，預期 **+0.5-1.5% F1** 提升

---

## 🚀 執行步驟

### 1. 確認依賴已安裝

檢查 [requirements.txt](../requirements.txt):
```txt
transformers>=4.30.0
torch>=2.0.0
```

安裝命令:
```powershell
pip install transformers torch
```

### 2. 執行實驗

**運行 MiniLM 實驗**:
```powershell
python main_train.py --config configs/exp3_exp2_with_bmi_minilm.yaml
```

**運行 TF-IDF 實驗**:
```powershell
python main_train.py --config configs/exp3_exp2_with_bmi_tfidf.yaml
```

### 3. 查看結果

實驗結果會自動記錄在:
- `experiments/experiment_log.csv` (總表)
- `experiments/exp_XXX_*/` (個別實驗資料夾)

對比 F1-Score、Accuracy、特徵維度變化。

---

## 💡 進階調整

### TF-IDF 超參數調整

如果 TF-IDF 效果不佳，可嘗試:
```yaml
tfidf_max_features: 50   # 減少特徵數（避免過擬合）
tfidf_max_features: 200  # 增加特徵數（捕捉更多資訊）
```

### MiniLM 替代模型

如需更快速度，可替換為更小的模型:
```python
# 在 TextEmbeddingTransformer 中修改
'sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2'  # 更小（22MB），速度更快
```

### 組合兩種方法

未來可考慮同時使用:
```yaml
text_cols: ["self_intro"]
text_embedding_method: "both"  # 同時使用 MiniLM + TF-IDF（需額外實現）
```

---

## 📝 後續規劃

1. **Exp 3-1 vs Exp 3-2 對比** → 選出最優文本嵌入方法
2. **文本清理**: 移除特殊符號、統一大小寫
3. **PCA 降維**: 如果維度過高導致過擬合，嘗試降維到 50-100 維
4. **嵌入微調**: 使用性別標籤 fine-tune MiniLM（進階）

---

## ✅ Checklist

- [x] 實現 `TextEmbeddingTransformer` 類別
- [x] 支援 MiniLM 和 TF-IDF
- [x] 整合到 `build_preprocessor()`
- [x] 創建兩個實驗配置檔案
- [x] 更新 requirements.txt
- [ ] 執行訓練並記錄結果
- [ ] 撰寫實驗分析報告

---

**參考文獻**:
- Sentence-BERT: https://www.sbert.net/
- TF-IDF: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
