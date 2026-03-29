# EXPERIMENT 3 全量對比（文本特徵方法）

資料來源：
- [experiments/experiment_record_exp3.csv](experiments/experiment_record_exp3.csv)
- [experiments/experiment_record_exp3_latest_comparison.csv](experiments/experiment_record_exp3_latest_comparison.csv)
- 共 18 組成功實驗（status=success）

## 0. method0 ~ method3 是什麼（Exp3 文本方法代碼對照）

Exp3 在同一個 CatBoost 主架構下，比較不同文本特徵策略與 PCA 設定。

後續若提到 method0~3，對照如下：

| 方法 | 對應文本方法 | 說明 |
|---|---|---|
| method0 | minilm | MiniLM 嵌入，使用語意向量 |
| method1 | tfidf | TF-IDF 詞頻向量 |
| method2 | both | MiniLM + TF-IDF 串接 |
| method3 | handcrafted | 手工文本特徵（低維可解釋） |

補充標記（非 method 編號）：

| 類別 | 設定重點 | 說明 |
|---|---|---|
| pca30 / pca64 | PCA 降維 | 針對高維嵌入降維 |
| enablesearch | search.enabled=true | 啟用超參數搜尋 |

註：Exp3 的 config 檔名多以 `tfidf` / `minilm` / `both` / `handcrafted` 命名，並非直接使用 method0~3；此處 method 代碼僅用於報告內統一閱讀。

## 1. 全體最佳組合

本輪最佳 F1 出現在 TF-IDF 路線：

- [configs/exp3_tfidf_exp1base.yaml](configs/exp3_tfidf_exp1base.yaml)

次佳為：

- [configs/exp3_exp2_with_bmi_tfidf.yaml](configs/exp3_exp2_with_bmi_tfidf.yaml)

最佳分數：
- Mean F1: 0.9221
- Mean Accuracy: 0.8843

次佳分數：
- Mean F1: 0.9201
- Mean Accuracy: 0.8819

## 2. 全部 18 組結果總表

說明：表格欄位為 Mean Accuracy / Mean F1 / Mean Precision / Mean Recall / F1 Std。

| 配置檔 | 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---|---:|---:|---:|---:|---:|
| exp3_both_exp1base.yaml | both | 0.8748 | 0.9182 | 0.8978 | 0.9400 | 0.0174 |
| exp3_both_pca30.yaml | both | 0.8748 | 0.9177 | 0.9001 | 0.9368 | 0.0142 |
| exp3_both_pca30_exp1base.yaml | both | 0.8747 | 0.9179 | 0.9005 | 0.9368 | 0.0126 |
| exp3_both_pca64.yaml | both | 0.8724 | 0.9163 | 0.8975 | 0.9368 | 0.0107 |
| exp3_both_pca64_exp1base.yaml | both | 0.8700 | 0.9148 | 0.8970 | 0.9337 | 0.0197 |
| exp3_exp2_with_bmi_both.yaml | both | 0.8701 | 0.9153 | 0.8920 | 0.9399 | 0.0107 |
| exp3_exp2_with_bmi_minilm.yaml | minilm | 0.8724 | 0.9167 | 0.8946 | 0.9400 | 0.0214 |
| exp3_exp2_with_bmi_tfidf.yaml | tfidf | 0.8819 | 0.9201 | 0.9213 | 0.9210 | 0.0332 |
| exp3_handcrafted15_exp1base.yaml | handcrafted | 0.8724 | 0.9138 | 0.9205 | 0.9083 | 0.0254 |
| exp3_handcrafted_exp1base.yaml | handcrafted | 0.8724 | 0.9138 | 0.9205 | 0.9083 | 0.0254 |
| exp3_handcrafted_exp1base_enablesearch.yaml | handcrafted | 0.8795 | 0.9182 | 0.9292 | 0.9084 | 0.0287 |
| exp3_minilm_exp1base.yaml | minilm | 0.8654 | 0.9119 | 0.8913 | 0.9336 | 0.0137 |
| exp3_minilm_pca30.yaml | minilm | 0.8606 | 0.9084 | 0.8910 | 0.9273 | 0.0181 |
| exp3_minilm_pca30_exp1base.yaml | minilm | 0.8653 | 0.9117 | 0.8939 | 0.9305 | 0.0135 |
| exp3_minilm_pca64.yaml | minilm | 0.8724 | 0.9169 | 0.8925 | 0.9431 | 0.0149 |
| exp3_minilm_pca64_exp1base.yaml | minilm | 0.8748 | 0.9176 | 0.9026 | 0.9336 | 0.0149 |
| exp3_tfidf_exp1base.yaml | tfidf | 0.8843 | 0.9221 | 0.9213 | 0.9242 | 0.0300 |
| exp3_tfidf_handcrafted_exp1base.yaml | tfidf | 0.8653 | 0.9088 | 0.9173 | 0.9020 | 0.0214 |

## 3. 方法橫向對比（跨配置平均）

| 方法 | 平均 Accuracy | 平均 F1 | 平均 F1 Std | 樣本數 |
|---|---:|---:|---:|---:|
| tfidf | 0.8772 | 0.9170 | 0.0282 | 3 |
| both | 0.8728 | 0.9167 | 0.0142 | 6 |
| handcrafted | 0.8748 | 0.9153 | 0.0265 | 3 |
| minilm | 0.8685 | 0.9139 | 0.0161 | 6 |

類別排名（以平均 F1）：
- 最佳類別：tfidf（平均 F1 = 0.9170）
- 次佳類別：both（平均 F1 = 0.9167）

若以「單一最佳配置」來看：
- 最佳配置：exp3_tfidf_exp1base（F1 = 0.9221）
- 次佳配置：exp3_exp2_with_bmi_tfidf（F1 = 0.9201）

## 4. 結果解讀（這次 Exp3 告訴我們什麼）

1. 效能天花板由 TF-IDF 拿到
- 單一最佳 F1 是 0.9221，配置為 `exp3_tfidf_exp1base`。
- 次佳也來自 TF-IDF（0.9201），顯示詞頻訊號在本資料規模下仍最有效。

1.1 為什麼這次 TF-IDF 會比較好（依本次數據）
- 原因 1：高峰分數出現在 TF-IDF 路線。單點最佳與次佳皆為 TF-IDF（0.9221 / 0.9201），高於 both 最佳（0.9182）與 minilm 最佳（0.9176）。
- 原因 2：加入 MiniLM 沒有帶來穩定增益。若語意向量有穩定加成，both 應明顯高於 tfidf；但本次類別平均 F1 為 tfidf 0.9170、both 0.9167，僅差 -0.0003。
- 原因 3：MiniLM 對降維較敏感。minilm_pca30 為 0.9084 / 0.9117，明顯低於 minilm_pca64 的 0.9169 / 0.9176，顯示在目前資料量下語意特徵壓縮更容易損失可分性。
- 原因 4：TF-IDF 特徵重要度更集中。最佳 TF-IDF 實驗前 4 個特徵重要度合計約 75.2%，而 minilm_pca64 約 47.9%，表示 TF-IDF 能更直接抓到少數高辨識度訊號。
- 原因 5：任務文本是短自介，分類訊號常來自關鍵詞與用詞分布；TF-IDF 的詞頻表示與此型態更匹配。
- 補充：這代表「目前略優」而非壓倒性優勢，因為類別平均 F1 的差距仍小（0.9170 vs 0.9167）。

2. both 平均表現不差，但沒有穩定超越 TF-IDF
- both 平均 F1 為 0.9167，略低於 TF-IDF 的 0.9170。
- 代表把 MiniLM 與 TF-IDF 直接拼接，整體上沒有穩定額外收益。

3. MiniLM 路線分數略低，穩定性中等
- minilm 平均 F1 為 0.9139，四種方法中最低。
- minilm 平均 F1 Std 為 0.0161，穩定度不差，但已不是四類中最低波動。

4. handcrafted 有競爭力，且可解釋性最好
- handcrafted 平均 F1 為 0.9153，略低於 both 與 TF-IDF。
- 若重視特徵可解釋性與工程可控性，handcrafted 是實用方案。

5. PCA30 對 MiniLM 幫助有限，PCA64 較佳
- minilm_pca30 在兩個配置分別為 0.9084 / 0.9117，整體偏弱。
- minilm_pca64 在兩個配置為 0.9169 / 0.9176，顯著優於 pca30。

6. 仍有部分 A/B 對照幾乎等價，但已非全部相同
- 例如 `both_pca30` 與 `both_pca30_exp1base`、`both_pca64` 與 `both_pca64_exp1base` 仍非常接近。
- 但 `minilm_pca30` 與 `minilm_pca30_exp1base` 已有明顯差距（0.9084 vs 0.9117），表示配置差異已反映到結果。

## 5. Feature Importance 重點（跨 18 組）

註：本輪已使用各實驗保存的 preprocessor 對照特徵欄位，`cv_results.json` 中可讀到 `num__*`、`cat__*`、`text__*`、`text_hc__*` 等實際名稱。

### 5.1 全體共同訊號（幾乎所有高分配置）

1. 前四大關鍵特徵高度一致：`num__height`、`num__weight`、`num__iq`、`log__fb_friends`。
2. 這四項在最佳配置中通常位於前 4 名，代表 Exp3 的文本方法差異主要發生在「次要增益區」，主幹仍是基礎數值特徵。

### 5.2 各方法的特徵重要度結構（平均占比）

| 方法 | num | log | cat | ord | text | text_hc |
|---|---:|---:|---:|---:|---:|---:|
| tfidf | 65.4% | 9.9% | 12.6% | 5.8% | 6.4% | 0.0% |
| both | 44.5% | 3.6% | 3.1% | 1.6% | 47.3% | 0.0% |
| minilm | 44.4% | 3.5% | 3.0% | 1.7% | 47.4% | 0.0% |
| handcrafted | 56.7% | 7.5% | 8.0% | 4.0% | 0.0% | 23.8% |
| tfidf+handcrafted | 55.6% | 7.3% | 7.7% | 4.5% | 3.6% | 21.3% |

補充：`tfidf+handcrafted` 對應 `exp3_tfidf_handcrafted_exp1base` 的單一配置，這裡獨立列出是為了觀察混合文本管線的特徵占比。

解讀：
- `both/minilm` 的文字向量占比接近 47%，但重要度分散在大量維度，單一維度訊號較弱。
- `tfidf` 的文字占比僅約 6.4%，但透過少數關鍵詞維度提供高品質增量，與強勢數值主幹互補後拿到最佳分數。
- `handcrafted` 的文字訊號集中在可解釋指標（如 `has_male_keyword`、`has_female_keyword`、`avg_word_len`），可讀性最佳。

### 5.3 代表性配置觀察

1. 最佳配置 `exp3_tfidf_exp1base`：
- Top4 為 `num__height`、`num__weight`、`num__iq`、`log__fb_friends`。
- 主要文字特徵為 `text__tfidf_i`、`text__tfidf_am`、`text__tfidf_a`、`text__tfidf_beautiful`、`text__tfidf_handsome`。

2. `both` 最佳配置 `exp3_both_exp1base`：
- Top4 同樣由數值主導，文字端主要是多個 `text__minilm_*` 維度分散貢獻。

3. `handcrafted` 最佳配置 `exp3_handcrafted_exp1base_enablesearch`：
- 除數值主幹外，`text_hc__has_male_keyword`、`text_hc__has_female_keyword`、`text_hc__avg_word_len`、`text_hc__upper_ratio` 排名前段，支持其可解釋性優勢。

## 6. 建議下一步

1. 主線先鎖定兩條
- 高分主線：TF-IDF（目前最佳 F1）
- 次主線：MiniLM + PCA64（分數明顯優於 MiniLM + PCA30）

2. 對 TF-IDF 做小範圍微調
- 先調 tfidf_max_features（例如 20, 30, 50）
- 同時觀察 F1 與 F1 Std 是否同步改善

3. 若要強化可解釋性，優先手工特徵
- 以 handcrafted15 + 搜尋為下一輪優化基線
- 補做關鍵手工特徵的消融實驗

4. 精簡等價配置
- 對分數極接近的 A/B 配置可酌量減少重跑
- 把資源集中在 TF-IDF 調參與 MiniLM-PCA64 對照
