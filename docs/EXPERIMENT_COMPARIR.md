# EXPERIMENT 文件對比與落差分析（新舊 Config）

更新時間：2026-03-29

本文件對比以下三組文件，說明為何結果會有落差：

1. [docs/EXPERIMENT_1_ALL.md](docs/EXPERIMENT_1_ALL.md) vs [docs/EXPERIMENT_1_SUMMARY.md](docs/EXPERIMENT_1_SUMMARY.md)
2. [docs/EXPERIMENT_2_ALL.md](docs/EXPERIMENT_2_ALL.md) vs [docs/EXPERIMENT_2_SUMMARY.md](docs/EXPERIMENT_2_SUMMARY.md)
3. [docs/EXPERIMENT_3_ALL.md](docs/EXPERIMENT_3_ALL.md) vs [docs/EXPERIMENT_3_COMPARISON.md](docs/EXPERIMENT_3_COMPARISON.md)

另外補充新舊 config 的代表性差異（主要比對 [configs/old_config/](configs/old_config/) 與 [configs/](configs/)）。

---

## 1) Exp1：ALL vs SUMMARY

### 1.1 觀察到的落差

| 面向 | EXPERIMENT_1_SUMMARY | EXPERIMENT_1_ALL | 落差來源 |
|---|---|---|---|
| 統計範圍 | 4 組（CatBoost × 4 補值） | 16 組（4 模型 × 4 補值） | 比較母體不同 |
| 排名依據 | 單一模型（CatBoost）絕對分數 | 跨模型平均 + 模型別拆解 | 統計口徑不同 |
| 方法排序 | Median > Paper Range Midpoint > Mean > Cluster Mean | Mean > Median > Paper Range Midpoint > Cluster Mean（跨模型平均） | 平均方式不同 |
| 單點最佳 | Median F1=0.9315 | Median F1=0.9303（CatBoost） | 接近，但批次不同 |

重點：
- SUMMARY 的結論偏向「CatBoost 單模型最佳化」。
- ALL 的結論偏向「跨模型可遷移性」。

### 1.2 新舊 config 代表差異（Exp1）

對照檔案：
- 舊版：[configs/old_config/exp1_p99_method1_global_mean.yaml](configs/old_config/exp1_p99_method1_global_mean.yaml)
- 新版：[configs/exp1_CB_method1.yaml](configs/exp1_CB_method1.yaml)

主要差異：
1. CatBoost 深度：`depth 6 -> 8`（Mean / Paper Range Midpoint / Cluster Mean 在新版被拉到與 Median 同層級）
2. CatBoost 學習率：`0.05 -> 0.03`
3. CatBoost 類別權重：`auto_class_weights: "Balanced" -> null`，改由 training.class_weight 統一
4. 訓練策略：新版新增 early stopping（rounds=50）
5. XGBoost 參數基線：`learning_rate 0.1 -> 0.03`（雖 Exp1_SUMMARY 是 CatBoost 報告，但 ALL 含四模型時會影響跨模型平均）
6. 新版補上 `pre_imputation_clip_cols` 與統一模板欄位（text/pca/search selection）

### 1.3 為什麼會有數字落差

1. 同一個補值方法在新舊版不完全是同一套模型容量與訓練規則。
2. SUMMARY 看的是早期單模型快照；ALL 是後續完整批次、跨模型口徑。
3. Mean 在新配置（尤其 CatBoost depth 統一）下被明顯「校正」，使跨模型平均名次上升。

---

## 2) Exp2：ALL vs SUMMARY

### 2.1 觀察到的落差

| 面向 | EXPERIMENT_2_SUMMARY | EXPERIMENT_2_ALL | 落差來源 |
|---|---|---|---|
| 統計範圍 | 3 組（CatBoost only：baseline/weight ratio/BMI） | 16 組（4 模型 × 4 方法，含 PI） | 方法集合不同 |
| 最佳方法 | +BMI（F1=0.9276） | baseline（跨模型平均最佳，且 CatBoost 單點也最佳） | 口徑與 config 版本不同 |
| 是否納入 PI | 未納入 | 納入 PI 且為跨模型次佳 | 候選集擴充 |

重點：
- SUMMARY 只有「CatBoost + 三種特徵版本」。
- ALL 是「四模型 + 四種特徵方法」的完整對照，因此 ranking 會變。

### 2.2 新舊 config 代表差異（Exp2）

對照檔案：
- 舊版：[configs/old_config/exp2_method0_with_bmi.yaml](configs/old_config/exp2_method0_with_bmi.yaml)
- 新版：[configs/exp2_CB_method2.yaml](configs/exp2_CB_method2.yaml)

主要差異：
1. CatBoost 參數：`depth 6 -> 8`，`learning_rate 0.05 -> 0.03`
2. 資料路徑：`train.csv/test.csv -> dataset/train.csv/dataset/test.csv`
3. 訓練策略：新版新增 early stopping（rounds=50）
4. 搜尋策略：新版新增 selection_mode/gap_lambda
5. 特徵開關與模板統一：新版以 `add_bmi/add_weight_height_ratio/add_ponderal_index` 為主，並加上 text/pca 保留欄位
6. 舊版多為 CatBoost 單模型導向；新版 ALL 報告以四模型平均為主

### 2.3 為什麼會有數字落差

1. SUMMARY 與 ALL 不是同一個比較問題：前者是「單模型選特徵」，後者是「跨模型選通用方案」。
2. Exp2 檔名後綴 `method0/1/2/3` 代表「特徵法」（baseline/weight ratio/BMI/PI），不是 imputation_method；若誤讀會導致解讀偏差。
3. 納入 PI 後，方法競爭格局改變（XGBoost 對 PI 有明顯增益，拉動跨模型結論）。

---

## 3) Exp3：ALL vs COMPARISON

### 3.1 觀察到的落差

| 面向 | EXPERIMENT_3_COMPARISON | EXPERIMENT_3_ALL | 落差來源 |
|---|---|---|---|
| 文件定位 | 快速對照與操作導向（A/B/C 組混合） | 18 組成功實驗的全量統計報告 | 目的不同 |
| 結果口徑 | 混合不同 baseline（Exp2-base、Exp1-base、search on/off） | 同一報告中統一以 record 檔彙整 | 可比性不同 |
| 代表數值 | 如 B 組 TF-IDF 曾記錄 0.9274 | 現行最佳 TF-IDF 為 0.9221 | 設定與批次版本差異 |

重點：
- COMPARISON 比較像「路線圖 + 快速觀察」。
- ALL 才是「最新批次的正式統計口徑」。

### 3.2 新舊 config 代表差異（Exp3）

代表對照一：
- 舊版：[configs/old_config/exp3_tfidf_exp1base.yaml](configs/old_config/exp3_tfidf_exp1base.yaml)
- 新版：[configs/exp3_tfidf_exp1base.yaml](configs/exp3_tfidf_exp1base.yaml)

主要差異：
1. TF-IDF 維度：`tfidf_max_features 50 -> 20`
2. 數值特徵：由含 `bmi` + `add_bmi: true`，改為不加 BMI（更接近純 TF-IDF 路線）
3. 新版新增 early stopping、selection_mode、gap_lambda
4. 文本模型版本資訊被固定（`text_model_revision`）以提高可重現性

代表對照二：
- 舊版：[configs/old_config/exp3_minilm_pca30.yaml](configs/old_config/exp3_minilm_pca30.yaml)
- 新版：[configs/exp3_minilm_pca30.yaml](configs/exp3_minilm_pca30.yaml)

主要差異：
1. CatBoost 深度：`6 -> 8`
2. 舊版含 `use_lm_embedding/use_tfidf_embedding` 等舊欄位；新版改為統一 `text_embedding_method + use_pca` 管理
3. 新版統一 early stopping 與 full/quick param grid 框架

代表對照三：
- 舊版：[configs/old_config/exp3_handcrafted_exp1base_enablesearch.yaml](configs/old_config/exp3_handcrafted_exp1base_enablesearch.yaml)
- 新版：[configs/exp3_handcrafted_exp1base_enablesearch.yaml](configs/exp3_handcrafted_exp1base_enablesearch.yaml)

主要差異：
1. 舊版帶 `bmi`，新版改成不帶 BMI（避免把文字路線與 BMI 路線混在一起）
2. 新版將 search 選分規則明確化（`mean_minus_std_minus_gap`）

### 3.3 為什麼會有數字落差

1. COMPARISON 混合了不同 base config（depth 6/8、是否 search、是否帶 BMI），本身就非完全 apples-to-apples。
2. ALL 用最新 record 檔重算且只納入成功實驗，口徑較乾淨。
3. TF-IDF 維度與是否帶 BMI 的調整，會直接改變文本訊號的有效密度與過擬合風險。

---

## 4) 綜合結論（可直接引用）

1. 這三組文件的差異，主因不是單一模型波動，而是「統計口徑 + config 版本」同時改變。
2. 若要嚴格比較新舊結果，必須固定：模型集合、特徵集合、search 設定、early stopping、以及文本維度。
3. 建議把 [docs/EXPERIMENT_1_ALL.md](docs/EXPERIMENT_1_ALL.md)、[docs/EXPERIMENT_2_ALL.md](docs/EXPERIMENT_2_ALL.md)、[docs/EXPERIMENT_3_ALL.md](docs/EXPERIMENT_3_ALL.md) 作為正式決策依據，SUMMARY/COMPARISON 保留為歷史快照。

---

## 5) 建議下一步（避免再次出現口徑落差）

1. 每次報告都加一段「統計口徑宣告」：單模型或跨模型、候選方法數量、是否含 search。
2. 在實驗紀錄表新增 config hash（或關鍵欄位快照），確保同名 config 變更可追溯。
3. 對 Exp3 另做一份「嚴格同條件比較表」（固定是否帶 BMI、固定 depth、固定 search），再判斷純文本方法優劣。
