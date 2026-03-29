# EXPERIMENT 1 全量對比（4 模型 × 4 補值方法）

資料來源：
- [experiments/experiment_record.csv](experiments/experiment_record.csv)
- 共 16 組成功實驗（status=success）

## 0. method0 ~ method3 是什麼（補值方法代碼對照）

四個方法都使用相同的離群值處理（1%~99% percentile clipping），差別只在 Missing Value 補值策略：

後續若提到 method0~3，對照如下：

本節定義以 [docs/Experiment1_Imputation.md](docs/Experiment1_Imputation.md) 的「補值方法說明」為準。

| 方法 | 補值策略 | 規則說明 |
|---|---|---|
| method0（Median） | 全局 Median（Baseline） | 用整個訓練集的中位數補值，通常最穩健。 |
| method1（Mean） | 全局 Mean | 用整個訓練集的平均值補值。 |
| method2（Paper Range Midpoint） | 論文範圍中點（按性別） | 依性別套用文獻生理範圍上下限中點做補值。 |
| method3（Cluster Mean） | 分群 Mean（按 star_sign） | 依 `star_sign` 分群後，用各群組平均值補值。 |

定義來源：
- [docs/Experiment1_Imputation.md](docs/Experiment1_Imputation.md)
- [configs/exp1_CB_method0.yaml](configs/exp1_CB_method0.yaml)
- [configs/exp1_CB_method1.yaml](configs/exp1_CB_method1.yaml)
- [configs/exp1_CB_method2.yaml](configs/exp1_CB_method2.yaml)
- [configs/exp1_CB_method3.yaml](configs/exp1_CB_method3.yaml)

## 1. 全體最佳組合

- 最佳 F1：CatBoost + Median
- Mean F1：0.9303
- Mean Accuracy：0.8961

## 2. 各模型在四種方法下的成效

說明：表格欄位為 Mean Accuracy / Mean F1 / Mean Precision / Mean Recall / F1 Std。

### CatBoost

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| Median | 0.8961 | 0.9303 | 0.9316 | 0.9305 | 0.0232 |
| Mean | 0.8890 | 0.9243 | 0.9318 | 0.9181 | 0.0397 |
| Paper Range Midpoint | 0.8583 | 0.9042 | 0.9113 | 0.8988 | 0.0277 |
| Cluster Mean | 0.8442 | 0.8943 | 0.9008 | 0.8895 | 0.0413 |

### XGBoost

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| Median | 0.8724 | 0.9120 | 0.9440 | 0.8829 | 0.0121 |
| Mean | 0.8677 | 0.9092 | 0.9336 | 0.8862 | 0.0195 |
| Paper Range Midpoint | 0.8369 | 0.8883 | 0.9109 | 0.8671 | 0.0091 |
| Cluster Mean | 0.8158 | 0.8738 | 0.8951 | 0.8546 | 0.0307 |

### Random Forest

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| Median | 0.8511 | 0.8981 | 0.9217 | 0.8766 | 0.0137 |
| Mean | 0.8653 | 0.9073 | 0.9338 | 0.8831 | 0.0277 |
| Paper Range Midpoint | 0.8322 | 0.8860 | 0.8998 | 0.8734 | 0.0154 |
| Cluster Mean | 0.8087 | 0.8672 | 0.9001 | 0.8387 | 0.0312 |

### LightGBM

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| Median | 0.8489 | 0.8954 | 0.9265 | 0.8672 | 0.0259 |
| Mean | 0.8630 | 0.9044 | 0.9470 | 0.8671 | 0.0242 |
| Paper Range Midpoint | 0.8276 | 0.8788 | 0.9251 | 0.8386 | 0.0269 |
| Cluster Mean | 0.8158 | 0.8714 | 0.9080 | 0.8389 | 0.0413 |

## 3. 補值方法橫向對比（跨模型平均）

| 方法 | 平均 Accuracy | 平均 F1 | 平均 F1 Std |
|---|---:|---:|---:|
| Mean | 0.8712 | 0.9113 | 0.0278 |
| Median | 0.8671 | 0.9090 | 0.0183 |
| Paper Range Midpoint | 0.8387 | 0.8893 | 0.0203 |
| Cluster Mean | 0.8211 | 0.8767 | 0.0336 |

## 4. 重點結論

1. **Mean 現已超越 Median**：跨模型平均 F1 由 0.9090 上升至 0.9113，Mean 排名第一。
2. **CatBoost + Median 仍為單一最佳分數**：F1=0.9303 > Mean 在大多數模型的表現。
3. **Median 與 Mean 差距已極小**：平均 F1 差距僅 0.0023，兩者基本同等有效。
4. **Paper Range Midpoint 與 Cluster Mean 落後明顯**：F1 分別為 0.8893 與 0.8767，與前兩者差距 >0.02。
5. **推薦策略**：
   - 單一最佳分數→ CatBoost + Median
   - 跨模型穩定性→ Mean（與 Median 互補）
   - 降低風險→ 同時嘗試 Median 與 Mean

### 4.1 為什麼 Mean 這次跨模型平均較好（依本次數據）

1. **Mean 的整體勝出來自 RF 與 LightGBM 的提升**：
   - RandomForest：Mean 相較 Median，F1 由 0.8981 提升到 0.9073（+0.0092）
   - LightGBM：Mean 相較 Median，F1 由 0.8954 提升到 0.9044（+0.0090）
2. **雖然 CatBoost 與 XGBoost 仍偏好 Median，但總平均被 RF/LGBM 拉高**：
   - CatBoost：Mean 相較 Median 下降 0.0060
   - XGBoost：Mean 相較 Median 下降 0.0028
   - 跨模型平均後，Mean 仍以 0.9113 小勝 Median 的 0.9090（+0.0023）
3. **Mean 的優勢主要來自更高的 Precision（在 RF/LGBM 特別明顯）**：
   - RandomForest Precision：0.9338（Mean）> 0.9217（Median）
   - LightGBM Precision：0.9470（Mean）> 0.9265（Median）
4. **但 Mean 並非全面壓勝，穩定性仍是 Median 較好**：
   - 平均 F1 Std：Mean=0.0278，高於 Median=0.0183
   - 因此實務上建議把 Mean 視為「平均略優」，把 Median 視為「更穩定基線」

## 5. Feature Importance 重點（跨 16 組）

註：以下統計基於本輪 16 組實驗對應的 `cv_results.json` 特徵重要度彙整。

### 5.1 全體共同訊號

1. 最關鍵特徵高度集中在基礎數值欄位：`num__height` 在 16 組中有 15 組排名第 1。
2. 高分配置的前段特徵大致固定為：`num__height`、`num__weight`、`log__fb_friends`、`num__iq`。
3. 補值方法改變分數，但不會改變「主幹訊號」來自身高/體重/社交活躍度這個事實。

### 5.2 各補值方法的重要度結構（平均占比）

| 方法 | num | log | cat | ord |
|---|---:|---:|---:|---:|
| Median | 64.8% | 12.4% | 17.8% | 5.1% |
| Mean | 65.9% | 11.5% | 17.9% | 4.7% |
| Paper Range Midpoint | 69.5% | 10.0% | 16.1% | 4.5% |
| Cluster Mean | 75.7% | 7.8% | 13.8% | 2.8% |

解讀：
- Cluster Mean 的數值占比最高，但 cat/log/ord 訊號占比明顯下降，代表模型對單一訊號來源依賴較高，與其分數落後、波動較大一致。
- Median、Mean 在群組占比分布上較均衡，也對應到較佳整體表現。

### 5.3 各方法代表配置（取該方法最佳 F1）

| 方法 | 代表配置 | F1 | Top3 特徵 |
|---|---|---:|---|
| Median | exp_039_exp1_CB_method0 | 0.9303 | num__height / num__weight / log__fb_friends |
| Mean | exp_040_exp1_CB_method1 | 0.9243 | num__height / num__weight / log__fb_friends |
| Paper Range Midpoint | exp_041_exp1_CB_method2 | 0.9042 | num__height / num__weight / log__fb_friends |
| Cluster Mean | exp_042_exp1_CB_method3 | 0.8943 | num__height / num__weight / num__iq |

- 四種方法的 Top 特徵結構高度一致，說明 Exp1 的差異主要來自「補值品質」而不是「特徵語意改變」。

## 6. 建議下一步

1. **Exp2 聚焦於 weight ratio**：在新特徵版本下，weight ratio 效能已接近或超越 baseline，值得深入探索。
2. **模型微調**：基於 CatBoost + {Median/Mean} 進行 learning_rate、max_depth 等超參數微調。
3. **淘汰 Cluster Mean**：Cluster Mean 表現最差且不穩定（Std 最高），後續實驗移出主要候選。
4. **保留 Paper Range Midpoint 備選**：Paper Range Midpoint 介於 Median/Mean 與 Cluster Mean 之間，作為穩定性驗證用。
