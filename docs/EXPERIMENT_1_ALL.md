# EXPERIMENT 1 全量對比（4 模型 × 4 補值方法）

資料來源：
- [experiments/experiment_record.csv](experiments/experiment_record.csv)
- 共 16 組成功實驗（status=success）

## 0. method0 ~ method3 是什麼

四個方法都使用相同的離群值處理（1%~99% percentile clipping），差別只在 Missing Value 補值策略：

本節定義以 [docs/Experiment1_Imputation.md](docs/Experiment1_Imputation.md) 的「補值方法說明」為準。

| 方法 | 補值策略 | 規則說明 |
|---|---|---|
| method0 | 全局 Median（Baseline） | 用整個訓練集的中位數補值，通常最穩健。 |
| method1 | 全局 Mean | 用整個訓練集的平均值補值。 |
| method2 | 論文範圍中點（按性別） | 依性別套用文獻生理範圍上下限中點做補值。 |
| method3 | 分群 Mean（按 star_sign） | 依 `star_sign` 分群後，用各群組平均值補值。 |

定義來源：
- [docs/Experiment1_Imputation.md](docs/Experiment1_Imputation.md)
- [configs/exp1_CB_method0.yaml](configs/exp1_CB_method0.yaml)
- [configs/exp1_CB_method1.yaml](configs/exp1_CB_method1.yaml)
- [configs/exp1_CB_method2.yaml](configs/exp1_CB_method2.yaml)
- [configs/exp1_CB_method3.yaml](configs/exp1_CB_method3.yaml)

## 1. 全體最佳組合

- 最佳 F1：CatBoost + method0
- Mean F1：0.9303
- Mean Accuracy：0.8961

## 2. 各模型在四種方法下的成效

說明：表格欄位為 Mean Accuracy / Mean F1 / Mean Precision / Mean Recall / F1 Std。

### CatBoost

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8961 | 0.9303 | 0.9316 | 0.9305 | 0.0232 |
| method1 | 0.8890 | 0.9243 | 0.9318 | 0.9181 | 0.0397 |
| method2 | 0.8583 | 0.9042 | 0.9113 | 0.8988 | 0.0277 |
| method3 | 0.8442 | 0.8943 | 0.9008 | 0.8895 | 0.0413 |

### XGBoost

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8724 | 0.9120 | 0.9440 | 0.8829 | 0.0121 |
| method1 | 0.8677 | 0.9092 | 0.9336 | 0.8862 | 0.0195 |
| method2 | 0.8369 | 0.8883 | 0.9109 | 0.8671 | 0.0091 |
| method3 | 0.8158 | 0.8738 | 0.8951 | 0.8546 | 0.0307 |

### Random Forest

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8511 | 0.8981 | 0.9217 | 0.8766 | 0.0137 |
| method1 | 0.8653 | 0.9073 | 0.9338 | 0.8831 | 0.0277 |
| method2 | 0.8322 | 0.8860 | 0.8998 | 0.8734 | 0.0154 |
| method3 | 0.8087 | 0.8672 | 0.9001 | 0.8387 | 0.0312 |

### LightGBM

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8489 | 0.8954 | 0.9265 | 0.8672 | 0.0259 |
| method1 | 0.8630 | 0.9044 | 0.9470 | 0.8671 | 0.0242 |
| method2 | 0.8276 | 0.8788 | 0.9251 | 0.8386 | 0.0269 |
| method3 | 0.8158 | 0.8714 | 0.9080 | 0.8389 | 0.0413 |

## 3. 補值方法橫向對比（跨模型平均）

| 方法 | 平均 Accuracy | 平均 F1 | 平均 F1 Std |
|---|---:|---:|---:|
| method1 | 0.8712 | 0.9113 | 0.0278 |
| method0 | 0.8671 | 0.9090 | 0.0183 |
| method2 | 0.8387 | 0.8893 | 0.0203 |
| method3 | 0.8211 | 0.8767 | 0.0336 |

## 4. 重點結論

1. **method1 現已超越 method0**：跨模型平均 F1 由 0.9090 上升至 0.9113，method1 排名第一。
2. **CatBoost + method0 仍為單一最佳分數**：F1=0.9303 > method1 在大多數模型的表現。
3. **method0 與 method1 差距已極小**：平均 F1 差距僅 0.0023，兩者基本同等有效。
4. **method2 與 method3 落後明顯**：F1 分別為 0.8893 與 0.8767，與前兩者差距 >0.02。
5. **推薦策略**：
   - 單一最佳分數→ CatBoost + method0
   - 跨模型穩定性→ method1（與 method0 互補）
   - 降低風險→ 同時嘗試 method0 與 method1

## 5. 建議下一步

1. **Exp2 聚焦於 method1**：在新特徵版本下，method1 效能已接近或超越 method0，值得深入探索。
2. **模型微調**：基於 CatBoost + {method0/method1} 進行 learning_rate、max_depth 等超參數微調。
3. **淘汰 method3**：method3 表現最差且不穩定（Std 最高），後續實驗移出主要候選。
4. **保留 method2 備選**：method2 介於 method0/1 與 method3 之間，作為穩定性驗證用。
