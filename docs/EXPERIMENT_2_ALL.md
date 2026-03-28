# EXPERIMENT 2 全量對比（4 模型 × 4 特徵處理）

資料來源：
- [experiments/experiment_record_exp2.csv](experiments/experiment_record_exp2.csv)
- 共 16 組成功實驗（status=success）

## 0. method0 ~ method3 是什麼（Exp2 特徵處理）

本實驗固定使用 Exp1 最佳補值策略（method0, median），只改動 weight / height 相關的衍生特徵。

| 方法 | 特徵處理策略 | 規則說明 |
|---|---|---|
| method0 | Base features | 不加任何衍生特徵（add_bmi=false, add_weight_height_ratio=false, add_ponderal_index=false） |
| method1 | + weight_height_ratio | 啟用體重身高比（add_weight_height_ratio=true） |
| method2 | + BMI | 啟用 BMI（add_bmi=true） |
| method3 | + Ponderal Index | 啟用 PI（add_ponderal_index=true） |

定義來源：
- [configs/default_config_all.yaml](configs/default_config_all.yaml)
- [configs/exp2_CB_method0.yaml](configs/exp2_CB_method0.yaml)
- [configs/exp2_CB_method1.yaml](configs/exp2_CB_method1.yaml)
- [configs/exp2_CB_method2.yaml](configs/exp2_CB_method2.yaml)
- [configs/exp2_CB_method3.yaml](configs/exp2_CB_method3.yaml)

## 1. 全體最佳組合

- 最佳 F1：CatBoost + method0
- Mean F1：0.9303
- Mean Accuracy：0.8961

## 2. 各模型在四種特徵處理下的成效

說明：表格欄位為 Mean Accuracy / Mean F1 / Mean Precision / Mean Recall / F1 Std。

### CatBoost

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8961 | 0.9303 | 0.9316 | 0.9305 | 0.0232 |
| method1 | 0.8890 | 0.9253 | 0.9307 | 0.9210 | 0.0249 |
| method2 | 0.8796 | 0.9188 | 0.9234 | 0.9147 | 0.0274 |
| method3 | 0.8890 | 0.9249 | 0.9302 | 0.9210 | 0.0290 |

### XGBoost

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8724 | 0.9120 | 0.9440 | 0.8829 | 0.0121 |
| method1 | 0.8653 | 0.9071 | 0.9371 | 0.8797 | 0.0169 |
| method2 | 0.8606 | 0.9038 | 0.9335 | 0.8765 | 0.0161 |
| method3 | 0.8772 | 0.9162 | 0.9385 | 0.8955 | 0.0162 |

### Random Forest

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8511 | 0.8981 | 0.9217 | 0.8766 | 0.0137 |
| method1 | 0.8487 | 0.8963 | 0.9219 | 0.8735 | 0.0245 |
| method2 | 0.8534 | 0.8996 | 0.9244 | 0.8766 | 0.0279 |
| method3 | 0.8463 | 0.8952 | 0.9154 | 0.8766 | 0.0205 |

### LightGBM

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8489 | 0.8954 | 0.9265 | 0.8672 | 0.0259 |
| method1 | 0.8512 | 0.8969 | 0.9304 | 0.8671 | 0.0168 |
| method2 | 0.8535 | 0.8992 | 0.9249 | 0.8766 | 0.0224 |
| method3 | 0.8488 | 0.8957 | 0.9242 | 0.8702 | 0.0228 |

## 3. 特徵處理方法橫向對比（跨模型平均）

| 方法 | 平均 Accuracy | 平均 F1 | 平均 F1 Std |
|---|---:|---:|---:|
| method0 | 0.8671 | 0.9090 | 0.0187 |
| method3 | 0.8653 | 0.9080 | 0.0222 |
| method1 | 0.8636 | 0.9064 | 0.0208 |
| method2 | 0.8618 | 0.9054 | 0.0234 |

## 4. 哪種 weight / height 特徵處理較有效

1. **method0 維持最佳**：跨模型平均 F1=0.9090，仍是首選方案。
2. **method3（PI）快速上升**：現已排名第二（F1=0.9080），與 method0 差距僅 0.001。
3. **method1 與 method2 接近**：F1 分別為 0.9064 與 0.9054，波動在 0.001 以內。
4. **模型別最佳方法變化**：
   - CatBoost：method0 最佳（F1=0.9303）
   - XGBoost：method0 最佳（F1=0.9120）
   - RandomForest：method2（BMI）最佳（F1=0.8996）
   - LightGBM：method2（BMI）最佳（F1=0.8992）
5. **重要發現**：與之前不同，method3（PI）現已成為次佳方案，LightGBM 與 RF 對 BMI 反應更佳。

## 5. Overfitting 觀察

1. 四模型仍有「Full Train 分數高於 CV 分數」的泛化落差，這在目前資料量與模型容量下屬常見現象。
2. 以 method0 代表觀察：
   - CatBoost：full_train_f1=1.0000，cv_f1=0.9259
   - XGBoost：full_train_f1=0.9936，cv_f1=0.9148
   - RandomForest：full_train_f1=0.9446，cv_f1=0.8981
   - LightGBM：full_train_f1=0.9491，cv_f1=0.8957
3. 各方法的 CV 標準差約落在 0.0137~0.0309 區間，沒有失控波動，但仍建議持續監控 train-vs-cv gap。
4. full_train 指標屬於 train_set_only_no_validation，決策時應以 CV 指標為主。

## 6. 重點結論

1. **method0 仍為通用首選**：跨模型平均 F1=0.9090，穩定性最佳。
2. **method3（PI）新興選項**：F1=0.9080 緊追 method0，特別適合 CatBoost 與 XGBoost。
3. **method2（BMI）特定場景優化**：RF 與 LightGBM 對 BMI 反應較佳，值得嘗試。
4. **method1 略遜一籌**：F1=0.9064，整體效果不如其他方法。
5. **建議策略**：
   - 生產模型→ method0（最穩定可靠）
   - CatBoost/XGBoost 專用→ method3（略勝一籌）
   - RandomForest/LightGBM 專用→ method2（模型相容性最佳）
