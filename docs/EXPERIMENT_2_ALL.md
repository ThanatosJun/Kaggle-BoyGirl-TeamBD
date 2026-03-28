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
- Mean F1：0.9259
- Mean Accuracy：0.8890

## 2. 各模型在四種特徵處理下的成效

說明：表格欄位為 Mean Accuracy / Mean F1 / Mean Precision / Mean Recall / F1 Std。

### CatBoost

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8890 | 0.9259 | 0.9197 | 0.9336 | 0.0309 |
| method1 | 0.8772 | 0.9180 | 0.9129 | 0.9241 | 0.0220 |
| method2 | 0.8772 | 0.9177 | 0.9154 | 0.9209 | 0.0256 |
| method3 | 0.8819 | 0.9210 | 0.9186 | 0.9241 | 0.0241 |

### XGBoost

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8746 | 0.9148 | 0.9326 | 0.8987 | 0.0193 |
| method1 | 0.8534 | 0.8997 | 0.9215 | 0.8797 | 0.0183 |
| method2 | 0.8700 | 0.9116 | 0.9288 | 0.8955 | 0.0189 |
| method3 | 0.8676 | 0.9092 | 0.9318 | 0.8891 | 0.0233 |

### Random Forest

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8511 | 0.8981 | 0.9217 | 0.8766 | 0.0137 |
| method1 | 0.8487 | 0.8959 | 0.9246 | 0.8703 | 0.0233 |
| method2 | 0.8582 | 0.9031 | 0.9248 | 0.8830 | 0.0255 |
| method3 | 0.8416 | 0.8920 | 0.9122 | 0.8734 | 0.0173 |

### LightGBM

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8488 | 0.8957 | 0.9271 | 0.8671 | 0.0197 |
| method1 | 0.8489 | 0.8958 | 0.9239 | 0.8703 | 0.0241 |
| method2 | 0.8464 | 0.8945 | 0.9209 | 0.8703 | 0.0230 |
| method3 | 0.8465 | 0.8943 | 0.9205 | 0.8703 | 0.0222 |

## 3. 特徵處理方法橫向對比（跨模型平均）

| 方法 | 平均 Accuracy | 平均 F1 | 平均 F1 Std |
|---|---:|---:|---:|
| method0 | 0.8659 | 0.9086 | 0.0209 |
| method2 | 0.8630 | 0.9067 | 0.0232 |
| method3 | 0.8594 | 0.9041 | 0.0217 |
| method1 | 0.8570 | 0.9024 | 0.0219 |

## 4. 哪種 weight / height 特徵處理較有效

1. 從跨模型平均來看，method0（不加衍生特徵）仍是最佳，F1=0.9086。
2. method2（BMI）是次佳，F1=0.9067，與 method0 差距小於 0.002。
3. method3（PI）與 method1（ratio）整體略弱，其中 method1 平均 F1 最低（0.9024）。
4. 模型別最佳方法不完全一致：
   - CatBoost：method0 最佳
   - XGBoost：method0 最佳
   - RandomForest：method2（BMI）最佳
   - LightGBM：method1（ratio）略優於其餘方法

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

1. 新數據顯示四種特徵處理已出現差異，不再是完全同分。
2. 整體而言 method0 最穩定、method2（BMI）最接近，method1 與 method3 略遜。
3. 若要一個通用預設，建議先用 method0；若使用 RandomForest，可優先嘗試 method2。
