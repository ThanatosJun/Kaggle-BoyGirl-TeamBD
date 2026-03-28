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
- Mean F1：0.9259
- Mean Accuracy：0.8890

## 2. 各模型在四種方法下的成效

說明：表格欄位為 Mean Accuracy / Mean F1 / Mean Precision / Mean Recall / F1 Std。

### CatBoost

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8890 | 0.9259 | 0.9197 | 0.9336 | 0.0309 |
| method1 | 0.7565 | 0.8419 | 0.8190 | 0.8674 | 0.0614 |
| method2 | 0.8560 | 0.9043 | 0.8959 | 0.9146 | 0.0265 |
| method3 | 0.7187 | 0.8114 | 0.8084 | 0.8168 | 0.0499 |

### XGBoost

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8746 | 0.9148 | 0.9326 | 0.8987 | 0.0193 |
| method1 | 0.7234 | 0.8086 | 0.8345 | 0.7851 | 0.0763 |
| method2 | 0.8323 | 0.8852 | 0.9049 | 0.8670 | 0.0151 |
| method3 | 0.6810 | 0.7828 | 0.7949 | 0.7724 | 0.0365 |

### Random Forest

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8511 | 0.8981 | 0.9217 | 0.8766 | 0.0137 |
| method1 | 0.7283 | 0.8145 | 0.8371 | 0.7945 | 0.0692 |
| method2 | 0.8322 | 0.8860 | 0.8998 | 0.8734 | 0.0154 |
| method3 | 0.6810 | 0.7847 | 0.7891 | 0.7819 | 0.0599 |

### LightGBM

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| method0 | 0.8488 | 0.8957 | 0.9271 | 0.8671 | 0.0197 |
| method1 | 0.6714 | 0.7634 | 0.8253 | 0.7124 | 0.0792 |
| method2 | 0.8111 | 0.8691 | 0.8997 | 0.8417 | 0.0294 |
| method3 | 0.6574 | 0.7599 | 0.7924 | 0.7313 | 0.0658 |

## 3. 補值方法橫向對比（跨模型平均）

| 方法 | 平均 Accuracy | 平均 F1 | 平均 F1 Std |
|---|---:|---:|---:|
| method0 | 0.8659 | 0.9086 | 0.0209 |
| method2 | 0.8329 | 0.8862 | 0.0216 |
| method1 | 0.7199 | 0.8071 | 0.0715 |
| method3 | 0.6845 | 0.7847 | 0.0530 |

## 4. 重點結論

1. method0 在四個模型下全部拿到該模型最佳 F1。
2. method2 穩定排第二，且與 method0 的差距明顯小於 method1、method3。
3. method1 與 method3 的整體表現顯著較差，且 F1 波動（Std）較大。
4. 若目標是單一最佳分數，首選 CatBoost + method0。
5. 若目標是穩定且可擴展，method0 仍是最一致的基線方案。

## 5. 建議下一步

1. 以 method0 為固定補值策略，進入模型超參數深度搜尋。
2. 保留 method2 作為備援方案，檢查在不同隨機種子與特徵版本下是否有反超機會。
3. 在後續實驗報告中，將 method1、method3 移出主要候選，降低試驗成本。
