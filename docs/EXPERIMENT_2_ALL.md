# EXPERIMENT 2 全量對比（4 模型 × 4 特徵處理）

資料來源：
- [experiments/experiment_record_exp2.csv](experiments/experiment_record_exp2.csv)
- 共 16 組成功實驗（status=success）

## 0. method0 ~ method3 是什麼（Exp2 特徵處理代碼對照）

本實驗固定使用 Exp1 最佳補值策略（Median），只改動 weight / height 相關的衍生特徵。

後續若提到 method0~3，對照如下：

| 方法 | 特徵處理策略 | 規則說明 |
|---|---|---|
| method0（baseline） | Base features | 不加任何衍生特徵（add_bmi=false, add_weight_height_ratio=false, add_ponderal_index=false） |
| method1（weight ratio） | + weight_height_ratio | 啟用體重身高比（add_weight_height_ratio=true） |
| method2（BMI） | + BMI | 啟用 BMI（add_bmi=true） |
| method3（PI） | + Ponderal Index | 啟用 PI（add_ponderal_index=true） |

定義來源：
- [configs/default_config_all.yaml](configs/default_config_all.yaml)
- [configs/exp2_CB_method0.yaml](configs/exp2_CB_method0.yaml)
- [configs/exp2_CB_method1.yaml](configs/exp2_CB_method1.yaml)
- [configs/exp2_CB_method2.yaml](configs/exp2_CB_method2.yaml)
- [configs/exp2_CB_method3.yaml](configs/exp2_CB_method3.yaml)

## 1. 全體最佳組合

- 最佳 F1：CatBoost + baseline
- Mean F1：0.9303
- Mean Accuracy：0.8961

## 2. 各模型在四種特徵處理下的成效

說明：表格欄位為 Mean Accuracy / Mean F1 / Mean Precision / Mean Recall / F1 Std。

### CatBoost

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| baseline | 0.8961 | 0.9303 | 0.9316 | 0.9305 | 0.0232 |
| weight ratio | 0.8890 | 0.9253 | 0.9307 | 0.9210 | 0.0249 |
| BMI | 0.8796 | 0.9188 | 0.9234 | 0.9147 | 0.0274 |
| PI | 0.8890 | 0.9249 | 0.9302 | 0.9210 | 0.0290 |

### XGBoost

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| baseline | 0.8724 | 0.9120 | 0.9440 | 0.8829 | 0.0121 |
| weight ratio | 0.8653 | 0.9071 | 0.9371 | 0.8797 | 0.0169 |
| BMI | 0.8606 | 0.9038 | 0.9335 | 0.8765 | 0.0161 |
| PI | 0.8772 | 0.9162 | 0.9385 | 0.8955 | 0.0162 |

### Random Forest

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| baseline | 0.8511 | 0.8981 | 0.9217 | 0.8766 | 0.0137 |
| weight ratio | 0.8487 | 0.8963 | 0.9219 | 0.8735 | 0.0245 |
| BMI | 0.8534 | 0.8996 | 0.9244 | 0.8766 | 0.0279 |
| PI | 0.8463 | 0.8952 | 0.9154 | 0.8766 | 0.0205 |

### LightGBM

| 方法 | Accuracy | F1 | Precision | Recall | F1 Std |
|---|---:|---:|---:|---:|---:|
| baseline | 0.8489 | 0.8954 | 0.9265 | 0.8672 | 0.0259 |
| weight ratio | 0.8512 | 0.8969 | 0.9304 | 0.8671 | 0.0168 |
| BMI | 0.8535 | 0.8992 | 0.9249 | 0.8766 | 0.0224 |
| PI | 0.8488 | 0.8957 | 0.9242 | 0.8702 | 0.0228 |

## 3. 特徵處理方法橫向對比（跨模型平均）

| 方法 | 平均 Accuracy | 平均 F1 | 平均 F1 Std |
|---|---:|---:|---:|
| baseline | 0.8671 | 0.9090 | 0.0187 |
| PI | 0.8653 | 0.9080 | 0.0222 |
| weight ratio | 0.8636 | 0.9064 | 0.0208 |
| BMI | 0.8618 | 0.9054 | 0.0234 |

## 4. 哪種 weight / height 特徵處理較有效

1. **baseline 維持最佳**：跨模型平均 F1=0.9090，仍是首選方案。
2. **PI 快速上升**：現已排名第二（F1=0.9080），與 baseline 差距僅 0.001。
3. **weight ratio 與 BMI 接近**：F1 分別為 0.9064 與 0.9054，波動在 0.001 以內。
4. **模型別最佳方法變化**：
   - CatBoost：baseline 最佳（F1=0.9303）
   - XGBoost：PI 最佳（F1=0.9162）
   - RandomForest：BMI 最佳（F1=0.8996）
   - LightGBM：BMI 最佳（F1=0.8992）
5. **重要發現**：與之前不同，PI 現已成為次佳方案；同時 LightGBM 與 RF 對 BMI 反應更佳。

### 4.1 為什麼 baseline 仍是跨模型首選（依本次數據）

1. **baseline 仍握有最強單點與最高跨模型平均**：
   - 單點最佳：CatBoost + baseline，F1=0.9303
   - 跨模型平均：baseline=0.9090（四法最高）
2. **baseline 的優勢來自關鍵模型與整體均衡表現**：
   - CatBoost 的最佳方法為 baseline（0.9303），且這也是全表單點最佳
   - 雖然 XGBoost 偏好 PI，但 baseline 在四模型平均仍維持最高，代表 base features 具有良好通用性
3. **baseline 穩定性最佳，泛化風險最低**：
   - 平均 F1 Std 為 0.0187（四法最低）
   - 代表不加衍生特徵時，fold 間波動最小
4. **衍生特徵帶來的是局部收益，不是全面收益**：
   - weight ratio 平均 F1=0.9064（比 baseline 低 0.0026）
   - BMI 平均 F1=0.9054（比 baseline 低 0.0036）
   - PI 平均 F1=0.9080（僅比 baseline 低 0.0010）

### 4.2 為什麼 PI 會成為次佳（依本次數據）

1. **PI 對 XGBoost 的加成最明顯**：
   - XGBoost：PI=0.9162，高於 baseline=0.9120（+0.0042）
2. **在 CatBoost 上也維持接近最佳**：
   - CatBoost：PI=0.9249，僅小幅落後 baseline=0.9303
3. **PI 可能提供了有效的體型非線性訊號**：
   - 與單純比例（weight ratio）相比，PI 的跨模型平均更高（0.9080 > 0.9064）
4. **但 PI 仍非最穩定方案**：
   - 平均 F1 Std=0.0222，高於 baseline 的 0.0187
   - 因此定位上較適合當次主線候選，而非取代 baseline

## 5. Overfitting 觀察

1. 四模型仍有「Full Train 分數高於 CV 分數」的泛化落差，這在目前資料量與模型容量下屬常見現象。
2. 以 baseline 代表觀察：
   - CatBoost：full_train_f1=1.0000，cv_f1=0.9259
   - XGBoost：full_train_f1=0.9936，cv_f1=0.9148
   - RandomForest：full_train_f1=0.9446，cv_f1=0.8981
   - LightGBM：full_train_f1=0.9491，cv_f1=0.8957
3. 各方法的 CV 標準差約落在 0.0137~0.0309 區間，沒有失控波動，但仍建議持續監控 train-vs-cv gap。
4. full_train 指標屬於 train_set_only_no_validation，決策時應以 CV 指標為主。

## 6. 重點結論

1. **baseline 仍為通用首選**：跨模型平均 F1=0.9090，穩定性最佳。
2. **PI 新興選項**：F1=0.9080 緊追 baseline，對 XGBoost 特別有效。
3. **BMI 特定場景優化**：RF 與 LightGBM 對 BMI 反應較佳，值得嘗試。
4. **weight ratio 略遜一籌**：F1=0.9064，整體效果不如其他方法。
5. **建議策略**：
   - 生產模型→ baseline（最穩定可靠）
   - XGBoost 專用→ PI（有明顯增益）
   - RandomForest/LightGBM 專用→ BMI（模型相容性最佳）
   - CatBoost 專用→ baseline（仍是最佳）

## 7. Feature Importance 重點（跨 16 組）

註：以下統計基於本輪 16 組實驗對應的 `cv_results.json` 特徵重要度彙整。

### 7.1 全體共同訊號

1. `num__height` 在 16/16 組都是第 1 重要特徵，屬於最穩定主訊號。
2. 多數高分配置前段特徵仍是 `num__height`、`num__weight`、`log__fb_friends`、`num__iq`，衍生特徵通常作為增益訊號而非主導訊號。

### 7.2 各特徵方法的重要度結構（平均占比）

| 方法 | num | log | cat | ord |
|---|---:|---:|---:|---:|
| baseline | 64.8% | 12.4% | 17.8% | 5.1% |
| weight ratio | 69.7% | 10.6% | 15.2% | 4.4% |
| BMI | 70.2% | 10.2% | 15.1% | 4.5% |
| PI | 70.2% | 10.6% | 14.7% | 4.5% |

解讀：
- 加入衍生特徵後，`num` 占比整體提高，代表模型確實有使用這些體型衍生訊號。
- baseline 的整體分布較均衡，仍是跨模型平均最佳。

### 7.3 衍生特徵本身的平均重要度占比

| 方法 | BMI 佔比 | weight_height_ratio 佔比 | PI 佔比 |
|---|---:|---:|---:|
| baseline | 0.0000 | 0.0000 | 0.0000 |
| weight ratio | 0.0000 | 0.1327 | 0.0000 |
| BMI | 0.1226 | 0.0000 | 0.0000 |
| PI | 0.0000 | 0.0000 | 0.1139 |

解讀：
- 三種衍生特徵都能取得實質權重（約 0.11~0.13），不是無效特徵。
- 但其增益受模型相容性影響明顯：
   - XGBoost + PI 為該模型最佳，且 `num__ponderal_index` 進入前段重要度。
   - RF/LGBM 在 BMI 表現較佳，`num__bmi` 在最佳配置中排名靠前。

### 7.4 模型別代表觀察（各模型最佳配置）

1. CatBoost 最佳：`exp_055_exp2_CB_method0`，Top5 為 `num__height`、`num__weight`、`log__fb_friends`、`num__iq`、`ord__sleepiness`。
2. LightGBM 最佳：`exp_061_exp2_LGBM_method2`，`num__bmi` 位於前段，支持 BMI 對 LGBM 的加成。
3. RandomForest 最佳：`exp_065_exp2_RF_method2`，`num__bmi` 進入 Top3。
4. XGBoost 最佳：`exp_070_exp2_XGB_method3`，`num__ponderal_index` 進入前段，支持 PI 路線優勢。
