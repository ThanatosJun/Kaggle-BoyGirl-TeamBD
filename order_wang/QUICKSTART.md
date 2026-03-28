# 快速入門指南

## 📦 安裝步驟

1. **進入專案目錄**
   ```powershell
   cd c:\Users\USER\Documents\projects\Kaggle-BoyGirl-TeamBD\wang
   ```

2. **安裝依賴套件**
   ```powershell
   pip install -r requirements.txt
   ```
   
   如果遇到網路問題，可以使用國內鏡像：
   ```powershell
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

## 🚀 執行訓練

**一鍵執行完整流程：**
```powershell
python train.py
```

這會自動完成：
- ✅ 資料載入與清理
- ✅ 異常值處理
- ✅ 特徵工程
- ✅ 文字特徵提取 (S-BERT)
- ✅ 訓練 XGBoost 和 CatBoost
- ✅ 5-Fold 交叉驗證
- ✅ 選擇最佳模型
- ✅ 生成提交檔案

## 📝 預期輸出

執行完成後，會看到類似的輸出：

```
==================================================
性別預測模型 - 完整流程
==================================================

[步驟 1/5] 載入資料...
訓練集: (423, 11)
測試集: (426, 11)

[步驟 2/5] 資料預處理...
YT 轉換完成，缺失值: 90
修正 2 筆 height/weight 對調
異常值處理完成，創建 9 個異常指標
特徵工程完成，新增特徵數: 22
S-BERT 特徵提取完成，維度: 384
預處理完成！
訓練集最終形狀: (423, 419)

[步驟 3/5] 準備特徵...
特徵數量: 407
訓練樣本: 423

[步驟 4/5] 訓練與比較模型...

方案 A: XGBoost
交叉驗證 F1-Score: 0.8523 (+/- 0.0312)

方案 B: CatBoost
交叉驗證 F1-Score: 0.8612 (+/- 0.0289)

✅ 選擇 CatBoost 作為最終模型

[步驟 5/5] 儲存結果...
✅ 提交檔案已儲存: wang/output/submission_CatBoost_20260322_123456.csv
✅ 模型已儲存: wang/models/CatBoost_model_20260322_123456.pkl

==================================================
流程完成！
==================================================
```

## 📤 輸出檔案

訓練完成後會產生：

1. **提交檔案** 📄
   - 路徑：`output/submission_CatBoost_20260322_123456.csv`
   - 格式：包含 id 和 gender 兩欄
   - 用途：直接上傳至 Kaggle

2. **模型檔案** 🤖
   - 路徑：`models/CatBoost_model_20260322_123456.pkl`
   - 用途：後續評估或預測

3. **特徵重要性** 📊
   - 路徑：`models/CatBoost_feature_importance_20260322_123456.csv`
   - 內容：所有特徵的重要性排序

4. **元數據** 📋
   - 路徑：`models/CatBoost_metadata_20260322_123456.json`
   - 內容：模型配置、訓練時間、CV 分數等

## 🔍 評估已訓練的模型

如果想詳細分析模型表現：

```powershell
python eval.py
```

這會顯示：
- ✅ 交叉驗證詳細結果
- ✅ 訓練集表現分析
- ✅ Top 20 重要特徵
- ✅ 錯誤樣本分析

## ⏱️ 執行時間

- **首次執行**：約 5-10 分鐘（需下載 S-BERT 模型）
- **後續執行**：約 2-3 分鐘

## 📍 注意事項

1. **資料路徑**
   - 確保 `data/raw/train.csv` 和 `data/raw/test.csv` 存在
   - 如果路徑不同，修改 `process.py` 中的 `DATA_DIR`

2. **記憶體需求**
   - 建議至少 4GB RAM
   - S-BERT 模型需要 ~500MB

3. **Python 版本**
   - 建議使用 Python 3.8 或更高版本

## 🎯 下一步

1. **檢視結果**
   - 查看 `output/` 目錄中的提交檔案
   - 上傳至 Kaggle 檢視分數

2. **調整參數** (可選)
   - 修改 `config.json` 中的模型參數
   - 重新執行 `python train.py`

3. **深入分析** (可選)
   - 執行 `python eval.py` 查看詳細評估
   - 分析錯誤樣本，改進特徵工程

## ❓ 遇到問題？

請查看 [README.md](README.md) 的常見問題部分，或檢查：
- [完整工作流程文檔](docs/complete_workflow.md)
- [資料處理方式說明](docs/process_way.md)
