### 資料集

- 訓練集: Path.home() / 'data' / 'raw' / 'train.csv'
- 測試集: Path.home() / 'data' / 'raw' / 'test.csv'
- 目標: 預測 gender (1=男生, 2=女生)
- 類別分布: 男生 316 筆, 女生 107 筆 (不平衡 3:1)

### 資料處理方法 (process.py)

#### 1. 資料清理
- **YT 欄位**: 轉換為數值型 `pd.to_numeric(df['yt'], errors='coerce')`
- **異常值修正**: 檢測並修正 height/weight 明顯對調的情況
- **ID 欄位**: 保留用於輸出，但不用於訓練

#### 2. 異常值處理
- **height**: clip 到 140-200 cm
- **weight**: clip 到 35-120 kg  
- **iq**: clip 到 70-150
- **fb_friends**: clip 到 0-2000 (去除梗數字如 9487, 6666)
- **sleepiness**: clip 到 1-5
- **yt**: clip 到 0-200

#### 3. 特徵工程
- **BMI**: `weight / (height/100)²` (重要！)
- **height_weight_ratio**: `height / weight`
- **social_activity**: `fb_friends + yt` (社交活躍度)
- **data_completeness**: 計算非缺失欄位數量
- **缺失值指標**: `is_height_missing`, `is_weight_missing` 等 (0/1)

#### 4. 缺失值處理策略

**重要原則**: XGBoost/CatBoost 可原生處理 NaN，保留缺失值可能比填補更好

**數值欄位** (height, weight, iq, sleepiness, fb_friends, yt):

**推薦方案 A: 保留 NaN** (用於 XGBoost/CatBoost) ⭐⭐⭐⭐⭐
- **不填補**，直接保留 NaN
- 樹模型會自動學習缺失模式
- 例如：某性別更傾向不填某欄位
- 保留最多信息

**方案 B: 填補 + 缺失指標** (用於其他模型)
- 步驟 1: 創建缺失指標 `is_height_missing` 等 (0/1)
- 步驟 2: 使用 `SimpleImputer(strategy='median')`
- ⚠️ 在訓練集上 fit，訓練和測試集上 transform
- ❌ **絕對不要**按性別分組填補（測試集無 gender）

**類別欄位** (star_sign, phone_os):
- 填補為 `'Unknown'` 字串
- One-hot encoding (若不用 CatBoost)
- CatBoost 可直接處理類別特徵

**文字欄位** (self_intro):
- 填補為空字串 `''`

#### 5. 文字特徵提取 (self_intro)

**主要方法**: Sentence-BERT (S-BERT)
- 使用預訓練模型將文本轉換為向量 (384 或 768 維)
- 模型: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- 理解語義和語境，自動捕捉文字中的性別特徵

#### 6. 特徵編碼
- **star_sign, phone_os**: One-hot encoding (`pd.get_dummies()`)
- **數值特徵**: StandardScaler 標準化 (建議)

### 訓練方法 (train.py)

#### 模型選擇 (監督學習分類器)

**方案 A: XGBoost** (推薦最佳性能)
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=316/107,  # 處理類別不平衡
    random_state=42,
    eval_metric='logloss'
)
```

**方案 B: CatBoost** (推薦處理類別特徵)
```python
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# 建立模型
model = CatBoostClassifier(
    iterations=100,
    depth=5,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='Accuracy',
    auto_class_weights='Balanced',  # 自動處理類別不平衡
    random_seed=42,
    verbose=False,
    use_best_model=True,  # 使用驗證集上的最佳模型
    cat_features=['star_sign', 'phone_os']  # 自動處理類別特徵，不需 one-hot
)

# 分割驗證集（用於監控過擬合）
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# 訓練時使用 eval_set 監控性能
model.fit(
    X_train_split, y_train_split,
    eval_set=(X_val_split, y_val_split),
    verbose=0,
    plot=True  # 繪製訓練曲線，可視化過擬合情況
)
```




#### 訓練策略

1. **交叉驗證** (評估模型性能)
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
```

2. **處理類別不平衡**
   - 使用 `class_weight='balanced'`
   - 或 XGBoost 的 `scale_pos_weight`
   - 或使用 SMOTE 過採樣

3. **訓練最終模型**
   - 用全部訓練集訓練
   - 預測測試集

4. **評估指標**
   - F1-score (weighted/macro)
   - Accuracy
   - Confusion Matrix

### 預測

- 測試集 gender=0 代表未知，需要預測
- 預測值: 1=男生, 2=女生
- 輸出格式: CSV 包含 id 和 gender 欄位

### 檔案結構

```
wang/
├── process.py       # 資料預處理、特徵工程
├── train.py         # 模型訓練、預測、輸出
├── requirements.txt # 依賴套件
└── docs/
   └── process_way.md
└── data/ # 處理後的資料存放位置
   └── raw/
       ├── train.csv
       └── test.csv
```

### 其他需求

- **Random Seed**: 42 (所有隨機操作)
  - `random_state=42` (模型)
  - `shuffle=True, random_state=42` (交叉驗證)
- **修改範圍**: 只能修改 `wang/` 資料夾內的檔案
- **主要框架**: sklearn 及相容套件 (sklearn API)
- **模組分離**: 
  - `process.py`: 資料處理、特徵工程
  - `train.py`: 模型訓練、評估、預測
- 為避免之後修改問題，請以模組化、函式化的方式撰寫程式碼，並添加適當註解說明每個步驟的目的和方法

### 依賴套件

```
pandas
numpy
scikit-learn
sentence-transformers  # S-BERT
xgboost               # 推薦，性能優異
catboost              # 推薦，自動處理類別特徵
```

