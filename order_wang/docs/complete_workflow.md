# 性別預測完整流程文件

## 📋 專案概述

- **任務**: 二元分類 - 預測性別 (1=男生, 2=女生)
- **訓練集**: 423 筆資料，gender 已知
- **測試集**: 426 筆資料，gender=0 (未知，需要預測)
- **類別分布**: 男生 316 筆 (74.7%), 女生 107 筆 (25.3%) - **不平衡**
- **評估**: 使用 5-Fold 交叉驗證評估模型性能
- **最新結果**: XGBoost CV F1=0.8977 ✅

---

## 🔄 完整流程概覽

```
1. 載入資料
   ↓
2. 資料清理 (YT轉數值、修正錯誤)
   ↓
3. 異常值處理 (clip + 異常指標)
   ↓
4. 特徵工程 (BMI, 比例, 缺失指標)
   ↓
5. 缺失值處理 (保留NaN - 方案A)
   ↓
6. 文字特徵提取 (S-BERT + PCA 降維)    ← 更新: 加入 PCA
   ↓
7. 標籤轉換 (1,2 → 0,1)               ← 更新: XGBoost 要求
   ↓
8. Optuna 超參數搜尋 (各50 trials)      ← 更新: 自動調參
   ↓
9. 模型訓練
   ├─ XGBoost + Optuna (方案A)
   └─ CatBoost + Optuna (方案B)
   ↓
10. 交叉驗證評估 + 自動選擇最佳模型
    ↓
11. 預測測試集 (標籤轉回 0,1 → 1,2)
    ↓
12. 輸出結果 (submission.csv + 模型 + 元數據)
```

---

## 📁 專案結構

```
wang/
├── config.json              # 配置文件
├── process.py              # 資料處理模組
├── train.py                # 模型訓練模組 (含 Optuna 調參)
├── eval.py                 # 模型評估模組
├── requirements.txt        # 依賴套件
├── README.md              # 專案說明
├── QUICKSTART.md          # 快速入門
├── docs/
│   ├── complete_workflow.md  # 本文件
│   └── process_way.md
├── models/                # 模型儲存 (自動創建)
│   ├── {Model}_model_{timestamp}.pkl
│   ├── {Model}_feature_importance_{timestamp}.csv
│   └── {Model}_metadata_{timestamp}.json
└── output/                # 輸出結果 (自動創建)
    └── submission_{Model}_{timestamp}.csv
```

---

## 📁 資料路徑

```python
from pathlib import Path

# 資料路徑
DATA_DIR = Path.home() / 'Documents' / 'projects' / 'Kaggle-BoyGirl-TeamBD' / 'data' / 'raw'
TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'

# 輸出路徑
OUTPUT_DIR = Path('wang') / 'output'

# 模型路徑
MODEL_DIR = Path('wang') / 'models'
```

---

## 📊 資料處理流程 (process.py)

### **步驟 1: 資料載入**

```python
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

def load_data():
    """載入訓練集和測試集"""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    
    print(f"訓練集: {train.shape}")
    print(f"測試集: {test.shape}")
    print(f"\n類別分布:\n{train['gender'].value_counts()}")
    
    return train, test
```

### **步驟 2: 資料清理**

```python
def clean_data(df):
    """
    資料清理
    1. YT 轉為數值型
    2. 修正 height/weight 明顯對調的情況
    """
    df = df.copy()
    
    # YT 轉數值
    df['yt'] = pd.to_numeric(df['yt'], errors='coerce')
    print(f"YT 轉換完成，缺失值: {df['yt'].isna().sum()}")
    
    # 修正 height/weight 對調
    # 規則: height < 130 且 weight > 130 → 可能對調
    swap_mask = (df['height'] < 130) & (df['weight'] > 130)
    swap_count = swap_mask.sum()
    
    if swap_count > 0:
        df.loc[swap_mask, ['height', 'weight']] = \
            df.loc[swap_mask, ['weight', 'height']].values
        print(f"修正 {swap_count} 筆 height/weight 對調")
    
    return df
```

### **步驟 3: 異常值處理**

```python
def handle_outliers(df):
    """
    異常值處理
    1. 創建異常指標特徵 (保留異常信息)
    2. Clip 到合理範圍
    """
    df = df.copy()
    
    # === 創建異常指標 ===
    df['is_height_low'] = (df['height'] < 140).astype(int)
    df['is_height_high'] = (df['height'] > 200).astype(int)
    df['is_weight_low'] = (df['weight'] < 35).astype(int)
    df['is_weight_high'] = (df['weight'] > 120).astype(int)
    df['is_iq_low'] = (df['iq'] < 70).astype(int)
    df['is_iq_high'] = (df['iq'] > 150).astype(int)
    df['is_fb_meme'] = df['fb_friends'].isin([9487, 6666, 5566, 1314, 520]).astype(int)
    df['is_fb_high'] = (df['fb_friends'] > 2000).astype(int)
    df['is_sleepiness_abnormal'] = ((df['sleepiness'] < 1) | 
                                     (df['sleepiness'] > 5)).astype(int)
    
    # 總異常計數
    outlier_cols = [col for col in df.columns if col.startswith('is_')]
    df['total_outliers'] = df[outlier_cols].sum(axis=1)
    
    # === Clip 到合理範圍 ===
    df['height'] = df['height'].clip(140, 200)
    df['weight'] = df['weight'].clip(35, 120)
    df['iq'] = df['iq'].clip(70, 150)
    df['fb_friends'] = df['fb_friends'].clip(0, 2000)
    df['sleepiness'] = df['sleepiness'].clip(1, 5)
    df['yt'] = df['yt'].clip(0, 200)
    
    return df
```

### **步驟 4: 特徵工程**

```python
def engineer_features(df):
    """
    特徵工程 - 創建組合特徵和缺失值指標
    """
    df = df.copy()
    
    # === 組合特徵 ===
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['height_weight_ratio'] = df['height'] / df['weight']
    df['social_activity'] = df['fb_friends'] + df['yt']
    
    original_cols = ['height', 'weight', 'iq', 'sleepiness', 'fb_friends', 'yt', 
                     'star_sign', 'phone_os', 'self_intro']
    df['data_completeness'] = df[original_cols].notna().sum(axis=1)
    
    # === 缺失值指標 ===
    df['is_height_missing'] = df['height'].isna().astype(int)
    df['is_weight_missing'] = df['weight'].isna().astype(int)
    df['is_iq_missing'] = df['iq'].isna().astype(int)
    df['is_sleepiness_missing'] = df['sleepiness'].isna().astype(int)
    df['is_fb_missing'] = df['fb_friends'].isna().astype(int)
    df['is_yt_missing'] = df['yt'].isna().astype(int)
    df['is_star_sign_missing'] = df['star_sign'].isna().astype(int)
    df['is_phone_os_missing'] = df['phone_os'].isna().astype(int)
    df['is_self_intro_missing'] = df['self_intro'].isna().astype(int)
    
    missing_cols = [col for col in df.columns if col.startswith('is_') and col.endswith('_missing')]
    df['total_missing'] = df[missing_cols].sum(axis=1)
    
    return df
```

### **步驟 5: 缺失值處理 (方案A: 保留NaN)**

```python
def handle_missing_values(df):
    """
    數值特徵: 保留NaN (XGBoost/CatBoost可處理)
    類別特徵: 填補 'Unknown'
    文字特徵: 填補空字串
    """
    df = df.copy()
    df['star_sign'] = df['star_sign'].fillna('Unknown')
    df['phone_os'] = df['phone_os'].fillna('Unknown')
    df['self_intro'] = df['self_intro'].fillna('')
    return df
```

### **步驟 6: 文字特徵提取 (S-BERT + PCA 降維)** ← 已更新

```python
def extract_text_features(train_df, test_df, pca_dim=30):
    """
    使用 S-BERT 提取文字特徵，並用 PCA 降維
    
    Parameters:
        pca_dim: PCA 降維目標維度 (None=不降維)
    
    降維原因:
    - 原始 S-BERT 輸出 384 維，對 423 筆訓練資料而言維度過高
    - 容易造成過擬合 (特徵數 >> 樣本數)
    - PCA 降至 30 維，保留 ~75.7% 方差，大幅減少噪音
    """
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 提取文字向量
    train_texts = train_df['self_intro'].fillna('').tolist()
    train_embeddings = model.encode(train_texts, show_progress_bar=True)
    test_texts = test_df['self_intro'].fillna('').tolist()
    test_embeddings = model.encode(test_texts, show_progress_bar=True)
    
    original_dim = train_embeddings.shape[1]  # 384
    
    # PCA 降維: 384 → 30
    pca = None
    if pca_dim is not None and pca_dim < original_dim:
        pca = PCA(n_components=pca_dim, random_state=42)
        train_embeddings = pca.fit_transform(train_embeddings)
        test_embeddings = pca.transform(test_embeddings)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"PCA 保留方差比例: {explained_var:.4f} ({explained_var:.1%})")
    
    # 轉為 DataFrame
    final_dim = train_embeddings.shape[1]
    embedding_cols = [f'sbert_{i}' for i in range(final_dim)]
    
    train_sbert_df = pd.DataFrame(train_embeddings, columns=embedding_cols)
    test_sbert_df = pd.DataFrame(test_embeddings, columns=embedding_cols)
    
    return train_sbert_df, test_sbert_df, pca
```

### **步驟 7: 完整預處理函數**

```python
def preprocess_data(train_df, test_df):
    """完整資料預處理流程"""
    # 1. 資料清理
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)
    
    # 2. 異常值處理
    train_df = handle_outliers(train_df)
    test_df = handle_outliers(test_df)
    
    # 3. 特徵工程
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # 4. 缺失值處理
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)
    
    # 5. 文字特徵提取 + PCA 降維 (384 → 30)
    train_sbert, test_sbert, pca = extract_text_features(train_df, test_df, pca_dim=30)
    
    # 6. 合併所有特徵
    train_final = pd.concat([train_df, train_sbert], axis=1)
    test_final = pd.concat([test_df, test_sbert], axis=1)
    
    return train_final, test_final
```

---

## 🤖 模型訓練流程 (train.py)

### **標籤轉換** ← 已更新

> **重要**: XGBoost 要求標籤為 `[0, 1]`，原始資料的 gender 為 `[1, 2]`。
> 訓練前需轉換 `y = gender - 1`，預測後需轉回 `gender = prediction + 1`。

### **準備訓練資料**

```python
def prepare_features(train_df, test_df):
    """準備特徵和標籤"""
    exclude_cols = ['id', 'gender', 'self_intro', 'star_sign', 'phone_os']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df['gender'] - 1  # 轉換標籤: 1,2 → 0,1
    
    X_test = test_df[feature_cols]
    test_ids = test_df['id']
    
    return X_train, y_train, X_test, test_ids, feature_cols
```

### **方案 A: XGBoost + Optuna 超參數調優** ← 已更新

```python
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def train_xgboost(X_train, y_train, X_test):
    """使用 Optuna 調參後訓練 XGBoost 模型"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'enable_categorical': False
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=skf,
                                 scoring='f1_weighted', n_jobs=-1)
        return scores.mean()
    
    # Optuna 搜尋
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    # 用最佳參數訓練最終模型
    best_params = study.best_params
    best_params.update({'random_state': 42, 'eval_metric': 'logloss',
                        'tree_method': 'hist', 'enable_categorical': False})
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    y_test_pred = model.predict(X_test)
    
    return model, y_test_pred, study.best_value, feature_importance
```

**Optuna 搜尋的超參數範圍:**

| 參數 | 範圍 | 說明 |
|------|------|------|
| `n_estimators` | 50-500 | 樹的數量 |
| `max_depth` | 3-10 | 最大深度 |
| `learning_rate` | 0.01-0.3 (log) | 學習率 |
| `min_child_weight` | 1-10 | 最小葉節點權重 |
| `subsample` | 0.6-1.0 | 行採樣比例 |
| `colsample_bytree` | 0.4-1.0 | 列採樣比例 |
| `reg_alpha` | 1e-8 - 10.0 (log) | L1 正則化 |
| `reg_lambda` | 1e-8 - 10.0 (log) | L2 正則化 |
| `scale_pos_weight` | 1.0 - 5.0 | 類別不平衡權重 |

### **方案 B: CatBoost + Optuna 超參數調優** ← 已更新

```python
from catboost import CatBoostClassifier

def train_catboost(X_train, y_train, X_test):
    """使用 Optuna 調參後訓練 CatBoost 模型"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'loss_function': 'Logloss',
            'auto_class_weights': 'Balanced',
            'random_seed': 42,
            'verbose': 0
        }
        model = CatBoostClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=skf,
                                 scoring='f1_weighted', n_jobs=-1)
        return scores.mean()
    
    # Optuna 搜尋
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    # 用最佳參數在全部訓練集上訓練
    best_params = study.best_params
    best_params.update({'loss_function': 'Logloss', 'auto_class_weights': 'Balanced',
                        'random_seed': 42, 'verbose': 0})
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, verbose=0)
    
    y_test_pred = model.predict(X_test).astype(int)
    
    return model, y_test_pred, study.best_value, feature_importance
```

**CatBoost Optuna 搜尋的超參數範圍:**

| 參數 | 範圍 | 說明 |
|------|------|------|
| `iterations` | 50-500 | 迭代次數 |
| `depth` | 3-10 | 樹深度 |
| `learning_rate` | 0.01-0.3 (log) | 學習率 |
| `l2_leaf_reg` | 1e-8 - 10.0 (log) | L2 正則化 |
| `bagging_temperature` | 0.0-10.0 | Bagging 溫度 |
| `random_strength` | 0.0-10.0 | 隨機強度 |
| `border_count` | 32-255 | 分桶數量 |

### **模型比較與選擇**

```python
def compare_models(X_train, y_train, X_test):
    """比較兩種模型，選擇 CV F1 最高的"""
    xgb_model, xgb_pred, xgb_score, xgb_importance = train_xgboost(X_train, y_train, X_test)
    cat_model, cat_pred, cat_score, cat_importance = train_catboost(X_train, y_train, X_test)
    
    if xgb_score >= cat_score:
        return xgb_model, xgb_pred, "XGBoost", xgb_score, xgb_importance
    else:
        return cat_model, cat_pred, "CatBoost", cat_score, cat_importance
```

---

## 📤 結果輸出

```python
def save_submission(test_ids, predictions, model_name):
    """儲存提交結果 (標籤轉回 1,2)"""
    submission = pd.DataFrame({
        'id': test_ids,
        'gender': predictions + 1  # 轉回原始標籤: 0,1 → 1,2
    })
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'submission_{model_name}_{timestamp}.csv'
    filepath = OUTPUT_DIR / filename
    submission.to_csv(filepath, index=False)
    
    return submission, filepath


def save_model_and_metadata(model, model_name, cv_score, feature_importance, feature_cols):
    """儲存模型、特徵重要性和元數據"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 儲存模型 (.pkl)
    model_path = MODEL_DIR / f'{model_name}_model_{timestamp}.pkl'
    joblib.dump(model, model_path)
    
    # 儲存特徵重要性 (.csv)
    importance_path = MODEL_DIR / f'{model_name}_feature_importance_{timestamp}.csv'
    feature_importance.to_csv(importance_path, index=False)
    
    # 儲存元數據 (.json)
    metadata = {
        'model_name': model_name,
        'cv_f1_score': float(cv_score),
        'timestamp': timestamp,
        'n_features': len(feature_cols),
        'feature_names': feature_cols
    }
    metadata_path = MODEL_DIR / f'{model_name}_metadata_{timestamp}.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return model_path, importance_path, metadata_path
```

---

## 🚀 主程式

```python
def main():
    """主程式 - 完整流程"""
    np.random.seed(42)
    
    # 1. 載入資料
    train_raw, test_raw = load_data()
    
    # 2. 資料預處理 (含 PCA 降維)
    train_processed, test_processed = preprocess_data(train_raw, test_raw)
    
    # 3. 準備特徵 (標籤轉換 1,2 → 0,1)
    X_train, y_train, X_test, test_ids, feature_cols = prepare_features(
        train_processed, test_processed)
    
    # 4. Optuna 調參 + 訓練 + 比較模型
    best_model, best_pred, best_name, best_score, best_importance = compare_models(
        X_train, y_train, X_test)
    
    # 5. 儲存結果 (標籤轉回 0,1 → 1,2)
    submission, submission_path = save_submission(test_ids, best_pred, best_name)
    model_path, importance_path, metadata_path = save_model_and_metadata(
        best_model, best_name, best_score, best_importance, feature_cols)

if __name__ == "__main__":
    main()
```

---

## 📦 依賴套件 (requirements.txt)

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
sentence-transformers>=2.2.0
torch>=2.0.0
xgboost>=1.7.0
catboost>=1.2.0
optuna>=3.0.0
joblib>=1.2.0
tqdm>=4.65.0
```

安裝指令:
```bash
cd wang
pip install -r requirements.txt
```

---

## 🎯 執行步驟

### **1. 準備環境**
```bash
cd wang
pip install -r requirements.txt
```

### **2. 執行預處理 (可選，單獨測試)**
```bash
python process.py
```

### **3. 執行完整流程 (推薦)**
```bash
python train.py
```

### **4. 評估已訓練的模型**
```bash
python eval.py
```

### **5. 查看結果**
- 提交檔案: `wang/output/submission_{模型}_{時間戳}.csv`
- 模型檔案: `wang/models/{模型}_model_{時間戳}.pkl`
- 特徵重要性: `wang/models/{模型}_feature_importance_{時間戳}.csv`
- 元數據: `wang/models/{模型}_metadata_{時間戳}.json`

---

## 📊 實際執行輸出

```
==================================================
性別預測模型 - 完整流程
==================================================

[步驟 1/5] 載入資料...
訓練集: (423, 11)
測試集: (426, 11)

[步驟 2/5] 資料預處理...
YT 轉換完成，缺失值: 91
修正 1 筆 height/weight 對調
異常值處理完成，創建 9 個異常指標
特徵工程完成，新增特徵數: 25
缺失值處理完成 (方案A: 數值特徵保留NaN)
S-BERT 原始維度: 384
執行 PCA 降維: 384 → 30...
PCA 保留方差比例: 0.7567 (75.7%)
S-BERT 特徵提取完成，最終維度: 30
預處理完成！
訓練集最終形狀: (423, 65)
測試集最終形狀: (426, 65)

[步驟 3/5] 準備特徵...
特徵數量: 60
訓練樣本: 423
測試樣本: 426
標籤分布 (轉換後): {0: 316, 1: 107}

[步驟 4/5] 訓練與比較模型...

==================================================
方案 A: XGBoost + Optuna 調參
==================================================
執行 Optuna 超參數搜尋 (50 trials)...
最佳 CV F1-Score: 0.8977

訓練最終模型...
Accuracy: 1.0000
F1-Score: 1.0000

Top 10 重要特徵:
 feature  importance
  height    0.074516
  weight    0.055934
sbert_23    0.040727
 sbert_5    0.037332
sbert_21    0.035302
sbert_14    0.028346
sbert_11    0.028039
sbert_15    0.027712
sbert_16    0.026004
 sbert_8    0.025475

==================================================
方案 B: CatBoost + Optuna 調參
==================================================
執行 Optuna 超參數搜尋 (50 trials)...
最佳 CV F1-Score: 0.8964

Top 10 重要特徵:
            feature  importance
             height   19.826882
             weight    9.463727
                bmi    3.105995
height_weight_ratio    2.653063
                 iq    2.456815
                 yt    2.427395
           sbert_18    2.360790
         fb_friends    2.070209

==================================================
模型比較結果
==================================================
XGBoost   CV F1-Score: 0.8977
CatBoost  CV F1-Score: 0.8964

✅ 選擇 XGBoost 作為最終模型

預測分布:
gender
1    324
2    102

最終模型: XGBoost
交叉驗證 F1-Score: 0.8977
```

---

## 📈 性能演變記錄

| 版本 | 改動 | XGBoost CV F1 | CatBoost CV F1 | 最佳模型 |
|------|------|---------------|----------------|----------|
| v1.0 | 基線 (固定參數, 384維S-BERT) | ~0.85 | 0.8728 | CatBoost |
| **v2.0** | **+ PCA降維(30維) + Optuna調參(50trials)** | **0.8977** | **0.8964** | **XGBoost** |

### v2.0 改進細節

**1. S-BERT PCA 降維 (process.py)**
- **改動**: 384 維 → 30 維
- **效果**: 保留 75.7% 方差，總特徵從 ~407 降至 60
- **原因**: 423 筆訓練資料面對 384 維嵌入容易過擬合
- **實作**: 在 `extract_text_features()` 中加入 PCA，`fit_transform` 訓練集，`transform` 測試集

**2. Optuna 超參數調優 (train.py)**
- **改動**: 固定參數 → TPE 貝葉斯搜尋 50 trials
- **效果**: XGBoost CV F1 從 ~0.85 提升至 0.8977 (+5%)
- **搜尋空間**: 9 個超參數 (含正則化、採樣、深度等)
- **實作**: 在 `train_xgboost()` 和 `train_catboost()` 中使用 `optuna.create_study()`

**3. XGBoost 最佳參數 (實際搜尋結果)**
```json
{
  "n_estimators": 477,
  "max_depth": 9,
  "learning_rate": 0.0669,
  "min_child_weight": 2,
  "subsample": 0.784,
  "colsample_bytree": 0.627,
  "reg_alpha": 6.68e-06,
  "reg_lambda": 1.02e-05,
  "scale_pos_weight": 1.189
}
```

**4. CatBoost 最佳參數 (實際搜尋結果)**
```json
{
  "iterations": 469,
  "depth": 6,
  "learning_rate": 0.0119,
  "l2_leaf_reg": 0.0986,
  "bagging_temperature": 5.975,
  "random_strength": 1.396,
  "border_count": 100
}
```

---

## 📝 關鍵決策說明

### **1. 缺失值處理 - 選擇方案A (保留NaN)**

**理由:**
- ✅ XGBoost/CatBoost 原生支援 NaN
- ✅ "是否缺失"本身有判別力
- ✅ 避免引入虛假信息
- ✅ 保留最多原始信息

### **2. 異常值處理 - Clip + 異常指標**

**理由:**
- ✅ 保留異常信息 (is_*_abnormal)
- ✅ 防止極端值干擾
- ✅ 適合小樣本量

### **3. 文字特徵 - S-BERT + PCA 降維** ← 已更新

**理由:**
- ✅ S-BERT 語義理解能力強
- ✅ 避免刻板印象
- ✅ **PCA 降維防止過擬合** (384→30, 保留 75.7% 方差)
- ✅ 總特徵從 407 降至 60，模型更穩健

### **4. 超參數搜尋 - Optuna TPE** ← 已更新

**理由:**
- ✅ TPE (Tree-structured Parzen Estimator) 比 GridSearch 更高效
- ✅ 自動探索最佳參數組合
- ✅ 50 trials 足以找到良好參數
- ✅ 同時優化正則化參數，防止過擬合

### **5. 標籤轉換 - gender 1,2 → 0,1** ← 已更新

**理由:**
- ✅ XGBoost 要求標籤從 0 開始
- ✅ 訓練前轉換 `y = gender - 1`
- ✅ 預測後轉回 `gender = prediction + 1`
- ✅ 確保提交檔案使用原始標籤

### **6. 模型選擇 - 自動比較**

**XGBoost 優勢:**
- 性能優異，調參後反超 CatBoost
- 成熟穩定，廣泛使用

**CatBoost 優勢:**
- 自動處理類別特徵
- 內建類別不平衡處理

**策略:** Optuna 分別調參 → 交叉驗證比較 → 選擇最高分

---

## 🔧 進階優化建議 (尚未實作)

1. **模型融合 (Ensemble)**
   - Voting: 結合 XGBoost + CatBoost + RandomForest
   - Stacking: 用 meta-learner
   - yokummy 的 RF 測試集達 0.9058，可納入融合

2. **閾值優化**
   - 不用預設 0.5，用 CV 找最佳概率閾值

3. **SMOTE 過採樣**
   - 平衡訓練集的類別分布

4. **更多特徵工程**
   - 身高×體重交互特徵
   - self_intro 文字長度
   - star_sign 分組 (水象/火象/土象/風象)

5. **增加 Optuna trials**
   - 從 50 增加到 100-200 可能進一步改善

---

## 📞 問題排查

### **常見錯誤**

1. **XGBoost "Expected: [0 1], got [1 2]"**
   - 原因: XGBoost 要求標籤從 0 開始
   - 解決: 已在 `prepare_features()` 中轉換 `y = gender - 1`

2. **S-BERT 下載慢**
   - 解決: 首次執行需下載 ~420MB 模型，後續快取

3. **記憶體不足**
   - 解決: PCA 已將 S-BERT 維度從 384 降至 30，大幅減少記憶體需求

4. **CatBoost 找不到類別特徵**
   - 解決: 已填補類別缺失值為 'Unknown'

5. **Optuna 執行慢**
   - 50 trials × 5-fold CV × 2 models ≈ 3-5 分鐘
   - 可減少 `n_trials` 加速 (但可能犧牲性能)

---

**最後更新**: 2026-03-22  
**版本**: 2.0  
**作者**: Wang Team
