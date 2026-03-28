"""
性別預測 - 評估與預測模組
載入已訓練的模型（或即時訓練），對測試集預測並輸出 submission.csv
輸出格式: id, gender (1=男生, 2=女生)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# 自定義模組
from process import load_data, preprocess_data


# ==================== 路徑配置 ====================
MODEL_DIR = Path('models')
OUTPUT_DIR = Path('output')


# ==================== 工具函數 ====================
def prepare_features(train_df, test_df):
    """準備特徵和標籤"""
    exclude_cols = ['id', 'gender', 'self_intro', 'star_sign', 'phone_os']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df['gender'] - 1  # 1,2 → 0,1

    X_test = test_df[feature_cols]
    test_ids = test_df['id']

    return X_train, y_train, X_test, test_ids, feature_cols


def find_latest_model():
    """尋找最新的已儲存模型，回傳 (model_path, metadata_path) 或 (None, None)"""
    if not MODEL_DIR.exists():
        return None, None

    model_files = sorted(MODEL_DIR.glob('*_model_*.pkl'),
                         key=lambda p: p.stat().st_mtime)
    if not model_files:
        return None, None

    latest = model_files[-1]
    # 對應的 metadata
    timestamp = latest.stem.split('_')[-1]
    model_type = latest.stem.split('_model_')[0]
    metadata_path = MODEL_DIR / f'{model_type}_metadata_{timestamp}.json'

    return latest, metadata_path if metadata_path.exists() else None


def train_quick_model(X_train, y_train):
    """快速訓練一個 XGBoost 模型（不做 Optuna 搜尋）"""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=3.0,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist',
        enable_categorical=False
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=skf, scoring='f1_weighted', n_jobs=-1)
    cv_f1 = cv_scores.mean()
    print(f"快速訓練 CV F1-Score: {cv_f1:.4f}")

    model.fit(X_train, y_train)
    return model, cv_f1


# ==================== 主程式 ====================
def main():
    print("=" * 50)
    print("性別預測 - 評估與預測")
    print("=" * 50)

    # === 1. 載入並預處理資料 ===
    print("\n[步驟 1] 載入資料與預處理...")
    train_raw, test_raw = load_data()
    train_processed, test_processed = preprocess_data(train_raw, test_raw)

    X_train, y_train, X_test, test_ids, feature_cols = prepare_features(
        train_processed, test_processed)

    print(f"特徵數量: {len(feature_cols)}")
    print(f"訓練樣本: {len(X_train)}, 測試樣本: {len(X_test)}")

    # === 2. 載入或訓練模型 ===
    print("\n[步驟 2] 載入模型...")
    model_path, metadata_path = find_latest_model()

    if model_path is not None:
        model = joblib.load(model_path)
        print(f"✅ 已載入模型: {model_path.name}")

        if metadata_path is not None:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"   模型類型: {metadata['model_name']}")
            print(f"   CV F1-Score: {metadata['cv_f1_score']:.4f}")
    else:
        print("⚠️  找不到已訓練的模型，執行快速訓練...")
        model, cv_f1 = train_quick_model(X_train, y_train)

    # === 3. 訓練集交叉驗證 ===
    print("\n[步驟 3] 交叉驗證評估...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=skf, scoring='f1_weighted', n_jobs=-1)
    print(f"CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"各 Fold: {[f'{s:.4f}' for s in cv_scores]}")

    # 訓練集分類報告
    y_train_pred = model.predict(X_train)
    print(f"\n訓練集分類報告:")
    print(classification_report(y_train, y_train_pred,
                                target_names=['男生(0)', '女生(1)']))

    # === 4. 預測測試集 ===
    print("[步驟 4] 預測測試集...")
    y_test_pred = model.predict(X_test)

    # 轉回原始標籤: 0,1 → 1,2
    gender_pred = (np.array(y_test_pred) + 1).astype(int)

    # 建立 submission DataFrame
    submission = pd.DataFrame({
        'id': test_ids.values,
        'gender': gender_pred
    })

    # === 5. 顯示預測結果 ===
    print("\n" + "=" * 50)
    print("預測結果預覽 (前 20 筆)")
    print("=" * 50)
    print(submission.head(20).to_string(index=False))

    print(f"\n預測分布:")
    dist = submission['gender'].value_counts().sort_index()
    print(f"  1 (男生): {dist.get(1, 0)} 筆")
    print(f"  2 (女生): {dist.get(2, 0)} 筆")
    print(f"  合計:     {len(submission)} 筆")

    # === 6. 儲存 submission.csv ===
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = OUTPUT_DIR / f'submission_{timestamp}.csv'
    submission.to_csv(filepath, index=False)
    print(f"\n✅ 提交檔案已儲存: {filepath}")

    # 同時儲存一份固定名稱的版本方便提交
    fixed_path = OUTPUT_DIR / 'submission.csv'
    submission.to_csv(fixed_path, index=False)
    print(f"✅ 固定檔案已儲存: {fixed_path}")

    print("\n" + "=" * 50)
    print("評估與預測完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
