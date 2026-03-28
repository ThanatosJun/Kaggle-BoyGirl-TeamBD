"""
性別預測 - 模型訓練模組
包含 XGBoost 和 CatBoost 訓練、交叉驗證、模型比較和結果輸出
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 模型相關
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 自定義模組
from process import load_data, preprocess_data


# ==================== 路徑配置 ====================
OUTPUT_DIR = Path('output')
MODEL_DIR = Path('models')


# ==================== 準備訓練資料 ====================
def prepare_features(train_df, test_df):
    """
    準備特徵和標籤
    """
    # 排除的欄位
    exclude_cols = ['id', 'gender', 'self_intro', 'star_sign', 'phone_os']
    
    # 訓練特徵
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df['gender'] - 1  # 轉換標籤: 1,2 → 0,1 (XGBoost 要求)
    
    X_test = test_df[feature_cols]
    test_ids = test_df['id']
    
    print(f"特徵數量: {len(feature_cols)}")
    print(f"訓練樣本: {len(X_train)}")
    print(f"測試樣本: {len(X_test)}")
    print(f"標籤分布 (轉換後): {dict(y_train.value_counts())}")
    
    return X_train, y_train, X_test, test_ids, feature_cols


# ==================== XGBoost 訓練 ====================
def train_xgboost(X_train, y_train, X_test):
    """
    使用 Optuna 調參後訓練 XGBoost 模型
    """
    print("\n" + "=" * 50)
    print("方案 A: XGBoost + Optuna 調參")
    print("=" * 50)
    
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
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()
    
    print("\n執行 Optuna 超參數搜尋 (50 trials)...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    best_params = study.best_params
    best_cv_score = study.best_value
    print(f"\n最佳參數: {json.dumps(best_params, indent=2)}")
    print(f"最佳 CV F1-Score: {best_cv_score:.4f}")
    
    # === 用最佳參數訓練最終模型 ===
    print("\n訓練最終模型...")
    best_params['random_state'] = 42
    best_params['eval_metric'] = 'logloss'
    best_params['tree_method'] = 'hist'
    best_params['enable_categorical'] = False
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # 訓練集評估
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    
    print(f"\n訓練集表現:")
    print(f"Accuracy: {train_acc:.4f}")
    print(f"F1-Score: {train_f1:.4f}")
    print(classification_report(y_train, y_train_pred, target_names=['男生', '女生']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))
    
    # 預測測試集
    y_test_pred = model.predict(X_test)
    
    # 特徵重要性
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 重要特徵:")
    print(feature_importance.head(10).to_string(index=False))
    
    return model, y_test_pred, best_cv_score, feature_importance


# ==================== CatBoost 訓練 ====================
def train_catboost(X_train, y_train, X_test):
    """
    使用 Optuna 調參後訓練 CatBoost 模型
    """
    print("\n" + "=" * 50)
    print("方案 B: CatBoost + Optuna 調參")
    print("=" * 50)
    
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
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()
    
    print("\n執行 Optuna 超參數搜尋 (50 trials)...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    best_params = study.best_params
    best_cv_score = study.best_value
    print(f"\n最佳參數: {json.dumps(best_params, indent=2)}")
    print(f"最佳 CV F1-Score: {best_cv_score:.4f}")
    
    # === 用最佳參數在全部訓練集上訓練 ===
    print("\n在全部訓練集上訓練...")
    best_params['loss_function'] = 'Logloss'
    best_params['auto_class_weights'] = 'Balanced'
    best_params['random_seed'] = 42
    best_params['verbose'] = 0
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, verbose=0)
    
    # 訓練集評估
    y_train_pred = model.predict(X_train).astype(int)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    
    print(f"\n訓練集表現:")
    print(f"Accuracy: {train_acc:.4f}")
    print(f"F1-Score: {train_f1:.4f}")
    print(classification_report(y_train, y_train_pred, target_names=['男生', '女生']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))
    
    # 預測測試集
    y_test_pred = model.predict(X_test).astype(int)
    
    # 特徵重要性
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 重要特徵:")
    print(feature_importance.head(10).to_string(index=False))
    
    return model, y_test_pred, best_cv_score, feature_importance


# ==================== 模型比較 ====================
def compare_models(X_train, y_train, X_test):
    """
    訓練兩種模型並比較
    """
    print("\n" + "=" * 50)
    print("模型訓練與比較")
    print("=" * 50)
    
    # 訓練兩種模型
    xgb_model, xgb_pred, xgb_score, xgb_importance = train_xgboost(X_train, y_train, X_test)
    cat_model, cat_pred, cat_score, cat_importance = train_catboost(X_train, y_train, X_test)
    
    # === 比較所有模型 ===
    print("\n" + "=" * 50)
    print("模型比較結果")
    print("=" * 50)
    
    results = {
        'XGBoost': (xgb_model, xgb_pred, xgb_score, xgb_importance),
        'CatBoost': (cat_model, cat_pred, cat_score, cat_importance),
    }
    
    for name, (_, _, score, _) in results.items():
        marker = " ←" if score == max(v[2] for v in results.values()) else ""
        print(f"{name:15s} CV F1-Score: {score:.4f}{marker}")
    
    # 選擇最佳模型
    best_name = max(results, key=lambda k: results[k][2])
    best_model, best_pred, best_score, best_importance = results[best_name]
    print(f"\n✅ 選擇 {best_name} 作為最終模型")
    
    return best_model, best_pred, best_name, best_score, best_importance


# ==================== 儲存結果 ====================
def save_submission(test_ids, predictions, model_name):
    """
    儲存提交結果
    """
    submission = pd.DataFrame({
        'id': test_ids,
        'gender': predictions + 1  # 轉回原始標籤: 0,1 → 1,2
    })
    
    # 確保輸出目錄存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 儲存檔案
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'submission_{model_name}_{timestamp}.csv'
    filepath = OUTPUT_DIR / filename
    submission.to_csv(filepath, index=False)
    
    print(f"\n✅ 提交檔案已儲存: {filepath}")
    print(f"\n預測分布:")
    print(submission['gender'].value_counts())
    
    return submission, filepath


def save_model_and_metadata(model, model_name, cv_score, feature_importance, feature_cols):
    """
    儲存模型和相關元數據
    """
    # 確保模型目錄存在
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 儲存模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = MODEL_DIR / f'{model_name}_model_{timestamp}.pkl'
    joblib.dump(model, model_path)
    print(f"\n✅ 模型已儲存: {model_path}")
    
    # 儲存特徵重要性
    importance_path = MODEL_DIR / f'{model_name}_feature_importance_{timestamp}.csv'
    feature_importance.to_csv(importance_path, index=False)
    print(f"✅ 特徵重要性已儲存: {importance_path}")
    
    # 儲存元數據
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
    print(f"✅ 元數據已儲存: {metadata_path}")
    
    return model_path, importance_path, metadata_path


# ==================== 主程式 ====================
def main():
    """
    主程式 - 完整流程
    """
    print("=" * 50)
    print("性別預測模型 - 完整流程")
    print("=" * 50)
    
    # 設定隨機種子
    np.random.seed(42)
    
    # === 1. 載入資料 ===
    print("\n[步驟 1/5] 載入資料...")
    train_raw, test_raw = load_data()
    
    # === 2. 資料預處理 ===
    print("\n[步驟 2/5] 資料預處理...")
    train_processed, test_processed = preprocess_data(train_raw, test_raw)
    
    # === 3. 準備特徵 ===
    print("\n[步驟 3/5] 準備特徵...")
    X_train, y_train, X_test, test_ids, feature_cols = prepare_features(
        train_processed, test_processed
    )
    
    # === 4. 訓練與比較模型 ===
    print("\n[步驟 4/5] 訓練與比較模型...")
    best_model, best_pred, best_name, best_score, best_importance = compare_models(
        X_train, y_train, X_test
    )
    
    # === 5. 儲存結果 ===
    print("\n[步驟 5/5] 儲存結果...")
    
    # 儲存提交檔案
    submission, submission_path = save_submission(test_ids, best_pred, best_name)
    
    # 儲存模型和元數據
    model_path, importance_path, metadata_path = save_model_and_metadata(
        best_model, best_name, best_score, best_importance, feature_cols
    )
    
    print("\n" + "=" * 50)
    print("流程完成！")
    print("=" * 50)
    print(f"\n輸出檔案:")
    print(f"  - 提交檔案: {submission_path}")
    print(f"  - 模型檔案: {model_path}")
    print(f"  - 特徵重要性: {importance_path}")
    print(f"  - 元數據: {metadata_path}")
    print(f"\n最終模型: {best_name}")
    print(f"交叉驗證 F1-Score: {best_score:.4f}")


if __name__ == "__main__":
    main()
