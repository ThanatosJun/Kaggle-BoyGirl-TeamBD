import os
import yaml
import joblib
import json
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils.class_weight import compute_sample_weight

from src.data_loader import load_and_clean_data, split_X_y
from src.features import build_preprocessor, engineer_features
from src.models import get_model
from src.evaluate import cross_validate_with_smote
from src.imputation_strategies import get_imputer_from_config


def resolve_model_type(config):
    raw_model_type = config.get('model', {}).get('type', 'xgboost').lower()
    alias_map = {
        'xgboost': 'xgboost',
        'xgb': 'xgboost',
        'lightgbm': 'lightgbm',
        'lgbm': 'lightgbm',
        'random_forest': 'random_forest',
        'rf': 'random_forest',
        'catboost': 'catboost',
        'cat': 'catboost'
    }
    model_type = alias_map.get(raw_model_type)
    if model_type is None:
        raise ValueError(
            f"不支援的模型類型: {raw_model_type}，請使用 'xgboost'、'lightgbm'、'random_forest' 或 'catboost'"
        )
    return model_type


def resolve_param_grid(config, model_type, search_cfg):
    """Resolve selectable param grid by mode: quick/full."""
    model_cfg = config.get('model', {})
    grid_mode = str(search_cfg.get('param_grid_mode', 'full')).lower()

    if grid_mode == 'quick':
        return model_cfg.get('param_grid_quick', {}).get(model_type, {})

    return model_cfg.get('param_grid', {}).get(model_type, {})


def resolve_class_weight_config(config):
    """Return class_weight argument for sklearn utils from config, or None if disabled."""
    training_cfg = config.get('training', {})
    class_weight_cfg = training_cfg.get('class_weight', None)

    if isinstance(class_weight_cfg, str):
        lowered = class_weight_cfg.strip().lower()
        if lowered in {'', 'none', 'null', 'false'}:
            return None
        return lowered

    if isinstance(class_weight_cfg, dict):
        normalized = {}
        for key, value in class_weight_cfg.items():
            cls_key = int(key) if str(key).isdigit() else key
            normalized[cls_key] = float(value)
        return normalized

    if class_weight_cfg in (None, False):
        return None

    return class_weight_cfg


def _extract_feature_names(preprocessor):
    """Safely get transformed feature names from a fitted preprocessor."""
    if not hasattr(preprocessor, 'get_feature_names_out'):
        return None
    try:
        return [str(x) for x in preprocessor.get_feature_names_out()]
    except Exception:
        return None


def _default_feature_names(n_features):
    """Generate deterministic fallback names when feature names are unavailable."""
    return [f"feature_{i:04d}" for i in range(int(n_features))]


def _normalize_importance_series(series: pd.Series) -> pd.Series:
    """Normalize importances to sum=1 for easier cross-model comparison."""
    total = float(series.sum())
    if total <= 0:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return series / total


def _extract_model_importance(model):
    """Get feature importance array from different estimator APIs."""
    # CatBoost style
    if hasattr(model, 'get_feature_importance'):
        try:
            values = model.get_feature_importance()
            return np.asarray(values, dtype=float).ravel()
        except Exception:
            pass

    # sklearn / xgboost / lightgbm wrappers
    if hasattr(model, 'feature_importances_'):
        try:
            values = model.feature_importances_
            return np.asarray(values, dtype=float).ravel()
        except Exception:
            pass

    # LightGBM native booster API fallback
    if hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_importance'):
        try:
            values = model.booster_.feature_importance(importance_type='gain')
            return np.asarray(values, dtype=float).ravel()
        except Exception:
            pass

    # XGBoost native booster API fallback (keys like f0, f1, ...)
    if hasattr(model, 'get_booster'):
        try:
            booster = model.get_booster()
            score = booster.get_score(importance_type='gain')
            if score:
                max_idx = max(int(k[1:]) for k in score.keys() if k.startswith('f'))
                values = np.zeros(max_idx + 1, dtype=float)
                for key, val in score.items():
                    if key.startswith('f') and key[1:].isdigit():
                        values[int(key[1:])] = float(val)
                return values
        except Exception:
            pass

    # Linear models fallback
    if hasattr(model, 'coef_'):
        try:
            coef = np.asarray(model.coef_, dtype=float)
            if coef.ndim == 1:
                return np.abs(coef)
            return np.mean(np.abs(coef), axis=0)
        except Exception:
            pass

    return None


def summarize_cv_feature_importance(fold_models, fold_preprocessors):
    """Aggregate fold-wise feature importance and keep only normalized CV mean."""
    fold_series_norm = []

    for fold_idx, (fold_model, fold_prep) in enumerate(zip(fold_models, fold_preprocessors), start=1):
        feature_names = _extract_feature_names(fold_prep)
        importances = _extract_model_importance(fold_model)

        if importances is None:
            continue

        if feature_names is None:
            feature_names = _default_feature_names(len(importances))

        aligned_len = min(len(feature_names), len(importances))
        if aligned_len == 0:
            continue

        series = pd.Series(importances[:aligned_len], index=feature_names[:aligned_len], dtype=float)
        series.name = f'fold_{fold_idx}'
        fold_series_norm.append(_normalize_importance_series(series))

    if not fold_series_norm:
        return None

    fold_df_norm = pd.concat(fold_series_norm, axis=1).fillna(0.0)

    fold_mean_norm = fold_df_norm.mean(axis=1)

    summary_df = pd.DataFrame({
        'feature': fold_df_norm.index,
        'cv_mean_importance_norm': fold_mean_norm.values,
    }).sort_values('cv_mean_importance_norm', ascending=False)

    return {
        'cv_summary': summary_df.to_dict(orient='records'),
    }

def get_next_experiment_id(base_dir):
    """獲取下一個實驗編號"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return 1

    # 列出所有實驗資料夾
    exp_folders = [f for f in os.listdir(base_dir) if f.startswith('exp_')]
    if not exp_folders:
        return 1

    # 提取編號並找出最大值
    exp_ids = [int(f.split('_')[1]) for f in exp_folders if f.split('_')[1].isdigit()]
    return max(exp_ids) + 1 if exp_ids else 1

def save_experiment_log(base_dir, exp_folder, config, metrics, timestamp):
    """將實驗結果記錄到 CSV"""
    log_file = os.path.join(base_dir, 'experiment_log.csv')
    model_type = resolve_model_type(config)
    model_cfg = config.get('model', {})
    params_map = {
        'xgboost': model_cfg.get('xgb_params', {}),
        'lightgbm': model_cfg.get('lgbm_params', {}),
        'random_forest': model_cfg.get('random_forest_params', {}),
        'catboost': model_cfg.get('catboost_params', {}),
    }
    active_params = params_map.get(model_type, {})

    # 準備記錄
    record = {
        'exp_id': exp_folder,
        'timestamp': timestamp,
        'name': config['experiment']['name'],
        'description': config['experiment']['description'],
        'model_type': model_type,
        'use_smote': config['training']['use_smote'],
        'learning_rate': active_params.get('learning_rate'),
        'max_depth': active_params.get('max_depth', active_params.get('depth')),
        'mean_accuracy': metrics['mean_accuracy'],
        'std_accuracy': metrics['std_accuracy'],
        'mean_f1': metrics['mean_f1'],
        'std_f1': metrics['std_f1'],
        'mean_precision': metrics['mean_precision'],
        'std_precision': metrics['std_precision'],
        'mean_recall': metrics['mean_recall'],
        'std_recall': metrics['std_recall'],
        'full_train_accuracy': metrics.get('full_train_accuracy'),
        'full_train_f1': metrics.get('full_train_f1'),
        'full_train_precision': metrics.get('full_train_precision'),
        'full_train_recall': metrics.get('full_train_recall'),
        'full_train_metric_scope': metrics.get('full_train_metric_scope', 'train_set_only_no_validation')
    }

    # 如果檔案不存在，創建新檔案
    if not os.path.exists(log_file):
        df = pd.DataFrame([record])
        df.to_csv(log_file, index=False)
        print(f"📝 創建實驗日誌: {log_file}")
    else:
        # 追加到現有檔案
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(log_file, index=False)
        print(f"📝 更新實驗日誌: {log_file}")

def main():
    # 0. 解析命令行參數
    parser = argparse.ArgumentParser(description="訓練模型並進行實驗")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='配置檔路徑（預設: configs/default_config.yaml）')
    args = parser.parse_args()

    # 1. 讀取設定檔
    print(f"🔖 讀取實驗設定檔: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 創建實驗目錄
    base_dir = config['training']['save_dir']
    exp_id = get_next_experiment_id(base_dir)
    exp_name = config['experiment']['name']
    exp_folder = f"exp_{exp_id:03d}_{exp_name}"
    exp_path = os.path.join(base_dir, exp_folder)
    os.makedirs(exp_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print(f"🔬 實驗編號: {exp_folder}")
    print(f"📝 實驗名稱: {exp_name}")
    print(f"📄 描述: {config['experiment']['description']}")
    print(f"⏰ 時間: {timestamp}")
    print(f"📁 保存路徑: {exp_path}")
    print(f"{'='*60}\n")

    # 3. 保存配置
    config_path = os.path.join(exp_path, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print(f"💾 已保存配置: {config_path}")

    # 4. 讀取並清理資料
    print(f"📦 正在載入訓練資料: {config['data']['train_path']}")
    df_train = load_and_clean_data(config['data']['train_path'], is_train=True, config=config)

    # 5. 準備 X_train, y_train
    X, y = split_X_y(df_train, config)

    # 6. 建立 Pipeline 元件與模型
    print("🛠️ 正在建立 Preprocessor 特徵工程管線 與 模型...")
    preprocessor = build_preprocessor(config)

    # 7. 可選：執行搜尋網格挑選最佳參數
    search_cfg = config.get('search', {})
    search_enabled = search_cfg.get('enabled', False)
    search_metric = search_cfg.get('metric', 'f1')
    model_type = resolve_model_type(config)
    param_grid = resolve_param_grid(config, model_type, search_cfg)

    if search_enabled and param_grid:
        print(f"\n🔎 開始搜尋網格（model={model_type}, metric={search_metric}）...")
        best_score = float('-inf')
        best_params = None
        best_cv_metrics = None
        best_fold_models = None
        best_fold_preprocessors = None

        for idx, params in enumerate(ParameterGrid(param_grid), start=1):
            print(f"\n[{idx}] 測試參數: {params}")
            candidate_model = get_model(config, override_params=params)
            cv_metrics_tmp, fold_models_tmp, fold_preprocessors_tmp, fold_imputers_tmp = cross_validate_with_smote(
                X, y, preprocessor, candidate_model, config
            )

            if search_metric not in cv_metrics_tmp:
                raise ValueError(f"search.metric={search_metric} 不支援，可選: accuracy|f1|precision|recall")

            score = float(np.mean(cv_metrics_tmp[search_metric]))
            print(f"➡️ mean_{search_metric}: {score:.6f}")

            if score > best_score:
                best_score = score
                best_params = params
                best_cv_metrics = cv_metrics_tmp
                best_fold_models = fold_models_tmp
                best_fold_preprocessors = fold_preprocessors_tmp
                best_fold_imputers = fold_imputers_tmp

        print(f"\n✅ 搜尋網格完成，最佳參數: {best_params}")
        print(f"✅ 最佳 mean_{search_metric}: {best_score:.6f}")

        # 使用最佳參數建立最終模型，並沿用最佳 CV 結果
        model = get_model(config, override_params=best_params)
        cv_metrics = best_cv_metrics
        fold_models = best_fold_models
        fold_preprocessors = best_fold_preprocessors
        fold_imputers = best_fold_imputers

        # 保存最佳參數
        best_params_path = os.path.join(exp_path, 'best_params.json')
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_type': model_type,
                'search_metric': search_metric,
                'best_score': best_score,
                'best_params': best_params
            }, f, indent=2, ensure_ascii=False)
        print(f"💾 已保存最佳參數: {best_params_path}")
    else:
        model = get_model(config)
        cv_metrics, fold_models, fold_preprocessors, fold_imputers = cross_validate_with_smote(X, y, preprocessor, model, config)

    # 8. 計算平均指標
    metrics_summary = {
        'mean_accuracy': np.mean(cv_metrics['accuracy']),
        'std_accuracy': np.std(cv_metrics['accuracy']),
        'mean_f1': np.mean(cv_metrics['f1']),
        'std_f1': np.std(cv_metrics['f1']),
        'mean_precision': np.mean(cv_metrics['precision']),
        'std_precision': np.std(cv_metrics['precision']),
        'mean_recall': np.mean(cv_metrics['recall']),
        'std_recall': np.std(cv_metrics['recall']),
        'fold_results': {
            'accuracy': cv_metrics['accuracy'],
            'f1': cv_metrics['f1'],
            'precision': cv_metrics['precision'],
            'recall': cv_metrics['recall']
        }
    }

    if search_enabled and param_grid:
        metrics_summary['search_enabled'] = True
        metrics_summary['search_metric'] = search_metric
        metrics_summary['search_model_type'] = model_type

    # 9. 在「所有」訓練資料上，訓練最終正式模型
    print("\n🚀 正在使用「所有訓練集」訓練最終對外預測模型...")
    # 與 CV 保持一致：先做自訂補值，再做 preprocessor
    full_imputer = get_imputer_from_config(config)
    X_imputed = full_imputer.fit_transform(X, y)
    # 衍生特徵工程（與 CV fold 保持一致）
    X_imputed = engineer_features(X_imputed, config)
    X_trans = preprocessor.fit_transform(X_imputed)

    if config['training']['use_smote']:
        from imblearn.over_sampling import SMOTE
        smote_params = config.get('training', {}).get('smote_params', {'random_state': config['training']['random_state']})
        smote = SMOTE(**smote_params)
        X_trans, y = smote.fit_resample(X_trans, y)

    fit_kwargs = {}
    class_weight_cfg = resolve_class_weight_config(config)
    if class_weight_cfg is not None:
        fit_kwargs['sample_weight'] = compute_sample_weight(class_weight=class_weight_cfg, y=y)

    try:
        model.fit(X_trans, y, **fit_kwargs)
    except TypeError:
        model.fit(X_trans, y)

    # 10a. 計算 Full Train 指標（供實驗記錄對比 CV 與 Full Train）
    full_train_preds = model.predict(X_trans)
    metrics_summary['full_train_accuracy'] = accuracy_score(y, full_train_preds)
    metrics_summary['full_train_f1'] = f1_score(y, full_train_preds, zero_division=0)
    metrics_summary['full_train_precision'] = precision_score(y, full_train_preds, zero_division=0)
    metrics_summary['full_train_recall'] = recall_score(y, full_train_preds, zero_division=0)
    metrics_summary['full_train_metric_scope'] = 'train_set_only_no_validation'

    # 10b. 彙整並保存特徵重要度（CV 每 fold + CV 平均/標準差 + Full Train）
    cv_feature_importance = summarize_cv_feature_importance(fold_models, fold_preprocessors)
    if cv_feature_importance is not None:
        metrics_summary['feature_importance'] = cv_feature_importance

    # 10. 保存 CV + Full Train 結果
    results_path = os.path.join(exp_path, 'cv_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        # 將 numpy 類型轉換為 Python 原生類型
        json_metrics = {k: (v.tolist() if isinstance(v, np.ndarray) else
                           {k2: (v2.tolist() if isinstance(v2, np.ndarray) else v2)
                            for k2, v2 in v.items()} if isinstance(v, dict) else v)
                       for k, v in metrics_summary.items()}
        json.dump(json_metrics, f, indent=2, ensure_ascii=False)
    print(f"💾 已保存 CV + Full Train 結果: {results_path}")

    # 11. 儲存模型與特徵處理器
    imputer_path = os.path.join(exp_path, 'imputer.pkl')
    prep_path = os.path.join(exp_path, 'preprocessor.pkl')
    model_path = os.path.join(exp_path, 'model.pkl')

    joblib.dump(full_imputer, imputer_path)
    joblib.dump(preprocessor, prep_path)
    joblib.dump(model, model_path)
    print(f"💾 已保存 Imputer: {imputer_path}")
    print(f"💾 已保存 Preprocessor: {prep_path}")
    print(f"💾 已保存 Model: {model_path}")

    # 11a. 儲存 Fold 模型與 Fold Preprocessor（用於 Ensemble 預測）
    print(f"\n💾 儲存 Fold 模型與 Fold Preprocessor...")
    for fold_idx, (fold_model, fold_prep, fold_imputer) in enumerate(zip(fold_models, fold_preprocessors, fold_imputers)):
        fold_model_path = os.path.join(exp_path, f'fold_{fold_idx}_model.pkl')
        fold_prep_path = os.path.join(exp_path, f'fold_{fold_idx}_preprocessor.pkl')
        fold_imputer_path = os.path.join(exp_path, f'fold_{fold_idx}_imputer.pkl')
        joblib.dump(fold_model, fold_model_path)
        joblib.dump(fold_prep, fold_prep_path)
        joblib.dump(fold_imputer, fold_imputer_path)
        print(f"   ✓ Fold {fold_idx}: {fold_model_path}, {fold_prep_path}, {fold_imputer_path}")

    # 12. 更新實驗日誌
    save_experiment_log(base_dir, exp_folder, config, metrics_summary, timestamp)

    # 13. 顯示實驗總結
    print(f"\n{'='*60}")
    print(f"🎉 實驗 {exp_folder} 完成！")
    print(f"{'='*60}")
    print(f"📊 Cross-Validation 平均結果:")
    print(f"  - Accuracy:  {metrics_summary['mean_accuracy']:.4f} (± {metrics_summary['std_accuracy']:.4f})")
    print(f"  - F1-Score:  {metrics_summary['mean_f1']:.4f} (± {metrics_summary['std_f1']:.4f})")
    print(f"  - Precision: {metrics_summary['mean_precision']:.4f} (± {metrics_summary['std_precision']:.4f})")
    print(f"  - Recall:    {metrics_summary['mean_recall']:.4f} (± {metrics_summary['std_recall']:.4f})")
    print(f"\n📌 Full Train（訓練集表現，無 validation，僅供參考）:")
    print(f"  - Accuracy:  {metrics_summary['full_train_accuracy']:.4f}")
    print(f"  - F1-Score:  {metrics_summary['full_train_f1']:.4f}")
    print(f"  - Precision: {metrics_summary['full_train_precision']:.4f}")
    print(f"  - Recall:    {metrics_summary['full_train_recall']:.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
