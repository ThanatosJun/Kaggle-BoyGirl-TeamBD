import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE

from .imputation_strategies import get_imputer_from_config
from .features import (
    engineer_features,
    fit_pre_imputation_clip_bounds,
    apply_pre_imputation_clip_bounds,
)


def _resolve_class_weight_config(config):
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


def _resolve_early_stopping_config(config):
    """Resolve early stopping settings with backward-compatible defaults."""
    training_cfg = config.get('training', {})
    es_cfg = training_cfg.get('early_stopping', None)

    # New style:
    # training:
    #   early_stopping:
    #     enabled: true
    #     rounds: 50
    #     use_best_model: true
    #     verbose: false
    if isinstance(es_cfg, dict):
        enabled = bool(es_cfg.get('enabled', True))
        rounds = int(es_cfg.get('rounds', es_cfg.get('early_stopping_rounds', 50)))
        use_best_model = bool(es_cfg.get('use_best_model', True))
        verbose = bool(es_cfg.get('verbose', False))
        models = es_cfg.get('models', ['catboost', 'lightgbm', 'xgboost'])
        models = [str(x).lower() for x in models] if isinstance(models, (list, tuple)) else ['catboost', 'lightgbm', 'xgboost']
        return {
            'enabled': enabled and rounds > 0,
            'rounds': max(rounds, 0),
            'use_best_model': use_best_model,
            'verbose': verbose,
            'models': models,
        }

    # Legacy style:
    # training:
    #   early_stopping_rounds: 50
    rounds = int(training_cfg.get('early_stopping_rounds', 50))
    enabled = bool(training_cfg.get('early_stopping_enabled', True))
    use_best_model = bool(training_cfg.get('use_best_model', True))
    verbose = bool(training_cfg.get('early_stopping_verbose', False))
    return {
        'enabled': enabled and rounds > 0,
        'rounds': max(rounds, 0),
        'use_best_model': use_best_model,
        'verbose': verbose,
        'models': ['catboost', 'lightgbm', 'xgboost'],
    }


def _model_family(model) -> str:
    name = model.__class__.__name__.lower()
    if 'catboost' in name:
        return 'catboost'
    if 'lgbm' in name or 'lightgbm' in name:
        return 'lightgbm'
    if 'xgb' in name or 'xgboost' in name:
        return 'xgboost'
    return 'other'


def _fit_model_with_optional_early_stopping(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    sample_weight,
    early_stopping_cfg,
):
    """Fit estimator with best-effort early stopping and safe fallbacks."""
    def _is_lightgbm_gpu_backend_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        keywords = [
            'no opencl device found',
            'opencl',
            'gpu tree learner was not enabled',
            'gpu tree learner',
            'cannot use gpu',
        ]
        return any(k in msg for k in keywords)

    def _force_lightgbm_cpu(target_model):
        # Support both sklearn wrapper aliases across lightgbm versions.
        params = {}
        if hasattr(target_model, 'get_params'):
            current = target_model.get_params(deep=False)
            if 'device' in current:
                params['device'] = 'cpu'
            if 'device_type' in current:
                params['device_type'] = 'cpu'
            if 'gpu_device_id' in current:
                params['gpu_device_id'] = -1
        if params and hasattr(target_model, 'set_params'):
            target_model.set_params(**params)

    family = _model_family(model)
    fit_kwargs = {}

    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight

    allowed_models = set(early_stopping_cfg.get('models', ['catboost', 'lightgbm', 'xgboost']))
    if early_stopping_cfg.get('enabled', False) and family in allowed_models:
        fit_kwargs['eval_set'] = [(X_val, y_val)]
        fit_kwargs['early_stopping_rounds'] = int(early_stopping_cfg['rounds'])

        if family == 'catboost':
            fit_kwargs['use_best_model'] = bool(early_stopping_cfg.get('use_best_model', True))
            fit_kwargs['verbose'] = bool(early_stopping_cfg.get('verbose', False))
        elif family == 'xgboost':
            fit_kwargs['verbose'] = bool(early_stopping_cfg.get('verbose', False))

    first_error = None
    try:
        model.fit(X_train, y_train, **fit_kwargs)
        return
    except TypeError as exc:
        first_error = exc
    except Exception as exc:
        first_error = exc
        if family == 'lightgbm' and _is_lightgbm_gpu_backend_error(exc):
            print("⚠️ LightGBM GPU 不可用（OpenCL/後端問題），自動回退 CPU 重新訓練。")
            _force_lightgbm_cpu(model)
            try:
                model.fit(X_train, y_train, **fit_kwargs)
                return
            except Exception:
                pass

    # LightGBM 4.x compatibility: callback-based early stopping
    if (
        early_stopping_cfg.get('enabled', False)
        and family == 'lightgbm'
        and 'eval_set' in fit_kwargs
    ):
        try:
            import lightgbm as lgb

            lgb_kwargs = dict(fit_kwargs)
            lgb_kwargs.pop('early_stopping_rounds', None)
            lgb_kwargs['callbacks'] = [
                lgb.early_stopping(
                    stopping_rounds=int(early_stopping_cfg['rounds']),
                    verbose=bool(early_stopping_cfg.get('verbose', False)),
                )
            ]
            model.fit(X_train, y_train, **lgb_kwargs)
            return
        except Exception as exc:
            if _is_lightgbm_gpu_backend_error(exc):
                print("⚠️ LightGBM callback 早停於 GPU 不可用，改用 CPU 重試。")
                _force_lightgbm_cpu(model)
                try:
                    model.fit(X_train, y_train, **lgb_kwargs)
                    return
                except Exception:
                    pass

    # Fallback 1: keep sample_weight only
    if sample_weight is not None:
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight)
            return
        except TypeError:
            pass
        except Exception as exc:
            if family == 'lightgbm' and _is_lightgbm_gpu_backend_error(exc):
                print("⚠️ LightGBM GPU 在 sample_weight 路徑失敗，改用 CPU 重試。")
                _force_lightgbm_cpu(model)
                try:
                    model.fit(X_train, y_train, sample_weight=sample_weight)
                    return
                except Exception:
                    pass

    # Fallback 2: plain fit
    try:
        model.fit(X_train, y_train)
        return
    except Exception as exc:
        if family == 'lightgbm' and _is_lightgbm_gpu_backend_error(exc):
            print("⚠️ LightGBM plain fit 仍遇 GPU/OpenCL 問題，最後改 CPU 重試。")
            _force_lightgbm_cpu(model)
            model.fit(X_train, y_train)
            return
        # Re-raise first captured error when available to keep failure context.
        if first_error is not None:
            raise first_error
        raise

def cross_validate_with_smote(X, y, preprocessor, model, config):
    """
    執行 5-Fold 交叉驗證，並確保 SMOTE 只在每次的 Train Fold 內部執行，
    以避免 Data Leakage。
    同時在每個 Fold 的訓練集內進行自訂補值（方法 0-3）。
    返回：(metrics, fold_models, fold_preprocessors, fold_imputers)
    """
    training_cfg = config.get('training', {})
    if not isinstance(training_cfg, dict):
        training_cfg = {}

    n_splits = int(training_cfg.get('n_splits', 5))
    use_smote = bool(training_cfg.get('use_smote', False))
    random_state = int(training_cfg.get('random_state', 42))
    smote_params = training_cfg.get('smote_params', {'random_state': random_state})
    class_weight_cfg = _resolve_class_weight_config(config)
    early_stopping_cfg = _resolve_early_stopping_config(config)
    prep_cfg = config.get('preprocessing', {})
    clip_cols = prep_cfg.get('pre_imputation_clip_cols', ['height', 'weight'])
    clip_lower = prep_cfg.get('clipping_lower_percentile', 1)
    clip_upper = prep_cfg.get('clipping_upper_percentile', 99)

    # 獲取自訂補值器
    imputer = get_imputer_from_config(config)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics = {
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        # Train-fold metrics are tracked to support overfitting-aware model selection.
        'train_accuracy': [],
        'train_f1': [],
        'train_precision': [],
        'train_recall': [],
    }
    fold_models = []
    fold_preprocessors = []
    fold_imputers = []

    print(
        f"開始 {n_splits}-Fold 交叉驗證 (SMOTE={'開啟' if use_smote else '關閉'}, "
        f"EarlyStopping={'開啟' if early_stopping_cfg['enabled'] else '關閉'})..."
    )

    # 將 DataFrame 轉為 Numpy array 以便透過 idx 切片 (或是直接用 iloc 取值)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        # 1. 拆分訓練與驗證集
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        # 1b. 先以 train fold 的分位數界限做 outlier clipping（只針對設定欄位）
        clip_bounds = fit_pre_imputation_clip_bounds(
            X_train_fold,
            clip_cols,
            lower_percentile=clip_lower,
            upper_percentile=clip_upper,
        )
        X_train_fold = apply_pre_imputation_clip_bounds(X_train_fold, clip_bounds)
        X_val_fold = apply_pre_imputation_clip_bounds(X_val_fold, clip_bounds)

        # 2. 在訓練集內進行自訂補值（fit on train, transform on both train & val）
        # 重點：將 y_train_fold 傳遞給 imputer，以便利用性別資訊補值
        imputer_fold = copy.deepcopy(imputer)
        X_train_fold = imputer_fold.fit_transform(X_train_fold, y_train_fold)
        X_val_fold = imputer_fold.transform(X_val_fold)

        # 2b. 衍生特徵工程（在補值後計算 ratio/BMI）
        X_train_fold = engineer_features(X_train_fold, config)
        X_val_fold   = engineer_features(X_val_fold, config)

        # 3. 定義該次 Fold 的 Preprocessor，並 Fit_Transform 訓練集
        prep_fold = copy.deepcopy(preprocessor)
        X_train_trans = prep_fold.fit_transform(X_train_fold)
        X_val_trans = prep_fold.transform(X_val_fold) # ⚠️ Validation 只能 Transform

        # 保留未 SMOTE 的 train fold，供 overfitting gap 評估使用
        X_train_eval_trans = X_train_trans
        y_train_eval = y_train_fold

        # 4. 處理 Data Balance：對轉換後的訓練特徵進行 SMOTE
        if use_smote:
            smote = SMOTE(**smote_params)
            X_train_trans, y_train_fold = smote.fit_resample(X_train_trans, y_train_fold)

        # 5. 訓練模型
        model_fold = copy.deepcopy(model)
        sample_weight = None
        if class_weight_cfg is not None:
            sample_weight = compute_sample_weight(class_weight=class_weight_cfg, y=y_train_fold)

        _fit_model_with_optional_early_stopping(
            model_fold,
            X_train_trans,
            y_train_fold,
            X_val_trans,
            y_val_fold,
            sample_weight,
            early_stopping_cfg,
        )

        # 6. 驗證與評估
        preds = model_fold.predict(X_val_trans)
        train_preds = model_fold.predict(X_train_eval_trans)

        acc = accuracy_score(y_val_fold, preds)
        f1 = f1_score(y_val_fold, preds, zero_division=0)
        prec = precision_score(y_val_fold, preds, zero_division=0)
        rec = recall_score(y_val_fold, preds, zero_division=0)

        metrics['accuracy'].append(acc)
        metrics['f1'].append(f1)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['train_accuracy'].append(accuracy_score(y_train_eval, train_preds))
        metrics['train_f1'].append(f1_score(y_train_eval, train_preds, zero_division=0))
        metrics['train_precision'].append(precision_score(y_train_eval, train_preds, zero_division=0))
        metrics['train_recall'].append(recall_score(y_train_eval, train_preds, zero_division=0))

        # 保存 fold 模型、該 fold 的 preprocessor 與 imputer
        fold_models.append(model_fold)
        fold_preprocessors.append(prep_fold)
        fold_imputers.append(imputer_fold)

        print(f"Fold {fold+1}: Accuracy={acc:.4f}, F1-Score={f1:.4f}")

    print("\n--- 🏁 CV 最終平均結果 ---")
    for metric_name, values in metrics.items():
        print(f"Mean {metric_name.capitalize()}: {np.mean(values):.4f} (± {np.std(values):.4f})")

    return metrics, fold_models, fold_preprocessors, fold_imputers
