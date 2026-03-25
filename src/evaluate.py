import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE

from .imputation_strategies import get_imputer_from_config


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

def cross_validate_with_smote(X, y, preprocessor, model, config):
    """
    執行 5-Fold 交叉驗證，並確保 SMOTE 只在每次的 Train Fold 內部執行，
    以避免 Data Leakage。
    同時在每個 Fold 的訓練集內進行自訂補值（方法 0-3）。
    返回：(metrics, fold_models, fold_preprocessors)
    """
    n_splits = config['training']['n_splits']
    use_smote = config['training']['use_smote']
    random_state = config['training']['random_state']
    smote_params = config.get('training', {}).get('smote_params', {'random_state': random_state})
    class_weight_cfg = _resolve_class_weight_config(config)

    # 獲取自訂補值器
    imputer = get_imputer_from_config(config)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
    fold_models = []
    fold_preprocessors = []

    print(f"開始 {n_splits}-Fold 交叉驗證 (SMOTE={'開啟' if use_smote else '關閉'})...")

    # 將 DataFrame 轉為 Numpy array 以便透過 idx 切片 (或是直接用 iloc 取值)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        # 1. 拆分訓練與驗證集
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        # 2. 在訓練集內進行自訂補值（fit on train, transform on both train & val）
        # 重點：將 y_train_fold 傳遞給 imputer，以便利用性別資訊補值
        imputer_fold = copy.deepcopy(imputer)
        X_train_fold = imputer_fold.fit_transform(X_train_fold, y_train_fold)
        X_val_fold = imputer_fold.transform(X_val_fold)

        # 3. 定義該次 Fold 的 Preprocessor，並 Fit_Transform 訓練集
        prep_fold = copy.deepcopy(preprocessor)
        X_train_trans = prep_fold.fit_transform(X_train_fold)
        X_val_trans = prep_fold.transform(X_val_fold) # ⚠️ Validation 只能 Transform

        # 4. 處理 Data Balance：對轉換後的訓練特徵進行 SMOTE
        if use_smote:
            smote = SMOTE(**smote_params)
            X_train_trans, y_train_fold = smote.fit_resample(X_train_trans, y_train_fold)

        # 5. 訓練模型
        model_fold = copy.deepcopy(model)
        fit_kwargs = {}
        if class_weight_cfg is not None:
            fit_kwargs['sample_weight'] = compute_sample_weight(class_weight=class_weight_cfg, y=y_train_fold)

        try:
            model_fold.fit(X_train_trans, y_train_fold, **fit_kwargs)
        except TypeError:
            # Fallback for estimators that do not accept sample_weight.
            model_fold.fit(X_train_trans, y_train_fold)

        # 6. 驗證與評估
        preds = model_fold.predict(X_val_trans)

        acc = accuracy_score(y_val_fold, preds)
        f1 = f1_score(y_val_fold, preds, zero_division=0)
        prec = precision_score(y_val_fold, preds, zero_division=0)
        rec = recall_score(y_val_fold, preds, zero_division=0)

        metrics['accuracy'].append(acc)
        metrics['f1'].append(f1)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)

        # 保存 fold 模型與該 fold 的 preprocessor
        fold_models.append(model_fold)
        fold_preprocessors.append(prep_fold)

        print(f"Fold {fold+1}: Accuracy={acc:.4f}, F1-Score={f1:.4f}")

    print("\n--- 🏁 CV 最終平均結果 ---")
    for metric_name, values in metrics.items():
        print(f"Mean {metric_name.capitalize()}: {np.mean(values):.4f} (± {np.std(values):.4f})")

    return metrics, fold_models, fold_preprocessors
