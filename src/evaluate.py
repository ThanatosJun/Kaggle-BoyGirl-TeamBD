import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

def cross_validate_with_smote(X, y, preprocessor, model, config):
    """
    執行 5-Fold 交叉驗證，並確保 SMOTE 只在每次的 Train Fold 內部執行，
    以避免 Data Leakage。
    """
    n_splits = config['training']['n_splits']
    use_smote = config['training']['use_smote']
    random_state = config['training']['random_state']

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}

    print(f"開始 {n_splits}-Fold 交叉驗證 (SMOTE={'開啟' if use_smote else '關閉'})...")

    # 將 DataFrame 轉為 Numpy array 以便透過 idx 切片 (或是直接用 iloc 取值)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        
        # 1. 拆分訓練與驗證集
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        # 2. 定義該次 Fold 的 Preprocessor，並 Fit_Transform 訓練集
        prep_fold = copy.deepcopy(preprocessor)
        X_train_trans = prep_fold.fit_transform(X_train_fold)
        X_val_trans = prep_fold.transform(X_val_fold) # ⚠️ Validation 只能 Transform

        # 3. 處理 Data Balance：對轉換後的訓練特徵進行 SMOTE
        if use_smote:
            smote = SMOTE(random_state=random_state)
            X_train_trans, y_train_fold = smote.fit_resample(X_train_trans, y_train_fold)

        # 4. 訓練模型
        model_fold = copy.deepcopy(model)
        model_fold.fit(X_train_trans, y_train_fold)

        # 5. 驗證與評估
        preds = model_fold.predict(X_val_trans)
        
        acc = accuracy_score(y_val_fold, preds)
        f1 = f1_score(y_val_fold, preds, zero_division=0)
        prec = precision_score(y_val_fold, preds, zero_division=0)
        rec = recall_score(y_val_fold, preds, zero_division=0)

        metrics['accuracy'].append(acc)
        metrics['f1'].append(f1)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)

        print(f"Fold {fold+1}: Accuracy={acc:.4f}, F1-Score={f1:.4f}")

    print("\n--- 🏁 CV 最終平均結果 ---")
    for metric_name, values in metrics.items():
        print(f"Mean {metric_name.capitalize()}: {np.mean(values):.4f} (± {np.std(values):.4f})")

    return metrics
