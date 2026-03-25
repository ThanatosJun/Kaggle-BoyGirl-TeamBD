import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

from .imputation_strategies import get_imputer_from_config

class ClippingTransformer(BaseEstimator, TransformerMixin):
    """自訂 Transformer：針對數值型特徵進行極端值剪裁 (Clipping)"""
    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        self.lower_bounds_ = np.nanpercentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.nanpercentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X, y=None):
        # 建立副本以避免覆寫原始資料
        X_clipped = np.copy(X)
        for i in range(X.shape[1]):
            X_clipped[:, i] = np.clip(X_clipped[:, i], self.lower_bounds_[i], self.upper_bounds_[i])
        return X_clipped


def clip_min_value(X, min_value=0):
    """
    將值 clip 到下界（用於 log1p 前處理）
    這是一個普通函數（非 lambda），可以被 pickle 序列化
    """
    return np.clip(X, min_value, None)


def get_scaler_from_config(scaler_name):
    if scaler_name == 'standard':
        return StandardScaler()
    if scaler_name == 'minmax':
        return MinMaxScaler()
    if scaler_name == 'robust':
        return RobustScaler()
    if scaler_name == 'none':
        return 'passthrough'
    raise ValueError(f"不支援的 scaler 設定: {scaler_name}")


def build_preprocessor(config):
    """
    根據 config 設定，建立完整的 Pipeline：
    1. 數值特徵 (一般): 中位數補值 -> 剪裁 (1%-99%) -> StandardScaler
    2. 數值特徵 (長尾): 中位數補值 -> 移除負值 (clip to 0) -> log(1+x) -> StandardScaler
       ⚠️ 重要: 數據中 fb_friends 存在 -1000 等負值，必須在 log1p 前處理，否則會產生 NaN
    3. 無序類別特徵: 可切換新版/舊版補值 -> One-Hot Encoding
    4. 有序類別特徵: 可切換新版/舊版補值 -> 維持原數值大小 (不做 One-Hot)
    """

    prep_cfg = config.get('preprocessing', {})
    imputation_mode = str(prep_cfg.get('imputation_mode', 'new')).lower()

    if imputation_mode in {'new', 'v2'}:
        default_cat_strategy = 'constant'
        default_cat_fill = '-1'
        default_ord_strategy = 'constant'
        default_ord_fill = -1
    elif imputation_mode in {'old', 'legacy', 'v1'}:
        default_cat_strategy = 'most_frequent'
        default_cat_fill = None
        default_ord_strategy = 'most_frequent'
        default_ord_fill = None
    else:
        raise ValueError("preprocessing.imputation_mode 僅支援: new | old")

    numeric_imputer_strategy = prep_cfg.get('numeric_imputer_strategy', 'median')
    # 允許手動覆寫策略；未設定時由 imputation_mode 決定
    categorical_imputer_strategy = prep_cfg.get('categorical_imputer_strategy', default_cat_strategy)
    categorical_fill_value = prep_cfg.get('categorical_fill_value', default_cat_fill)
    ordinal_imputer_strategy = prep_cfg.get('ordinal_imputer_strategy', default_ord_strategy)
    ordinal_fill_value = prep_cfg.get('ordinal_fill_value', default_ord_fill)
    clipping_lower = prep_cfg.get('clipping_lower_percentile', 1)
    clipping_upper = prep_cfg.get('clipping_upper_percentile', 99)
    log_clip_min = prep_cfg.get('log_clip_min', 0)
    scaler_name = prep_cfg.get('scaler', 'standard')
    onehot_handle_unknown = prep_cfg.get('onehot_handle_unknown', 'ignore')
    onehot_sparse_output = prep_cfg.get('onehot_sparse_output', False)

    scaler = get_scaler_from_config(scaler_name)

    # 1. 數值特徵 Pipeline (Clipping + Scaling)
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_imputer_strategy)),
        ('clipper', ClippingTransformer(lower_percentile=clipping_lower, upper_percentile=clipping_upper)),
        ('scaler', scaler)
    ])

    # 2. 數值特徵長尾分佈 Pipeline (Log1p + Scaling)
    # ⚠️ 注意：必須先移除負值，否則 log1p 會產生 NaN 導致 SMOTE 失敗
    # Pipeline: 中位數補值 → clip 負值到 0 → log(1+x) → StandardScaler
    log_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_imputer_strategy)),
        ('clip_min', FunctionTransformer(clip_min_value, kw_args={'min_value': log_clip_min}, validate=False)),
        ('log1p', FunctionTransformer(np.log1p, validate=False)),
        ('scaler', scaler)
    ])

    # 3. 無序類別特徵 Pipeline (OneHot Encoding)
    cat_imputer_kwargs = {'strategy': categorical_imputer_strategy}
    if categorical_imputer_strategy == 'constant':
        cat_imputer_kwargs['fill_value'] = categorical_fill_value

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(**cat_imputer_kwargs)),
        ('onehot', OneHotEncoder(handle_unknown=onehot_handle_unknown, sparse_output=onehot_sparse_output))
    ])

    # 4. 有序類別特徵 Pipeline (只做補值，因在 data_loader 已轉成 Float)
    ord_imputer_kwargs = {'strategy': ordinal_imputer_strategy}
    if ordinal_imputer_strategy == 'constant':
        ord_imputer_kwargs['fill_value'] = ordinal_fill_value

    ord_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(**ord_imputer_kwargs))
    ])

    # 組合所有 Pipeline 到 ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, config['features']['numeric_cols']),
            ('log', log_pipeline, config['features']['numeric_log_cols']),
            ('cat', cat_pipeline, config['features']['categorical_cols']),
            ('ord', ord_pipeline, config['features']['ordinal_cols'])
        ],
        remainder='drop'  # 沒有設定到的特徵直接丟棄
    )

    return preprocessor
