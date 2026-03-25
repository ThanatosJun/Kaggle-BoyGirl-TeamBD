"""
補值策略模組

實現 4 種補值方法的自訂 Transformer：
- 方法 0: 全局 Median (Baseline)
- 方法 1: 全局 Mean
- 方法 2: 論文範圍中點（按性別）
- 方法 3: 分群平均值（按 star_sign）
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


# ============= 論文範圍數據 =============
PAPER_MIDPOINT = {
    1: {  # 男
        'weight': 72.2,   # kg
        'height': 171.2   # cm
    },
    2: {  # 女
        'weight': 57.4,   # kg
        'height': 158.4   # cm
    }
}


class GlobalMedianImputer(BaseEstimator, TransformerMixin):
    """方法 0: 全局 Median Imputer (Baseline)"""

    def __init__(self, columns_to_impute=None):
        self.columns_to_impute = columns_to_impute
        self.medians_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_work = X
        else:
            X_work = pd.DataFrame(X, columns=self.columns_to_impute)

        self.medians_ = {}
        columns = self.columns_to_impute or X_work.columns
        for col in columns:
            if col in X_work.columns:
                self.medians_[col] = X_work[col].median()

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
        else:
            X_transformed = pd.DataFrame(X, columns=self.columns_to_impute)

        for col, median_val in self.medians_.items():
            if col in X_transformed.columns:
                # 直接用 loc 賦值，避免 ChainedAssignmentError
                mask = X_transformed[col].isna()
                X_transformed.loc[mask, col] = median_val

        if not isinstance(X, pd.DataFrame):
            return X_transformed.values
        return X_transformed


class GlobalMeanImputer(BaseEstimator, TransformerMixin):
    """方法 1: 全局 Mean Imputer"""

    def __init__(self, columns_to_impute=None):
        self.columns_to_impute = columns_to_impute
        self.means_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_work = X
        else:
            X_work = pd.DataFrame(X, columns=self.columns_to_impute)

        self.means_ = {}
        columns = self.columns_to_impute or X_work.columns
        for col in columns:
            if col in X_work.columns:
                self.means_[col] = X_work[col].mean()

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
        else:
            X_transformed = pd.DataFrame(X, columns=self.columns_to_impute)

        for col, mean_val in self.means_.items():
            if col in X_transformed.columns:
                # 直接用 loc 賦值，避免 ChainedAssignmentError
                mask = X_transformed[col].isna()
                X_transformed.loc[mask, col] = mean_val

        if not isinstance(X, pd.DataFrame):
            return X_transformed.values
        return X_transformed


class PaperRangeImputer(BaseEstimator, TransformerMixin):
    """方法 2: 論文範圍中點補值（按性別）

    使用論文 PMC8306797 提供的上下限範圍中點值補值，依性別區分。
    如果提供了 y（性別標籤），使用 y 來確定每個樣本的性別。
    否則如果 gender 列存在，使用 gender 列。
    都不存在時，使用全局平均值（男女中點的平均）。
    """

    def __init__(self, gender_col='gender', columns_to_impute=None):
        self.gender_col = gender_col
        self.columns_to_impute = columns_to_impute
        self.global_means_ = None
        self.gender_mapping_ = None  # 從 y 學習的性別對應

    def fit(self, X, y=None):
        # 預先計算全局平均值（作為 fallback）
        if isinstance(X, pd.DataFrame):
            X_work = X
        else:
            X_work = pd.DataFrame(X, columns=self.columns_to_impute)

        self.global_means_ = {}
        columns = self.columns_to_impute or ['weight', 'height']
        for col in columns:
            if col in X_work.columns:
                # 計算兩個性別範圍的平均
                values = []
                for gender in [1, 2]:
                    if gender in PAPER_MIDPOINT and col in PAPER_MIDPOINT[gender]:
                        values.append(PAPER_MIDPOINT[gender][col])
                self.global_means_[col] = np.mean(values) if values else X_work[col].mean()

        # 如果提供了 y，詳記性別對應，以便在 transform 時使用
        if y is not None:
            y_series = y if isinstance(y, pd.Series) else pd.Series(y)
            self.gender_mapping_ = y_series  # 保存索引和性別對應

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
        else:
            X_transformed = pd.DataFrame(X, columns=self.columns_to_impute)

        columns = self.columns_to_impute or ['weight', 'height']

        # 優先級 1：使用通過 fit 保存的 y 性別對應
        if self.gender_mapping_ is not None:
            gender_values = self.gender_mapping_.values
            for idx, gender in enumerate(gender_values):
                if idx < len(X_transformed):
                    if gender in PAPER_MIDPOINT:
                        for col in columns:
                            if col in PAPER_MIDPOINT[gender] and pd.isna(X_transformed.iloc[idx][col]):
                                fill_value = PAPER_MIDPOINT[gender][col]
                                X_transformed.iloc[idx, X_transformed.columns.get_loc(col)] = fill_value
        # 優先級 2：檢查 X 中是否存在 gender 列
        elif self.gender_col in X_transformed.columns:
            # 依性別補值
            for gender in [1, 2]:  # 1=男, 2=女
                mask = X_transformed[self.gender_col] == gender

                if gender in PAPER_MIDPOINT:
                    for col in columns:
                        if col in PAPER_MIDPOINT[gender]:
                            fill_value = PAPER_MIDPOINT[gender][col]
                            # 直接用 loc 賦值，避免 ChainedAssignmentError
                            mask_na = mask & X_transformed[col].isna()
                            X_transformed.loc[mask_na, col] = fill_value
        else:
            # 優先級 3：Fallback - gender 列不存在且沒有 y，使用全局平均
            for col in columns:
                if col in X_transformed.columns and col in self.global_means_:
                    mask = X_transformed[col].isna()
                    X_transformed.loc[mask, col] = self.global_means_[col]

        if not isinstance(X, pd.DataFrame):
            return X_transformed.values
        return X_transformed


class GroupedMeanImputer(BaseEstimator, TransformerMixin):
    """方法 3: 分群平均值補值（按 star_sign 分群）

    先按 star_sign + gender 分群計算平均值，再補值。
    如果提供了 y（性別標籤），使用 y 來確定每個樣本的性別。
    否則如果 gender 列存在，使用 gender 列。
    都不存在時，使用全局平均。
    """

    def __init__(self, gender_col='gender', group_col='star_sign', columns_to_impute=None):
        self.gender_col = gender_col
        self.group_col = group_col
        self.columns_to_impute = columns_to_impute
        self.group_means_ = None
        self.fallback_means_ = None
        self.global_means_ = None
        self.gender_mapping_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_work = X
        else:
            X_work = pd.DataFrame(X, columns=self.columns_to_impute)

        self.group_means_ = {}
        self.fallback_means_ = {}
        self.global_means_ = {}

        columns = self.columns_to_impute or X_work.columns

        # 先計算全局平均（作為最後的 fallback）
        for col in columns:
            if col in X_work.columns:
                self.global_means_[col] = X_work[col].mean()

        # 如果提供了 y，詳記性別對應
        if y is not None:
            y_series = y if isinstance(y, pd.Series) else pd.Series(y)
            self.gender_mapping_ = y_series

            # 使用 y 和 X 計算分群平均
            for gender in [1, 2]:
                gender_mask = y_series == gender
                gender_indices = y_series.index[gender_mask] if hasattr(y_series, 'index') else np.where(gender_mask)[0]

                # 計算該性別的 fallback（該性別全局平均）
                self.fallback_means_[gender] = {}
                X_gender = X_work.loc[gender_indices] if hasattr(X_work, 'loc') else X_work.iloc[gender_indices]
                for col in columns:
                    if col in X_gender.columns:
                        self.fallback_means_[gender][col] = X_gender[col].mean()

                # 按 star_sign 分組計算平均
                self.group_means_[gender] = {}
                if self.group_col in X_gender.columns:
                    groups = X_gender[self.group_col].unique()
                    for group_val in groups:
                        if pd.isna(group_val):
                            continue
                        group_mask = X_gender[self.group_col] == group_val
                        self.group_means_[gender][group_val] = {}
                        for col in columns:
                            if col in X_gender.columns:
                                self.group_means_[gender][group_val][col] = X_gender.loc[group_mask, col].mean()
            return self

        # 如果沒有 y，嘗試從 X 中的 gender 列學習
        if self.gender_col in X_work.columns:
            for gender in [1, 2]:
                mask_gender = X_work[self.gender_col] == gender

                # 計算 fallback（性別全局平均）
                self.fallback_means_[gender] = {}
                for col in columns:
                    if col in X_work.columns:
                        self.fallback_means_[gender][col] = X_work.loc[mask_gender, col].mean()

                # 按 star_sign 分組計算平均
                self.group_means_[gender] = {}
                if self.group_col in X_work.columns:
                    groups = X_work.loc[mask_gender, self.group_col].unique()
                    for group_val in groups:
                        if pd.isna(group_val):
                            continue
                        mask_group = mask_gender & (X_work[self.group_col] == group_val)
                        self.group_means_[gender][group_val] = {}
                        for col in columns:
                            if col in X_work.columns:
                                self.group_means_[gender][group_val][col] = X_work.loc[mask_group, col].mean()

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
        else:
            X_transformed = pd.DataFrame(X, columns=self.columns_to_impute)

        columns = self.columns_to_impute or X_transformed.columns

        # 優先級 1：使用通過 fit 保存的 y 性別對應
        if self.gender_mapping_ is not None:
            gender_values = self.gender_mapping_.values
            for idx, gender in enumerate(gender_values):
                if idx < len(X_transformed):
                    for col in columns:
                        if col not in X_transformed.columns:
                            continue
                        if pd.isna(X_transformed.iloc[idx][col]):
                            fill_value = np.nan

                            # 優先使用分組平均
                            if gender in self.group_means_ and col in X_transformed.columns:
                                if self.group_col in X_transformed.columns:
                                    group_val = X_transformed.iloc[idx][self.group_col]
                                    if gender in self.group_means_ and group_val in self.group_means_[gender]:
                                        fill_value = self.group_means_[gender][group_val].get(col, np.nan)

                            # Fallback 到性別平均
                            if pd.isna(fill_value) and gender in self.fallback_means_:
                                fill_value = self.fallback_means_[gender].get(col, np.nan)

                            # 最後 fallback 到全局平均
                            if pd.isna(fill_value) and col in self.global_means_:
                                fill_value = self.global_means_[col]

                            if not pd.isna(fill_value):
                                X_transformed.iloc[idx, X_transformed.columns.get_loc(col)] = fill_value

        # 優先級 2：檢查 X 中是否存在 gender 列
        elif self.gender_col in X_transformed.columns:
            for gender in [1, 2]:
                mask_gender = X_transformed[self.gender_col] == gender

                if self.group_col in X_transformed.columns:
                    groups = X_transformed.loc[mask_gender, self.group_col].unique()
                    for group_val in groups:
                        if pd.isna(group_val):
                            continue
                        mask_group = mask_gender & (X_transformed[self.group_col] == group_val)

                        for col in columns:
                            if col in X_transformed.columns:
                                # 優先使用分組平均
                                if gender in self.group_means_ and group_val in self.group_means_[gender]:
                                    fill_value = self.group_means_[gender][group_val].get(col, np.nan)
                                else:
                                    fill_value = np.nan

                                # Fallback 到性別平均
                                if pd.isna(fill_value) and gender in self.fallback_means_:
                                    fill_value = self.fallback_means_[gender].get(col, np.nan)

                                # 最後 fallback 到全局平均
                                if pd.isna(fill_value) and col in self.global_means_:
                                    fill_value = self.global_means_[col]

                                if not pd.isna(fill_value):
                                    # 直接用 loc 賦值，避免 ChainedAssignmentError
                                    mask_na = mask_group & X_transformed[col].isna()
                                    X_transformed.loc[mask_na, col] = fill_value
        else:
            # 優先級 3：Fallback - gender 列不存在且沒有 y，使用全局平均
            for col in columns:
                if col in X_transformed.columns and col in self.global_means_:
                    mask = X_transformed[col].isna()
                    X_transformed.loc[mask, col] = self.global_means_[col]

        if not isinstance(X, pd.DataFrame):
            return X_transformed.values
        return X_transformed


def get_imputer_from_config(config):
    """根據 config 選擇合適的補值器"""

    method = config.get('preprocessing', {}).get('imputation_method', 'method0')
    columns_to_impute = config.get('features', {}).get('numeric_cols', ['height', 'weight'])

    if method == 'method0':
        return GlobalMedianImputer(columns_to_impute=columns_to_impute)
    elif method == 'method1':
        return GlobalMeanImputer(columns_to_impute=columns_to_impute)
    elif method == 'method2':
        return PaperRangeImputer(
            gender_col='gender',
            columns_to_impute=columns_to_impute
        )
    elif method == 'method3':
        return GroupedMeanImputer(
            gender_col='gender',
            group_col='star_sign',
            columns_to_impute=columns_to_impute
        )
    else:
        raise ValueError(f"未知的補值方法: {method}")
