import pandas as pd
import numpy as np

def load_and_clean_data(file_path, is_train=True, config=None):
    """
    讀取資料並進行基礎清理：包含丟掉無用特徵與格式轉換。
    """
    df = pd.read_csv(file_path)
    
    # 1. 丟棄不需要計算的特徵
    cols_to_drop = [c for c in config['data']['drop_cols'] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 2. 基本清理：將 sleepiness 強制轉為數值型態 (None/NaN 會保留為空值交給後續 Imputer)
    if 'sleepiness' in df.columns:
        df['sleepiness'] = pd.to_numeric(df['sleepiness'], errors='coerce')

    # 3. 如果是訓練集，處理 Target 欄位並移除 Target 是空值的異常資料
    if is_train and config['data']['target_col'] in df.columns:
        target = config['data']['target_col']
        # 防呆機制：移除 target 為空的列
        df = df.dropna(subset=[target])
        
        # 將性別進行 Label Encoding -> 男: 1, 女: 0 (依據您的定義)
        # 這裡支援字串或原生的轉換寫法
        # 原始數據: 1=男, 2=女
        mapping = {'男': 1, '女': 0, 'Male': 1, 'Female': 0, 1: 1, 2: 0, 0: 0, '1': 1, '0': 0}
        df[target] = df[target].map(mapping)
        
    return df

def split_X_y(df, config):
    """
    將 DataFrame 分離出 特徵 (X) 與 標籤 (y)
    """
    target = config['data']['target_col']
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
