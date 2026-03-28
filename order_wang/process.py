"""
性別預測 - 資料處理模組
僅保留原始欄位 + S-BERT 文字向量，不新增任何額外特徵欄位
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# ==================== 資料路徑配置 ====================
DATA_DIR = Path.home() / 'Documents' / 'projects' / 'Kaggle-BoyGirl-TeamBD' / 'data' / 'raw'
TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'


# ==================== 資料載入 ====================
def load_data():
    """載入訓練集和測試集"""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    
    print(f"訓練集: {train.shape}")
    print(f"測試集: {test.shape}")
    print(f"\n類別分布:\n{train['gender'].value_counts()}")
    
    return train, test


# ==================== 資料清理 ====================
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


# ==================== 異常值處理 ====================
def handle_outliers(df):
    """
    異常值處理
    僅將數值 Clip 到合理範圍，不新增任何額外欄位
    """
    df = df.copy()
    
    # Clip 到合理範圍 (限制極端值，不改變欄位數)
    df['height'] = df['height'].clip(140, 200)
    df['weight'] = df['weight'].clip(35, 120)
    df['iq'] = df['iq'].clip(70, 150)
    df['fb_friends'] = df['fb_friends'].clip(0, 2000)
    df['sleepiness'] = df['sleepiness'].clip(1, 5)
    df['yt'] = df['yt'].clip(0, 200)
    
    print("異常值處理完成 (僅 Clip，不新增欄位)")
    
    return df


# ==================== 缺失值處理 ====================
def handle_missing_values(df):
    """
    缺失值處理 - 方案A: 保留NaN
    
    數值特徵: 保留NaN (XGBoost/CatBoost可處理)
    類別特徵: 填補 'Unknown'
    文字特徵: 填補空字串
    """
    df = df.copy()
    
    # === 數值特徵: 保留NaN ===
    # XGBoost/CatBoost 會自動學習缺失模式
    # 不需要填補！
    
    # === 類別特徵: 填補 'Unknown' ===
    df['star_sign'] = df['star_sign'].fillna('Unknown')
    df['phone_os'] = df['phone_os'].fillna('Unknown')
    
    # === 文字特徵: 填補空字串 ===
    df['self_intro'] = df['self_intro'].fillna('')
    
    print("缺失值處理完成 (方案A: 數值特徵保留NaN)")
    
    return df


# ==================== 文字特徵提取 ====================
def extract_text_features(train_df, test_df, pca_dim=30):
    """
    使用 S-BERT 提取文字特徵，並用 PCA 降維
    
    Parameters:
        pca_dim: PCA 降維目標維度 (None=不降維)
    """
    # 載入預訓練模型
    print("載入 S-BERT 模型...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 提取訓練集文字向量
    print("提取訓練集文字特徵...")
    train_texts = train_df['self_intro'].fillna('').tolist()
    train_embeddings = model.encode(train_texts, show_progress_bar=True)
    
    # 提取測試集文字向量
    print("提取測試集文字特徵...")
    test_texts = test_df['self_intro'].fillna('').tolist()
    test_embeddings = model.encode(test_texts, show_progress_bar=True)
    
    original_dim = train_embeddings.shape[1]
    print(f"S-BERT 原始維度: {original_dim}")
    
    # PCA 降維
    pca = None
    if pca_dim is not None and pca_dim < original_dim:
        print(f"執行 PCA 降維: {original_dim} → {pca_dim}...")
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
    
    print(f"S-BERT 特徵提取完成，最終維度: {final_dim}")
    
    return train_sbert_df, test_sbert_df, pca


# ==================== 完整預處理流程 ====================
def preprocess_data(train_df, test_df):
    """
    完整資料預處理流程
    """
    print("=" * 50)
    print("開始資料預處理")
    print("=" * 50)
    
    # 1. 資料清理 (YT 轉數值 + 修正身高體重對調)
    print("\n[1/4] 資料清理...")
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)
    
    # 2. 異常值處理 (僅 Clip，不新增欄位)
    print("\n[2/4] 異常值處理...")
    train_df = handle_outliers(train_df)
    test_df = handle_outliers(test_df)
    
    # 3. 缺失值處理 (數值保留NaN、類別填 Unknown、文字填空字串)
    print("\n[3/4] 缺失值處理...")
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)
    
    # 4. 文字特徵提取 (S-BERT 384維 → PCA 降至 30維)
    print("\n[4/4] 文字特徵提取 + PCA 降維...")
    train_sbert, test_sbert, pca = extract_text_features(train_df, test_df, pca_dim=30)
    
    # 6. 合併所有特徵
    train_final = pd.concat([train_df, train_sbert], axis=1)
    test_final = pd.concat([test_df, test_sbert], axis=1)
    
    print("\n" + "=" * 50)
    print("預處理完成！")
    print(f"訓練集最終形狀: {train_final.shape}")
    print(f"測試集最終形狀: {test_final.shape}")
    print("=" * 50)
    
    return train_final, test_final


# ==================== 主程式 (測試用) ====================
if __name__ == "__main__":
    # 載入資料
    train, test = load_data()
    
    # 執行預處理
    train_processed, test_processed = preprocess_data(train, test)
    
    # 顯示結果
    print("\n訓練集前5筆:")
    print(train_processed.head())
    
    print("\n特徵列表:")
    print(train_processed.columns.tolist())
