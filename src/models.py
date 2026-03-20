from xgboost import XGBClassifier

def get_model(config):
    """
    實例化 XGBoost 模型，參數由 config 檔案決定
    """
    params = config['model']['xgb_params']
    
    # XGBoost 預設為樹狀演算法，可以支援缺失值，也能彈性調整參數
    model = XGBClassifier(**params)
    
    return model
