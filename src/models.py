from sklearn.ensemble import RandomForestClassifier

def _resolve_model_type(config):
    model_cfg = config.get('model', {})
    raw_model_type = model_cfg.get('type', 'xgboost').lower()

    alias_map = {
        'xgboost': 'xgboost',
        'xgb': 'xgboost',
        'lightgbm': 'lightgbm',
        'lgbm': 'lightgbm',
        'random_forest': 'random_forest',
        'rf': 'random_forest'
    }
    model_type = alias_map.get(raw_model_type)

    if model_type is None:
        raise ValueError(
            f"不支援的模型類型: {raw_model_type}，請使用 'xgboost'、'lightgbm' 或 'random_forest'"
        )
    return model_type


def get_model(config, override_params=None):
    """
    根據 config 實例化模型。
    支援：xgboost / lightgbm / random_forest
    （向下相容：xgb / lgbm / rf）
    """
    model_cfg = config.get('model', {})
    model_type = _resolve_model_type(config)

    if model_type == 'xgboost':
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "model.type 設為 'xgboost'，但環境未安裝 xgboost。"
                "請先執行: pip install xgboost"
            ) from exc
        params = dict(model_cfg['xgb_params'])
        if override_params:
            params.update(override_params)
        return XGBClassifier(**params)

    if model_type == 'lightgbm':
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "model.type 設為 'lightgbm'，但環境未安裝 lightgbm。"
                "請先執行: pip install lightgbm"
            ) from exc
        params = dict(model_cfg['lgbm_params'])
        if override_params:
            params.update(override_params)
        return LGBMClassifier(**params)

    if model_type == 'random_forest':
        params = dict(model_cfg['random_forest_params'])
        if override_params:
            params.update(override_params)
        return RandomForestClassifier(**params)
