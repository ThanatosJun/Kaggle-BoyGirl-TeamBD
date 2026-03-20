import os
import yaml
import joblib

from src.data_loader import load_and_clean_data, split_X_y
from src.features import build_preprocessor
from src.models import get_model
from src.evaluate import cross_validate_with_smote

def main():
    # 1. 讀取設定檔
    print("🔖 讀取實驗設定檔: configs/default_config.yaml")
    with open("configs/default_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 讀取並清理資料
    print(f"📦 正在載入訓練資料: {config['data']['train_path']}")
    df_train = load_and_clean_data(config['data']['train_path'], is_train=True, config=config)
    
    # 3. 準備 X_train, y_train
    X, y = split_X_y(df_train, config)
    
    # 4. 建立 Pipeline 元件與模型
    print("🛠️ 正在建立 Preprocessor 特徵工程管線 與 模型...")
    preprocessor = build_preprocessor(config)
    model = get_model(config)

    # 5. 執行 5-Fold CV 進行可靠的效能評估
    cross_validate_with_smote(X, y, preprocessor, model, config)

    # 6. 在 "所有" 訓練資料上，訓練最終正式模型 (為了產出 Predict 用)
    print("\n🚀 正在使用「所有訓練集」訓練最終對外預測模型...")
    X_trans = preprocessor.fit_transform(X)
    
    if config['training']['use_smote']:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=config['training']['random_state'])
        X_trans, y = smote.fit_resample(X_trans, y)
        
    model.fit(X_trans, y)

    # 7. 儲存模型與特徵處理器給 Test 使用
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    prep_path = os.path.join(save_dir, 'preprocessor.pkl')
    model_path = os.path.join(save_dir, 'model.pkl')
    
    joblib.dump(preprocessor, prep_path)
    joblib.dump(model, model_path)
    print(f"💾 儲存成功！已將前處理器存至 {prep_path}，模型存至 {model_path}")

if __name__ == "__main__":
    main()
