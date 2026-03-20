import os
import yaml
import joblib
import pandas as pd

from src.data_loader import load_and_clean_data

def main():
    # 1. 讀取設定檔
    with open("configs/default_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"📦 正在載入測試資料: {config['data']['test_path']}")
    
    # 為了最後輸出 submission，我們需要先讀一次原始的 test.csv 把 ID 截取出來
    raw_test_df = pd.read_csv(config['data']['test_path'])
    
    # 載入並進行基礎清理：(is_train=False 代表不需要處理 Target 欄位)
    df_test = load_and_clean_data(config['data']['test_path'], is_train=False, config=config)

    # 2. 載入已經 Train 好的編碼器與模型
    save_dir = config['training']['save_dir']
    prep_path = os.path.join(save_dir, 'preprocessor.pkl')
    model_path = os.path.join(save_dir, 'model.pkl')

    print(f"🧠 載入特徵處理器與模型自: {save_dir}")
    preprocessor = joblib.load(prep_path)
    model = joblib.load(model_path)

    # 3. 執行特徵轉換 (⚠️ 注意：這裡只呼叫 transform)
    print("✨ 進行特徵轉換...")
    X_test_trans = preprocessor.transform(df_test)

    # 4. 進行預測
    print("🔮 進行預測...")
    preds = model.predict(X_test_trans)

    # 5. 輸出提交檔
    print("💾 儲存預測結果至 submission.csv")
    submission = pd.DataFrame()
    
    # 如果原始測試集有 id，就放進 submission 中
    if 'id' in raw_test_df.columns:
        submission['id'] = raw_test_df['id']
    else:
        submission['id'] = range(1, len(preds) + 1)
        
    submission[config['data']['target_col']] = preds
    
    submission.to_csv("submission.csv", index=False)
    print("✅ 完成！")

if __name__ == "__main__":
    main()
