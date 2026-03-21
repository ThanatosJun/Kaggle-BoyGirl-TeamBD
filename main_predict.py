import os
import yaml
import joblib
import pandas as pd
import numpy as np
import sys

from src.data_loader import load_and_clean_data

def get_latest_experiment(base_dir):
    """獲取最新的實驗資料夾"""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"實驗目錄不存在: {base_dir}")

    # 列出所有實驗資料夾
    exp_folders = [f for f in os.listdir(base_dir) if f.startswith('exp_') and os.path.isdir(os.path.join(base_dir, f))]
    if not exp_folders:
        raise FileNotFoundError(f"在 {base_dir} 中找不到任何實驗資料夾")

    # 按照編號排序，取最新的
    exp_folders.sort(reverse=True)
    return exp_folders[0]

def main():
    # 1. 讀取設定檔
    with open("configs/default_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_dir = config['training']['save_dir']

    # 決定預測模式：fold 或 full
    predict_mode = "full"  # 預設為 full
    if len(sys.argv) > 2:
        predict_mode = sys.argv[2].lower()
    
    if predict_mode not in ["fold", "full"]:
        print(f"❌ 無效的預測模式: {predict_mode} (應為 'fold' 或 'full')")
        return

    # 2. 決定使用哪個實驗（預設使用最新的）
    if len(sys.argv) > 1:
        # 如果有提供參數，使用指定的實驗
        exp_id = sys.argv[1]
        exp_folder = f"exp_{int(exp_id):03d}_*"
        matching = [f for f in os.listdir(base_dir) if f.startswith(f"exp_{int(exp_id):03d}_")]
        if not matching:
            print(f"❌ 找不到實驗 {exp_id}")
            return
        exp_folder = matching[0]
    else:
        # 使用最新的實驗
        exp_folder = get_latest_experiment(base_dir)

    exp_path = os.path.join(base_dir, exp_folder)

    print(f"{'='*60}")
    print(f"🔮 使用實驗: {exp_folder}")
    print(f"📁 路徑: {exp_path}")
    print(f"🎯 預測模式: {predict_mode.upper()} ({'5-Fold Ensemble' if predict_mode == 'fold' else 'Full Train 單模型'})")
    print(f"{'='*60}\n")

    print(f"📦 正在載入測試資料: {config['data']['test_path']}")

    # 為了最後輸出 submission，我們需要先讀一次原始的 test.csv 把 ID 截取出來
    raw_test_df = pd.read_csv(config['data']['test_path'])

    # 載入並進行基礎清理：(is_train=False 代表不需要處理 Target 欄位)
    df_test = load_and_clean_data(config['data']['test_path'], is_train=False, config=config)

    # 3. 根據模式載入相應的模型
    if predict_mode == "fold":
        # 載入 5 個 fold 的模型與 preprocessor
        print(f"🧠 載入 5 個 Fold 的模型與 Preprocessor...")
        fold_models = []
        fold_preprocessors = []
        
        for fold_idx in range(5):
            fold_model_path = os.path.join(exp_path, f'fold_{fold_idx}_model.pkl')
            fold_prep_path = os.path.join(exp_path, f'fold_{fold_idx}_preprocessor.pkl')
            
            if not os.path.exists(fold_model_path) or not os.path.exists(fold_prep_path):
                print(f"❌ 找不到 Fold {fold_idx} 的模型或 Preprocessor")
                return
            
            fold_models.append(joblib.load(fold_model_path))
            fold_preprocessors.append(joblib.load(fold_prep_path))
            print(f"   ✓ Fold {fold_idx} 已載入")

        # 進行 Ensemble 預測（投票）
        print("🔮 進行 Ensemble 預測（投票模式）...")
        all_preds = []
        for fold_idx, (fold_model, fold_prep) in enumerate(zip(fold_models, fold_preprocessors)):
            X_test_trans = fold_prep.transform(df_test)
            preds = fold_model.predict(X_test_trans)
            all_preds.append(preds)
        
        # 投票：取多數意見
        all_preds = np.array(all_preds)
        preds = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=all_preds)
        
    else:
        # 載入 Full Train 模型
        prep_path = os.path.join(exp_path, 'preprocessor.pkl')
        model_path = os.path.join(exp_path, 'model.pkl')

        if not os.path.exists(prep_path) or not os.path.exists(model_path):
            print(f"❌ 找不到模型檔案:")
            print(f"   - {prep_path}")
            print(f"   - {model_path}")
            return

        print(f"🧠 載入特徵處理器與模型...")
        preprocessor = joblib.load(prep_path)
        model = joblib.load(model_path)

        # 4. 執行特徵轉換
        print("✨ 進行特徵轉換...")
        X_test_trans = preprocessor.transform(df_test)

        # 5. 進行預測
        print("🔮 進行預測...")
        preds = model.predict(X_test_trans)

    # 6. 將預測結果轉換回原始格式
    # 模型輸出: 0=女, 1=男
    # 需要轉換為: 2=女, 1=男（符合原始數據格式）
    preds_original = preds.copy()
    preds_original[preds == 0] = 2  # 女生：0 → 2
    # 男生保持 1 不變

    # 7. 輸出提交檔
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"submission_{exp_folder}_{predict_mode}.csv")
    latest_file = os.path.join(output_dir, f"submission_{predict_mode}.csv")
    print(f"💾 儲存預測結果至 {output_file}")
    submission = pd.DataFrame()

    # 如果原始測試集有 id，就放進 submission 中
    if 'id' in raw_test_df.columns:
        submission['id'] = raw_test_df['id']
    else:
        submission['id'] = range(1, len(preds_original) + 1)

    # 確保 gender 是整數型（1=男 或 2=女）
    submission[config['data']['target_col']] = preds_original.astype(int)

    submission.to_csv(output_file, index=False)
    submission.to_csv(latest_file, index=False)

    print(f"✅ 完成！")
    print(f"   - {output_file} (帶實驗編號與模式)")
    print(f"   - {latest_file} (模式版本)")
    print(f"\n📊 預測統計:")
    print(f"   - 男生 (1): {(preds_original == 1).sum()} 筆")
    print(f"   - 女生 (2): {(preds_original == 2).sum()} 筆")
    print(f"   - 總計: {len(preds_original)} 筆")

if __name__ == "__main__":
    main()
