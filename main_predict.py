import os
import yaml
import joblib
import pandas as pd
import numpy as np
import sys
from datetime import datetime

from src.data_loader import load_and_clean_data
from src.features import engineer_features

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
    # 1. 先讀取 default config（取得 base_dir 等基礎設定）
    with open("configs/default_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_dir = config['training']['save_dir']
    pred_cfg = config.get('prediction', {})
    output_dir = pred_cfg.get('output_dir', 'result')
    default_mode = pred_cfg.get('default_mode', 'full').lower()
    valid_modes = ["fold", "full"]

    # 2. 解析命令列參數
    # 支援：
    # - python main_predict.py                -> 最新實驗 + full
    # - python main_predict.py fold           -> 最新實驗 + fold
    # - python main_predict.py 3              -> 指定實驗 + full
    # - python main_predict.py 3 fold         -> 指定實驗 + fold
    args = sys.argv[1:]
    predict_mode = default_mode
    exp_id = None

    if len(args) == 1:
        if args[0].lower() in valid_modes:
            predict_mode = args[0].lower()
        else:
            exp_id = args[0]
    elif len(args) >= 2:
        exp_id = args[0]
        predict_mode = args[1].lower()

    if predict_mode not in valid_modes:
        print(f"❌ 無效的預測模式: {predict_mode} (應為 'fold' 或 'full')")
        return

    # 3. 決定使用哪個實驗（預設使用最新的）
    if exp_id is not None:
        if not str(exp_id).isdigit():
            print(f"❌ 無效的實驗編號: {exp_id}（應為整數，例如 3）")
            return
        matching = [f for f in os.listdir(base_dir) if f.startswith(f"exp_{int(exp_id):03d}_")]
        if not matching:
            print(f"❌ 找不到實驗 {exp_id}")
            return
        exp_folder = matching[0]
    else:
        exp_folder = get_latest_experiment(base_dir)

    exp_path = os.path.join(base_dir, exp_folder)

    # ✅ 優先使用實驗資料夾裡保存的 config（確保 drop_cols、text_cols 等與訓練時一致）
    exp_config_path = os.path.join(exp_path, 'config.yaml')
    if os.path.exists(exp_config_path):
        with open(exp_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"📋 使用實驗配置: {exp_config_path}")
    else:
        print(f"⚠️ 實驗資料夾中無 config.yaml，使用 default_config.yaml")

    # 重新讀取 prediction 設定（可能被實驗 config 覆蓋）
    pred_cfg = config.get('prediction', {})
    output_dir = pred_cfg.get('output_dir', 'result')

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
        n_splits = config['training']['n_splits']
        print(f"🧠 載入 {n_splits} 個 Fold 模型、Imputer 與對應 Preprocessor...")
        
        # 載入每個 fold 的模型與對應 preprocessor
        fold_models = []
        fold_imputers = []
        fold_preprocessors = []
        for fold_idx in range(n_splits):
            fold_model_path = os.path.join(exp_path, f'fold_{fold_idx}_model.pkl')
            fold_imputer_path = os.path.join(exp_path, f'fold_{fold_idx}_imputer.pkl')
            fold_prep_path = os.path.join(exp_path, f'fold_{fold_idx}_preprocessor.pkl')
            
            if not os.path.exists(fold_model_path) or not os.path.exists(fold_prep_path):
                print(f"❌ 找不到 Fold {fold_idx} 的模型或 Preprocessor")
                return
            
            fold_models.append(joblib.load(fold_model_path))
            if os.path.exists(fold_imputer_path):
                fold_imputers.append(joblib.load(fold_imputer_path))
            else:
                fold_imputers.append(None)
            fold_preprocessors.append(joblib.load(fold_prep_path))
            print(f"   ✓ Fold {fold_idx} 模型、Imputer 與 Preprocessor 已載入")

        # 進行 Ensemble 預測（投票）
        print("🔮 進行 Ensemble 預測（投票模式）...")
        all_preds = []
        for fold_idx, (fold_model, fold_imputer, fold_prep) in enumerate(zip(fold_models, fold_imputers, fold_preprocessors)):
            X_test_input = fold_imputer.transform(df_test) if fold_imputer is not None else df_test
            X_test_input = engineer_features(X_test_input, config)
            X_test_trans = fold_prep.transform(X_test_input)
            preds = fold_model.predict(X_test_trans)
            all_preds.append(preds)
        
        # 投票：取多數意見
        all_preds = np.array(all_preds)
        preds = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=all_preds)
        
    else:
        # 載入 Full Train 模型
        imputer_path = os.path.join(exp_path, 'imputer.pkl')
        prep_path = os.path.join(exp_path, 'preprocessor.pkl')
        model_path = os.path.join(exp_path, 'model.pkl')

        if not os.path.exists(prep_path) or not os.path.exists(model_path):
            print(f"❌ 找不到模型檔案:")
            print(f"   - {prep_path}")
            print(f"   - {model_path}")
            return

        print(f"🧠 載入特徵處理器與模型...")
        imputer = joblib.load(imputer_path) if os.path.exists(imputer_path) else None
        preprocessor = joblib.load(prep_path)
        model = joblib.load(model_path)

        # 4. 執行特徵轉換（補值 → 衍生特徵 → preprocessor，與訓練流程一致）
        print("✨ 進行特徵轉換...")
        X_test_input = imputer.transform(df_test) if imputer is not None else df_test
        X_test_input = engineer_features(X_test_input, config)
        X_test_trans = preprocessor.transform(X_test_input)

        # 5. 進行預測
        print("🔮 進行預測...")
        preds = model.predict(X_test_trans)

    # 6. 將預測結果轉換回原始格式
    # 模型輸出: 0=女, 1=男
    # 需要轉換為: 2=女, 1=男（符合原始數據格式）
    output_mapping_raw = config.get('data', {}).get('target_output_mapping', {0: 2, 1: 1})
    output_mapping = {int(k): int(v) for k, v in output_mapping_raw.items()}
    preds_original = pd.Series(preds).map(output_mapping).fillna(pd.Series(preds)).astype(int).to_numpy()

    # 7. 輸出提交檔
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

    # 8. 記錄每次預測的男/女生數量
    male_count = int((preds_original == 1).sum())
    female_count = int((preds_original == 2).sum())
    total_count = int(len(preds_original))
    stats_log_file = os.path.join(output_dir, 'prediction_stats_log.csv')
    stats_record = pd.DataFrame([
        {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment': exp_folder,
            'mode': predict_mode,
            'male_count': male_count,
            'female_count': female_count,
            'total_count': total_count,
            'output_file': output_file,
        }
    ])

    if os.path.exists(stats_log_file):
        stats_record.to_csv(stats_log_file, mode='a', index=False, header=False)
    else:
        stats_record.to_csv(stats_log_file, index=False)

    print(f"✅ 完成！")
    print(f"   - {output_file} (帶實驗編號與模式)")
    print(f"   - {latest_file} (模式版本)")
    print(f"   - {stats_log_file} (預測統計紀錄)")
    print(f"\n📊 預測統計:")
    print(f"   - 男生 (1): {male_count} 筆")
    print(f"   - 女生 (2): {female_count} 筆")
    print(f"   - 總計: {total_count} 筆")

if __name__ == "__main__":
    main()
