import os
import yaml
import joblib
import json
import pandas as pd
from datetime import datetime

from src.data_loader import load_and_clean_data, split_X_y
from src.features import build_preprocessor
from src.models import get_model
from src.evaluate import cross_validate_with_smote

def get_next_experiment_id(base_dir):
    """獲取下一個實驗編號"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return 1

    # 列出所有實驗資料夾
    exp_folders = [f for f in os.listdir(base_dir) if f.startswith('exp_')]
    if not exp_folders:
        return 1

    # 提取編號並找出最大值
    exp_ids = [int(f.split('_')[1]) for f in exp_folders if f.split('_')[1].isdigit()]
    return max(exp_ids) + 1 if exp_ids else 1

def save_experiment_log(base_dir, exp_folder, config, metrics, timestamp):
    """將實驗結果記錄到 CSV"""
    log_file = os.path.join(base_dir, 'experiment_log.csv')

    # 準備記錄
    record = {
        'exp_id': exp_folder,
        'timestamp': timestamp,
        'name': config['experiment']['name'],
        'description': config['experiment']['description'],
        'use_smote': config['training']['use_smote'],
        'learning_rate': config['model']['xgb_params']['learning_rate'],
        'max_depth': config['model']['xgb_params']['max_depth'],
        'mean_accuracy': metrics['mean_accuracy'],
        'std_accuracy': metrics['std_accuracy'],
        'mean_f1': metrics['mean_f1'],
        'std_f1': metrics['std_f1'],
        'mean_precision': metrics['mean_precision'],
        'std_precision': metrics['std_precision'],
        'mean_recall': metrics['mean_recall'],
        'std_recall': metrics['std_recall']
    }

    # 如果檔案不存在，創建新檔案
    if not os.path.exists(log_file):
        df = pd.DataFrame([record])
        df.to_csv(log_file, index=False)
        print(f"📝 創建實驗日誌: {log_file}")
    else:
        # 追加到現有檔案
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(log_file, index=False)
        print(f"📝 更新實驗日誌: {log_file}")

def main():
    # 1. 讀取設定檔
    print("🔖 讀取實驗設定檔: configs/default_config.yaml")
    with open("configs/default_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 創建實驗目錄
    base_dir = config['training']['save_dir']
    exp_id = get_next_experiment_id(base_dir)
    exp_name = config['experiment']['name']
    exp_folder = f"exp_{exp_id:03d}_{exp_name}"
    exp_path = os.path.join(base_dir, exp_folder)
    os.makedirs(exp_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print(f"🔬 實驗編號: {exp_folder}")
    print(f"📝 實驗名稱: {exp_name}")
    print(f"📄 描述: {config['experiment']['description']}")
    print(f"⏰ 時間: {timestamp}")
    print(f"📁 保存路徑: {exp_path}")
    print(f"{'='*60}\n")

    # 3. 保存配置
    config_path = os.path.join(exp_path, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print(f"💾 已保存配置: {config_path}")

    # 4. 讀取並清理資料
    print(f"📦 正在載入訓練資料: {config['data']['train_path']}")
    df_train = load_and_clean_data(config['data']['train_path'], is_train=True, config=config)

    # 5. 準備 X_train, y_train
    X, y = split_X_y(df_train, config)

    # 6. 建立 Pipeline 元件與模型
    print("🛠️ 正在建立 Preprocessor 特徵工程管線 與 模型...")
    preprocessor = build_preprocessor(config)
    model = get_model(config)

    # 7. 執行 5-Fold CV 進行可靠的效能評估
    cv_metrics, fold_models, fold_preprocessors = cross_validate_with_smote(X, y, preprocessor, model, config)

    # 8. 計算平均指標
    import numpy as np
    metrics_summary = {
        'mean_accuracy': np.mean(cv_metrics['accuracy']),
        'std_accuracy': np.std(cv_metrics['accuracy']),
        'mean_f1': np.mean(cv_metrics['f1']),
        'std_f1': np.std(cv_metrics['f1']),
        'mean_precision': np.mean(cv_metrics['precision']),
        'std_precision': np.std(cv_metrics['precision']),
        'mean_recall': np.mean(cv_metrics['recall']),
        'std_recall': np.std(cv_metrics['recall']),
        'fold_results': {
            'accuracy': cv_metrics['accuracy'],
            'f1': cv_metrics['f1'],
            'precision': cv_metrics['precision'],
            'recall': cv_metrics['recall']
        }
    }

    # 9. 保存 CV 結果
    results_path = os.path.join(exp_path, 'cv_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        # 將 numpy 類型轉換為 Python 原生類型
        json_metrics = {k: (v.tolist() if isinstance(v, np.ndarray) else
                           {k2: (v2.tolist() if isinstance(v2, np.ndarray) else v2)
                            for k2, v2 in v.items()} if isinstance(v, dict) else v)
                       for k, v in metrics_summary.items()}
        json.dump(json_metrics, f, indent=2, ensure_ascii=False)
    print(f"💾 已保存 CV 結果: {results_path}")

    # 10. 在「所有」訓練資料上，訓練最終正式模型
    print("\n🚀 正在使用「所有訓練集」訓練最終對外預測模型...")
    X_trans = preprocessor.fit_transform(X)

    if config['training']['use_smote']:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=config['training']['random_state'])
        X_trans, y = smote.fit_resample(X_trans, y)

    model.fit(X_trans, y)

    # 11. 儲存模型與特徵處理器
    prep_path = os.path.join(exp_path, 'preprocessor.pkl')
    model_path = os.path.join(exp_path, 'model.pkl')

    joblib.dump(preprocessor, prep_path)
    joblib.dump(model, model_path)
    print(f"💾 已保存 Preprocessor: {prep_path}")
    print(f"💾 已保存 Model: {model_path}")

    # 11a. 儲存 5 個 Fold 的模型與 Preprocessor（用於 Ensemble 預測）
    print(f"\n💾 儲存 Fold 模型與 Preprocessor...")
    for fold_idx, (fold_model, fold_prep) in enumerate(zip(fold_models, fold_preprocessors)):
        fold_model_path = os.path.join(exp_path, f'fold_{fold_idx}_model.pkl')
        fold_prep_path = os.path.join(exp_path, f'fold_{fold_idx}_preprocessor.pkl')
        joblib.dump(fold_model, fold_model_path)
        joblib.dump(fold_prep, fold_prep_path)
        print(f"   ✓ Fold {fold_idx}: {fold_model_path}, {fold_prep_path}")

    # 12. 更新實驗日誌
    save_experiment_log(base_dir, exp_folder, config, metrics_summary, timestamp)

    # 13. 顯示實驗總結
    print(f"\n{'='*60}")
    print(f"🎉 實驗 {exp_folder} 完成！")
    print(f"{'='*60}")
    print(f"📊 Cross-Validation 平均結果:")
    print(f"  - Accuracy:  {metrics_summary['mean_accuracy']:.4f} (± {metrics_summary['std_accuracy']:.4f})")
    print(f"  - F1-Score:  {metrics_summary['mean_f1']:.4f} (± {metrics_summary['std_f1']:.4f})")
    print(f"  - Precision: {metrics_summary['mean_precision']:.4f} (± {metrics_summary['std_precision']:.4f})")
    print(f"  - Recall:    {metrics_summary['mean_recall']:.4f} (± {metrics_summary['std_recall']:.4f})")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
