#!/usr/bin/env python3
"""
批量統一所有實驗配置文件的模型參數
"""
import yaml
from pathlib import Path

# 統一的模型參數設定
UNIFIED_PARAMS = {
    'catboost': {
        'depth': 6,  # 統一使用 6
    },
    'random_forest': {
        'max_depth': 12,  # 統一使用 12
        'remove_class_weight': True,  # 移除 class_weight，由 training.class_weight 統一控制
    },
    'training': {
        'class_weight': "balanced",  # 統一使用字符串格式
    }
}

def unify_config(config_path):
    """統一單個配置文件的模型參數"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    changes = []

    # 1. 統一 CatBoost depth
    if 'model' in config and 'catboost_params' in config['model']:
        old_depth = config['model']['catboost_params'].get('depth')
        new_depth = UNIFIED_PARAMS['catboost']['depth']
        if old_depth != new_depth:
            config['model']['catboost_params']['depth'] = new_depth
            changes.append(f"  CatBoost depth: {old_depth} → {new_depth}")

    # 2. 統一 Random Forest max_depth
    if 'model' in config and 'random_forest_params' in config['model']:
        old_max_depth = config['model']['random_forest_params'].get('max_depth')
        new_max_depth = UNIFIED_PARAMS['random_forest']['max_depth']
        if old_max_depth != new_max_depth:
            config['model']['random_forest_params']['max_depth'] = new_max_depth
            changes.append(f"  RF max_depth: {old_max_depth} → {new_max_depth}")

        # 3. 移除 Random Forest class_weight
        if 'class_weight' in config['model']['random_forest_params']:
            old_cw = config['model']['random_forest_params'].pop('class_weight')
            changes.append(f"  RF class_weight: {old_cw} → (removed, use training.class_weight)")

    # 4. 統一 training.class_weight
    if 'training' in config:
        old_cw = config['training'].get('class_weight')
        new_cw = UNIFIED_PARAMS['training']['class_weight']
        # 標準化比較（balanced 和 "balanced" 視為相同）
        if str(old_cw).lower().strip('"\'') != str(new_cw).lower().strip('"\''):
            config['training']['class_weight'] = new_cw
            changes.append(f"  training.class_weight: {old_cw} → {new_cw}")

    # 保存修改
    if changes:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        return True, changes
    return False, []

def main():
    configs_dir = Path('configs')
    exp_configs = sorted(configs_dir.glob('exp*.yaml'))

    print(f"🔧 開始統一 {len(exp_configs)} 個實驗配置文件的模型參數...\n")

    modified_count = 0
    for config_path in exp_configs:
        modified, changes = unify_config(config_path)
        if modified:
            print(f"✅ {config_path.name}")
            for change in changes:
                print(change)
            print()
            modified_count += 1
        else:
            print(f"⏭️  {config_path.name} (已是統一參數，跳過)")

    print(f"\n{'='*60}")
    print(f"🎉 完成！共修改 {modified_count}/{len(exp_configs)} 個配置文件")
    print(f"{'='*60}")
    print(f"\n📋 統一後的參數:")
    print(f"  - CatBoost depth: 6")
    print(f"  - Random Forest max_depth: 12")
    print(f"  - Random Forest class_weight: (removed)")
    print(f"  - training.class_weight: \"balanced\"")

if __name__ == '__main__':
    main()
