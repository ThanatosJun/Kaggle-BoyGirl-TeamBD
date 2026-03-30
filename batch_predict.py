import os
import subprocess
import sys

def main():
    # 取得模式參數，預設為 full
    mode = "full"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ["fold", "full"]:
            print(f"❌ 錯誤：不支援的模式 '{mode}'。請填寫 'fold' 或 'full'。")
            return

    base_dir = "experiments"
    if not os.path.exists(base_dir):
        print(f"❌ 錯誤：找不到目錄 {base_dir}")
        return

    # 找出所有實驗資料夾
    exp_folders = [f for f in os.listdir(base_dir) if f.startswith('exp_') and os.path.isdir(os.path.join(base_dir, f))]
    
    # 按照編號排序
    exp_folders.sort()

    print(f"🔍 找到 {len(exp_folders)} 個實驗，準備進行 Batch Predict...\n")

    for folder in exp_folders:
        # 解析 exp_id，例如 'exp_047_exp1_RF_method0' -> '047' -> 47
        parts = folder.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            exp_id_int = int(parts[1])
            
            print(f"\n{'='*60}")
            print(f"🚀 開始預測實驗: {folder} (ID: {exp_id_int}) - 模式: {mode}")
            print(f"{'='*60}")
            
            # 呼叫 main_predict.py <exp_id> <mode>
            cmd = ["python", "main_predict.py", str(exp_id_int), mode]
            try:
                # 使用 subprocess 執行，並將輸出印在螢幕上
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ 預測實驗 {folder} 時發生錯誤: {e}")
        else:
            print(f"⚠️ 無法解析實驗資料夾名稱: {folder}，跳過...")

    print("\n🎉 所有實驗 batch 預測完成！")

if __name__ == "__main__":
    main()
