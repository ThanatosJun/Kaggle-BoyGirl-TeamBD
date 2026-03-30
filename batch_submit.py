import os
import subprocess
import time

def format_message(file_name):
    """
    將檔名轉換成可讀的 Message。
    預期檔名格式例如: submission_exp_063_exp2_RF_method0_fold.csv
    目標輸出格式例如: Exp2: RF method0-Median (fold)
    """
    # 移除 .csv
    name = file_name.replace(".csv", "")
    
    # 判斷是 fold 還是 full
    mode = "full"
    if name.endswith("_fold"):
        mode = "fold"
        name = name[:-5] # 移除 _fold
    elif name.endswith("_full"):
        name = name[:-5] # 移除 _full
        
    parts = name.split('_')
    # 解析各個部分
    # submission_exp_063_exp2_RF_method0 -> parts: ['submission', 'exp', '063', 'exp2', 'RF', 'method0']
    # 對於更長的，例如 submission_exp_078_exp3_exp2_with_bmi_tfidf -> ...
    
    if len(parts) >= 6:
        # 從第 3 個元素開始 (index 3) 就是實驗的大分類，例如 exp1, exp2, exp3
        exp_group = parts[3].capitalize() # Exp1, Exp2, Exp3
        
        # 剩下的部分重組
        model_and_method = "_".join(parts[4:])
        
        # 根據方法名給予可讀的名稱翻譯 (method0 -> Median, method1 -> Mean, method2 -> Mode, method3 -> Baseline 等)
        if "method0" in model_and_method:
            model_and_method = model_and_method.replace("method0", "method0-Median")
        elif "method1" in model_and_method:
            model_and_method = model_and_method.replace("method1", "method1-Mean")
        elif "method2" in model_and_method:
            model_and_method = model_and_method.replace("method2", "method2-Mode")
        elif "method3" in model_and_method:
            model_and_method = model_and_method.replace("method3", "method3-Baseline")
            
        # 針對特徵的優化顯示 (若為 exp3 開頭的特殊命名)
        model_and_method = model_and_method.replace("_", " ")

        return f"{exp_group}: {model_and_method} ({mode})"
    else:
        # 如果解析失敗，退回使用原始名稱
        return name

def main():
    result_dir = "result"
    competition_name = "boy-or-girl-2026-new"
    
    if not os.path.exists(result_dir):
        print(f"❌ 錯誤：找不到目錄 {result_dir}")
        return

    # 找出所有包含 exp_ 的 submission 檔案，且實驗編號 >= 17 
    files = []
    for f in os.listdir(result_dir):
        if f.startswith('submission_exp_') and f.endswith('.csv'):
            parts = f.split('_')
            # 解析檔名中的實驗編號，例如 submission_exp_017_... -> parts[2] 就是 '017'
            if len(parts) >= 3 and parts[2].isdigit():
                exp_id = int(parts[2])
                if exp_id >= 71:
                    files.append(f)
    files.sort()

    if not files:
        print(f"⚠️ 在 {result_dir}/ 中找不到任何符合條件的提交檔案。")
        return

    print(f"🔍 找到 {len(files)} 個 submission 檔案，準備上傳...\n")
    print("⚠️ 警告：Kaggle 通常一天有提交次數限制（5~20次不等），請確認你不會超過每日上限！\n")

    for file_name in files:
        file_path = os.path.join(result_dir, file_name)
        # 產生可讀的 submission message
        message = format_message(file_name)
        
        print(f"{'='*60}")
        print(f"🚀 提交檔案: {file_name}")
        print(f"📝 訊息: {message}")
        print(f"{'='*60}")
        
        cmd = [
            "kaggle", "competitions", "submit",
            "-c", competition_name,
            "-f", file_path,
            "-m", message
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ 提交成功！")
        except subprocess.CalledProcessError as e:
            print(f"❌ 提交失敗: {e}")
            
        # 加上一點延遲防止 Kaggle API 阻擋頻繁請求
        time.sleep(3)

    print("\n🎉 所有批次上傳指令執行完畢！")

if __name__ == "__main__":
    main()
