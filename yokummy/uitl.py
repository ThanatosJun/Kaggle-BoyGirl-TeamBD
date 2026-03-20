import kagglehub
import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
import shutil

def need_caculate_feature(data: pd.DataFrame):
    """返回 DataFrame 中所有數值型欄位的名稱列表"""
    return data.select_dtypes(include=[np.number]).columns.tolist()

def download_dataset(name:str,store_path:str):
    #檢查檔案是否存在
    os.makedirs(store_path, exist_ok=True)
    csv_file=[f for f in os.listdir(store_path) if f.endswith('.csv')]
    if csv_file:
        print(f"Dataset already exists at {store_path}. Skipping download.")
        return 
    else:
        print(f"Downloading dataset '{name}' to {store_path}...")
        download_path = kagglehub.competition_download('boy-or-girl-2026-new')
        
        # 將下載的檔案複製到指定路徑
        if os.path.exists(download_path):
            for file in os.listdir(download_path):
                if file.endswith('.csv'):
                    src = os.path.join(download_path, file)
                    dst = os.path.join(store_path, file)
                    shutil.copy2(src, dst)
                    print(f"Dataset '{name}' downloaded successfully to {store_path}.")
            return True
        else:
            print(f"Failed to download dataset '{name}'.")
            return False
        
def caculate_mean(data, path, col, store_path):
    """計算指定欄位的平均數並保存到 JSON"""
    
    feature_name = need_caculate_feature(data)
    if col not in feature_name:
        print(f"Column '{col}' is not in the list of features to calculate.")
        return None
    mean_value = data[col].mean()
    
    json_file = f"{store_path}"
    
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = {}

    if col not in result:
        result[col] = {}

    result[col]['mean'] = float(mean_value)
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"{col} 的平均數 ({mean_value:.2f}) 已保存至 {json_file}")
    

def caculate_median(data,path,col,store_path):
    feature_name = need_caculate_feature(data)
    if col not in feature_name:
        print(f"Column '{col}' is not in the list of features to calculate.")
        return None
    median_value = data[col].median()
    
    json_file = f"{store_path}"
    
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = {}

    if col not in result:
        result[col] = {}

    result[col]['median'] = float(median_value)
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"{col} 的中位數 ({median_value:.2f}) 已保存至 {json_file}")

def process_missing_values(data: pd.DataFrame, store_path: str, record_path: str = "results/stastic/missing_value.json"):
    """
    處理缺失值的函數
    - 統計缺失值數量
    - 丟棄有缺失值的資料行
    - 保存處理後的數據和統計報告
    """
    data_copy = data.copy()
    missing_values_count = data_copy.isnull().sum()
    total_missing_values = missing_values_count.sum()
    idx = data_copy[data_copy.isnull().any(axis=1)].index.tolist()
    
    print(f"原始資料筆數: {len(data_copy)}")
    print(f"總缺失值數量: {total_missing_values}")
    print(f"有缺失值的資料筆數: {len(idx)}")
    data_copy.drop(idx, inplace=True)
    data_copy.reset_index(drop=True, inplace=True)
    
    print(f"處理後資料筆數: {len(data_copy)}")
    os.makedirs(store_path, exist_ok=True)
    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    
    output_csv = f"{store_path}/dataset-filtered.csv"
    data_copy.to_csv(output_csv, index=False)
    print(f"處理後的數據已保存至: {output_csv}")
    
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump({
            "original_rows": len(data),
            "filtered_rows": len(data_copy),
            "dropped_rows_count": len(idx),
            "total_missing_values": int(total_missing_values),
            "missing_values_by_column": missing_values_count.to_dict(),
            "dropped_row_indices": idx
        }, f, indent=4, ensure_ascii=False)
    
    print(f"缺失值報告已保存至: {record_path}")
    
    return data_copy
    
def col_process(data: pd.DataFrame, cols=["grade_level", "ai_tools_used"], store_path: str = "."):
    """
    處理特定欄位：
    - grade_level: 移除 " Year" 並轉為整數
    - ai_tools_used: 拆分欄位為 "toolname_usage" 0/1
    """
    data_copy = data.copy()
    
    for col in cols:
        if col not in data_copy.columns:
            print(f"Column '{col}' does not exist in the DataFrame. Skipping...")
            continue
        
        if col == "grade_level":
            data_copy[col] = (data_copy[col]
                .str.replace(" Year", "", regex=False)
                .str.replace(r'(st|nd|rd|th)', '', regex=True)
                .str.strip()
                .astype(int))
            print(f"{col} 已處理：轉換為整數 (1, 2, 3, 4, 10, 11, 12)")
    
        # 處理 ai_tools_used
        elif col == "ai_tools_used":
            tool_names = set()
            
            for tools_str in data_copy[col]:
                if pd.notna(tools_str) and str(tools_str).lower().strip() != "none":
                    tool_list = str(tools_str).split("+")
                    for tool in tool_list:
                        tool_cleaned = tool.strip()
                        if tool_cleaned:
                            tool_names.add(tool_cleaned)
            
            for tool in sorted(tool_names):
                data_copy[f"{tool}_usage"] = data_copy[col].apply(
                    lambda x: 1 if (pd.notna(x) and 
                                    str(x).lower().strip() != "none" and 
                                    tool in str(x).split("+")) 
                              else 0
                )
            
            print(f"{col} 已處理：創建 {len(tool_names)} 個工具欄位")
            data_copy.drop(columns=[col], inplace=True)
    
    os.makedirs(store_path, exist_ok=True)
    output_csv = f"{store_path}/dataset-convert.csv"
    data_copy.to_csv(output_csv, index=False)
    print(f"處理後的數據已保存至: {output_csv}")
    
    return data_copy

    


def find_extreme_values(data):
    pass