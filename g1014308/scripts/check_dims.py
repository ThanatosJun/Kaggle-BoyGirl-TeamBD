import pandas as pd

def investigate_columns():
    # Load raw data
    df = pd.read_csv("boy or girl 2025 train_missingValue.csv")
    
    print("--- 1. All Columns in raw training data ---")
    print(df.columns.tolist())
    
    # Check for string / object types
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("\n--- 2. Columns identified as string/object ---")
    print(object_cols)
    
    # Known structual and categorical exclusions
    cols_to_exclude = ['id', 'is_train', 'gender', 'star_sign', 'phone_os', 'self_intro', 'yt', 'sleepiness']
    
    # Find the culprits
    suspicious = [c for c in object_cols if c not in cols_to_exclude]
    print("\n--- 3. SUSPICIOUS STRING COLUMNS (Not explicitly excluded) ---")
    print(suspicious)
    
    for col in suspicious:
        print(f"\nUnique values in '{col}': {df[col].nunique()}")
        print(f"Top 5 values in '{col}':\n{df[col].value_counts().head(5)}")

if __name__ == "__main__":
    investigate_columns()
