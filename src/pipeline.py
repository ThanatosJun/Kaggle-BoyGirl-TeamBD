import pandas as pd
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def a1_unified_ingestion():
    """
    Task A1: Unified Ingestion & Duplicate Eradication
    """
    logging.info("Starting A1: Unified Ingestion & Duplicate Eradication")
    train_path = "boy or girl 2025 train_missingValue.csv"
    test_path = "boy or girl 2025 test no ans_missingValue.csv"
    sample_sub_path = "Boy_or_girl_test_sandbox_sample_submission.csv"

    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        sample_sub = pd.read_csv(sample_sub_path)
    except FileNotFoundError as e:
        logging.fatal(f"File not found: {e}")
        sys.exit(1)

    train['is_train'] = 1
    test['is_train'] = 0

    # Validation Criteria 1: Deduplication
    cols_to_check = [col for col in train.columns if col not in ['id', 'is_train']]
    initial_train_len = len(train)
    train = train.drop_duplicates(subset=cols_to_check, keep='first').copy()
    dropped_rows = initial_train_len - len(train)
    logging.info(f"Dropped {dropped_rows} duplicate rows from train.")

    merged_dataframe = pd.concat([train, test], axis=0, ignore_index=True)

    # Validation Criteria 2: Fatal check on test length
    test_len = len(merged_dataframe[merged_dataframe['is_train'] == 0])
    sub_len = len(sample_sub)
    if test_len != sub_len:
        logging.fatal(f"A1 Validation Failed: Test set length ({test_len}) != Sample Submission length ({sub_len})")
        sys.exit(1)
    
    logging.info(f"A1 validation passed. Merged dataframe has {len(merged_dataframe)} rows.")
    return merged_dataframe

def a2_schema_coercion(df):
    """
    Task A2: Schema Coercion & Anomaly Melting
    """
    logging.info("Starting A2: Schema Coercion & Anomaly Melting")
    
    # Target columns that should be float64
    numeric_cols = ['yt', 'height', 'weight', 'sleepiness', 'iq', 'fb_friends']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Validation Criteria
    for col in ['yt', 'height', 'weight']:
        if col in df.columns:
            if not pd.api.types.is_float_dtype(df[col]):
                logging.fatal(f"A2 Validation Failed: Column '{col}' is not float64. Found {df[col].dtype}")
                sys.exit(1)
                
    logging.info("A2 validation passed. Schema aligned.")
    return df

def a3_boundary_clipping(df):
    """
    Task A3: Physical Boundary Clipping
    """
    logging.info("Starting A3: Physical Boundary Clipping")
    
    if 'height' in df.columns:
        df['height'] = df['height'].clip(lower=140, upper=210)
    if 'weight' in df.columns:
        df['weight'] = df['weight'].clip(lower=35, upper=150)
        
    # Validation Criteria
    if 'height' in df.columns:
        h_max, h_min = df['height'].max(), df['height'].min()
        if h_max > 210 or h_min < 140:
            logging.fatal(f"A3 Validation Failed: height bounds exceeded ({h_min}, {h_max})")
            sys.exit(1)
            
    if 'weight' in df.columns:
        w_max, w_min = df['weight'].max(), df['weight'].min()
        if w_max > 150 or w_min < 35:
            logging.fatal(f"A3 Validation Failed: weight bounds exceeded ({w_min}, {w_max})")
            sys.exit(1)
            
    logging.info("A3 validation passed. Physical boundaries clipped.")
    return df

if __name__ == "__main__":
    df_a1 = a1_unified_ingestion()
    df_a2 = a2_schema_coercion(df_a1)
    df_a3 = a3_boundary_clipping(df_a2)
