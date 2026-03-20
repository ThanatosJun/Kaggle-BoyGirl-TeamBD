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

if __name__ == "__main__":
    df = a1_unified_ingestion()
