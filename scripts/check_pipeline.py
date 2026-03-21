from src.pipeline import *
import pandas as pd

df_a1 = a1_unified_ingestion()
df_a2 = a2_schema_coercion(df_a1)
df_a3 = a3_boundary_clipping(df_a2)
df_a4 = a4_text_normalization(df_a3)
train_df, test_df, kfold_dict = a5_stratified_shuffle_indexing(df_a4)
train_b1, test_b1 = b1_anthropometric_features(train_df, test_df)
train_b2, test_b2 = b2_regression_residual(train_b1, test_b1)
train_b3, test_b3 = b3_noise_pruning(train_b2, test_b2)
train_b4_rules, test_b4_rules, train_b4_tfidf, test_b4_tfidf = b4_text_vectorization(train_b3, test_b3)

cols_to_exclude = ['id', 'is_train', 'gender', 'star_sign', 'phone_os', 'self_intro']
num_df = train_b3.drop(columns=[c for c in cols_to_exclude if c in train_b3.columns], errors='ignore')

print("\n=== PIPELINE DIAGNOSTICS ===")
print("1. Before get_dummies shape:", num_df.shape)
print("\n2. Columns in num_df:", num_df.columns.tolist())
print("\n3. Dtypes before encoding:\n", num_df.dtypes)

num_df_encoded = pd.get_dummies(num_df)
print("\n4. After get_dummies shape:", num_df_encoded.shape)
if num_df_encoded.shape[1] > num_df.shape[1]:
    print("NEW COLUMNS CREATED BY GET_DUMMIES:")
    print([c for c in num_df_encoded.columns if c not in num_df.columns])

print("\n5. TF-IDF Matrix shape:", train_b4_tfidf.shape)
print("6. Rule-based Matrix shape:", train_b4_rules.shape)
