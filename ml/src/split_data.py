import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.src.data import process_dataset, TARGET_COLUMN


def run_exploration_and_split(
    raw_data_path: str = "data/ai4i2020.csv",
    output_dir: str = "data",
    seed: int = 42
) -> None:
    print("--- 1. Loading and Cleaning Data ---")
    df = process_dataset(raw_data_path)
    print(f"Cleaned dataset shape: {df.shape}")
    
    print("\n--- 2. Target Class Imbalance Exploration ---")
    total_rows = len(df)
    failures = df[TARGET_COLUMN].sum()
    failure_rate = (failures / total_rows) * 100
    print(f"Total Records : {total_rows}")
    print(f"Total Failures: {failures}")
    print(f"Failure Rate  : {failure_rate:.2f}%")
    
    print("\n--- 3. Stratified Train/Test Split (80/20) ---")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_fail_rate = (y_train.sum() / len(y_train)) * 100
    test_fail_rate = (y_test.sum() / len(y_test)) * 100
    
    print(f"Train set shape: {train_df.shape} | Failure Rate: {train_fail_rate:.2f}%")
    print(f"Test set shape : {test_df.shape} | Failure Rate: {test_fail_rate:.2f}%")
    
    print("\n--- 4. Saving Processed Splits ---")
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    run_exploration_and_split()
