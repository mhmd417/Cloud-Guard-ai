"""Explore class imbalance and feature distributions, then save stratified train/test splits."""

from pathlib import Path
import pandas as pd

from ml.src.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    drop_leakage_columns,
    load_raw_data,
    split_data,
    validate_raw_data,
)

RAW_DATA_PATH = Path("data/ai4i2020.csv")
TRAIN_OUTPUT_PATH = Path("data/train.csv")
TEST_OUTPUT_PATH = Path("data/test.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def print_exploration_report(df: pd.DataFrame) -> None:
    """Print class balance and feature distribution summaries."""
    print("=" * 60)
    print("CLOUDGUARD AI - DATA EXPLORATION REPORT")
    print("=" * 60)
    print(f"Total rows: {len(df)} | Total columns: {df.shape[1]}")

    # 1. Class Imbalance
    print("\n--- 1. Target Class Distribution (Machine failure) ---")
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    rates = df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100
    for label in counts.index:
        status = "Normal (0)" if label == 0 else "Failure (1)"
        print(f"  {status:<14}: {counts[label]:>5} rows ({rates[label]:.2f}%)")

    # 2. Categorical Feature Distribution
    print("\n--- 2. Machine Type Distribution (Type) ---")
    type_counts = df["Type"].value_counts()
    type_rates = df["Type"].value_counts(normalize=True) * 100
    for machine_type in ["L", "M", "H"]:
        count = type_counts.get(machine_type, 0)
        rate = type_rates.get(machine_type, 0.0)
        failure_rate_by_type = df[df["Type"] == machine_type][TARGET_COLUMN].mean() * 100
        print(
            f"  Type {machine_type}: {count:>5} rows ({rate:>5.2f}%) | "
            f"Failure rate: {failure_rate_by_type:.2f}%"
        )

    # 3. Numeric Feature Distributions
    print("\n--- 3. Numeric Sensor Distributions (Overall) ---")
    numeric_cols = [col for col in FEATURE_COLUMNS if col != "Type"]
    stats = df[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    print(stats.to_string(float_format=lambda x: f"{x:.2f}"))
    print("=" * 60)


def main() -> None:
    raw_df = load_raw_data(RAW_DATA_PATH)
    validate_raw_data(raw_df)
    clean_df = drop_leakage_columns(raw_df)

    print_exploration_report(clean_df)

    train_df, test_df = split_data(
        clean_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    TRAIN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)

    print("\n--- 4. Stratified Split Verification ---")
    print(
        f"Train shape: {train_df.shape} | "
        f"Failures: {train_df[TARGET_COLUMN].sum()} "
        f"({train_df[TARGET_COLUMN].mean() * 100:.2f}%)"
    )
    print(
        f"Test shape : {test_df.shape} | "
        f"Failures: {test_df[TARGET_COLUMN].sum()} "
        f"({test_df[TARGET_COLUMN].mean() * 100:.2f}%)"
    )
    print(f"Saved splits to '{TRAIN_OUTPUT_PATH}' and '{TEST_OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()
