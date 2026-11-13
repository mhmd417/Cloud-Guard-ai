"""Data loading, validation, leakage removal, and stratified splitting for CloudGuard AI."""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

EXPECTED_COLUMNS: list[str] = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

LEAKAGE_COLUMNS: list[str] = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

FEATURE_COLUMNS: list[str] = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET_COLUMN: str = "Machine failure"
VALID_TYPES: set[str] = {"L", "M", "H"}


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw UCI AI4I 2020 CSV file into a DataFrame."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    return pd.read_csv(path)


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate schema (14 columns), missing values, and basic value constraints."""
    if df.empty:
        raise ValueError("Dataset is empty.")

    actual_columns = list(df.columns)
    if len(actual_columns) != len(EXPECTED_COLUMNS):
        raise ValueError(
            f"Expected {len(EXPECTED_COLUMNS)} columns, but found {len(actual_columns)}."
        )

    if set(actual_columns) != set(EXPECTED_COLUMNS):
        missing = sorted(set(EXPECTED_COLUMNS) - set(actual_columns))
        extra = sorted(set(actual_columns) - set(EXPECTED_COLUMNS))
        raise ValueError(
            f"Column mismatch. Missing columns: {missing}; Unexpected columns: {extra}"
        )

    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise ValueError(
            f"Dataset contains missing values: {cols_with_nulls.to_dict()}"
        )

    invalid_types = set(df["Type"].unique()) - VALID_TYPES
    if invalid_types:
        raise ValueError(
            f"Invalid 'Type' values found: {invalid_types}. Expected subset of {VALID_TYPES}."
        )

    invalid_targets = set(df[TARGET_COLUMN].unique()) - {0, 1}
    if invalid_targets:
        raise ValueError(
            f"Invalid '{TARGET_COLUMN}' values found: {invalid_targets}. Expected only 0 or 1."
        )


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the DataFrame with identifier and failure-mode leakage columns removed."""
    clean_df = df.drop(columns=LEAKAGE_COLUMNS).copy()
    expected_remaining = FEATURE_COLUMNS + [TARGET_COLUMN]
    return clean_df[expected_remaining]


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a stratified train/test split on the target column."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
