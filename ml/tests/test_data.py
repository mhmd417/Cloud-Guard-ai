"""Unit tests for ml.src.data validation, leakage removal, and splitting."""

import numpy as np
import pandas as pd
import pytest

from ml.src.data import (
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    drop_leakage_columns,
    split_data,
    validate_raw_data,
)


def _make_sample_raw_df(n_rows: int = 20, n_failures: int = 2) -> pd.DataFrame:
    """Create a small, valid 14-column DataFrame matching the UCI AI4I 2020 schema."""
    types = ["L", "M", "H"]
    failures = [1 if i < n_failures else 0 for i in range(n_rows)]
    return pd.DataFrame(
        {
            "UDI": list(range(1, n_rows + 1)),
            "Product ID": [f"{types[i % 3]}{14860 + i}" for i in range(n_rows)],
            "Type": [types[i % 3] for i in range(n_rows)],
            "Air temperature [K]": [298.1 + (i * 0.1) for i in range(n_rows)],
            "Process temperature [K]": [308.6 + (i * 0.1) for i in range(n_rows)],
            "Rotational speed [rpm]": [1500 + (i * 10) for i in range(n_rows)],
            "Torque [Nm]": [40.0 + (i * 0.5) for i in range(n_rows)],
            "Tool wear [min]": [i * 5 for i in range(n_rows)],
            "Machine failure": failures,
            "TWF": [1 if i == 0 and n_failures > 0 else 0 for i in range(n_rows)],
            "HDF": [1 if i == 1 and n_failures > 1 else 0 for i in range(n_rows)],
            "PWF": [0] * n_rows,
            "OSF": [0] * n_rows,
            "RNF": [0] * n_rows,
        }
    )


def test_validate_raw_data_valid() -> None:
    df = _make_sample_raw_df()
    assert list(df.columns) == EXPECTED_COLUMNS
    validate_raw_data(df)


def test_validate_raw_data_empty() -> None:
    empty_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    with pytest.raises(ValueError, match="empty"):
        validate_raw_data(empty_df)


def test_validate_raw_data_missing_column() -> None:
    df = _make_sample_raw_df().drop(columns=["Torque [Nm]"])
    with pytest.raises(ValueError, match="Expected 14 columns"):
        validate_raw_data(df)


def test_validate_raw_data_extra_column() -> None:
    df = _make_sample_raw_df()
    df["Extra Sensor"] = 123.4
    with pytest.raises(ValueError, match="Expected 14 columns"):
        validate_raw_data(df)


def test_validate_raw_data_wrong_column_name() -> None:
    df = _make_sample_raw_df().rename(columns={"Torque [Nm]": "Torque_Nm"})
    with pytest.raises(ValueError, match="Column mismatch"):
        validate_raw_data(df)


def test_validate_raw_data_null_values() -> None:
    df = _make_sample_raw_df()
    df.loc[3, "Air temperature [K]"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_raw_data(df)


def test_validate_raw_data_invalid_type() -> None:
    df = _make_sample_raw_df()
    df.loc[0, "Type"] = "X"
    with pytest.raises(ValueError, match="Invalid 'Type'"):
        validate_raw_data(df)


def test_validate_raw_data_invalid_target() -> None:
    df = _make_sample_raw_df()
    df.loc[0, "Machine failure"] = 2
    with pytest.raises(ValueError, match="Invalid 'Machine failure'"):
        validate_raw_data(df)


def test_drop_leakage_columns() -> None:
    df = _make_sample_raw_df()
    clean_df = drop_leakage_columns(df)

    for col in LEAKAGE_COLUMNS:
        assert col not in clean_df.columns

    assert list(clean_df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert clean_df.shape == (len(df), 7)


def test_split_data_stratification() -> None:
    # 4 failures out of 40 rows (10%) splits cleanly 75/25 into 3/30 (10%) and 1/10 (10%)
    df = drop_leakage_columns(_make_sample_raw_df(n_rows=40, n_failures=4))
    train_df, test_df = split_data(df, test_size=0.25, random_state=42)

    assert train_df.shape == (30, 7)
    assert test_df.shape == (10, 7)
    assert pytest.approx(train_df[TARGET_COLUMN].mean()) == df[TARGET_COLUMN].mean()
    assert pytest.approx(test_df[TARGET_COLUMN].mean()) == df[TARGET_COLUMN].mean()
