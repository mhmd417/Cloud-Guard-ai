from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from train import (
    ALL_FEATURES,
    RANDOM_SEED,
    TARGET_COL,
    build_preprocessor,
    load_training_data,
)

# Model identity & hyperparameters selected from validation in train.py
MODEL_VERSION = "1.0.0"
MODEL_FILENAME = f"cloudguard_rf_v{MODEL_VERSION}.joblib"
METADATA_FILENAME = f"cloudguard_rf_v{MODEL_VERSION}_metadata.json"

BEST_RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 1,
}
CHOSEN_THRESHOLD = 0.43


def load_test_data(test_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the held-out test split and verify required columns exist."""
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test file not found at {test_path}. Ensure data/test.csv exists."
        )

    df = pd.read_csv(test_path)
    missing_cols = [c for c in ALL_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in test data: {missing_cols}")

    X_test = df[ALL_FEATURES].copy()
    y_test = df[TARGET_COL].astype(int).copy()
    return X_test, y_test


def build_final_pipeline() -> Pipeline:
    """Build the single end-to-end pipeline (preprocessing + chosen classifier)."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=BEST_RF_PARAMS["n_estimators"],
                    max_depth=BEST_RF_PARAMS["max_depth"],
                    min_samples_leaf=BEST_RF_PARAMS["min_samples_leaf"],
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def measure_single_row_latency(
    pipeline: Pipeline, X_test: pd.DataFrame, num_samples: int = 200
) -> dict[str, float]:
    """Measure per-prediction latency (ms) simulating single API requests."""
    sample_count = min(num_samples, len(X_test))
    latencies_ms: list[float] = []

    # Warm-up call so first-call overhead does not skew timings
    _ = pipeline.predict_proba(X_test.iloc[[0]])

    for i in range(sample_count):
        single_row = X_test.iloc[[i]]
        start = time.perf_counter()
        _ = pipeline.predict_proba(single_row)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

    return {
        "samples_timed": sample_count,
        "mean_ms_per_prediction": round(float(np.mean(latencies_ms)), 4),
        "p95_ms_per_prediction": round(float(np.percentile(latencies_ms, 95)), 4),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    train_path = repo_root / "data" / "train.csv"
    test_path = repo_root / "data" / "test.csv"
    models_dir = repo_root / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load train and test splits
    X_train, y_train = load_training_data(train_path)
    X_test, y_test = load_test_data(test_path)
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    # 2. Fit final pipeline on full training set
    print("Training final pipeline on full training split...")
    final_pipeline = build_final_pipeline()
    final_pipeline.fit(X_train, y_train)

    # 3. Evaluate ONCE on the held-out test set
    print(f"Evaluating ONCE on test split (threshold = {CHOSEN_THRESHOLD:.3f})...")
    y_test_probs = final_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_probs >= CHOSEN_THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    test_metrics = {
        "pr_auc": round(float(average_precision_score(y_test, y_test_probs)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_test_probs)), 4),
        "precision": round(float(precision_score(y_test, y_test_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_test_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_test_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    # 4. Measure inference latency per single prediction
    latency_metrics = measure_single_row_latency(final_pipeline, X_test)

    print("\n" + "=" * 55)
    print(f"FINAL TEST EVALUATION (Model v{MODEL_VERSION})")
    print("=" * 55)
    print(f"PR-AUC    : {test_metrics['pr_auc']:.4f}")
    print(f"ROC-AUC   : {test_metrics['roc_auc']:.4f}")
    print(f"Precision : {test_metrics['precision']:.4f}")
    print(f"Recall    : {test_metrics['recall']:.4f}")
    print(f"F1-Score  : {test_metrics['f1_score']:.4f}")
    print(f"Confusion : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(
        f"Latency   : {latency_metrics['mean_ms_per_prediction']:.2f} ms mean "
        f"(p95: {latency_metrics['p95_ms_per_prediction']:.2f} ms)"
    )

    # 5. Save full pipeline (preprocessing + model) as one joblib file
    model_path = models_dir / MODEL_FILENAME
    joblib.dump(final_pipeline, model_path)
    print(f"\nSaved pipeline artifact to: {model_path}")

    # 6. Save metadata JSON alongside the model artifact
    metadata = {
        "model_name": "CloudGuard AI Predictive Maintenance Classifier",
        "model_version": MODEL_VERSION,
        "artifact_filename": MODEL_FILENAME,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "UCI AI4I 2020 Predictive Maintenance Dataset",
            "is_synthetic": True,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_positive_rate": round(float(y_test.mean()), 4),
        },
        "algorithm": "RandomForestClassifier",
        "hyperparameters": BEST_RF_PARAMS,
        "random_seed": RANDOM_SEED,
        "features": ALL_FEATURES,
        "target": TARGET_COL,
        "excluded_leakage_columns": ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"],
        "decision_threshold": CHOSEN_THRESHOLD,
        "test_metrics": test_metrics,
        "inference_latency": latency_metrics,
        "library_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    metadata_path = models_dir / METADATA_FILENAME
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata JSON to   : {metadata_path}")


if __name__ == "__main__":
    main()