from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Exact feature definitions matching UCI AI4I 2020 (leakage columns excluded)
CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COL = "Machine failure"
RANDOM_SEED = 42


def load_training_data(train_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load ONLY the training split and verify required columns exist."""
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found at {train_path}. Make sure P2 generated data/train.csv."
        )

    df = pd.read_csv(train_path)
    missing_cols = [c for c in ALL_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in training data: {missing_cols}")

    X_train = df[ALL_FEATURES].copy()
    y_train = df[TARGET_COL].astype(int).copy()
    return X_train, y_train


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing step: One-Hot encode Type, StandardScale numeric features."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def evaluate_oof_predictions(
    y_true: pd.Series, y_probs: np.ndarray, threshold: float = 0.50
) -> dict:
    """Compute imbalance-aware metrics from Out-Of-Fold validation probabilities."""
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_probs),
        "confusion_matrix": cm,
    }


def print_metrics_report(model_name: str, metrics: dict) -> None:
    """Print clean formatted validation report."""
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{'=' * 55}")
    print(f"MODEL: {model_name} (Threshold = {metrics['threshold']:.2f})")
    print(f"{'=' * 55}")
    print(f"PR-AUC    : {metrics['pr_auc']:.4f}  <-- Primary ranking metric")
    print(f"Precision : {metrics['precision']:.4f}  (When it alerts, how often is it right?)")
    print(f"Recall    : {metrics['recall']:.4f}  (What % of real failures did it catch?)")
    print(f"F1-Score  : {metrics['f1']:.4f}  (Harmonic mean of Precision & Recall)")
    print("\nValidation Confusion Matrix (Out-of-Fold):")
    print(f"  True Negatives  (Normal correctly ignored) : {tn}")
    print(f"  False Positives (False alarm inspections)  : {fp}")
    print(f"  False Negatives (MISSED failures!)         : {fn}")
    print(f"  True Positives  (Failures caught!)         : {tp}")


def analyze_thresholds(y_true: pd.Series, y_probs: np.ndarray) -> float:
    """Compare candidate decision thresholds to weigh False Positives vs False Negatives."""
    print(f"\n{'=' * 55}")
    print("THRESHOLD TRADE-OFF ANALYSIS (Validation Folds)")
    print(f"{'=' * 55}")
    print(f"{'Thresh':<8} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'FP (Alarms)':<12} | {'FN (Missed)':<12}")
    print("-" * 65)

    best_f1 = -1.0
    best_thresh = 0.50

    for thresh in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        y_pred = (y_probs >= thresh).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print(f"{thresh:<8.2f} | {prec:<7.3f} | {rec:<7.3f} | {f1:<7.3f} | {fp:<12} | {fn:<12}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # Also compute exact precision-recall curve optimal F1 threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    optimal_idx = int(np.argmax(f1_scores))
    optimal_thresh = float(thresholds[optimal_idx])

    print("-" * 65)
    print(f"Best grid threshold by F1 : {best_thresh:.2f}")
    print(f"Exact PR-curve max-F1 cut : {optimal_thresh:.3f} (F1 = {f1_scores[optimal_idx]:.4f})")
    return optimal_thresh


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    train_path = repo_root / "data" / "train.csv"

    print(f"Loading training split from: {train_path}")
    X_train, y_train = load_training_data(train_path)
    print(f"Training rows: {len(X_train)} | Positive failures: {y_train.sum()} ({y_train.mean():.2%})")

    # 5-fold Stratified CV preserves the ~3.4% failure ratio in every fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # ---------------------------------------------------------
    # MODEL 1: Logistic Regression Baseline
    # ---------------------------------------------------------
    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    lr_grid = GridSearchCV(
        estimator=lr_pipeline,
        param_grid={"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="average_precision",  # Optimize PR-AUC
        cv=cv,
        n_jobs=-1,
    )
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_
    print(f"\n[Logistic Regression] Best params: {lr_grid.best_params_}")

    # Out-of-fold validation probabilities (every sample predicted by a model that didn't train on it)
    lr_oof_probs = cross_val_predict(best_lr, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    lr_metrics = evaluate_oof_predictions(y_train, lr_oof_probs, threshold=0.50)
    print_metrics_report("Logistic Regression (Baseline)", lr_metrics)

    # ---------------------------------------------------------
    # MODEL 2: Random Forest Classifier
    # ---------------------------------------------------------
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    rf_grid = GridSearchCV(
        estimator=rf_pipeline,
        param_grid={
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [8, 12, None],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
        scoring="average_precision",  # Optimize PR-AUC
        cv=cv,
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"\n[Random Forest] Best params: {rf_grid.best_params_}")

    rf_oof_probs = cross_val_predict(best_rf, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    rf_metrics = evaluate_oof_predictions(y_train, rf_oof_probs, threshold=0.50)
    print_metrics_report("Random Forest (Tuned)", rf_metrics)

    # ---------------------------------------------------------
    # THRESHOLD SELECTION ON WINNING MODEL (Random Forest)
    # ---------------------------------------------------------
    recommended_threshold = analyze_thresholds(y_train, rf_oof_probs)
    tuned_rf_metrics = evaluate_oof_predictions(y_train, rf_oof_probs, threshold=recommended_threshold)
    print_metrics_report(f"Random Forest (Chosen Threshold = {recommended_threshold:.3f})", tuned_rf_metrics)


if __name__ == "__main__":
    main()
    