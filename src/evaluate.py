import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from src.config import DATA_PROCESSED, MODELS, SCORE_THRESHOLD


def _auto_threshold_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    eps = 1e-12
    p = precision[1:]
    r = recall[1:]
    f1 = 2 * p * r / (p + r + eps)
    f2 = (1 + 2 ** 2) * p * r / (2 ** 2 * p + r + eps)
    if thresholds.size == 0:
        return 0.0, {"P": float(p[-1]) if p.size else 0.0, "R": float(r[-1]) if r.size else 0.0, "F1": 0.0, "F2": 0.0}
    idx = int(np.nanargmax(f1)) if f1.size > 0 else 0
    return float(thresholds[idx]), {"P": float(p[idx]), "R": float(r[idx]), "F1": float(f1[idx]), "F2": float(f2[idx])}


def eval_iforest(X_test: pd.DataFrame, y_test: np.ndarray):
    path = MODELS / "isolation_forest.joblib"
    if not path.exists():
        return None
    clf = load(path)
    scores = clf.decision_function(X_test)
    anom_scores = -scores  # larger = more anomalous

    if SCORE_THRESHOLD is None:
        thr, stats = _auto_threshold_f1(y_test, anom_scores)
        print(
            f"Chosen threshold (IF, auto): {thr:.6f} | P={stats['P']:.4f} R={stats['R']:.4f} F1={stats['F1']:.4f} F2={stats['F2']:.4f}"
        )
    else:
        thr = float(SCORE_THRESHOLD)
        print(f"Using configured threshold for IF: {thr:.6f}")

    y_pred = (anom_scores >= thr).astype(int)

    print("\nIF Evaluation Results\n")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, anom_scores)
    print(f"ROC AUC: {auc:.4f}")

    return {"score": scores, "anom_score": anom_scores, "pred": y_pred}


def eval_supervised(X_test: pd.DataFrame, y_test: np.ndarray):
    path = MODELS / "supervised_sgd.joblib"
    if not path.exists():
        return None
    clf = load(path)
    scores = clf.decision_function(X_test)
    thr, stats = _auto_threshold_f1(y_test, scores)
    print(
        f"Supervised threshold (auto): {thr:.6f} | P={stats['P']:.4f} R={stats['R']:.4f} F1={stats['F1']:.4f}"
    )
    y_pred = (scores >= thr).astype(int)
    print("\nSupervised Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nSupervised Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, scores)
    print(f"Supervised ROC AUC: {auc:.4f}")
    return {"sup_score": scores, "sup_pred": y_pred}


def main():
    # Load processed test data
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].values.ravel()

    # Evaluate models if available
    out = {"label": y_test}
    res_if = eval_iforest(X_test, y_test)
    if res_if is not None:
        out.update(res_if)
    res_sup = eval_supervised(X_test, y_test)
    if res_sup is not None:
        out.update(res_sup)

    # Save combined scores
    out_df = pd.DataFrame(out)
    out_path = DATA_PROCESSED / "test_scored.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved results -> {out_path}")


if __name__ == "__main__":
    main()

