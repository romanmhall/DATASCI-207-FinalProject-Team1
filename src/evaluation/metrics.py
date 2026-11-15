from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the probability decision threshold that maximizes the F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    # The last element of precision/recall corresponds to threshold=1 (empty)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    credit_amounts: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive classification and economic cost metrics."""
    flagged = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        'ROC-AUC': float(roc_auc_score(y_true, y_prob)),
        'PR-AUC': float(average_precision_score(y_true, y_prob)),
        'Brier': float(brier_score_loss(y_true, y_prob)),
        'Threshold': float(threshold),
        'F1': float(f1_score(y_true, flagged, zero_division=0)),
        'Precision': float(precision_score(y_true, flagged, zero_division=0)),
        'Recall': float(recall_score(y_true, flagged, zero_division=0)),
        'Specificity': float(specificity),
        'Pct_Flagged': float(flagged.mean()),
        'Defaults_Caught': float(flagged[y_true == 1].mean()) if (y_true == 1).sum() > 0 else 0.0,
    }

    if credit_amounts is not None:
        metrics['Predicted_Loss_M'] = float((y_prob * credit_amounts).sum() / 1e6)
        metrics['Actual_Loss_M'] = float((y_true * credit_amounts).sum() / 1e6)

    return metrics


def evaluate_threshold_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Generate precision, recall, and specificity trade-off table across thresholds."""
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.50]

    rows = []
    for tau in thresholds:
        m = evaluate_predictions(y_true, y_prob, threshold=tau)
        rows.append({
            'Threshold': tau,
            'Precision': m['Precision'],
            'Recall (Sensitivity)': m['Recall'],
            'Specificity': m['Specificity'],
            'F1 Score': m['F1'],
            'Flagged Rate': m['Pct_Flagged'],
            'Defaults Caught': m['Defaults_Caught'],
        })
    return pd.DataFrame(rows)


def bootstrap_subgroup_test(
    y_a: np.ndarray,
    p_trad_a: np.ndarray,
    p_hyb_a: np.ndarray,
    y_b: np.ndarray,
    p_trad_b: np.ndarray,
    p_hyb_b: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 1234,
) -> Dict[str, Any]:
    """Test Hypothesis 2: Does alternative data provide greater lift for thin-file applicants?

    Group A: Bureau-backed applicants.
    Group B: Thin-file applicants lacking bureau history.
    Lift = AUC(Hybrid) - AUC(Traditional).
    """
    rng = np.random.default_rng(seed)
    n_a = len(y_a)
    n_b = len(y_b)

    deltas_a = []
    deltas_b = []
    diffs = []

    for _ in range(n_bootstraps):
        idx_a = rng.integers(0, n_a, n_a)
        idx_b = rng.integers(0, n_b, n_b)

        # Check for single-class sample edge cases
        if len(np.unique(y_a[idx_a])) < 2 or len(np.unique(y_b[idx_b])) < 2:
            continue

        auc_trad_a = roc_auc_score(y_a[idx_a], p_trad_a[idx_a])
        auc_hyb_a = roc_auc_score(y_a[idx_a], p_hyb_a[idx_a])
        delta_a = auc_hyb_a - auc_trad_a

        auc_trad_b = roc_auc_score(y_b[idx_b], p_trad_b[idx_b])
        auc_hyb_b = roc_auc_score(y_b[idx_b], p_hyb_b[idx_b])
        delta_b = auc_hyb_b - auc_trad_b

        deltas_a.append(delta_a)
        deltas_b.append(delta_b)
        diffs.append(delta_b - delta_a)

    deltas_a = np.array(deltas_a)
    deltas_b = np.array(deltas_b)
    diffs = np.array(diffs)

    return {
        'mean_lift_group_a': float(np.mean(deltas_a)),
        'ci95_lift_group_a': (float(np.percentile(deltas_a, 2.5)), float(np.percentile(deltas_a, 97.5))),
        'mean_lift_group_b': float(np.mean(deltas_b)),
        'ci95_lift_group_b': (float(np.percentile(deltas_b, 2.5)), float(np.percentile(deltas_b, 97.5))),
        'mean_diff_lift (B - A)': float(np.mean(diffs)),
        'ci95_diff_lift': (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
        'p_value_one_sided': float(np.mean(diffs <= 0)),
    }
