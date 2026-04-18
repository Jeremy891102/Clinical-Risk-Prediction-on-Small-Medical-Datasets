from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute binary classification metrics. y_proba shape: (n_samples, 2)."""
    proba_pos = y_proba[:, 1]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, proba_pos),
        "brier_score": brier_score_loss(y_true, proba_pos),
    }
