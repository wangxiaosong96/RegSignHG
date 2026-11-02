import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def compute_metrics(labels, positive_probabilities, threshold=0.5):
    y, p = np.asarray(labels, dtype=int), np.asarray(positive_probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all() or len(np.unique(y)) != 2:
        raise ValueError("Metrics require finite probabilities and both signs")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities outside [0,1]")
    return {"AUC": float(roc_auc_score(y, p)), "AUPR": float(average_precision_score(y, p)),
            "Binary-F1": float(f1_score(y, p >= threshold, pos_label=1, zero_division=0))}
