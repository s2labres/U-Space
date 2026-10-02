"""AUROC, AUPRC and AURC with errors as the positive class (y = 1 wrong, higher score =
more likely wrong), and their length-matched versions."""
from __future__ import annotations

import numpy as np


def auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y); s = np.asarray(score, dtype=np.float64)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    n1, n0 = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y); s = np.asarray(score, dtype=np.float64)
    if not (y == 1).any():
        return None
    order = np.argsort(-s, kind="mergesort")
    hits = y[order] == 1
    prec = np.cumsum(hits) / np.arange(1, len(y) + 1)
    return float(prec[hits].sum() / hits.sum())


def aurc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y); s = np.asarray(score, dtype=np.float64)
    if len(y) == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    err = np.asarray(y, dtype=int)[order]
    return float((np.cumsum(err) / np.arange(1, len(y) + 1)).mean())


METRICS = {"auroc": auroc, "auprc": auprc, "aurc": aurc}
LOWER_IS_BETTER = {"aurc"}


def length_matched(metric, y: np.ndarray, score: np.ndarray, length: np.ndarray,
                   n_bins: int = 10, min_bin: int = 20) -> float | None:
    """`metric` within quantile bins of length, averaged weighted by bin size."""
    y = np.asarray(y); s = np.asarray(score, dtype=np.float64); L = np.asarray(length, float)
    q = np.quantile(L, np.linspace(0, 1, n_bins + 1))
    vals, w = [], []
    for i in range(n_bins):
        m = (L >= q[i]) & ((L <= q[i + 1]) if i == n_bins - 1 else (L < q[i + 1]))
        if m.sum() < min_bin:
            continue
        v = metric(y[m], s[m])
        if v is not None:
            vals.append(v); w.append(int(m.sum()))
    return float(np.average(vals, weights=w)) if vals else None
