"""Mean next-token entropy over the generation (Malinin & Gales, 2021)."""
import numpy as np

NEEDS = {"forward"}


def score(cell):
    return np.array([np.mean(h) for h in cell.per_trace("tok_entropy")])
