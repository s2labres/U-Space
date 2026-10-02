"""Maximum next-token entropy over the generation."""
import numpy as np

NEEDS = {"forward"}


def score(cell):
    return np.array([np.max(h) for h in cell.per_trace("tok_entropy")])
