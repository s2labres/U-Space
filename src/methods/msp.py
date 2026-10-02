"""Maximum softmax probability of the first generated token (Hendrycks & Gimpel, 2017)."""
import numpy as np

NEEDS = {"forward"}


def score(cell):
    return np.array([-np.exp(lp[0]) for lp in cell.per_trace("tok_logprob")])
