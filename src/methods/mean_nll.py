"""Mean negative log-likelihood of the generation."""
import numpy as np

NEEDS = {"forward"}


def score(cell):
    return np.array([-np.mean(lp) for lp in cell.per_trace("tok_logprob")])
