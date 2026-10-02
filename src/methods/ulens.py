"""U-Lens: A_cone of the end-of-think state x mean token entropy over the thinking block."""
import numpy as np

from src import uspace

NEEDS = {"forward", "basis"}


def score(cell):
    think_entropy = np.array([np.mean(h[:e + 1]) for h, e in
                              zip(cell.per_trace("tok_entropy"), cell.forward["eot_tok"])])
    return uspace.a_cone(cell.forward["eot_state"], cell.basis) * think_entropy
