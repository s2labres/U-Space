"""Self-Certainty (Kang et al., 2025): mean KL(uniform || p_t), negated."""
import numpy as np

NEEDS = {"forward"}


# https://github.com/backprop07/Self-Certainty/blob/b17d021eac56609741671b7a730e5aa8b3cd9546/src/confidence_list.py#L12-L25
def score(cell):
    V = int(cell.forward["vocab_size"])
    return np.array([-np.mean(-s / V - np.log(V)) for s in cell.per_trace("tok_logprob_sum")])
