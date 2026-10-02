"""DeepConf (Fu et al., 2026): mean over tokens of the top-20 log-probability mean."""
import numpy as np

NEEDS = {"forward"}


# https://github.com/facebookresearch/deepconf/blob/675529d172ef6d240aa1f21cb60940d427cc1ea6/deepconf/processors.py#L29-L34
# https://github.com/facebookresearch/deepconf/blob/675529d172ef6d240aa1f21cb60940d427cc1ea6/deepconf/utils.py#L93-L101
def score(cell):
    return np.array([np.mean(np.asarray(t, np.float64).mean(axis=1))
                     for t in cell.per_trace("topk_logprob")])
