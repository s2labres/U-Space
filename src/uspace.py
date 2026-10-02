"""U-Space basis from lens-transported word directions, and the A_cone readout."""
from __future__ import annotations

import json

import numpy as np

CONE_D = 4


def unit(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def lowdin(M: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(M.T @ M)
    U = M @ (V @ np.diag(np.maximum(w, 1e-12) ** -0.5) @ V.T)
    return U * np.sign(np.einsum("hc,hc->c", U, M))[None, :]


def single_token_id(tok, word: str) -> int | None:
    for s in (" " + word, word):
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


def unembedding_rows(hf: str, revision: str, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(W_U[ids], final-norm gain), read from the safetensors shards without loading the model."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    weights = json.load(open(hf_hub_download(hf, "model.safetensors.index.json",
                                             revision=revision)))["weight_map"]

    def key(*suffixes):
        hits = [k for s in suffixes for k in weights if k.endswith(s)]
        return sorted(hits, key=lambda k: ("language_model" not in k, len(k)))[0]

    w_key = key("lm_head.weight") if any(k.endswith("lm_head.weight") for k in weights) \
        else key("embed_tokens.weight")
    g_key = key("model.norm.weight", ".norm.weight")
    with safe_open(hf_hub_download(hf, weights[w_key], revision=revision), framework="pt") as f:
        sl = f.get_slice(w_key)
        W = np.stack([sl[i:i + 1, :].float().numpy()[0] for i in ids])
    with safe_open(hf_hub_download(hf, weights[g_key], revision=revision), framework="pt") as f:
        gamma = f.get_tensor(g_key).float().numpy()
    return W, gamma


def lens_matrix(lens: dict, block: int) -> np.ndarray:
    if "path" in lens:
        from src.config import ROOT
        return np.load(ROOT / lens["path"]).astype(np.float32)
    import torch
    from huggingface_hub import hf_hub_download
    z = torch.load(hf_hub_download(lens["repo"], lens["file"], revision=lens["revision"]),
                   map_location="cpu", weights_only=False)
    if "J" in z:
        return z["J"][block].float().numpy()
    n = z.get("n_done") or z.get("n_prompts")
    return (z["jacobian_sum"][block].float() / n).numpy()


def basis(model: dict, terms: dict) -> tuple[np.ndarray, list[str], dict]:
    """U [H, 4]: per category, unit(anchor centroid - pole centroid) of the lens-transported
    word directions at the readout block, Loewdin-orthonormalised."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model["hf"], revision=model["revision"])
    ids = {w: single_token_id(tok, w) for c in terms.values() for side in c.values() for w in side}
    ids = {w: i for w, i in ids.items() if i is not None}
    order = sorted(set(ids.values()))
    row = {t: j for j, t in enumerate(order)}
    W, gamma = unembedding_rows(model["hf"], model["revision"], order)
    G = unit((W * gamma) @ lens_matrix(model["lens"], model["readout_block"]))

    def centroid(words):
        return G[[row[ids[w]] for w in words if w in ids]].mean(0)

    cats = list(terms)
    U = lowdin(np.stack([unit(centroid(terms[c]["anchors"]) - centroid(terms[c]["poles"]))
                         for c in cats], 1))
    used = {c: {s: [w for w in terms[c][s] if w in ids] for s in ("anchors", "poles")} for c in cats}
    return U, cats, used


def a_cone(H: np.ndarray, U: np.ndarray) -> np.ndarray:
    """|| softplus(sqrt(d) * zhat) ||, zhat = U^T unit(h) normalised."""
    z = unit(H) @ U
    zhat = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-30)
    return np.linalg.norm(np.log1p(np.exp(np.sqrt(CONE_D) * zhat)), axis=1)
