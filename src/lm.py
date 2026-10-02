"""Loading a causal LM and reading per-token statistics from a teacher-forced pass."""
from __future__ import annotations

import re

import numpy as np
import torch

TOPK = 20
CHUNK = 256


def load(model: dict):
    from transformers import AutoModelForCausalLM
    from src.chat import load_tokenizer
    lm = AutoModelForCausalLM.from_pretrained(model["hf"], revision=model["revision"],
                                              dtype=torch.bfloat16, device_map="auto").eval()
    return lm, load_tokenizer(model["hf"], model["revision"])


def text_config(lm):
    return getattr(lm.config, "text_config", lm.config)


def decoder(lm):
    return getattr(lm, "model", lm)


def decoder_blocks(lm) -> list:
    n = text_config(lm).num_hidden_layers
    groups = {}
    for name, mod in lm.named_modules():
        m = re.match(r"(.*)\.layers\.(\d+)$", name)
        if m:
            groups.setdefault(m.group(1), {})[int(m.group(2))] = mod
    blocks = next(g for g in groups.values() if len(g) == n)
    return [blocks[i] for i in range(n)]


def log_softmax(lm, h: torch.Tensor) -> torch.Tensor:
    z = lm.get_output_embeddings()(h.to(lm.dtype)).float()
    cap = getattr(text_config(lm), "final_logit_softcapping", None)
    if cap:
        z = torch.tanh(z / cap) * cap
    return torch.log_softmax(z, dim=-1)


@torch.no_grad()
def token_stats(lm, h: torch.Tensor, targets: torch.Tensor) -> dict[str, np.ndarray]:
    """Per position: log p(target), entropy, sum of log p over the vocabulary, top-k log p."""
    out = {k: [] for k in ("logprob", "entropy", "logprob_sum", "topk")}
    targets = targets.to(h.device)
    for i in range(0, h.shape[0], CHUNK):
        g = log_softmax(lm, h[i:i + CHUNK])
        out["logprob"].append(g.gather(1, targets[i:i + CHUNK, None]).squeeze(1))
        out["entropy"].append(-(g.exp() * g).sum(-1))
        out["logprob_sum"].append(g.sum(-1))
        out["topk"].append(g.topk(TOPK, dim=-1).values)
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


@torch.no_grad()
def hidden(lm, ids: list[int]) -> torch.Tensor:
    x = torch.tensor([ids], device=lm.device)
    return decoder(lm)(input_ids=x, use_cache=False).last_hidden_state[0]
