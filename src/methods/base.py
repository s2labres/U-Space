"""The method interface.

A method is a module in `src/methods/` that defines

    NEEDS: set[str]                     inputs it reads from the Cell
    def score(cell: Cell) -> np.ndarray one value per labelled item, in `cell.ids` order;
                                        higher = more likely wrong

NEEDS names Cell attributes:

    forward   per-token log-probabilities and the end-of-think state      (step 4)
    basis     the U-Space basis U                                         (step 5)
"""
from __future__ import annotations

import json
from functools import cached_property

import numpy as np


class Cell:
    """One (model, benchmark, seed)."""

    def __init__(self, cfg, model: str, ds: str, seed: int):
        self.cfg, self.model_key, self.ds, self.seed = cfg, model, ds, seed

    def missing(self, needs: set[str]) -> list[str]:
        """The inputs in `needs` that have not been produced yet."""
        paths = dict(forward=self.cfg.forward_path(self.model_key, self.ds, self.seed),
                     basis=self.cfg.uspace_path(self.model_key))
        return [n for n in sorted(needs) if not paths[n].exists()]

    @cached_property
    def labels(self) -> list[dict]:
        rows = map(json.loads, open(self.cfg.label_path(self.model_key, self.ds, self.seed)))
        return [r for r in rows if r["correct"] is not None]

    @property
    def ids(self) -> np.ndarray:
        return np.array([r["id"] for r in self.labels])

    @property
    def error(self) -> np.ndarray:
        return np.array([int(not r["correct"]) for r in self.labels])

    @property
    def gen_length(self) -> np.ndarray:
        return np.array([r["tlen"] for r in self.labels], float)

    @cached_property
    def forward(self) -> dict[str, np.ndarray]:
        z = dict(np.load(self.cfg.forward_path(self.model_key, self.ds, self.seed)))
        assert list(z["id"]) == list(self.ids)
        return z

    def per_trace(self, key: str):
        """Slices of a per-token forward array, one per item."""
        z = self.forward
        return (z[key][o:o + n] for o, n in zip(z["off"], z["n"]))

    @cached_property
    def basis(self) -> np.ndarray:
        return np.load(self.cfg.uspace_path(self.model_key))["U"]
