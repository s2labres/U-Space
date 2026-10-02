"""Run configuration and output paths."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    @property
    def data(self) -> Path:
        return ROOT / self["data_dir"]

    def items_path(self, ds: str) -> Path:
        return self.data / "items" / f"{ds}.jsonl"

    def trace_path(self, model: str, ds: str, seed: int) -> Path:
        return self.data / "traces" / model / f"{ds}_s{seed}.jsonl.gz"

    def label_path(self, model: str, ds: str, seed: int) -> Path:
        return self.data / "labels" / model / f"{ds}_s{seed}.jsonl"

    def forward_path(self, model: str, ds: str, seed: int) -> Path:
        return self.data / "forward" / model / f"{ds}_s{seed}.npz"

    def uspace_path(self, model: str) -> Path:
        return self.data / "uspace" / f"{model}.npz"

    def score_path(self, model: str, ds: str, seed: int, method: str) -> Path:
        return self.data / "scores" / model / f"{ds}_s{seed}" / f"{method}.npz"


def load(path: str | Path) -> Config:
    return Config(yaml.safe_load(Path(path).read_text()))
