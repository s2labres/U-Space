"""Score every labelled item with the given methods (modules in src/methods/).

    python pipeline/06_score.py --config configs/reproduce.yaml --model gemma --methods ulens a_cone msp
"""
import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.methods.base import Cell

METHODS = ["gen_length", "msp", "predictive_entropy", "max_entropy", "mean_nll", "self_certainty",
           "deepconf", "a_cone", "ulens"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    args = ap.parse_args()
    cfg = config.load(args.config)

    for ds in args.datasets or list(cfg["datasets"]):
        for seed in args.seeds or cfg["generation"]["seeds"]:
            cell = Cell(cfg, args.model, ds, seed)
            for name in args.methods:
                out = cfg.score_path(args.model, ds, seed, name)
                if out.exists():
                    continue
                method = importlib.import_module(f"src.methods.{name}")
                if missing := cell.missing(method.NEEDS):
                    print(f"[{args.model}/{ds}/s{seed}] {name}: skipped, missing {missing}")
                    continue
                s = np.asarray(method.score(cell), float)
                assert s.shape == cell.ids.shape
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez(out, id=cell.ids, error=cell.error, gen_length=cell.gen_length, score=s)
                print(f"[{args.model}/{ds}/s{seed}] {name}: {np.isfinite(s).sum()} items scored")


if __name__ == "__main__":
    main()
