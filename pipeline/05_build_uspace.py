"""Build the U-Space basis for one model.

    python pipeline/05_build_uspace.py --config configs/reproduce.yaml --model gemma
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, uspace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    cfg = config.load(args.config)
    terms = yaml.safe_load((config.ROOT / cfg["uspace_terms"]).read_text())

    U, cats, used = uspace.basis(cfg["models"][args.model], terms)
    out = cfg.uspace_path(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, U=U, categories=np.array(cats))
    out.with_suffix(".json").write_text(json.dumps(used, indent=1))
    print(f"[{args.model}] U {U.shape} -> {out}; terms used: "
          + ", ".join(f"{c} {len(used[c]['anchors'])}/{len(used[c]['poles'])}" for c in cats))


if __name__ == "__main__":
    main()
