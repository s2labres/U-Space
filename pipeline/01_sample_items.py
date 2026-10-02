"""Sample the evaluation items.

    python pipeline/01_sample_items.py --config configs/reproduce.yaml
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--datasets", nargs="*")
    args = ap.parse_args()
    cfg = config.load(args.config)

    for ds in args.datasets or list(cfg["datasets"]):
        src = cfg["datasets"][ds]
        pool = data.LOADERS[ds](src, cfg.data / "raw")
        ids = data.sample_ids(pool, cfg["items"]["n"], cfg["items"]["seed"], ds)
        digest = data.ids_digest(ids)
        if src.get("ids_sha256") and src["ids_sha256"] != digest:
            raise SystemExit(f"[{ds}] the draw does not match the config: sha256 {digest}, "
                             f"expected {src['ids_sha256']}")
        by_id = {it["id"]: it for it in pool}
        out = cfg.items_path(ds)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(by_id[i]) + "\n" for i in ids))
        strata = collections.Counter(str(by_id[i]["stratum"]) for i in ids)
        out.with_suffix(".meta.json").write_text(json.dumps(
            dict(dataset=ds, n=len(ids), pool=len(pool), sha256=digest, **src,
                 strata=dict(sorted(strata.items()))), indent=1))
        print(f"[{ds}] {len(ids)} of {len(pool)} items -> {out}  (sha256 {digest[:12]})")


if __name__ == "__main__":
    main()
