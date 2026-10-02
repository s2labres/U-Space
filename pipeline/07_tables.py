"""Tables 1 and 2 (and the per-model raw table) from the per-item scores of step 6.

Per (model, benchmark, seed): AUROC, AUPRC, AURC, raw and length-matched (10 quantile bins of
generation length). Per model: mean over the benchmarks, then mean and sample s.d. over seeds.
Table 1: mean and s.d. across models. Bold: best per column; generation length is not eligible.

    python pipeline/07_tables.py --config configs/reproduce.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.metrics import LOWER_IS_BETTER, METRICS, length_matched

ROWS = [("gen_length", "Generation length"), ("msp", "MSP"), ("max_entropy", "Max entropy"),
        ("mean_nll", "Mean NLL"), ("self_certainty", "Self-Certainty"), ("deepconf", "DeepConf"),
        ("predictive_entropy", "Predictive entropy"), ("a_cone", "A_cone"), ("ulens", "U-Lens")]
NOT_ELIGIBLE = {"gen_length"}


def cell_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    z = np.load(path)
    ok = np.isfinite(z["score"])
    y, s, L = z["error"][ok], z["score"][ok], z["gen_length"][ok]
    res = {(m, "raw"): f(y, s) for m, f in METRICS.items()} | \
          {(m, "lm"): length_matched(f, y, s, L) for m, f in METRICS.items()}
    return {k: np.nan if v is None else v for k, v in res.items()}


def model_cells(cfg, model: str) -> tuple[dict, list]:
    """{(method, metric, scale): (mean, sd)} over seeds of the benchmark macro, in percent."""
    out = {}
    for key, _ in ROWS:
        per_seed = {}
        for seed in cfg["generation"]["seeds"]:
            got = [cell_metrics(cfg.score_path(model, ds, seed, key)) for ds in cfg["datasets"]]
            if all(g is not None for g in got):
                for k in got[0]:
                    per_seed.setdefault(k, []).append(100 * np.mean([g[k] for g in got]))
        for (m, sc), v in per_seed.items():
            if len(v) == len(cfg["generation"]["seeds"]):
                out[(key, m, sc)] = (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0)
    acc = []
    for seed in cfg["generation"]["seeds"]:
        paths = [cfg.score_path(model, ds, seed, "gen_length") for ds in cfg["datasets"]]
        if all(p.exists() for p in paths):
            acc.append(100 * np.mean([1 - np.load(p)["error"].mean() for p in paths]))
    return out, acc


def bold(values: dict, metric: str) -> set:
    pool = {k: round(v, 1) for k, v in values.items() if k not in NOT_ELIGIBLE and np.isfinite(v)}
    if not pool:
        return set()
    best = (min if metric in LOWER_IS_BETTER else max)(pool.values())
    return {k for k, v in pool.items() if v == best}


def fmt(v, winner: bool) -> str:
    if not np.isfinite(v[0]):
        return "–"
    s = f"{v[0]:.1f} ± {v[1]:.1f}"
    return f"**{s}**" if winner else s


def per_model_table(cells: dict, accs: dict, scale: str) -> str:
    models = list(cells)
    head = "| method | " + " | ".join(f"{mm} {m.upper()}" for mm in models
                                        for m in ("auroc", "auprc", "aurc")) + " |"
    lines = [" | ".join(f"{mm}: accuracy {np.mean(a):.1f} ± {np.std(a, ddof=1) if len(a) > 1 else 0:.1f}"
                        for mm, a in accs.items() if a), "", head,
             "|---|" + "---|" * (3 * len(models))]
    winners = {(mm, m): bold({k: v[0] for (k, mt, sc), v in cells[mm].items()
                              if mt == m and sc == scale}, m)
               for mm in models for m in METRICS}
    for key, name in ROWS:
        row = [fmt(cells[mm][(key, m, scale)], key in winners[(mm, m)])
               if (key, m, scale) in cells[mm] else "–" for mm in models for m in METRICS]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def cross_model_table(cells: dict) -> str:
    avg = {}
    for key, _ in ROWS:
        for m in METRICS:
            for sc in ("raw", "lm"):
                v = [cells[mm][(key, m, sc)][0] for mm in cells if (key, m, sc) in cells[mm]]
                if len(v) == len(cells):
                    avg[(key, m, sc)] = (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0)
    winners = {m: bold({k: v[0] for (k, mt, sc), v in avg.items()
                        if mt == m and sc == "raw" and k != "a_cone"}, m) for m in METRICS}
    lines = ["| method | AUROC | AUPRC | AURC | AUROC length-matched | Δ |", "|---|---|---|---|---|---|"]
    for key, name in ROWS:
        if key == "a_cone" or (key, "auroc", "raw") not in avg:
            continue
        raw, lm = avg[(key, "auroc", "raw")][0], avg[(key, "auroc", "lm")][0]
        cols = [fmt(avg[(key, m, "raw")], key in winners[m]) for m in METRICS]
        tail = f" | {lm:.1f} | {lm - raw:+.1f} |" if np.isfinite(lm) else " | – | – |"
        lines.append(f"| {name} | " + " | ".join(cols) + tail)
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = config.load(args.config)
    cells, accs = {}, {}
    for mm in cfg["models"]:
        cells[mm], accs[mm] = model_cells(cfg, mm)
    out = cfg.data / "tables"
    out.mkdir(parents=True, exist_ok=True)
    tables = {"table1.md": cross_model_table(cells),
              "table2_length_matched.md": per_model_table(cells, accs, "lm"),
              "table_raw.md": per_model_table(cells, accs, "raw")}
    for name, text in tables.items():
        (out / name).write_text(text)
        print(f"== {name}\n{text}")


if __name__ == "__main__":
    main()
