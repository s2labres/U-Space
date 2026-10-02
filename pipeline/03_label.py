"""Label every trace: end-of-think position, exclusion, correctness.

    python pipeline/03_label.py --config configs/reproduce.yaml --model gemma
"""
import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import chat, config, data, grading


def label(tr: dict, it: dict, tok, eot_id: int) -> dict:
    rec = dict(id=tr["id"], tlen=tr["tlen"], eot_tok=None, pred=None, correct=None, excluded=None)
    if tr["finish"] == "length":
        rec["excluded"] = "truncated"
        return rec
    at = [i for i, t in enumerate(tr["gen_ids"]) if t == eot_id]
    if len(at) != 1:
        rec["excluded"] = "no_marker" if not at else "multi_marker"
        return rec
    rec["eot_tok"] = at[0]
    answer = tok.decode(tr["gen_ids"][at[0] + 1:], skip_special_tokens=True).strip()
    rec["pred"], rec["correct"] = grading.grade(it, answer)
    if rec["correct"] is None:
        rec["excluded"] = "no_answer"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    args = ap.parse_args()
    cfg = config.load(args.config)
    m = cfg["models"][args.model]
    tok = chat.load_tokenizer(m["hf"], m["revision"])
    eot_id = tok.convert_tokens_to_ids(m["think_end"])

    for ds in args.datasets or list(cfg["datasets"]):
        items = {it["id"]: it for it in data.read_items(cfg.items_path(ds))}
        for seed in args.seeds or cfg["generation"]["seeds"]:
            with gzip.open(cfg.trace_path(args.model, ds, seed), "rt") as f:
                traces = [json.loads(l) for l in f]
            rows = [label(tr, items[tr["id"]], tok, eot_id) for tr in traces]
            out = cfg.label_path(args.model, ds, seed)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps(r) + "\n" for r in rows))
            kept = [r for r in rows if r["correct"] is not None]
            excl = collections.Counter(r["excluded"] for r in rows if r["excluded"])
            print(f"[{args.model}/{ds}/s{seed}] {len(kept)} labelled, "
                  f"accuracy {sum(r['correct'] for r in kept) / max(len(kept), 1):.3f}, "
                  f"excluded {dict(excl)}")


if __name__ == "__main__":
    main()
