"""Teacher-forced pass over every labelled trace: per-token log-probabilities and the
end-of-think residual state at the readout block.

    python pipeline/04_forward.py --config configs/reproduce.yaml --model gemma
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, lm as LM


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
    datasets = args.datasets or list(cfg["datasets"])
    seeds = args.seeds or cfg["generation"]["seeds"]
    if all(cfg.forward_path(args.model, ds, s).exists() for ds in datasets for s in seeds):
        print(f"[{args.model}] all forward passes exist, skipped")
        return
    lm, _ = LM.load(m)
    block = LM.decoder_blocks(lm)[m["readout_block"]]

    for ds in datasets:
        for seed in seeds:
            out = cfg.forward_path(args.model, ds, seed)
            if out.exists():
                print(f"[{args.model}/{ds}/s{seed}] exists, skipped")
                continue
            labels = {r["id"]: r for r in map(json.loads, open(cfg.label_path(args.model, ds, seed)))}
            with gzip.open(cfg.trace_path(args.model, ds, seed), "rt") as f:
                traces = [t for t in map(json.loads, f) if labels[t["id"]]["correct"] is not None]
            stats, states = [], []
            for k, tr in enumerate(traces):
                p = len(tr["prompt_ids"])
                eot = p + labels[tr["id"]]["eot_tok"]
                grab = {}
                hook = block.register_forward_hook(
                    lambda _m, _i, o: grab.update(h=(o[0] if isinstance(o, tuple) else o)[0, eot]))
                h = LM.hidden(lm, tr["prompt_ids"] + tr["gen_ids"])
                hook.remove()
                stats.append(LM.token_stats(lm, h[p - 1:-1], torch.tensor(tr["gen_ids"])))
                states.append(grab["h"].float().cpu().numpy())
                if (k + 1) % 100 == 0:
                    print(f"  {k + 1}/{len(traces)}", flush=True)

            n = np.array([len(t["gen_ids"]) for t in traces])
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out,
                     id=np.array([t["id"] for t in traces]),
                     error=np.array([int(not labels[t["id"]]["correct"]) for t in traces]),
                     eot_tok=np.array([labels[t["id"]]["eot_tok"] for t in traces]),
                     n=n, off=np.concatenate([[0], np.cumsum(n)[:-1]]),
                     tok_logprob=np.concatenate([s["logprob"] for s in stats]).astype(np.float32),
                     tok_entropy=np.concatenate([s["entropy"] for s in stats]).astype(np.float32),
                     tok_logprob_sum=np.concatenate([s["logprob_sum"] for s in stats]).astype(np.float32),
                     topk_logprob=np.concatenate([s["topk"] for s in stats]).astype(np.float16),
                     eot_state=np.stack(states), readout_block=m["readout_block"],
                     vocab_size=lm.get_output_embeddings().weight.shape[0])
            print(f"[{args.model}/{ds}/s{seed}] {len(traces)} traces -> {out}")


if __name__ == "__main__":
    main()
