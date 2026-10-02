"""Generate one reasoning trace per item and seed.

    python pipeline/02_generate.py --config configs/reproduce.yaml --model gemma
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import chat, config, data, prompts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=50)
    args = ap.parse_args()
    cfg = config.load(args.config)
    m = cfg["models"][args.model]

    datasets = args.datasets or list(cfg["datasets"])
    seeds = args.seeds or cfg["generation"]["seeds"]
    if all(cfg.trace_path(args.model, ds, s).exists() for ds in datasets for s in seeds):
        print(f"[{args.model}] all traces exist, skipped")
        return

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = chat.load_tokenizer(m["hf"], m["revision"])
    llm = LLM(model=m["hf"], revision=m["revision"], tensor_parallel_size=args.tensor_parallel,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=m["budget"] + cfg["generation"]["prompt_headroom"])

    for ds in datasets:
        items = data.read_items(cfg.items_path(ds))
        reqs = [TokensPrompt(prompt_token_ids=chat.prompt_ids(
            tok, m, prompts.build(it))) for it in items]
        for seed in seeds:
            out = cfg.trace_path(args.model, ds, seed)
            if out.exists():
                print(f"[{args.model}/{ds}/s{seed}] exists, skipped")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            part = out.with_name(out.name + ".part")
            done = set()
            if part.exists():
                with gzip.open(part, "rt") as f:
                    done = {json.loads(l)["id"] for l in f}
            todo = [i for i, it in enumerate(items) if it["id"] not in done]
            print(f"[{args.model}/{ds}/s{seed}] {len(todo)} to generate "
                  f"({len(done)} already on disk), budget {m['budget']}")
            sp = SamplingParams(max_tokens=m["budget"], seed=seed, **m["sampling"])
            t0 = time.time()
            for k in range(0, len(todo), args.chunk):
                batch = todo[k:k + args.chunk]
                outs = llm.generate([reqs[i] for i in batch], sp)
                with gzip.open(part, "at") as f:
                    for i, o in zip(batch, outs):
                        g = o.outputs[0]
                        f.write(json.dumps(dict(id=items[i]["id"],
                                                prompt_ids=list(o.prompt_token_ids),
                                                gen_ids=list(g.token_ids),
                                                finish=g.finish_reason,
                                                tlen=len(g.token_ids))) + "\n")
                print(f"  {k + len(batch)}/{len(todo)} ({(time.time() - t0) / 60:.0f} min)")

            with gzip.open(part, "rt") as f:
                rows = [json.loads(l) for l in f]
            part.rename(out)
            n = len(rows)
            out.with_name(out.name.replace(".jsonl.gz", ".meta.json")).write_text(json.dumps(
                dict(model=args.model, hf=m["hf"], revision=m["revision"], dataset=ds,
                     seed=seed, budget=m["budget"], sampling=m["sampling"],
                     chat_kwargs=m["chat_kwargs"], n=n, vllm=vllm.__version__,
                     truncated=sum(r["finish"] == "length" for r in rows),
                     mean_tlen=round(sum(r["tlen"] for r in rows) / max(n, 1))), indent=1))
            print(f"[{args.model}/{ds}/s{seed}] {n} traces -> {out}")


if __name__ == "__main__":
    main()
