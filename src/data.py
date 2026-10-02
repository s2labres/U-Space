"""Benchmark loaders and stratified sampling."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def load_mmlupro(src: dict, raw_dir: Path) -> list[dict]:
    from datasets import load_dataset
    rows = load_dataset(src["repo"], split="test", revision=src["revision"])
    return [dict(id=str(r["question_id"]), ds="mmlupro", q=r["question"],
                 options=list(r["options"]), gold=r["answer"], category=r["category"],
                 stratum=r["category"]) for r in rows]


def load_supergpqa(src: dict, raw_dir: Path) -> list[dict]:
    from datasets import load_dataset
    rows = load_dataset(src["repo"], split="train", revision=src["revision"])
    return [dict(id=r["uuid"], ds="supergpqa", q=r["question"],
                 options=[str(o) for o in r["options"]], gold=r["answer_letter"],
                 category=str(r["discipline"]), stratum=str(r["discipline"]))
            for r in rows if r["difficulty"] == "hard" and 2 <= len(r["options"]) <= 10]


def _difficulty_band(d: float) -> str:
    return ("d1-2.5" if d <= 2.5 else "d3-4" if d <= 4 else "d4.5-5" if d <= 5
            else "d5.25-6" if d <= 6 else "d6.5-7.5" if d <= 7.5 else "d8+")


def load_omnimath(src: dict, raw_dir: Path) -> list[dict]:
    import requests
    f = raw_dir / f"omni_math_rule_{src['revision'][:8]}.jsonl"
    if not f.exists():
        url = (f"https://raw.githubusercontent.com/{src['repo']}/{src['revision']}"
               "/omni_math_rule.jsonl")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(requests.get(url, timeout=120).text)
    items = []
    for i, line in enumerate(l for l in f.read_text().splitlines() if l.strip()):
        r = json.loads(line)
        d = float(r["difficulty"])
        if d >= src["min_difficulty"]:
            items.append(dict(id=f"omni-{i}", ds="omnimath", q=r["problem"],
                              gold=str(r["answer"]), difficulty=d,
                              stratum=_difficulty_band(d)))
    return items


def load_triviaqa(src: dict, raw_dir: Path) -> list[dict]:
    from datasets import load_dataset
    rows = load_dataset(src["repo"], "rc.nocontext", split="validation",
                        revision=src["revision"])
    items, seen = [], set()
    for r in rows:
        if r["question_id"] in seen:
            continue
        seen.add(r["question_id"])
        items.append(dict(id=r["question_id"], ds="triviaqa", q=r["question"],
                          gold=r["answer"]["value"],
                          aliases=list(r["answer"]["normalized_aliases"]), stratum=None))
    return items


LOADERS = dict(mmlupro=load_mmlupro, omnimath=load_omnimath,
               supergpqa=load_supergpqa, triviaqa=load_triviaqa)


def sample_ids(items: list[dict], n: int, seed: int, ds: str) -> list[str]:
    if len(items) <= n:
        return [it["id"] for it in items]
    rng = random.Random(f"{seed}:{ds}")
    strata = defaultdict(list)
    for it in items:
        strata[it["stratum"]].append(it["id"])
    if len(strata) == 1:
        return sorted(rng.sample(next(iter(strata.values())), n))
    quota = {s: n * len(v) / len(items) for s, v in strata.items()}
    alloc = {s: int(q) for s, q in quota.items()}
    for s in sorted(quota, key=lambda s: quota[s] - alloc[s], reverse=True):
        if sum(alloc.values()) == n:
            break
        alloc[s] += 1
    out = []
    for s in sorted(strata, key=str):
        out += rng.sample(strata[s], min(alloc[s], len(strata[s])))
    return sorted(out)


def ids_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def read_items(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
