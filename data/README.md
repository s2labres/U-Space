# data/

| path | contents |
|---|---|
| `uspace_terms.yaml` | anchor and pole terms per category |
| `lens/magistral_block26.npy` | Jacobian lens of Magistral-Small-2507 at block 26 |

Written by the pipeline:

| path | step | contents |
|---|---|---|
| `raw/` | 01 | Omni-MATH jsonl |
| `items/<ds>.jsonl` | 01 | sampled items |
| `items/<ds>.meta.json` | 01 | pool size, strata, sha256 of the ids |
| `traces/<model>/<ds>_s<seed>.jsonl.gz` | 02 | id, prompt_ids, gen_ids, finish, tlen |
| `traces/<model>/<ds>_s<seed>.meta.json` | 02 | model, budget, sampling, truncation count |
| `labels/<model>/<ds>_s<seed>.jsonl` | 03 | id, eot_tok, pred, correct, excluded |
| `forward/<model>/<ds>_s<seed>.npz` | 04 | per-token logprob, entropy, vocabulary logprob sum, top-20 logprobs; eot_state |
| `uspace/<model>.npz` | 05 | U [hidden, 4], categories |
| `uspace/<model>.json` | 05 | terms used (single-token in the model's vocabulary) |
| `scores/<model>/<ds>_s<seed>/<method>.npz` | 06 | id, error, gen_length, score |
| `tables/*.md` | 07 | Table 1, Table 2 (length-matched), per-model raw table |
