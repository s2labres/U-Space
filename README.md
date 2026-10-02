# U-Lens

Code for U-Lens and the single-pass baselines in the uncertainty-estimation results (Tables 1
and 2) on Gemma-4-31B-it, Qwen3.5-27B and Magistral-Small-2507, over MMLU-Pro, Omni-MATH,
SuperGPQA and TriviaQA, seeds 41, 42 and 43.

## Setup

    pip install -r requirements.txt

## Configs

- `configs/reproduce.yaml`: paper settings.
- `configs/smoke.yaml`: 10 TriviaQA items, one seed, Qwen3.5-4B.

## Run

    ./run.sh          # configs/reproduce.yaml
    ./run_smoke.sh    # configs/smoke.yaml

`run.sh <config>` runs every step below in order; finished outputs are skipped.

## Pipeline

Inputs and outputs live in `data/` (see `data/README.md`).

| step | description | hardware |
|---|---|---|
| `01_sample_items.py` | sample 1,000 items per benchmark | CPU |
| `02_generate.py` | one reasoning trace per item and seed | 1 GPU per model |
| `03_label.py` | end-of-think position, exclusion, correctness | CPU |
| `04_forward.py` | per-token log-probabilities, end-of-think residual state | 1 GPU per model |
| `05_build_uspace.py` | U-Space basis from the terms in `data/uspace_terms.yaml` | CPU |
| `06_score.py` | per-item scores of every method in `src/methods/` | CPU |
| `07_tables.py` | Tables 1 and 2 and the per-model raw table | CPU |

```
python pipeline/01_sample_items.py --config configs/reproduce.yaml
python pipeline/02_generate.py     --config configs/reproduce.yaml --model gemma
python pipeline/03_label.py        --config configs/reproduce.yaml --model gemma
python pipeline/04_forward.py      --config configs/reproduce.yaml --model gemma
python pipeline/05_build_uspace.py --config configs/reproduce.yaml --model gemma
python pipeline/06_score.py        --config configs/reproduce.yaml --model gemma
python pipeline/07_tables.py       --config configs/reproduce.yaml
```

Run steps 2 to 6 once per model: `gemma`, `qwen`, `magistral`.

## Methods

Each method is one module in `src/methods/` with `NEEDS` and `score(cell)`; the interface is
documented in `src/methods/base.py`.

| method | module | needs |
|---|---|---|
| U-Lens | `ulens` | forward, basis |
| A_cone | `a_cone` | forward, basis |
| Generation length | `gen_length` | |
| MSP | `msp` | forward |
| Predictive entropy | `predictive_entropy` | forward |
| Max entropy | `max_entropy` | forward |
| Mean NLL | `mean_nll` | forward |
| Self-Certainty | `self_certainty` | forward |
| DeepConf | `deepconf` | forward |
