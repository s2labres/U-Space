# U-Space

Official code for **[U-Space: Uncovering When and Why Uncertainty Arises in Language Models](https://arxiv.org/abs/2610.09087)**.

U-Space tracks ambiguity, incomplete information, conflicting evidence, and general uncertainty through language-model generation. U-Lens combines this verbalizable signal with predictive entropy into a label-free, single-pass uncertainty score.

**[Project page](https://s2labres.github.io/U-Space/) · [Paper](https://arxiv.org/abs/2610.09087) · [PDF](https://arxiv.org/pdf/2610.09087) · [Reproduction guide](#setup)**

This repository reproduces U-Lens and the single-pass baselines in Tables 1 and 2 on Gemma-4-31B-it, Qwen3.5-27B, and Magistral-Small-2507 across MMLU-Pro, Omni-MATH, SuperGPQA, and TriviaQA with seeds 41, 42, and 43.

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

## License

The code is released under the [BSD 3-Clause License](LICENSE).

Third-party material keeps its own terms:

- `data/lens/magistral_block26.npy` is derived from [Magistral-Small-2507](https://huggingface.co/mistralai/Magistral-Small-2507) (Apache 2.0) and was fitted with Anthropic's [jacobian-lens](https://github.com/anthropics/jacobian-lens) reference implementation (Apache 2.0).
- The Gemma and Qwen lenses are downloaded at run time from [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens) and are not redistributed here.
- The anchor terms in `data/uspace_terms.yaml` are drawn from the uncertainty cues of Chen et al. (2018, *Journal of Informetrics*) and the certainty norm of Rocklage et al. (2023, *Journal of Marketing Research*).
- Models and benchmark datasets are downloaded at run time and are subject to their own licenses.

## Citation

```bibtex
@article{braun2026uspace,
  title   = {U-Space: Uncovering When and Why Uncertainty Arises in Language Models},
  author  = {Braun, Tobias and Loose, Nils and Herzog, Alexander and Ceccatelli, Virginia and Rohrbach, Marcus and Eisenbarth, Thomas and Cavallaro, Lorenzo},
  journal = {arXiv preprint arXiv:2610.09087},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.09087}
}
```
