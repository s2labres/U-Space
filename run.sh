#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CONFIG="${1:-configs/reproduce.yaml}"
MODELS=$(python -c "import sys, yaml; print(' '.join(yaml.safe_load(open(sys.argv[1]))['models']))" "$CONFIG")

python pipeline/01_sample_items.py --config "$CONFIG"
for MODEL in $MODELS; do
    for STEP in 02_generate 03_label 04_forward 05_build_uspace 06_score; do
        python "pipeline/$STEP.py" --config "$CONFIG" --model "$MODEL"
    done
done
python pipeline/07_tables.py --config "$CONFIG"
