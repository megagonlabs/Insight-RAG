#!/usr/bin/env bash
# Usage: ./run_insight-rag.sh DOMAIN MAIN_MODEL INSIGHT_MINER_MODEL

set -euo pipefail

DOMAIN="$1"
MAIN_MODEL="$2"
INSIGHT_MINER_MODEL="$3"

python code/insight-rag.py \
       --domain "$DOMAIN" \
       --main_model "$MAIN_MODEL" \
       --insight_miner_model_name "$INSIGHT_MINER_MODEL" 
