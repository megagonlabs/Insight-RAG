#!/usr/bin/env bash
# Usage: ./run_insight_miner.sh DOMAIN BASE_MODEL

set -euo pipefail

DOMAIN="$1"
BASE_MODEL="$2"

python code/insight_miner.py \
       --domain "$DOMAIN" \
       --model_name "$BASE_MODEL" 
