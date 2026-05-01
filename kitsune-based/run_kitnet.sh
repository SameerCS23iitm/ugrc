#!/bin/bash

# Usage:
# ./run_kitnet.sh <scenario> <validate|test> [extra args...]
#
# Examples:
# ./run_kitnet.sh cscada validate
# ./run_kitnet.sh external test --max-ae 32 --learning-rate 0.05
# ./run_kitnet.sh cscada validate --hidden-ratio 0.6 --threshold-quantile 0.99

if [ $# -lt 2 ]; then
    echo "Usage: $0 <scenario> <validate|test> [extra hyperparameters...]"
    exit 1
fi

SCENARIO=$1
MODE=$2
shift 2

# Remaining args = hyperparameters
EXTRA_ARGS=("$@")

# Validate scenario
if [[ "$SCENARIO" != "cscada" && "$SCENARIO" != "external" ]]; then
    echo "Error: scenario must be 'cscada' or 'external'"
    exit 1
fi

# Validate mode
if [[ "$MODE" != "validate" && "$MODE" != "test" ]]; then
    echo "Error: second argument must be 'validate' or 'test'"
    exit 1
fi

echo "Running training for scenario: $SCENARIO"
python3 kitsune_new.py \
    --mode train \
    --scenario "$SCENARIO" \
    "${EXTRA_ARGS[@]}"

echo "Running $MODE for scenario: $SCENARIO"
python3 kitsune_new.py \
    --mode "$MODE" \
    --scenario "$SCENARIO" \
    "${EXTRA_ARGS[@]}"