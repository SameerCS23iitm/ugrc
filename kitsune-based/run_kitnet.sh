#!/bin/bash

# Usage:
# ./run_kitnet.sh <scenario> <validate|test> [extra args...]
#
# Extra args are passed to kitsune_new.py. To force retraining before running
# test mode, include `--force-retrain` among the extra args. The script will
# detect and remove it before invoking Python.
#
# Examples:
# ./run_kitnet.sh cscada validate
# ./run_kitnet.sh external test --max-ae 32 --learning-rate 0.05
# ./run_kitnet.sh cscada test --force-retrain --max-ae 32
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

# Detect --force-retrain among EXTRA_ARGS; remove it and set flag
FORCE_RETRAIN=0
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    NEW_ARGS=()
    for a in "${EXTRA_ARGS[@]}"; do
        if [[ "$a" == "--force-retrain" ]]; then
            FORCE_RETRAIN=1
        else
            NEW_ARGS+=("$a")
        fi
    done
    EXTRA_ARGS=("${NEW_ARGS[@]}")
fi

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

if [[ "$MODE" == "validate" ]]; then
    echo "Running training for scenario: $SCENARIO"
    python3 kitsune_new.py \
        --mode train \
        --scenario "$SCENARIO" \
        "${EXTRA_ARGS[@]}"

    echo "Running validate for scenario: $SCENARIO"
    python3 kitsune_new.py \
        --mode validate \
        --scenario "$SCENARIO" \
        "${EXTRA_ARGS[@]}"
else
    # MODE == test
    if [[ "$FORCE_RETRAIN" -eq 1 ]]; then
        echo "--force-retrain detected: running training for scenario: $SCENARIO"
        python3 kitsune_new.py \
            --mode train \
            --scenario "$SCENARIO" \
            "${EXTRA_ARGS[@]}"
        echo "Running test for scenario: $SCENARIO (after forced retrain)"
        python3 kitsune_new.py \
            --mode test \
            --scenario "$SCENARIO" \
            "${EXTRA_ARGS[@]}"
    else
        # No force flag: do not retrain, only run test
        echo "Running test for scenario: $SCENARIO (no retrain)"
        python3 kitsune_new.py \
            --mode test \
            --scenario "$SCENARIO" \
            "${EXTRA_ARGS[@]}"
    fi
fi