#!/usr/bin/env bash

# batch_predict.sh
# Runs predict_material.py over material_1..10 and cycle_1..100
# Usage: ./batch_predict.sh [output_log]

OUTPUT_LOG=${1:-batch_predictions.txt}
> "$OUTPUT_LOG"

for n in {1..10}; do
  for m in {1..100}; do
    FLOW_PATH="material_${n}/cycle_${m}_A/flow.json"
    if [ -f "$FLOW_PATH" ]; then
      echo "Predicting for $FLOW_PATH..." | tee -a "$OUTPUT_LOG"
      python predict_material.py "$FLOW_PATH" | tee -a "$OUTPUT_LOG"
    else
      echo "Missing: $FLOW_PATH" | tee -a "$OUTPUT_LOG"
    fi
  done
done

echo "Batch prediction complete. Results saved to $OUTPUT_LOG."
