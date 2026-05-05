#!/bin/bash

set -e

echo "Evaluating KazVLM..."

python src/training/evaluate.py \
	--config configs/vision_pooling_lorallm.yaml \
	--checkpoints checkpoints/best_model

echo "Evaluation finished"