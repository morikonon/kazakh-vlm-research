#!bin/bash

set -e 

echo "Training Linear VLM"

python src/training/train.py \
	--config configs/vision_linear_llm.yaml \
	--experiment_name linear_vlm

echo "Training finished"