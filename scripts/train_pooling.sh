#!bin/bash

set -e 

echo "Training Pooling VLM"

python src/training/train.py \
	--config configs/vision_pooling_llm.yaml \
	--experiment_name pooling_vlm

echo "Training finished"