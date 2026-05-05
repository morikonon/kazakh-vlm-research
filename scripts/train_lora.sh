#!bin/bash

set -e 

echo "Training LoRA Pooling VLM"

python src/training/train.py \
	--config configs/vision_pooling_lorallm.yaml \
	--experiment_name lora_pooling_vlm

echo "Training finished"