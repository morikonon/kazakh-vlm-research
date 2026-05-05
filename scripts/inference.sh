#!bin/bash

set -e 

echo "Running inference..."

python src/training/inference.py \
	--config configs/vision_pooling_lorallm.yaml

echo "Inference finished"