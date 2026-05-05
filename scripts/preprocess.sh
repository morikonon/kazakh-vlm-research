#!/bin/bash

set -e

echo "Starting data preprocessing..."

python src/data/preprocess_data.py

echo "Preprocessing finished"