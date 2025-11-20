#!/bin/bash

# run.sh
# Runs multiple BiLSTM configurations and evaluates each on test set

echo "======================================"
echo "BiLSTM Hyperparameter Sweep"
echo "======================================"

# Create directories
mkdir -p checkpoints
mkdir -p results

# Configuration 1: Baseline (1 layer, small)
echo ""
echo "Running Config 1: 1 layer, embedding=128, hidden=128"
python train.py \
    --run_name "config1-1layer-128" \
    --num_layers 1 \
    --embedding_dim 128 \
    --hidden_dim 128 \
    --dropout 0.3 \
    --learning_rate 0.001 \
    --max_epochs 15

# Configuration 2: Medium capacity (1 layer, larger)
echo ""
echo "Running Config 2: 1 layer, embedding=256, hidden=256"
python train.py \
    --run_name "config2-1layer-256" \
    --num_layers 1 \
    --embedding_dim 256 \
    --hidden_dim 256 \
    --dropout 0.5 \
    --learning_rate 0.001 \
    --max_epochs 15

# Configuration 3: Higher capacity (1 layer, even larger)
echo ""
echo "Running Config 3: 1 layer, embedding=384, hidden=384"
python train.py \
    --run_name "config3-1layer-384" \
    --num_layers 1 \
    --embedding_dim 384 \
    --hidden_dim 384 \
    --dropout 0.5 \
    --learning_rate 0.001 \
    --max_epochs 15

# Configuration 4: Two layers (medium)
echo ""
echo "Running Config 4: 2 layers, embedding=256, hidden=256"
python train.py \
    --run_name "config4-2layers-256" \
    --num_layers 2 \
    --embedding_dim 256 \
    --hidden_dim 256 \
    --dropout 0.5 \
    --learning_rate 0.001 \
    --max_epochs 15

# Configuration 5: Two layers (larger)
echo ""
echo "Running Config 5: 2 layers, embedding=384, hidden=384"
python train.py \
    --run_name "config5-2layers-384" \
    --num_layers 2 \
    --embedding_dim 384 \
    --hidden_dim 384 \
    --dropout 0.5 \
    --learning_rate 0.0005 \
    --max_epochs 15

echo ""
echo "======================================"
echo "All configurations completed!"
echo "======================================"
echo ""
