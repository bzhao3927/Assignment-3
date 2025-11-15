#!/bin/bash

# Setup script for IMDB Sentiment Classification Project

echo "=========================================="
echo "IMDB Sentiment Classification Setup"
echo "=========================================="

# Create virtual environment
echo ""
echo "Step 1: Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "Step 2: Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

# Upgrade pip
echo ""
echo "Step 3: Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Step 4: Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Step 5: Creating directories..."
mkdir -p checkpoints
mkdir -p results

# Run quick test
echo ""
echo "Step 6: Running quick tests..."
python quick_test.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source venv/Scripts/activate"
else
    echo "   source venv/bin/activate"
fi
echo ""
echo "2. Configure W&B (if not already done):"
echo "   wandb login"
echo ""
echo "3. Start training:"
echo "   python train.py"
echo ""
echo "4. Evaluate model:"
echo "   python evaluate.py checkpoints/best_model.ckpt"
echo ""