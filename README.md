# IMDB Sentiment Classification using Bi-LSTM

Binary sentiment classification on the IMDB 50K movie review dataset using a Bidirectional LSTM with PyTorch Lightning.

## Team Members
- Cade Boiney
- Ken Lam
- Ognian Trajanov
- Benjamin Zhao

Hamilton College - CS-366 Deep Learning - Fall 2025

## Results
- **Test Accuracy:** 85.35%
- **Test Loss:** 0.359
- **Model Parameters:** 6.3M
- **Training Time:** ~3 minutes on NVIDIA RTX 5070 Ti

Full results and analysis available in `report.pdf`.

## Model Architecture
- Embedding: 128-dimensional, randomly initialized (vocab size: 30,522)
- Bi-LSTM: 2 layers, 256 hidden units per direction
- Dropout: 0.5
- Classifier: Binary cross-entropy loss

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

Or with virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset
The IMDB 50K dataset will be automatically downloaded on first run.

## Usage

### Training
Train the model from scratch:
```bash
python train.py
```

Training logs will be automatically uploaded to Weights & Biases.

### Evaluation
Evaluate a trained checkpoint:
```bash
python evaluate.py checkpoints/bilstm-epoch=03-val_loss=0.35-v1.ckpt
```

This will generate:
- Test accuracy and loss
- Confusion matrix (saved as `confusion_matrix.png`)
- Classification report with precision/recall/F1
- 3 misclassified examples

## File Structure
```
.
├── train.py              # Main training script
├── model.py              # Bi-LSTM model definition
├── data_module.py        # Data loading and preprocessing
├── evaluate.py           # Evaluation and analysis script
├── requirements.txt      # Python dependencies
├── report.pdf            # Full project report
├── confusion_matrix.png  # Generated confusion matrix
└── checkpoints/          # Saved model checkpoints
```

## Hyperparameters
- Batch size: 32
- Learning rate: 0.001 (AdamW)
- Weight decay: 1e-5
- Max sequence length: 256 tokens
- LR scheduler: Cosine Annealing (T_max=20)
- Early stopping patience: 3 epochs

## Reproducibility
All experiments use random seed 42 for reproducibility:
```python
pl.seed_everything(42)
```

## Weights & Biases
View training logs and metrics:
https://wandb.ai/bzhao-hamilton-college/imdb-sentiment-bilstm

## References
- Dataset: [IMDB 50K](https://ai.stanford.edu/~amaas/data/sentiment/)
- Framework: PyTorch Lightning
- Tokenizer: BERT (bert-base-uncased)

## License
Educational project for Hamilton College CS-366.
