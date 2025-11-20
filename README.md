# IMDB Sentiment Classification using Bi-LSTM

Binary sentiment classification on IMDB 50K dataset with systematic capacity scaling using Bidirectional LSTM and PyTorch Lightning.

## Team Members
- Cade Boiney
- Ken Lam
- Ognian Trajanov
- Benjamin Zhao

**Hamilton College - CS-366 Deep Learning - Fall 2025**

## Results

**Best Model (2048-dim):**
- **Test Accuracy:** 91.04%
- **Test Loss:** 0.222
- **Parameters:** 230M
- **Training:** NVIDIA RTX 5070 Ti with FP16

| Model | Params | Val Acc | Test Acc |
|-------|--------|---------|----------|
| 256-dim | 10.6M | 87.3% | 88.0% |
| 512-dim | 26.1M | 90.0% | 90.27% |
| 1024-dim | 73.2M | 90.0% | 90.71% |
| **2048-dim** | **230M** | **90.1%** | **91.04%** |
| 4096-dim | 796M | - | OOM |

Full results and analysis in [`report.pdf`](report.pdf).

## Model Architecture

Systematically scaled across four capacities (256/512/1024/2048 dimensions):
- **Embedding:** Randomly initialized, proportional to model size (vocab: 30,522)
- **Bi-LSTM:** 2 layers, hidden units match embedding dimension
- **Dropout:** 0.3
- **Classifier:** BCEWithLogitsLoss
- **Max Length:** Scaled with capacity (256/512/1024/2048 tokens)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dataset auto-downloads on first run.

## Usage

### Training
```bash
# Best model (2048-dim)
python train.py --embedding-dim 2048 --hidden-dim 2048 --max-length 2048

# Other scales
python train.py --embedding-dim 1024 --hidden-dim 1024 --max-length 1024
python train.py --embedding-dim 512 --hidden-dim 512 --max-length 512
```

### Evaluation
```bash
# Must specify matching max-length
python evaluate.py checkpoints/bilstm-epoch=01-val_loss=0.24-v2.ckpt --max-length 2048
```

Generates confusion matrix, classification report, and misclassified examples.

## Key Findings

1. **Capacity Scaling:** Systematic improvement from 88.0% (256-dim) to 91.04% (2048-dim)
2. **Generalization:** Larger models generalized better despite more parameters (7-8% train-val gap)
3. **Hardware Limits:** 4096-dim (796M params) exceeded 16GB GPU memory
4. **Diminishing Returns:** Each doubling of capacity yields progressively smaller gains

## Configuration

**Hyperparameters (held constant across scales):**
- Optimizer: AdamW (lr=0.001, wd=1e-5)
- Batch size: 32
- Dropout: 0.3
- LR scheduler: Cosine Annealing (T_max=20)
- Early stopping: patience=2
- Data split: 70/15/15 (35K train, 7.5K val, 7.5K test)
- Random seed: 42

## File Structure
```
.
├── train.py              # Training with capacity arguments
├── model.py              # Bi-LSTM definition
├── data_module.py        # Data loading
├── evaluate.py           # Evaluation (requires --max-length)
├── requirements.txt      # Dependencies
├── report.pdf            # Full analysis
└── checkpoints/          # Model weights
```

## Links

**W&B Project:** https://wandb.ai/bzhao-hamilton-college/imdb-sentiment-bilstm

**Code:** https://github.com/bzhao3927/assignment-3

**Best Run:** `2048-all` (Val: 90.1%, Test: 91.04%)

## References
- Dataset: [IMDB 50K](https://ai.stanford.edu/~amaas/data/sentiment/)
- Framework: PyTorch Lightning
- Tokenizer: BERT (bert-base-uncased)

## License
Educational project for Hamilton College CS-366.