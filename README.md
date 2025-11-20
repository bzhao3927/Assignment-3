# IMDB Sentiment Classification using Bi-LSTM

Binary sentiment classification on IMDB 50K dataset with systematic capacity scaling using Bidirectional LSTM and PyTorch Lightning.

## Team Members
- Cade Boiney
- Ken Lam
- Ognian Trajanov
- Benjamin Zhao

**Hamilton College - CS-366 Deep Learning - Fall 2025**

## Results

| Model | Params | Max Length | Train Acc | Val Acc |
|-------|--------|------------|-----------|---------|
| 256-dim | 10.6M | 256 | 97.2% | 87.9% |
| 512-dim | 26.1M | 512 | 98.6% | 90.3% |
| 1024-dim | 73.2M | 1024 | 98.1% | 90.6% |
| **2048-dim** | **230M** | **2048** | **98.2%** | **90.5%** |
| 4096-dim | 796M | 4096 | OOM | - |

We selected the **2048-dim model** for final evaluation based on its strong validation performance (90.5%) and greater model capacity.

**Final Test Results (2048-dim):**
- **Test Accuracy:** 91.04%
- **Test Loss:** 0.222
- **Training:** NVIDIA RTX 5070 Ti with FP16

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

Model checkpoints automatically saved via PyTorch Lightning to `wandb/` directory.

### Evaluation
```bash
# Evaluate saved checkpoint (specify matching max-length)
python evaluate.py <path-to-checkpoint.ckpt> --max-length 2048
```

Generates confusion matrix, classification report, and misclassified examples.

## Key Findings

1. **Capacity Scaling:** Systematic improvement from 87.9% to 90.6% validation accuracy
2. **Generalization:** Larger models generalized better despite more parameters (7-8% train-val gap)
3. **Hardware Limits:** 4096-dim (796M params) exceeded 16GB GPU memory
4. **Final Performance:** Best model achieved 91.04% test accuracy on held-out set

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
├── train.py              # Training script
├── model.py              # Bi-LSTM model definition
├── data_module.py        # Data loading and preprocessing
├── evaluate.py           # Evaluation script
├── requirements.txt      # Python dependencies
├── report.pdf            # Full analysis report
├── confusion_matrix.png  # Test set confusion matrix
└── wandb/                # W&B logs and model checkpoints
```

## Links

**W&B Project:** https://wandb.ai/bzhao-hamilton-college/imdb-sentiment-bilstm

**GitHub:** https://github.com/bzhao3927/Assignment-3

**Best Run:** `dim-2048` (Val: 90.5%, Test: 91.04%)

## References
- Dataset: [IMDB 50K](https://ai.stanford.edu/~amaas/data/sentiment/)
- Framework: PyTorch Lightning
- Tokenizer: BERT (bert-base-uncased)

## License
Educational project for Hamilton College CS-366.