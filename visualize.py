"""
Generate visualizations for the report from W&B data
"""
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

def download_wandb_data(project_name, run_name=None):
    """
    Download training metrics from W&B
    
    Args:
        project_name: W&B project name (e.g., 'imdb-sentiment-bilstm')
        run_name: Specific run name (optional)
    """
    api = wandb.Api()
    
    # Get runs from project
    runs = api.runs(f"YOUR_USERNAME/{project_name}")
    
    if run_name:
        runs = [r for r in runs if r.name == run_name]
    
    if not runs:
        print("No runs found!")
        return None
    
    # Get the most recent run
    run = runs[0]
    print(f"Loading data from run: {run.name}")
    
    # Get history
    history = run.history()
    
    return history

def plot_loss_curves(history, save_path='loss_curves.png'):
    """Plot training and validation loss curves"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Filter out NaN values
    train_data = history[['epoch', 'train_loss']].dropna()
    val_data = history[['epoch', 'val_loss']].dropna()
    
    ax.plot(train_data['epoch'], train_data['train_loss'], 
            label='Training Loss', marker='o', linewidth=2)
    ax.plot(val_data['epoch'], val_data['val_loss'], 
            label='Validation Loss', marker='s', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curves saved to {save_path}")

def plot_accuracy_curves(history, save_path='accuracy_curves.png'):
    """Plot training and validation accuracy curves"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Filter out NaN values
    train_data = history[['epoch', 'train_acc']].dropna()
    val_data = history[['epoch', 'val_acc']].dropna()
    
    ax.plot(train_data['epoch'], train_data['train_acc'] * 100, 
            label='Training Accuracy', marker='o', linewidth=2)
    ax.plot(val_data['epoch'], val_data['val_acc'] * 100, 
            label='Validation Accuracy', marker='s', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Accuracy curves saved to {save_path}")

def plot_combined_metrics(history, save_path='combined_metrics.png'):
    """Plot loss and accuracy in subplots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    train_loss = history[['epoch', 'train_loss']].dropna()
    val_loss = history[['epoch', 'val_loss']].dropna()
    
    ax1.plot(train_loss['epoch'], train_loss['train_loss'], 
             label='Training', marker='o', linewidth=2, markersize=6)
    ax1.plot(val_loss['epoch'], val_loss['val_loss'], 
             label='Validation', marker='s', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Loss Curves', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    train_acc = history[['epoch', 'train_acc']].dropna()
    val_acc = history[['epoch', 'val_acc']].dropna()
    
    ax2.plot(train_acc['epoch'], train_acc['train_acc'] * 100, 
             label='Training', marker='o', linewidth=2, markersize=6)
    ax2.plot(val_acc['epoch'], val_acc['val_acc'] * 100, 
             label='Validation', marker='s', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Curves', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined metrics saved to {save_path}")

def print_summary(history):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    train_loss = history['train_loss'].dropna()
    val_loss = history['val_loss'].dropna()
    train_acc = history['train_acc'].dropna()
    val_acc = history['val_acc'].dropna()
    
    print(f"\nFinal Training Loss: {train_loss.iloc[-1]:.4f}")
    print(f"Final Training Accuracy: {train_acc.iloc[-1] * 100:.2f}%")
    print(f"\nFinal Validation Loss: {val_loss.iloc[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_acc.iloc[-1] * 100:.2f}%")
    
    print(f"\nBest Validation Loss: {val_loss.min():.4f} (Epoch {val_loss.idxmin()})")
    print(f"Best Validation Accuracy: {val_acc.max() * 100:.2f}% (Epoch {val_acc.idxmax()})")

def main():
    """
    Main function - you need to update with your W&B username and project
    """
    import sys
    
    # You can pass project name as argument or modify here
    if len(sys.argv) > 1:
        project = sys.argv[1]
    else:
        project = 'imdb-sentiment-bilstm'
    
    print(f"Fetching data from W&B project: {project}")
    print("Note: Update 'YOUR_USERNAME' in the script with your actual W&B username")
    
    # If you want to fetch from W&B (requires your username):
    # history = download_wandb_data(project)
    
    # For now, create sample visualization instructions
    print("\n" + "="*60)
    print("INSTRUCTIONS FOR CREATING VISUALIZATIONS")
    print("="*60)
    print("""
1. Go to your W&B project: https://wandb.ai/YOUR_USERNAME/imdb-sentiment-bilstm

2. Click on your run to view metrics

3. You can export plots directly from W&B:
   - Click on any chart
   - Click the download icon to save as PNG
   
4. Or download data programmatically:
   - Update line 32 in this script with your W&B username
   - Run: python visualize.py
   
5. Metrics to include in your report:
   - Training vs Validation Loss (by epoch)
   - Training vs Validation Accuracy (by epoch)
   - Learning rate over time (if using scheduler)
   
6. Use the confusion matrix from evaluate.py
    """)

if __name__ == '__main__':
    main()