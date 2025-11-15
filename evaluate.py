import torch
import pytorch_lightning as pl
from data_module import IMDBDataModule
from model import BiLSTMClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")
    print("\nConfusion Matrix:")
    print(cm)

def find_misclassified_examples(model, data_module, num_examples=3):
    """Find and print misclassified examples"""
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Get test dataloader
    test_loader = data_module.test_dataloader()
    
    misclassified = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['label'].to(model.device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            # Find misclassified samples in this batch
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    misclassified.append({
                        'text': text,
                        'true_label': 'Positive' if labels[i].item() == 1 else 'Negative',
                        'predicted_label': 'Positive' if preds[i].item() == 1 else 'Negative'
                    })
                    
                    if len(misclassified) >= num_examples:
                        break
            
            if len(misclassified) >= num_examples:
                break
    
    return misclassified

def evaluate_model(checkpoint_path):
    """Load model and evaluate on test set"""
    pl.seed_everything(42)
    
    # Load data module
    data_module = IMDBDataModule(batch_size=32, max_length=256, num_workers=4)
    data_module.setup('test')
    
    # Load model from checkpoint
    model = BiLSTMClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Create trainer for testing
    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        logger=False
    )
    
    # Test the model
    test_results = trainer.test(model, data_module)
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Test Accuracy: {test_results[0]['test_acc']:.4f}")
    print(f"Test Loss: {test_results[0]['test_loss']:.4f}")
    
    # Get predictions for confusion matrix
    predictions = []
    true_labels = []
    
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['label']
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(labels.numpy())
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions)
    
    # Print classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(
        true_labels, 
        predictions,
        target_names=['Negative', 'Positive'],
        digits=4
    ))
    
    # Find misclassified examples
    print("\n" + "="*50)
    print("MISCLASSIFIED EXAMPLES")
    print("="*50)
    misclassified = find_misclassified_examples(model, data_module, num_examples=3)
    
    for i, example in enumerate(misclassified, 1):
        print(f"\nExample {i}:")
        print(f"True Label: {example['true_label']}")
        print(f"Predicted Label: {example['predicted_label']}")
        print(f"Text: {example['text'][:200]}...")  # First 200 chars
        print("-" * 50)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <checkpoint_path>")
        print("Example: python evaluate.py checkpoints/bilstm-epoch=05-val_loss=0.25.ckpt")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    evaluate_model(checkpoint_path)