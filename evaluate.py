import torch
import pytorch_lightning as pl
from data_module import IMDBDataModule
from model import BiLSTMClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import wandb
import argparse

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to {save_path}")
    print("\nConfusion Matrix:")
    print(cm)
    return plt.gcf()

def find_false_positives(model, data_module, num_examples=3):
    """Find false positives (pred=Positive, true=Negative)."""
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_loader = data_module.test_dataloader()
    false_positives = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['label'].to(model.device)
            logits = model(input_ids, attention_mask)
            preds = (torch.sigmoid(logits) > 0.5).long()

            for i in range(len(labels)):
                if preds[i] == 1 and labels[i] == 0:
                    text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    false_positives.append({
                        'text': text,
                        'true_label': 'Negative',
                        'predicted_label': 'Positive'
                    })
                    if len(false_positives) >= num_examples:
                        break
            if len(false_positives) >= num_examples:
                break
    return false_positives

def find_false_negatives(model, data_module, num_examples=3):
    """Find false negatives (pred=Negative, true=Positive)."""
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_loader = data_module.test_dataloader()
    false_negatives = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['label'].to(model.device)
            logits = model(input_ids, attention_mask)
            preds = (torch.sigmoid(logits) > 0.5).long()

            for i in range(len(labels)):
                if preds[i] == 0 and labels[i] == 1:
                    text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    false_negatives.append({
                        'text': text,
                        'true_label': 'Positive',
                        'predicted_label': 'Negative'
                    })
                    if len(false_negatives) >= num_examples:
                        break
            if len(false_negatives) >= num_examples:
                break
    return false_negatives

def evaluate_model(checkpoint_path, max_length, log_to_wandb=True):
    pl.seed_everything(42)
    print(f"Using max_length={max_length}")

    if log_to_wandb:
        wandb.init(
            project='imdb-sentiment-bilstm',
            name='evaluation',
            job_type='evaluation',
            config={'max_length': max_length}
        )

    data_module = IMDBDataModule(batch_size=32, max_length=max_length, num_workers=4)
    data_module.setup('test')
    model = BiLSTMClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()

    trainer = pl.Trainer(accelerator='auto', devices=1, logger=False)
    test_results = trainer.test(model, data_module)

    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Test Accuracy: {test_results[0]['test_acc']:.4f}")
    print(f"Test Loss: {test_results[0]['test_loss']:.4f}")

    predictions, true_labels = [], []
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for batch in data_module.test_dataloader():
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['label']
            logits = model(input_ids, attention_mask)
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.numpy())

    fig = plot_confusion_matrix(true_labels, predictions)

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(
        true_labels, predictions,
        target_names=['Negative', 'Positive'],
        digits=4, output_dict=True
    )
    print(classification_report(
        true_labels, predictions,
        target_names=['Negative', 'Positive'],
        digits=4
    ))

    # False Positives
    print("\n" + "="*50)
    print("FALSE POSITIVE EXAMPLES")
    print("="*50)
    false_positives = find_false_positives(model, data_module, num_examples=3)
    fp_table = []
    for i, ex in enumerate(false_positives, 1):
        print(f"\nFalse Positive {i}:")
        print(f"True Label: {ex['true_label']}")
        print(f"Predicted Label: {ex['predicted_label']}")
        print(f"Text: {ex['text'][:200]}...")
        print("-" * 50)
        fp_table.append([i, ex['true_label'], ex['predicted_label'], ex['text'][:200]])

    # False Negatives
    print("\n" + "="*50)
    print("FALSE NEGATIVE EXAMPLES")
    print("="*50)
    false_negatives = find_false_negatives(model, data_module, num_examples=3)
    fn_table = []
    for i, ex in enumerate(false_negatives, 1):
        print(f"\nFalse Negative {i}:")
        print(f"True Label: {ex['true_label']}")
        print(f"Predicted Label: {ex['predicted_label']}")
        print(f"Text: {ex['text'][:200]}...")
        print("-" * 50)
        fn_table.append([i, ex['true_label'], ex['predicted_label'], ex['text'][:200]])

    if log_to_wandb:
        wandb.log({
            'eval/test_accuracy': test_results[0]['test_acc'],
            'eval/test_loss': test_results[0]['test_loss'],
            'eval/precision_negative': report['Negative']['precision'],
            'eval/precision_positive': report['Positive']['precision'],
            'eval/recall_negative': report['Negative']['recall'],
            'eval/recall_positive': report['Positive']['recall'],
            'eval/f1_negative': report['Negative']['f1-score'],
            'eval/f1_positive': report['Positive']['f1-score'],
            'eval/macro_avg_f1': report['macro avg']['f1-score'],
            "eval/confusion_matrix": wandb.Image(fig),
            "eval/false_positives": wandb.Table(
                columns=["Example #", "True Label", "Predicted Label", "Text (truncated)"],
                data=fp_table
            ),
            "eval/false_negatives": wandb.Table(
                columns=["Example #", "True Label", "Predicted Label", "Text (truncated)"],
                data=fn_table
            )
        })
        wandb.finish()
        print("\n✓ Results logged to W&B")

    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate BiLSTM model on IMDB test set')
    parser.add_argument('checkpoint_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--max-length', type=int, required=True, help='Maximum sequence length (REQUIRED - must match training)')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint_path,
        max_length=args.max_length,
        log_to_wandb=not args.no_wandb
    )
