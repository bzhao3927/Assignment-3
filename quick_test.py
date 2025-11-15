"""
Quick test script to verify everything works before full training
"""
import torch
from transformers import BertTokenizer
from model import BiLSTMClassifier
from data_module import IMDBDataModule
import pytorch_lightning as pl

def test_tokenizer():
    print("Testing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    text = "I really enjoyed this movie!"
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    
    print(f"✓ Tokenizer working")
    print(f"  Input IDs shape: {encoding['input_ids'].shape}")
    print(f"  Attention mask shape: {encoding['attention_mask'].shape}")
    print(f"  First 10 tokens: {encoding['input_ids'][0][:10]}")
    return True

def test_model():
    print("\nTesting model architecture...")
    model = BiLSTMClassifier(
        vocab_size=30522,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    
    # Create dummy input
    batch_size = 4
    seq_len = 256
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    output = model(input_ids, attention_mask)
    
    print(f"✓ Model forward pass successful")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: [batch_size={batch_size}, num_classes=2]")
    
    assert output.shape == (batch_size, 2), "Output shape mismatch!"
    return True

def test_data_module():
    print("\nTesting data module...")
    pl.seed_everything(42)
    
    data_module = IMDBDataModule(batch_size=8, max_length=256, num_workers=0)
    data_module.prepare_data()
    data_module.setup('fit')
    
    # Get a batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"✓ Data module working")
    print(f"  Batch keys: {batch.keys()}")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Labels shape: {batch['label'].shape}")
    print(f"  Sample labels: {batch['label'][:5]}")
    
    return True

def test_training_step():
    print("\nTesting single training step...")
    pl.seed_everything(42)
    
    model = BiLSTMClassifier(
        vocab_size=30522,
        embedding_dim=64,  # Smaller for quick test
        hidden_dim=128,
        num_layers=1,
        dropout=0.3
    )
    
    data_module = IMDBDataModule(batch_size=8, max_length=128, num_workers=0)
    data_module.setup('fit')
    
    # Get a batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    # Training step
    loss = model.training_step(batch, 0)
    
    print(f"✓ Training step successful")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss is finite: {torch.isfinite(loss).item()}")
    
    return True

def main():
    print("="*60)
    print("RUNNING QUICK TESTS")
    print("="*60)
    
    try:
        test_tokenizer()
        test_model()
        test_data_module()
        test_training_step()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now run the full training with:")
        print("  python train.py")
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED!")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()