import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import BertTokenizer
import os
import numpy as np
import random
from pathlib import Path

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

class IMDBDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='aclImdb', batch_size=32, max_length=256, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def load_texts_from_folder(self, folder_path):
        """Load all text files from a folder"""
        texts = []
        folder = Path(folder_path)
        for file_path in sorted(folder.glob('*.txt')):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        return texts
        
    def prepare_data(self):
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory '{self.data_dir}' not found!")
        
    def setup(self, stage=None):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        
        # Load ALL data (50K total)
        train_pos = self.load_texts_from_folder(f'{self.data_dir}/train/pos')
        train_neg = self.load_texts_from_folder(f'{self.data_dir}/train/neg')
        test_pos = self.load_texts_from_folder(f'{self.data_dir}/test/pos')
        test_neg = self.load_texts_from_folder(f'{self.data_dir}/test/neg')
        
        # Combine ALL 50K reviews
        all_texts = train_pos + train_neg + test_pos + test_neg
        all_labels = (
            [1] * len(train_pos) + [0] * len(train_neg) +
            [1] * len(test_pos) + [0] * len(test_neg)
        )
        
        # Shuffle all data
        all_indices = list(range(len(all_texts)))
        np.random.shuffle(all_indices)
        all_texts = [all_texts[i] for i in all_indices]
        all_labels = [all_labels[i] for i in all_indices]
        
        # Split into 70% train / 15% val / 15% test
        total = len(all_texts)  # 50,000
        train_size = int(0.70 * total)  # 35,000
        val_size = int(0.15 * total)     # 7,500
        test_size = total - train_size - val_size  # 7,500
        
        # Create splits
        train_texts = all_texts[:train_size]
        train_labels = all_labels[:train_size]
        
        val_texts = all_texts[train_size:train_size + val_size]
        val_labels = all_labels[train_size:train_size + val_size]
        
        test_texts = all_texts[train_size + val_size:]
        test_labels = all_labels[train_size + val_size:]
        
        # Print split statistics
        print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = IMDBDataset(
                train_texts, train_labels, self.tokenizer, self.max_length
            )
            self.val_dataset = IMDBDataset(
                val_texts, val_labels, self.tokenizer, self.max_length
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = IMDBDataset(
                test_texts, test_labels, self.tokenizer, self.max_length
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )