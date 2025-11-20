import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy

class BiLSTMClassifier(pl.LightningModule):
    def __init__(
        self, 
        vocab_size, 
        embedding_dim, 
        hidden_dim, 
        num_layers, 
        dropout,
        learning_rate=0.001,
        weight_decay=1e-5,
        max_epochs=20
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        
        # Layers - all randomly initialized
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        
        # Loss and metrics
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.test_accuracy = Accuracy(task='binary')
    
    def forward(self, input_ids, attention_mask=None):
        # Embedding (randomly initialized)
        embedded = self.embedding(input_ids)
        
        # Pack padded sequence if attention_mask is provided
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Unpack if we packed
        if attention_mask is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Use last hidden states from both directions
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        hidden_fwd = hidden[-2]  # Forward direction last layer
        hidden_bwd = hidden[-1]  # Backward direction last layer
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        # Dropout and classifier
        hidden_concat = self.dropout(hidden_concat)
        logits = self.fc(hidden_concat)
        
        return logits.squeeze()
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label'].float()
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        preds = torch.sigmoid(logits) > 0.5
        acc = self.train_accuracy(preds, labels.long())
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label'].float()
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        preds = torch.sigmoid(logits) > 0.5
        acc = self.val_accuracy(preds, labels.long())
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label'].float()
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        preds = torch.sigmoid(logits) > 0.5
        acc = self.test_accuracy(preds, labels.long())
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Use Cosine Annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }