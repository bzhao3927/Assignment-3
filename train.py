import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import wandb
from data_module import IMDBDataModule
from model import BiLSTMClassifier

def main():
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Hyperparameters
    config = {
        'batch_size': 32,
        'max_length': 2048,
        'embedding_dim': 2048,
        'hidden_dim': 2048,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'max_epochs': 20,
        'num_workers': 4
    }
    
    # Initialize Weights & Biases
    wandb.login()
    wandb_logger = WandbLogger(
        project='imdb-sentiment-bilstm',
        name='dim-2048',
        config=config
    )
    
    # Initialize data module
    data_module = IMDBDataModule(
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        num_workers=config['num_workers']
    )
    
    # Initialize model
    model = BiLSTMClassifier(
        vocab_size=30522,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        max_epochs=config['max_epochs'],
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='bilstm-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=2,
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        accelerator='auto',
        devices=1,
        precision=16 if torch.cuda.is_available() else 32,
        log_every_n_steps=50,
        deterministic=True,
        gradient_clip_val=1.0
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Test the model
    trainer.test(model, data_module, ckpt_path='best')
    
    # Close wandb run
    wandb.finish()
    
    print("Training completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == '__main__':
    main()