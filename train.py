"""
Training script for Seq2Seq summarization model.

Usage:
    # Train with synthetic data (no real data needed)
    python train.py --use_synthetic --epochs 5
    
    # Train with real PubMed data
    python train.py --data_dir data/ --epochs 10
    
    # Quick test run
    python train.py --use_synthetic --epochs 1 --batch_size 4 --max_samples 100
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config import Config
from src.dataset import (
    SyntheticDataGenerator, 
    load_csv_data, 
    create_dataloaders
)
from src.seq2seq import create_model
from src.utils import save_checkpoint, EarlyStopping, plot_training_history


def train_epoch(model, dataloader, optimizer, criterion, clip_grad, device):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        src = batch['src'].to(device)
        trg = batch['trg'].to(device)
        src_lens = batch['src_lens']
        
        optimizer.zero_grad()
        
        # Forward pass with teacher forcing
        output = model(src, trg, src_lens, Config.TEACHER_FORCING_RATIO)
        
        # Reshape for loss calculation
        # output: (trg_len, batch, vocab_size) -> (trg_len * batch, vocab_size)
        # trg: (trg_len, batch) -> (trg_len * batch)
        output_dim = output.shape[-1]
        
        # Skip first token (SOS) in output and target
        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)
        
        # Calculate loss
        loss = criterion(output, trg)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)
            src_lens = batch['src_lens']
            
            # Forward pass without teacher forcing
            output = model(src, trg, src_lens, teacher_forcing_ratio=0)
            
            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def main(args):
    """Main training function."""
    print("=" * 60)
    print("Seq2Seq Summarization Training")
    print("=" * 60)
    print(Config.get_device_info())
    
    device = Config.DEVICE
    
    # Load or generate data
    if args.use_synthetic:
        print("\nGenerating synthetic data...")
        train_articles, train_summaries = SyntheticDataGenerator.generate_dataset(
            args.max_samples or 5000
        )
        # Use 10% for validation
        split_idx = int(len(train_articles) * 0.9)
        val_articles = train_articles[split_idx:]
        val_summaries = train_summaries[split_idx:]
        train_articles = train_articles[:split_idx]
        train_summaries = train_summaries[:split_idx]
    else:
        print(f"\nLoading data from {args.data_dir}...")
        train_path = os.path.join(args.data_dir, 'train.csv')
        val_path = os.path.join(args.data_dir, 'validation.csv')
        
        train_articles, train_summaries = load_csv_data(
            train_path, max_samples=args.max_samples
        )
        
        if os.path.exists(val_path):
            val_articles, val_summaries = load_csv_data(
                val_path, max_samples=args.max_samples // 10 if args.max_samples else None
            )
        else:
            # Split training data
            split_idx = int(len(train_articles) * 0.9)
            val_articles = train_articles[split_idx:]
            val_summaries = train_summaries[split_idx:]
            train_articles = train_articles[:split_idx]
            train_summaries = train_summaries[:split_idx]
    
    print(f"Training samples: {len(train_articles)}")
    print(f"Validation samples: {len(val_articles)}")
    
    # Create dataloaders
    print("\nBuilding vocabulary and dataloaders...")
    train_loader, val_loader, src_vocab, trg_vocab = create_dataloaders(
        train_articles, train_summaries,
        val_articles, val_summaries,
        batch_size=args.batch_size
    )
    
    # Save vocabulary
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    src_vocab.save(os.path.join(args.checkpoint_dir, 'vocab.json'))
    
    # Create model
    print("\nCreating model...")
    model = create_model(len(src_vocab), len(trg_vocab), device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            Config.CLIP_GRAD, device
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        elapsed = time.time() - start_time
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.checkpoint_dir, 'best_model.pt')
            )
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            os.path.join(args.checkpoint_dir, 'latest_model.pt')
        )
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Plot training history
    plot_training_history(
        train_losses, val_losses,
        os.path.join(args.checkpoint_dir, 'training_history.png')
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.checkpoint_dir}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Seq2Seq Summarization Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing train.csv and validation.csv')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic data for testing (no real data needed)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of training samples')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=Config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    main(args)
