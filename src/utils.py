"""
Utility functions for training and evaluation.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")
    return epoch, loss


def plot_attention(attention, src_tokens, trg_tokens, save_path=None):
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention: Attention weights (trg_len, src_len)
        src_tokens: List of source tokens
        trg_tokens: List of target tokens
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Truncate if too long
    max_display = 50
    attention = attention[:max_display, :max_display]
    src_tokens = src_tokens[:max_display]
    trg_tokens = trg_tokens[:max_display]
    
    im = ax.imshow(attention, cmap='Blues')
    
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(trg_tokens)))
    ax.set_xticklabels(src_tokens, rotation=90)
    ax.set_yticklabels(trg_tokens)
    
    ax.set_xlabel('Source')
    ax.set_ylabel('Target')
    ax.set_title('Attention Weights')
    
    plt.colorbar(im)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Attention plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(train_losses, val_losses=None, save_path=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label='Training Loss', color='blue')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', color='orange')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_bleu(predictions, references, max_n=4):
    """
    Calculate BLEU score (simplified implementation).
    
    For production, use nltk.translate.bleu_score or sacrebleu.
    
    Args:
        predictions: List of predicted sentences (list of token lists)
        references: List of reference sentences (list of token lists)
        max_n: Maximum n-gram to consider
    
    Returns:
        bleu_score: BLEU score (0-1)
    """
    from collections import Counter
    import math
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def count_matches(pred_ngrams, ref_ngrams):
        pred_counts = Counter(pred_ngrams)
        ref_counts = Counter(ref_ngrams)
        matches = 0
        for ngram, count in pred_counts.items():
            matches += min(count, ref_counts.get(ngram, 0))
        return matches
    
    precisions = []
    total_pred_len = 0
    total_ref_len = 0
    
    for n in range(1, max_n + 1):
        matches = 0
        total = 0
        
        for pred, ref in zip(predictions, references):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            
            matches += count_matches(pred_ngrams, ref_ngrams)
            total += len(pred_ngrams)
        
        if total > 0:
            precisions.append(matches / total)
        else:
            precisions.append(0)
    
    # Calculate brevity penalty
    for pred, ref in zip(predictions, references):
        total_pred_len += len(pred)
        total_ref_len += len(ref)
    
    if total_pred_len > total_ref_len:
        bp = 1
    elif total_pred_len == 0:
        bp = 0
    else:
        bp = math.exp(1 - total_ref_len / total_pred_len)
    
    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        log_precisions = [math.log(p) for p in precisions]
        geo_mean = math.exp(sum(log_precisions) / len(log_precisions))
    else:
        geo_mean = 0
    
    return bp * geo_mean


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
