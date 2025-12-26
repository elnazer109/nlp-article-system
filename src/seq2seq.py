"""
Full Seq2Seq model combining Encoder, Attention, and Decoder.
"""

import random
import torch
import torch.nn as nn

from src.config import Config
from src.encoder import Encoder
from src.decoder import Decoder


class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model with attention.
    
    Combines:
    - Bidirectional LSTM Encoder
    - Bahdanau Attention
    - LSTM Decoder with attention
    """
    
    def __init__(self, src_vocab_size, trg_vocab_size, 
                 embedding_dim=None, hidden_dim=None, attention_dim=None,
                 num_layers=None, dropout=None, padding_idx=0):
        """
        Args:
            src_vocab_size: Size of source vocabulary
            trg_vocab_size: Size of target vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden states
            attention_dim: Dimension of attention layer
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            padding_idx: Index of padding token
        """
        super().__init__()
        
        # Use config defaults if not specified
        embedding_dim = embedding_dim or Config.EMBEDDING_DIM
        hidden_dim = hidden_dim or Config.HIDDEN_DIM
        attention_dim = attention_dim or Config.ATTENTION_DIM
        num_layers = num_layers or Config.NUM_LAYERS
        dropout = dropout or Config.DROPOUT
        
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.padding_idx = padding_idx
        
        # Encoder (bidirectional)
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx
        )
        
        # Decoder with attention
        # Encoder output dim is hidden_dim * 2 (bidirectional)
        self.decoder = Decoder(
            vocab_size=trg_vocab_size,
            embedding_dim=embedding_dim,
            encoder_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx
        )
        
    def create_mask(self, src):
        """
        Create mask for padding tokens.
        
        Args:
            src: Source sequence (src_len, batch)
        
        Returns:
            mask: Boolean mask (batch, src_len), True = padding
        """
        # Transpose to (batch, src_len) and check for padding
        mask = (src == self.padding_idx).permute(1, 0)
        return mask
    
    def forward(self, src, trg, src_lens=None, teacher_forcing_ratio=0.5):
        """
        Forward pass with teacher forcing.
        
        Args:
            src: Source sequence (src_len, batch)
            trg: Target sequence (trg_len, batch)
            src_lens: Source lengths for packing
            teacher_forcing_ratio: Probability of using ground truth as next input
        
        Returns:
            outputs: Predicted token scores (trg_len, batch, trg_vocab_size)
        """
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(
            trg_len, batch_size, self.trg_vocab_size,
            device=src.device
        )
        
        # Create padding mask
        mask = self.create_mask(src)
        
        # Encode source sequence
        encoder_outputs, hidden, cell = self.encoder(src, src_lens)
        
        # First decoder input is SOS token
        input_token = trg[0]  # (batch,)
        
        # Decode step by step
        for t in range(1, trg_len):
            # Decode one step
            output, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask
            )
            
            # Store output
            outputs[t] = output
            
            # Decide whether to use teacher forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            # Get predicted token
            top1 = output.argmax(1)
            
            # Next input: either ground truth or prediction
            input_token = trg[t] if use_teacher_forcing else top1
        
        return outputs
    
    def generate(self, src, src_lens=None, max_len=None, sos_idx=None, eos_idx=None):
        """
        Generate summary for a source sequence (inference mode).
        
        Args:
            src: Source sequence (src_len, batch)
            src_lens: Source lengths
            max_len: Maximum output length
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
        
        Returns:
            predictions: Generated token indices (max_len, batch)
            attention_weights: Attention weights for visualization
        """
        max_len = max_len or Config.MAX_TARGET_LEN
        sos_idx = sos_idx or Config.SOS_IDX
        eos_idx = eos_idx or Config.EOS_IDX
        
        batch_size = src.shape[1]
        
        # Store predictions and attention
        predictions = torch.zeros(max_len, batch_size, dtype=torch.long, device=src.device)
        all_attention = []
        
        # Create mask and encode
        mask = self.create_mask(src)
        encoder_outputs, hidden, cell = self.encoder(src, src_lens)
        
        # Start with SOS token
        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=src.device)
        predictions[0] = input_token
        
        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        
        for t in range(1, max_len):
            output, hidden, cell, attention = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask
            )
            
            all_attention.append(attention)
            
            # Greedy decoding
            top1 = output.argmax(1)
            predictions[t] = top1
            
            # Check for EOS
            finished = finished | (top1 == eos_idx)
            if finished.all():
                break
            
            input_token = top1
        
        # Stack attention weights
        attention_weights = torch.stack(all_attention, dim=0) if all_attention else None
        
        return predictions, attention_weights
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(src_vocab_size, trg_vocab_size, device=None):
    """
    Factory function to create and initialize the model.
    
    Args:
        src_vocab_size: Size of source vocabulary
        trg_vocab_size: Size of target vocabulary
        device: Device to place model on
    
    Returns:
        model: Initialized Seq2Seq model
    """
    device = device or Config.DEVICE
    
    model = Seq2Seq(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size
    ).to(device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.1)
            if m.padding_idx is not None:
                nn.init.zeros_(m.weight[m.padding_idx])
    
    model.apply(init_weights)
    
    print(f"Model created with {model.count_parameters():,} trainable parameters")
    print(f"Device: {device}")
    
    return model
