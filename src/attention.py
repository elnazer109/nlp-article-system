"""
Bahdanau (Additive) Attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (Additive Attention).
    
    Computes attention weights between decoder hidden state and 
    all encoder outputs, then returns a weighted context vector.
    
    Reference: "Neural Machine Translation by Jointly Learning to Align and Translate"
               Bahdanau et al., 2015
    """
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        Args:
            encoder_dim: Dimension of encoder outputs (hidden_dim * 2 for bidirectional)
            decoder_dim: Dimension of decoder hidden state
            attention_dim: Dimension of attention layer
        """
        super().__init__()
        
        # Linear layers to transform encoder and decoder states
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        
        # Layer to compute energy scores
        self.v = nn.Linear(attention_dim, 1)
        
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Compute attention weights and context vector.
        
        Args:
            decoder_hidden: Current decoder hidden state (batch, decoder_dim)
            encoder_outputs: All encoder outputs (src_len, batch, encoder_dim)
            mask: Mask for padding positions (batch, src_len), True = ignore
        
        Returns:
            context: Weighted context vector (batch, encoder_dim)
            attention_weights: Attention distribution (batch, src_len)
        """
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        
        # Reshape for attention computation
        # encoder_outputs: (src_len, batch, encoder_dim) -> (batch, src_len, encoder_dim)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # Transform encoder outputs
        # (batch, src_len, encoder_dim) -> (batch, src_len, attention_dim)
        encoder_energy = self.encoder_att(encoder_outputs)
        
        # Transform decoder hidden state
        # (batch, decoder_dim) -> (batch, attention_dim)
        decoder_energy = self.decoder_att(decoder_hidden)
        
        # Add decoder energy to all encoder positions
        # (batch, 1, attention_dim) + (batch, src_len, attention_dim)
        # -> (batch, src_len, attention_dim)
        combined = torch.tanh(encoder_energy + decoder_energy.unsqueeze(1))
        
        # Compute energy scores
        # (batch, src_len, attention_dim) -> (batch, src_len, 1) -> (batch, src_len)
        energy = self.v(combined).squeeze(2)
        
        # Apply mask if provided (set padding positions to -inf)
        if mask is not None:
            energy = energy.masked_fill(mask, float('-inf'))
        
        # Compute attention weights via softmax
        attention_weights = F.softmax(energy, dim=1)
        
        # Compute context vector as weighted sum of encoder outputs
        # (batch, 1, src_len) @ (batch, src_len, encoder_dim) -> (batch, 1, encoder_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        # (batch, 1, encoder_dim) -> (batch, encoder_dim)
        context = context.squeeze(1)
        
        return context, attention_weights
