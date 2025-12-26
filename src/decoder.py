"""
LSTM Decoder with Attention for Seq2Seq model.
"""

import torch
import torch.nn as nn

from src.attention import BahdanauAttention


class Decoder(nn.Module):
    """
    LSTM Decoder with Bahdanau Attention.
    
    At each step:
    1. Embed the input token
    2. Compute attention over encoder outputs using previous hidden state
    3. Concatenate embedding with context vector
    4. Pass through LSTM
    5. Predict next token
    """
    
    def __init__(self, vocab_size, embedding_dim, encoder_dim, 
                 hidden_dim, attention_dim, num_layers=2, 
                 dropout=0.5, padding_idx=0):
        """
        Args:
            vocab_size: Size of target vocabulary
            embedding_dim: Dimension of embeddings
            encoder_dim: Dimension of encoder outputs (hidden_dim * 2 for bidirectional)
            hidden_dim: Dimension of decoder hidden state
            attention_dim: Dimension of attention layer
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            padding_idx: Index of padding token
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        # Attention mechanism
        self.attention = BahdanauAttention(
            encoder_dim=encoder_dim,
            decoder_dim=hidden_dim,
            attention_dim=attention_dim
        )
        
        # LSTM: input is [embedding; context]
        self.lstm = nn.LSTM(
            input_size=embedding_dim + encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False
        )
        
        # Output projection: [hidden; context; embedding] -> vocab
        self.fc_out = nn.Linear(
            hidden_dim + encoder_dim + embedding_dim, 
            vocab_size
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        """
        Forward pass for a single decoding step.
        
        Args:
            input_token: Input token indices (batch,)
            hidden: Previous hidden state (num_layers, batch, hidden_dim)
            cell: Previous cell state (num_layers, batch, hidden_dim)
            encoder_outputs: All encoder outputs (src_len, batch, encoder_dim)
            mask: Mask for padding (batch, src_len)
        
        Returns:
            output: Vocabulary scores (batch, vocab_size)
            hidden: Updated hidden state
            cell: Updated cell state
            attention_weights: Attention distribution (batch, src_len)
        """
        # input_token: (batch,) -> (1, batch)
        input_token = input_token.unsqueeze(0)
        
        # Embed input token
        # (1, batch) -> (1, batch, embedding_dim)
        embedded = self.dropout(self.embedding(input_token))
        
        # Compute attention using top layer hidden state
        # hidden[-1]: (batch, hidden_dim)
        context, attention_weights = self.attention(
            hidden[-1], 
            encoder_outputs, 
            mask
        )
        
        # Concatenate embedding with context
        # (1, batch, embedding_dim) cat (1, batch, encoder_dim)
        # -> (1, batch, embedding_dim + encoder_dim)
        lstm_input = torch.cat([embedded, context.unsqueeze(0)], dim=2)
        
        # LSTM step
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # lstm_output: (1, batch, hidden_dim) -> (batch, hidden_dim)
        lstm_output = lstm_output.squeeze(0)
        
        # Concatenate for output prediction
        # [hidden; context; embedding]
        embedded = embedded.squeeze(0)  # (batch, embedding_dim)
        output_input = torch.cat([lstm_output, context, embedded], dim=1)
        
        # Predict next token
        output = self.fc_out(output_input)
        
        return output, hidden, cell, attention_weights
