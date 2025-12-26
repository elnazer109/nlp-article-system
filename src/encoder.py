"""
Bidirectional LSTM Encoder for Seq2Seq model.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    """
    Bidirectional LSTM Encoder.
    
    Takes a sequence of token indices and produces:
    - encoder_outputs: All hidden states from both directions
    - hidden: Final hidden state (concatenated from both directions)
    - cell: Final cell state (concatenated from both directions)
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 num_layers=2, dropout=0.5, padding_idx=0):
        """
        Args:
            vocab_size: Size of source vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            padding_idx: Index of padding token
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=False  # (seq_len, batch, features)
        )
        
        # Linear layers to combine bidirectional states for decoder
        # Decoder expects hidden_dim, but we have hidden_dim * 2 from bidirectional
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_lens=None):
        """
        Forward pass.
        
        Args:
            src: Source sequence (seq_len, batch)
            src_lens: Lengths of each sequence in batch (for packing)
        
        Returns:
            encoder_outputs: (seq_len, batch, hidden_dim * 2)
            hidden: (num_layers, batch, hidden_dim)
            cell: (num_layers, batch, hidden_dim)
        """
        # Embed tokens
        # src: (seq_len, batch) -> embedded: (seq_len, batch, embedding_dim)
        embedded = self.dropout(self.embedding(src))
        
        # Pack sequence if lengths provided (more efficient)
        if src_lens is not None:
            # Ensure src_lens is on CPU for pack_padded_sequence
            src_lens_cpu = src_lens.cpu()
            packed = pack_padded_sequence(embedded, src_lens_cpu, enforce_sorted=True)
            packed_outputs, (hidden, cell) = self.lstm(packed)
            # Unpack
            encoder_outputs, _ = pad_packed_sequence(packed_outputs)
        else:
            encoder_outputs, (hidden, cell) = self.lstm(embedded)
        
        # encoder_outputs: (seq_len, batch, hidden_dim * 2)
        # hidden: (num_layers * 2, batch, hidden_dim)
        # cell: (num_layers * 2, batch, hidden_dim)
        
        # Combine bidirectional hidden states
        # Take the last layer's forward and backward hidden states
        # hidden[-2]: last forward, hidden[-1]: last backward
        batch_size = src.shape[1]
        
        # Reshape hidden and cell to separate directions
        # From (num_layers * 2, batch, hidden_dim) 
        # To (num_layers, 2, batch, hidden_dim)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
        
        # Concatenate forward and backward for each layer
        # Then project to decoder dimension
        # (num_layers, batch, hidden_dim * 2) -> (num_layers, batch, hidden_dim)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))
        
        return encoder_outputs, hidden, cell
