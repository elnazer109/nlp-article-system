"""
Configuration and hyperparameters for the Seq2Seq model.
"""

import torch

class Config:
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Vocabulary
    MAX_VOCAB_SIZE = 50000
    MIN_WORD_FREQ = 2
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'
    UNK_TOKEN = '<UNK>'
    
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3
    
    # Model architecture
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    ATTENTION_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.5
    
    # Sequence lengths
    MAX_SOURCE_LEN = 512  # Max article length (tokens)
    MAX_TARGET_LEN = 128  # Max summary length (tokens)
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10
    TEACHER_FORCING_RATIO = 0.5
    CLIP_GRAD = 1.0
    
    # Paths
    DATA_DIR = 'data'
    CHECKPOINT_DIR = 'checkpoints'
    
    @classmethod
    def get_device_info(cls):
        if cls.DEVICE.type == 'cuda':
            return f"Using GPU: {torch.cuda.get_device_name(0)}"
        return "Using CPU"
