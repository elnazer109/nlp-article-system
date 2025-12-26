# PubMed Seq2Seq Summarization

A PyTorch implementation of a seq2seq model with Bahdanau attention for scientific article summarization.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train with synthetic data (for testing)
python train.py --use_synthetic --epochs 5

# Train with real PubMed data
python train.py --data_dir data/

# Generate summaries
python evaluate.py --model_path checkpoints/best_model.pt --input "Your article text..."
```

## Project Structure

```
├── src/
│   ├── config.py       # Hyperparameters
│   ├── vocabulary.py   # Tokenization
│   ├── dataset.py      # Data loading
│   ├── encoder.py      # Bidirectional LSTM
│   ├── attention.py    # Bahdanau attention
│   ├── decoder.py      # LSTM with attention
│   └── seq2seq.py      # Full model
├── train.py            # Training script
└── evaluate.py         # Inference
```

## Model Architecture

- **Encoder**: Bidirectional LSTM
- **Attention**: Bahdanau (additive) attention
- **Decoder**: LSTM with attention context
