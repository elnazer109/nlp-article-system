"""
Vocabulary class for building word-to-index mappings.
"""

from collections import Counter
from src.config import Config


class Vocabulary:
    def __init__(self):
        self.word2idx = {
            Config.PAD_TOKEN: Config.PAD_IDX,
            Config.SOS_TOKEN: Config.SOS_IDX,
            Config.EOS_TOKEN: Config.EOS_IDX,
            Config.UNK_TOKEN: Config.UNK_IDX,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()
        
    def __len__(self):
        return len(self.word2idx)
    
    def add_word(self, word):
        """Add a word to the vocabulary."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def build_from_texts(self, texts, min_freq=None, max_size=None):
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of strings (already tokenized or not)
            min_freq: Minimum word frequency to include
            max_size: Maximum vocabulary size
        """
        min_freq = min_freq or Config.MIN_WORD_FREQ
        max_size = max_size or Config.MAX_VOCAB_SIZE
        
        # Count word frequencies
        for text in texts:
            if isinstance(text, str):
                words = text.lower().split()
            else:
                words = text
            self.word_freq.update(words)
        
        # Filter by frequency and add to vocab
        sorted_words = sorted(
            self.word_freq.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for word, freq in sorted_words:
            if freq < min_freq:
                break
            if len(self.word2idx) >= max_size:
                break
            self.add_word(word)
        
        print(f"Vocabulary built: {len(self)} words")
        return self
    
    def encode(self, text, max_len=None, add_sos=False, add_eos=False):
        """
        Convert text to list of indices.
        
        Args:
            text: String or list of tokens
            max_len: Maximum sequence length (truncate/pad)
            add_sos: Add start-of-sequence token
            add_eos: Add end-of-sequence token
        """
        if isinstance(text, str):
            words = text.lower().split()
        else:
            words = text
            
        indices = []
        
        if add_sos:
            indices.append(Config.SOS_IDX)
            
        for word in words:
            idx = self.word2idx.get(word, Config.UNK_IDX)
            indices.append(idx)
            
        if add_eos:
            indices.append(Config.EOS_IDX)
        
        # Truncate if needed
        if max_len:
            indices = indices[:max_len]
            
        return indices
    
    def decode(self, indices, skip_special=True):
        """
        Convert list of indices back to text.
        
        Args:
            indices: List of token indices
            skip_special: Skip PAD, SOS, EOS tokens
        """
        special_indices = {Config.PAD_IDX, Config.SOS_IDX, Config.EOS_IDX}
        
        words = []
        for idx in indices:
            if skip_special and idx in special_indices:
                if idx == Config.EOS_IDX:
                    break  # Stop at EOS
                continue
            words.append(self.idx2word.get(idx, Config.UNK_TOKEN))
            
        return ' '.join(words)
    
    def save(self, path):
        """Save vocabulary to file."""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': self.word2idx,
                'word_freq': dict(self.word_freq)
            }, f)
        print(f"Vocabulary saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load vocabulary from file."""
        import json
        vocab = cls()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        vocab.word_freq = Counter(data.get('word_freq', {}))
        print(f"Vocabulary loaded: {len(vocab)} words")
        return vocab
