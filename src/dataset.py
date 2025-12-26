"""
Dataset classes for loading and preprocessing data.
Includes synthetic data generation for testing without real data.
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.config import Config
from src.vocabulary import Vocabulary


class SummarizationDataset(Dataset):
    """Dataset for article-summary pairs."""
    
    def __init__(self, articles, summaries, src_vocab, trg_vocab, 
                 max_src_len=None, max_trg_len=None):
        """
        Args:
            articles: List of article texts
            summaries: List of summary texts
            src_vocab: Vocabulary for source (articles)
            trg_vocab: Vocabulary for target (summaries)
            max_src_len: Max source sequence length
            max_trg_len: Max target sequence length
        """
        self.articles = articles
        self.summaries = summaries
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_src_len = max_src_len or Config.MAX_SOURCE_LEN
        self.max_trg_len = max_trg_len or Config.MAX_TARGET_LEN
        
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = self.articles[idx]
        summary = self.summaries[idx]
        
        # Encode article (source)
        src_indices = self.src_vocab.encode(
            article, 
            max_len=self.max_src_len,
            add_eos=True
        )
        
        # Encode summary (target) - add SOS and EOS
        trg_indices = self.trg_vocab.encode(
            summary,
            max_len=self.max_trg_len,
            add_sos=True,
            add_eos=True
        )
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'trg': torch.tensor(trg_indices, dtype=torch.long),
            'src_len': len(src_indices),
            'trg_len': len(trg_indices)
        }


def collate_fn(batch):
    """
    Collate function for DataLoader with dynamic padding.
    
    Returns tensors of shape (seq_len, batch_size) for seq2seq.
    """
    # Sort by source length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x['src_len'], reverse=True)
    
    src_tensors = [item['src'] for item in batch]
    trg_tensors = [item['trg'] for item in batch]
    src_lens = [item['src_len'] for item in batch]
    trg_lens = [item['trg_len'] for item in batch]
    
    # Pad sequences
    src_padded = pad_sequence(src_tensors, padding_value=Config.PAD_IDX)
    trg_padded = pad_sequence(trg_tensors, padding_value=Config.PAD_IDX)
    
    return {
        'src': src_padded,  # (src_len, batch)
        'trg': trg_padded,  # (trg_len, batch)
        'src_lens': torch.tensor(src_lens),
        'trg_lens': torch.tensor(trg_lens)
    }


class SyntheticDataGenerator:
    """
    Generate synthetic article-summary pairs for testing.
    Mimics the structure of scientific text without real content.
    """
    
    # Scientific vocabulary for synthetic data
    SUBJECTS = [
        "the study", "the research", "the experiment", "the analysis",
        "the investigation", "the trial", "the review", "the model",
        "the method", "the approach", "the technique", "the framework"
    ]
    
    VERBS = [
        "demonstrates", "shows", "reveals", "indicates", "suggests",
        "confirms", "establishes", "validates", "examines", "analyzes",
        "investigates", "evaluates", "explores", "proposes", "develops"
    ]
    
    OBJECTS = [
        "significant improvements", "novel findings", "positive results",
        "statistical significance", "promising outcomes", "key insights",
        "important correlations", "notable patterns", "critical factors",
        "underlying mechanisms", "potential applications", "new approaches"
    ]
    
    CONTEXTS = [
        "in clinical trials", "in laboratory conditions", "in real-world scenarios",
        "across multiple datasets", "using advanced methods", "with high accuracy",
        "for medical applications", "in biological systems", "for drug discovery",
        "in patient populations", "under controlled conditions", "with novel techniques"
    ]
    
    CONCLUSIONS = [
        "These findings have important implications for future research.",
        "The results support the proposed hypothesis.",
        "Further studies are needed to validate these findings.",
        "This approach shows promise for clinical applications.",
        "The method outperforms existing techniques.",
        "These insights advance our understanding of the field."
    ]
    
    @classmethod
    def generate_sentence(cls):
        """Generate a single synthetic sentence."""
        subject = random.choice(cls.SUBJECTS)
        verb = random.choice(cls.VERBS)
        obj = random.choice(cls.OBJECTS)
        context = random.choice(cls.CONTEXTS)
        return f"{subject} {verb} {obj} {context}"
    
    @classmethod
    def generate_article(cls, num_sentences=None):
        """Generate a synthetic article."""
        if num_sentences is None:
            num_sentences = random.randint(10, 20)
        
        sentences = [cls.generate_sentence() for _ in range(num_sentences)]
        sentences.append(random.choice(cls.CONCLUSIONS))
        return ". ".join(sentences) + "."
    
    @classmethod
    def generate_summary(cls, num_sentences=None):
        """Generate a synthetic summary (shorter than article)."""
        if num_sentences is None:
            num_sentences = random.randint(2, 4)
        
        sentences = [cls.generate_sentence() for _ in range(num_sentences)]
        return ". ".join(sentences) + "."
    
    @classmethod
    def generate_dataset(cls, num_samples=1000):
        """
        Generate a synthetic dataset.
        
        Returns:
            articles: List of article texts
            summaries: List of summary texts
        """
        articles = []
        summaries = []
        
        for _ in range(num_samples):
            articles.append(cls.generate_article())
            summaries.append(cls.generate_summary())
        
        return articles, summaries


def load_csv_data(filepath, article_col='article', abstract_col='abstract', max_samples=None):
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to CSV file
        article_col: Name of article column
        abstract_col: Name of abstract column
        max_samples: Maximum number of samples to load
    
    Returns:
        articles: List of article texts
        summaries: List of summary texts
    """
    import pandas as pd
    
    df = pd.read_csv(filepath)
    
    # Try to find the right columns
    if article_col not in df.columns:
        # Common alternatives
        for col in ['text', 'document', 'content', 'body']:
            if col in df.columns:
                article_col = col
                break
    
    if abstract_col not in df.columns:
        for col in ['summary', 'abstract', 'highlights']:
            if col in df.columns:
                abstract_col = col
                break
    
    articles = df[article_col].fillna('').astype(str).tolist()
    summaries = df[abstract_col].fillna('').astype(str).tolist()
    
    if max_samples:
        articles = articles[:max_samples]
        summaries = summaries[:max_samples]
    
    print(f"Loaded {len(articles)} samples from {filepath}")
    return articles, summaries


def create_dataloaders(train_articles, train_summaries, 
                       val_articles=None, val_summaries=None,
                       batch_size=None, share_vocab=True):
    """
    Create DataLoaders for training and validation.
    
    Args:
        train_articles: Training articles
        train_summaries: Training summaries
        val_articles: Validation articles (optional)
        val_summaries: Validation summaries (optional)
        batch_size: Batch size
        share_vocab: Whether to share vocabulary between source and target
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (or None)
        src_vocab: Source vocabulary
        trg_vocab: Target vocabulary
    """
    batch_size = batch_size or Config.BATCH_SIZE
    
    # Build vocabularies
    src_vocab = Vocabulary()
    src_vocab.build_from_texts(train_articles)
    
    if share_vocab:
        # Use same vocab for source and target
        all_texts = train_articles + train_summaries
        src_vocab = Vocabulary()
        src_vocab.build_from_texts(all_texts)
        trg_vocab = src_vocab
    else:
        trg_vocab = Vocabulary()
        trg_vocab.build_from_texts(train_summaries)
    
    # Create training dataset
    train_dataset = SummarizationDataset(
        train_articles, train_summaries,
        src_vocab, trg_vocab
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create validation dataset if provided
    val_loader = None
    if val_articles and val_summaries:
        val_dataset = SummarizationDataset(
            val_articles, val_summaries,
            src_vocab, trg_vocab
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    return train_loader, val_loader, src_vocab, trg_vocab
