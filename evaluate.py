"""
Evaluation and inference script for Seq2Seq summarization model.

Usage:
    # Evaluate on test set
    python evaluate.py --model_path checkpoints/best_model.pt --data_path data/test.csv
    
    # Generate summary for custom input
    python evaluate.py --model_path checkpoints/best_model.pt --input "Your article text..."
    
    # Interactive mode
    python evaluate.py --model_path checkpoints/best_model.pt --interactive
"""

import argparse
import os
import torch

from src.config import Config
from src.vocabulary import Vocabulary
from src.dataset import load_csv_data, SyntheticDataGenerator
from src.seq2seq import Seq2Seq
from src.utils import calculate_bleu, plot_attention


def load_model_and_vocab(model_path, vocab_path, device):
    """Load trained model and vocabulary."""
    # Load vocabulary
    vocab = Vocabulary.load(vocab_path)
    
    # Create model
    model = Seq2Seq(
        src_vocab_size=len(vocab),
        trg_vocab_size=len(vocab)
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Trained for {checkpoint['epoch'] + 1} epochs, loss: {checkpoint['loss']:.4f}")
    
    return model, vocab


def summarize(model, vocab, text, device, max_len=None):
    """
    Generate summary for a given text.
    
    Args:
        model: Trained Seq2Seq model
        vocab: Vocabulary object
        text: Input article text
        device: Device to run on
        max_len: Maximum output length
    
    Returns:
        summary: Generated summary string
        attention: Attention weights (if available)
    """
    model.eval()
    max_len = max_len or Config.MAX_TARGET_LEN
    
    # Encode input
    src_indices = vocab.encode(text, max_len=Config.MAX_SOURCE_LEN, add_eos=True)
    src_tensor = torch.tensor(src_indices, dtype=torch.long, device=device).unsqueeze(1)
    
    # Generate
    with torch.no_grad():
        predictions, attention = model.generate(
            src_tensor, 
            max_len=max_len
        )
    
    # Decode output
    pred_tokens = predictions[:, 0].tolist()
    summary = vocab.decode(pred_tokens)
    
    return summary, attention


def evaluate_test_set(model, vocab, test_articles, test_summaries, device, num_samples=None):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        vocab: Vocabulary
        test_articles: List of test articles
        test_summaries: List of reference summaries
        device: Device
        num_samples: Number of samples to evaluate (None = all)
    
    Returns:
        bleu_score: BLEU score
        examples: List of (article, reference, generated) tuples
    """
    if num_samples:
        test_articles = test_articles[:num_samples]
        test_summaries = test_summaries[:num_samples]
    
    predictions = []
    references = []
    examples = []
    
    print(f"\nEvaluating on {len(test_articles)} samples...")
    
    for i, (article, ref_summary) in enumerate(zip(test_articles, test_summaries)):
        # Generate summary
        gen_summary, _ = summarize(model, vocab, article, device)
        
        # Tokenize for BLEU
        gen_tokens = gen_summary.lower().split()
        ref_tokens = ref_summary.lower().split()
        
        predictions.append(gen_tokens)
        references.append(ref_tokens)
        
        # Store example
        if len(examples) < 5:
            examples.append({
                'article': article[:200] + '...' if len(article) > 200 else article,
                'reference': ref_summary,
                'generated': gen_summary
            })
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(test_articles)}")
    
    # Calculate BLEU score
    bleu = calculate_bleu(predictions, references)
    
    return bleu, examples


def interactive_mode(model, vocab, device):
    """Interactive mode for generating summaries."""
    print("\n" + "=" * 60)
    print("Interactive Summary Generation")
    print("=" * 60)
    print("Enter article text (or 'quit' to exit):")
    print("-" * 60)
    
    while True:
        try:
            text = input("\n> ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            summary, _ = summarize(model, vocab, text, device)
            
            print("\n[Generated Summary]")
            print(summary)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main(args):
    """Main evaluation function."""
    device = Config.DEVICE
    print(Config.get_device_info())
    
    # Check for model
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first with: python train.py --use_synthetic")
        return
    
    # Load vocabulary
    vocab_path = os.path.join(os.path.dirname(args.model_path), 'vocab.json')
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary not found at {vocab_path}")
        return
    
    # Load model
    model, vocab = load_model_and_vocab(args.model_path, vocab_path, device)
    
    if args.interactive:
        # Interactive mode
        interactive_mode(model, vocab, device)
    
    elif args.input:
        # Single input
        print("\n[Input Article]")
        print(args.input[:500] + '...' if len(args.input) > 500 else args.input)
        
        summary, attention = summarize(model, vocab, args.input, device)
        
        print("\n[Generated Summary]")
        print(summary)
        
        # Plot attention if requested
        if args.save_attention and attention is not None:
            src_tokens = args.input.lower().split()[:50]
            trg_tokens = summary.split()[:50]
            plot_attention(
                attention[:len(trg_tokens), :len(src_tokens)].cpu().numpy(),
                src_tokens, trg_tokens,
                args.save_attention
            )
    
    elif args.data_path:
        # Evaluate on test set
        if args.use_synthetic:
            test_articles, test_summaries = SyntheticDataGenerator.generate_dataset(500)
        else:
            test_articles, test_summaries = load_csv_data(args.data_path)
        
        bleu, examples = evaluate_test_set(
            model, vocab, test_articles, test_summaries, 
            device, args.num_samples
        )
        
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"BLEU Score: {bleu:.4f}")
        
        print("\n" + "-" * 60)
        print("Sample Outputs:")
        print("-" * 60)
        
        for i, ex in enumerate(examples):
            print(f"\n[Example {i + 1}]")
            print(f"Article: {ex['article']}")
            print(f"Reference: {ex['reference']}")
            print(f"Generated: {ex['generated']}")
    
    else:
        # Demo with synthetic data
        print("\nNo input provided. Running demo with synthetic data...")
        
        # Generate a sample
        article = SyntheticDataGenerator.generate_article(num_sentences=5)
        
        print("\n[Sample Article]")
        print(article)
        
        summary, _ = summarize(model, vocab, article, device)
        
        print("\n[Generated Summary]")
        print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Seq2Seq Summarization Model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt',
                        help='Path to trained model checkpoint')
    
    # Input arguments
    parser.add_argument('--input', type=str, default=None,
                        help='Article text to summarize')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to test CSV file')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic data for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate')
    
    # Mode arguments
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    # Output arguments
    parser.add_argument('--save_attention', type=str, default=None,
                        help='Path to save attention plot')
    
    args = parser.parse_args()
    main(args)
