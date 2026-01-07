#!/usr/bin/env python3
"""
Phase 4: HyenaDNA Embedding Generation

Generates embeddings for variant sequences using HyenaDNA model.

HyenaDNA is a long-range genomic foundation model that can process
sequences up to 1M tokens. We use it to generate embeddings that
capture the functional context of each variant.

Usage:
    python scripts/generate_embeddings.py \
        --sequences data/sequences/variant_sequences.parquet \
        --output data/embeddings/variant_embeddings.zarr \
        --model-name hyenadna-small-32k \
        --batch-size 32

Models available:
    - hyenadna-tiny-1k (fastest, 1k context)
    - hyenadna-small-32k (balanced, 32k context)
    - hyenadna-medium-160k (160k context)
    - hyenadna-medium-450k (450k context)
    - hyenadna-large-1m (largest, 1M context)
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# HyenaDNA model configurations
HYENADNA_MODELS = {
    'hyenadna-tiny-1k': {
        'pretrained_model_name': 'LongSafari/hyenadna-tiny-1k-seqlen-hf',
        'max_length': 1024,
        'embedding_dim': 128,
    },
    'hyenadna-small-32k': {
        'pretrained_model_name': 'LongSafari/hyenadna-small-32k-seqlen-hf',
        'max_length': 32768,
        'embedding_dim': 256,
    },
    'hyenadna-medium-160k': {
        'pretrained_model_name': 'LongSafari/hyenadna-medium-160k-seqlen-hf',
        'max_length': 160000,
        'embedding_dim': 256,
    },
    'hyenadna-medium-450k': {
        'pretrained_model_name': 'LongSafari/hyenadna-medium-450k-seqlen-hf',
        'max_length': 450000,
        'embedding_dim': 256,
    },
    'hyenadna-large-1m': {
        'pretrained_model_name': 'LongSafari/hyenadna-large-1m-seqlen-hf',
        'max_length': 1000000,
        'embedding_dim': 256,
    },
}


def load_hyenadna_model(model_name: str, device: str = 'auto'):
    """
    Load HyenaDNA model and tokenizer.

    Args:
        model_name: One of the HYENADNA_MODELS keys
        device: 'cuda', 'cpu', or 'auto'

    Returns:
        Tuple of (model, tokenizer, config)
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError(
            "transformers and torch required. Install with:\n"
            "pip install transformers torch"
        )

    if model_name not in HYENADNA_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(HYENADNA_MODELS.keys())}")

    config = HYENADNA_MODELS[model_name]
    pretrained_name = config['pretrained_model_name']

    logger.info(f"Loading model: {pretrained_name}")

    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_name,
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        pretrained_name,
        trust_remote_code=True
    )

    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded. Embedding dim: {config['embedding_dim']}, Max length: {config['max_length']}")

    return model, tokenizer, config, device


def generate_embeddings_batch(
    model,
    tokenizer,
    sequences: List[str],
    max_length: int,
    device: str,
    pooling: str = 'mean'
) -> np.ndarray:
    """
    Generate embeddings for a batch of sequences.

    Args:
        model: HyenaDNA model
        tokenizer: HyenaDNA tokenizer
        sequences: List of DNA sequences
        max_length: Maximum sequence length
        device: Device to use
        pooling: 'mean', 'cls', or 'last'

    Returns:
        Array of embeddings (batch_size, embedding_dim)
    """
    import torch

    # Tokenize
    inputs = tokenizer(
        sequences,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Get hidden states
    hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

    # Apply pooling
    if pooling == 'mean':
        # Mean pooling over sequence length
        attention_mask = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
        mask = attention_mask.unsqueeze(-1).float()
        embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
    elif pooling == 'cls':
        # Use first token
        embeddings = hidden_states[:, 0, :]
    elif pooling == 'last':
        # Use last token
        embeddings = hidden_states[:, -1, :]
    else:
        raise ValueError(f"Unknown pooling: {pooling}")

    return embeddings.cpu().numpy()


def generate_embeddings(args):
    """Main function to generate embeddings for all variants."""
    logger.info("=" * 50)
    logger.info("Phase 4: HyenaDNA Embedding Generation")
    logger.info("=" * 50)

    # Load sequences
    logger.info(f"Loading sequences from {args.sequences}")
    df = pd.read_parquet(args.sequences)
    logger.info(f"Loaded {len(df):,} variants")

    # Check for sequence columns
    if 'ref_sequence' not in df.columns or 'alt_sequence' not in df.columns:
        logger.error("Missing sequence columns. Run construct_sequences.py first.")
        return

    # Filter out empty sequences
    valid_mask = (df['ref_sequence'] != '') & (df['alt_sequence'] != '')
    n_valid = valid_mask.sum()
    logger.info(f"Valid sequences: {n_valid:,}/{len(df):,}")

    if n_valid == 0:
        logger.error("No valid sequences found!")
        return

    # Load model
    model, tokenizer, config, device = load_hyenadna_model(args.model_name, args.device)

    # Prepare output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embedding_dim = config['embedding_dim']
    max_length = min(config['max_length'], args.max_length) if args.max_length else config['max_length']

    # Initialize storage
    try:
        import zarr
        use_zarr = True
        ref_embeddings = zarr.open(
            str(output_path.with_suffix('.ref.zarr')),
            mode='w',
            shape=(len(df), embedding_dim),
            chunks=(min(1000, len(df)), embedding_dim),
            dtype='float32'
        )
        alt_embeddings = zarr.open(
            str(output_path.with_suffix('.alt.zarr')),
            mode='w',
            shape=(len(df), embedding_dim),
            chunks=(min(1000, len(df)), embedding_dim),
            dtype='float32'
        )
    except ImportError:
        use_zarr = False
        ref_embeddings = np.zeros((len(df), embedding_dim), dtype=np.float32)
        alt_embeddings = np.zeros((len(df), embedding_dim), dtype=np.float32)

    # Generate embeddings in batches
    batch_size = args.batch_size
    n_batches = (n_valid + batch_size - 1) // batch_size

    logger.info(f"Generating embeddings in {n_batches} batches of {batch_size}...")
    logger.info(f"Max sequence length: {max_length}")
    logger.info(f"Pooling strategy: {args.pooling}")

    valid_indices = df.index[valid_mask].tolist()

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_valid)
        batch_indices = valid_indices[start:end]

        # Get sequences for this batch
        ref_seqs = df.loc[batch_indices, 'ref_sequence'].tolist()
        alt_seqs = df.loc[batch_indices, 'alt_sequence'].tolist()

        # Generate embeddings
        try:
            ref_emb = generate_embeddings_batch(
                model, tokenizer, ref_seqs, max_length, device, args.pooling
            )
            alt_emb = generate_embeddings_batch(
                model, tokenizer, alt_seqs, max_length, device, args.pooling
            )

            # Store embeddings
            for i, idx in enumerate(batch_indices):
                ref_embeddings[idx] = ref_emb[i]
                alt_embeddings[idx] = alt_emb[i]

        except Exception as e:
            logger.warning(f"Batch {batch_idx + 1} failed: {e}")
            # Fill with zeros for failed batch
            for idx in batch_indices:
                ref_embeddings[idx] = np.zeros(embedding_dim, dtype=np.float32)
                alt_embeddings[idx] = np.zeros(embedding_dim, dtype=np.float32)

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            logger.info(f"Processed {batch_idx + 1}/{n_batches} batches ({end:,}/{n_valid:,} variants)")

    # Save embeddings
    if not use_zarr:
        np.save(output_path.with_suffix('.ref.npy'), ref_embeddings)
        np.save(output_path.with_suffix('.alt.npy'), alt_embeddings)
        logger.info(f"Saved embeddings to {output_path.with_suffix('.ref.npy')}")
        logger.info(f"Saved embeddings to {output_path.with_suffix('.alt.npy')}")
    else:
        logger.info(f"Saved embeddings to {output_path.with_suffix('.ref.zarr')}")
        logger.info(f"Saved embeddings to {output_path.with_suffix('.alt.zarr')}")

    # Also save difference embeddings (alt - ref) which capture variant effect
    diff_embeddings = np.array(alt_embeddings) - np.array(ref_embeddings)

    if use_zarr:
        diff_zarr = zarr.open(
            str(output_path.with_suffix('.diff.zarr')),
            mode='w',
            shape=diff_embeddings.shape,
            chunks=(min(1000, len(df)), embedding_dim),
            dtype='float32'
        )
        diff_zarr[:] = diff_embeddings
        logger.info(f"Saved difference embeddings to {output_path.with_suffix('.diff.zarr')}")
    else:
        np.save(output_path.with_suffix('.diff.npy'), diff_embeddings)
        logger.info(f"Saved difference embeddings to {output_path.with_suffix('.diff.npy')}")

    # Save metadata
    metadata = {
        'model_name': args.model_name,
        'embedding_dim': embedding_dim,
        'max_length': max_length,
        'pooling': args.pooling,
        'n_variants': len(df),
        'n_valid': n_valid,
    }

    import json
    with open(output_path.with_suffix('.meta.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("")
    logger.info("=" * 50)
    logger.info("EMBEDDING GENERATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Variants processed: {n_valid:,}")
    logger.info(f"Output files:")
    logger.info(f"  Reference embeddings: {output_path.with_suffix('.ref.zarr' if use_zarr else '.ref.npy')}")
    logger.info(f"  Alternate embeddings: {output_path.with_suffix('.alt.zarr' if use_zarr else '.alt.npy')}")
    logger.info(f"  Difference embeddings: {output_path.with_suffix('.diff.zarr' if use_zarr else '.diff.npy')}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HyenaDNA embeddings for variant sequences"
    )
    parser.add_argument(
        '--sequences', required=True,
        help='Path to sequences parquet file (from construct_sequences.py)'
    )
    parser.add_argument(
        '--output', '-o', required=True,
        help='Output path prefix for embedding files'
    )
    parser.add_argument(
        '--model-name', default='hyenadna-small-32k',
        choices=list(HYENADNA_MODELS.keys()),
        help='HyenaDNA model to use (default: hyenadna-small-32k)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for inference (default: 32)'
    )
    parser.add_argument(
        '--max-length', type=int, default=None,
        help='Override maximum sequence length (default: model max)'
    )
    parser.add_argument(
        '--device', default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)'
    )
    parser.add_argument(
        '--pooling', default='mean',
        choices=['mean', 'cls', 'last'],
        help='Pooling strategy for sequence embedding (default: mean)'
    )

    args = parser.parse_args()
    generate_embeddings(args)


if __name__ == "__main__":
    main()
