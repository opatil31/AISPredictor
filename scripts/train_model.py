#!/usr/bin/env python3
"""
Phase 5: AIS Prediction Model Training

Implements the hierarchical attention architecture from the implementation plan:
- Input projection with positional encoding
- Region-type attention pooling (8 region types)
- Region combination layer
- Classification head

Architecture (~100k parameters):
1. Input: dosage-scaled delta embeddings (batch, n_variants, 256)
2. Input Projection: Linear(256→64) → LayerNorm → GELU → Dropout
3. Positional Encoding: Sinusoidal based on genomic position
4. Region Attention: Learned query per region → attention → region embedding
5. Region Combination: Concat(8×64) → MLP → patient embedding
6. Classifier: LayerNorm → MLP → P(AIS)

Usage:
    python scripts/train_model.py \
        --embeddings data/embeddings/variant_embeddings \
        --annotations data/variants/pruned/variants_vep_annotated.parquet \
        --dosages data/variants/pruned/chr6_pruned_dosages.parquet \
        --cohort data/cohorts/ais_cohort.parquet \
        --output models/ais_model \
        --model-type hierarchical

"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, classification_report
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# Region type mapping (from VEP consequences)
REGION_TYPES = ['promoter', 'utr5', 'exon', 'intron', 'utr3', 'splice', 'downstream', 'other']
REGION_TYPE_MAP = {
    'upstream_gene_variant': 'promoter',
    '5_prime_UTR_variant': 'utr5',
    'missense_variant': 'exon',
    'synonymous_variant': 'exon',
    'stop_gained': 'exon',
    'stop_lost': 'exon',
    'start_lost': 'exon',
    'frameshift_variant': 'exon',
    'inframe_insertion': 'exon',
    'inframe_deletion': 'exon',
    'intron_variant': 'intron',
    '3_prime_UTR_variant': 'utr3',
    'splice_acceptor_variant': 'splice',
    'splice_donor_variant': 'splice',
    'splice_region_variant': 'splice',
    'downstream_gene_variant': 'downstream',
    'intergenic_variant': 'other',
    'regulatory_region_variant': 'other',
}


def read_zarr_v3_array(zarr_path: Path) -> np.ndarray:
    """
    Read a zarr v3 format array manually.

    Zarr v3 uses zarr.json for metadata and stores chunks in subdirectories.
    This function provides compatibility when zarr library is v2.
    Handles zstd compression which is common in zarr v3.
    """
    import json

    zarr_json_path = zarr_path / 'zarr.json'
    if not zarr_json_path.exists():
        raise FileNotFoundError(f"zarr.json not found at {zarr_path}")

    with open(zarr_json_path, 'r') as f:
        metadata = json.load(f)

    logger.info(f"Zarr v3 metadata: {json.dumps(metadata, indent=2)[:500]}...")

    # Extract array metadata
    shape = metadata.get('shape', [])
    chunks = metadata.get('chunk_grid', {}).get('configuration', {}).get('chunk_shape', shape)
    dtype_info = metadata.get('data_type', 'float32')
    codecs = metadata.get('codecs', [])

    # Check for compression
    compression = None
    for codec in codecs:
        codec_name = codec.get('name', '')
        if codec_name in ['zstd', 'gzip', 'blosc', 'lz4']:
            compression = codec_name
            break

    logger.info(f"Compression: {compression}")

    # Handle dtype
    if isinstance(dtype_info, dict):
        dtype = dtype_info.get('name', 'float32')
    else:
        dtype = str(dtype_info)

    # Map zarr dtype names to numpy
    dtype_map = {
        'float32': np.float32,
        'float64': np.float64,
        'int32': np.int32,
        'int64': np.int64,
        'int8': np.int8,
        'uint8': np.uint8,
        'bool': bool,
    }
    np_dtype = dtype_map.get(dtype, np.float32)

    logger.info(f"Zarr v3 array: shape={shape}, chunks={chunks}, dtype={dtype}")

    # Setup decompressor
    decompress = None
    if compression == 'zstd':
        try:
            import zstandard
            dctx = zstandard.ZstdDecompressor()
            decompress = lambda data: dctx.decompress(data)
            logger.info("Using zstandard decompression")
        except ImportError:
            try:
                import zstd
                decompress = lambda data: zstd.decompress(data)
                logger.info("Using zstd decompression")
            except ImportError:
                raise ImportError("zstd compression requires 'zstandard' package. Install with: pip install zstandard")
    elif compression == 'gzip':
        import gzip
        decompress = lambda data: gzip.decompress(data)
        logger.info("Using gzip decompression")
    elif compression == 'lz4':
        try:
            import lz4.frame
            decompress = lambda data: lz4.frame.decompress(data)
            logger.info("Using lz4 decompression")
        except ImportError:
            raise ImportError("lz4 compression requires 'lz4' package. Install with: pip install lz4")

    # Initialize output array
    result = np.zeros(shape, dtype=np_dtype)

    # Find chunk directory (usually 'c' for zarr v3)
    chunk_dir = zarr_path / 'c'
    if not chunk_dir.exists():
        chunk_dir = zarr_path

    logger.info(f"Looking for chunks in: {chunk_dir}")

    # Calculate expected number of chunks
    n_chunks_per_dim = [int(np.ceil(s / c)) for s, c in zip(shape, chunks)]
    total_expected = int(np.prod(n_chunks_per_dim))
    logger.info(f"Expected chunks: {n_chunks_per_dim} = {total_expected} total")

    # Read chunks by iterating through expected indices
    chunks_read = 0
    for i in range(n_chunks_per_dim[0]):
        for j in range(n_chunks_per_dim[1]) if len(shape) > 1 else [0]:
            # Build chunk path
            if len(shape) == 1:
                chunk_path = chunk_dir / str(i)
            else:
                chunk_path = chunk_dir / str(i) / str(j)

            if not chunk_path.exists():
                continue

            try:
                # Read raw bytes
                with open(chunk_path, 'rb') as f:
                    raw_data = f.read()

                # Decompress if needed
                if decompress:
                    raw_data = decompress(raw_data)

                # Convert to numpy array
                chunk_data = np.frombuffer(raw_data, dtype=np_dtype)

                # Zarr stores chunks at full chunk size, even for edge chunks
                # First reshape to full chunk size, then slice for edges
                if len(shape) == 1:
                    # Reshape to full chunk size
                    chunk_data = chunk_data.reshape((chunks[0],))

                    # Calculate what portion of the array this chunk fills
                    start_i = i * chunks[0]
                    end_i = min(start_i + chunks[0], shape[0])
                    actual_size = end_i - start_i

                    # Slice if edge chunk
                    if actual_size < chunks[0]:
                        chunk_data = chunk_data[:actual_size]

                    result[start_i:end_i] = chunk_data
                else:
                    # Reshape to full chunk size
                    chunk_data = chunk_data.reshape((chunks[0], chunks[1]))

                    # Calculate what portion of the array this chunk fills
                    start_i = i * chunks[0]
                    end_i = min(start_i + chunks[0], shape[0])
                    start_j = j * chunks[1]
                    end_j = min(start_j + chunks[1], shape[1])

                    actual_rows = end_i - start_i
                    actual_cols = end_j - start_j

                    # Slice if edge chunk
                    chunk_slice = chunk_data[:actual_rows, :actual_cols]

                    result[start_i:end_i, start_j:end_j] = chunk_slice

                chunks_read += 1

                if chunks_read % 50 == 0:
                    logger.info(f"Read {chunks_read}/{total_expected} chunks...")

            except Exception as e:
                logger.warning(f"Error reading chunk {chunk_path}: {e}")
                continue

    logger.info(f"Read {chunks_read}/{total_expected} chunks from zarr v3 store")

    if chunks_read == 0:
        raise ValueError(f"No chunks could be read from {zarr_path}. "
                        f"Install zstandard if using zstd compression: pip install zstandard")

    return result


def load_embeddings(embeddings_path: str) -> Dict[str, np.ndarray]:
    """Load HyenaDNA embeddings from zarr or numpy files."""
    embeddings_path = Path(embeddings_path)
    embeddings = {}

    for key in ['ref', 'alt', 'diff']:
        zarr_path = embeddings_path.with_suffix(f'.{key}.zarr')
        npy_path = embeddings_path.with_suffix(f'.{key}.npy')

        if zarr_path.exists():
            try:
                import zarr
                embeddings[key] = np.array(zarr.open(str(zarr_path), mode='r'))
                logger.info(f"Loaded {key} embeddings from zarr: {embeddings[key].shape}")
            except ImportError:
                pass

        if key not in embeddings and npy_path.exists():
            embeddings[key] = np.load(npy_path)
            logger.info(f"Loaded {key} embeddings from numpy: {embeddings[key].shape}")

    if not embeddings:
        raise FileNotFoundError(f"No embedding files found at {embeddings_path}")

    return embeddings


def get_region_type(consequence: str) -> int:
    """Map VEP consequence to region type index."""
    region = REGION_TYPE_MAP.get(consequence, 'other')
    return REGION_TYPES.index(region)


def build_hierarchical_model(
    embedding_dim: int = 256,
    d_model: int = 64,
    n_heads: int = 2,
    n_regions: int = 8,
    hidden_dim: int = 128,
    dropout: float = 0.3
):
    """
    Build the hierarchical attention model as specified in the implementation plan.

    Architecture:
    - Input projection: Linear(256→64) → LayerNorm → GELU → Dropout
    - Positional encoding: Sinusoidal
    - Region attention pooling: Learned query per region
    - Region combination: MLP
    - Classifier head

    ~100k parameters total
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding based on genomic position."""

        def __init__(self, d_model: int, max_position: int = 200_000_000):
            super().__init__()
            self.d_model = d_model
            self.scale = 1e8  # Normalize positions

        def forward(self, positions: torch.Tensor) -> torch.Tensor:
            """
            Args:
                positions: (batch, n_variants) genomic positions
            Returns:
                (batch, n_variants, d_model) positional embeddings
            """
            positions = positions.float() / self.scale

            # Create dimension indices
            dim = torch.arange(self.d_model, device=positions.device).float()
            div_term = torch.exp(dim * (-math.log(10000.0) / self.d_model))

            # Compute sinusoidal encoding
            positions = positions.unsqueeze(-1)  # (batch, n_variants, 1)
            pe = torch.zeros(*positions.shape[:-1], self.d_model, device=positions.device)
            pe[..., 0::2] = torch.sin(positions * div_term[0::2])
            pe[..., 1::2] = torch.cos(positions * div_term[1::2])

            return pe

    class RegionAttentionPooling(nn.Module):
        """Attention pooling for each region type with learned queries."""

        def __init__(self, d_model: int, n_regions: int, n_heads: int, dropout: float):
            super().__init__()
            self.d_model = d_model
            self.n_regions = n_regions
            self.n_heads = n_heads

            # Learned query vector for each region type
            self.region_queries = nn.Parameter(torch.randn(n_regions, d_model))

            # Key and Value projections
            self.key_proj = nn.Linear(d_model, d_model)
            self.value_proj = nn.Linear(d_model, d_model)

            # Learned embedding for empty regions
            self.empty_embedding = nn.Parameter(torch.randn(n_regions, d_model))

            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(d_model)

        def forward(
            self,
            x: torch.Tensor,
            region_types: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Args:
                x: (batch, n_variants, d_model) variant embeddings
                region_types: (batch, n_variants) region type indices (0-7)
                mask: (batch, n_variants) True for valid variants

            Returns:
                (batch, n_regions, d_model) region embeddings
            """
            batch_size = x.shape[0]
            device = x.device

            # Project to keys and values
            keys = self.key_proj(x)  # (batch, n_variants, d_model)
            values = self.value_proj(x)  # (batch, n_variants, d_model)

            # Initialize region embeddings
            region_embeddings = torch.zeros(batch_size, self.n_regions, self.d_model, device=device)

            for r in range(self.n_regions):
                # Get query for this region
                query = self.region_queries[r]  # (d_model,)

                # Create mask for variants in this region
                region_mask = (region_types == r)  # (batch, n_variants)
                if mask is not None:
                    region_mask = region_mask & mask

                # Check if any variants in this region
                has_variants = region_mask.any(dim=1)  # (batch,)

                if has_variants.any():
                    # Compute attention scores
                    scores = torch.einsum('d,bnd->bn', query, keys) / self.scale  # (batch, n_variants)

                    # Mask out variants not in this region (use large negative instead of -inf for stability)
                    scores = scores.masked_fill(~region_mask, -1e9)

                    # Softmax attention
                    attn = F.softmax(scores, dim=-1)  # (batch, n_variants)
                    attn = self.dropout(attn)

                    # Handle any remaining NaN (shouldn't happen but safety check)
                    attn = torch.nan_to_num(attn, nan=0.0)

                    # Weighted sum of values
                    region_emb = torch.einsum('bn,bnd->bd', attn, values)  # (batch, d_model)

                    # Use empty embedding for samples with no variants in this region
                    region_emb = torch.where(
                        has_variants.unsqueeze(-1),
                        region_emb,
                        self.empty_embedding[r].unsqueeze(0).expand(batch_size, -1)
                    )

                    region_embeddings[:, r] = region_emb
                else:
                    # All samples have no variants in this region
                    region_embeddings[:, r] = self.empty_embedding[r].unsqueeze(0).expand(batch_size, -1)

            return region_embeddings

    class HierarchicalAISModel(nn.Module):
        """
        Hierarchical attention model for AIS prediction.

        Architecture from implementation plan:
        1. Input projection: Linear(256→64) → LayerNorm → GELU → Dropout
        2. Add positional encoding
        3. Region attention pooling with learned queries
        4. Region combination: Concat → MLP
        5. Classification head
        """

        def __init__(
            self,
            embedding_dim: int = 256,
            d_model: int = 64,
            n_heads: int = 2,
            n_regions: int = 8,
            hidden_dim: int = 128,
            dropout: float = 0.3
        ):
            super().__init__()

            self.embedding_dim = embedding_dim
            self.d_model = d_model
            self.n_regions = n_regions

            # Input projection (256 → 64)
            self.input_proj = nn.Sequential(
                nn.Linear(embedding_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )

            # Positional encoding
            self.pos_encoding = PositionalEncoding(d_model)

            # Region attention pooling
            self.region_attention = RegionAttentionPooling(d_model, n_regions, n_heads, dropout)

            # Region combination layer
            # Input: concatenated region embeddings (8 × 64 = 512)
            # Output: patient embedding (64)
            self.region_combination = nn.Sequential(
                nn.Linear(n_regions * d_model, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model)
            )

            # Classification head
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, 2)  # [control, case] logits
            )

            self._init_weights()

        def _init_weights(self):
            """Initialize weights."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(
            self,
            embeddings: torch.Tensor,
            positions: torch.Tensor,
            region_types: torch.Tensor,
            dosages: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Forward pass.

            Args:
                embeddings: (batch, n_variants, embedding_dim) delta embeddings
                positions: (batch, n_variants) genomic positions
                region_types: (batch, n_variants) region type indices
                dosages: (batch, n_variants) genotype dosages
                mask: (batch, n_variants) True for valid variants

            Returns:
                (batch, 2) logits for [control, case]
            """
            # Scale embeddings by dosage
            x = embeddings * dosages.unsqueeze(-1)  # (batch, n_variants, embedding_dim)

            # Input projection
            x = self.input_proj(x)  # (batch, n_variants, d_model)

            # Add positional encoding
            pos_emb = self.pos_encoding(positions)  # (batch, n_variants, d_model)
            x = x + pos_emb

            # Region attention pooling
            region_emb = self.region_attention(x, region_types, mask)  # (batch, n_regions, d_model)

            # Flatten region embeddings
            region_flat = region_emb.view(region_emb.shape[0], -1)  # (batch, n_regions * d_model)

            # Region combination
            patient_emb = self.region_combination(region_flat)  # (batch, d_model)

            # Classification
            logits = self.classifier(patient_emb)  # (batch, 2)

            return logits

        def get_region_embeddings(
            self,
            embeddings: torch.Tensor,
            positions: torch.Tensor,
            region_types: torch.Tensor,
            dosages: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """Get intermediate region embeddings for attribution."""
            x = embeddings * dosages.unsqueeze(-1)
            x = self.input_proj(x)
            pos_emb = self.pos_encoding(positions)
            x = x + pos_emb
            region_emb = self.region_attention(x, region_types, mask)
            return region_emb

    return HierarchicalAISModel(embedding_dim, d_model, n_heads, n_regions, hidden_dim, dropout)


def build_baseline_model(embedding_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.3):
    """Build simple mean-pooling baseline model."""
    import torch.nn as nn

    class MeanPoolingBaseline(nn.Module):
        """Baseline: mean pooling + MLP classifier."""

        def __init__(self, embedding_dim, hidden_dim, dropout):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2)
            )

        def forward(self, embeddings, positions, region_types, dosages, mask=None):
            # Dosage-weighted mean pooling
            x = embeddings * dosages.unsqueeze(-1)
            if mask is not None:
                x = x * mask.unsqueeze(-1).float()
                weights = dosages * mask.float()
            else:
                weights = dosages

            # Weighted mean
            weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            pooled = (x * dosages.unsqueeze(-1)).sum(dim=1) / weight_sum.squeeze(-1).unsqueeze(-1).clamp(min=1e-8)

            return self.classifier(pooled)

    return MeanPoolingBaseline(embedding_dim, hidden_dim, dropout)


class AISDataset:
    """Dataset for AIS prediction with hierarchical model."""

    def __init__(
        self,
        embeddings: np.ndarray,
        positions: np.ndarray,
        region_types: np.ndarray,
        dosages: np.ndarray,
        labels: np.ndarray,
        sample_ids: List[str] = None
    ):
        self.embeddings = embeddings
        self.positions = positions
        self.region_types = region_types
        self.dosages = dosages
        self.labels = labels
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'embeddings': self.embeddings,  # Shared across samples
            'positions': self.positions,
            'region_types': self.region_types,
            'dosages': self.dosages[idx],
            'label': self.labels[idx]
        }


def train_model_cv(
    embeddings: np.ndarray,
    positions: np.ndarray,
    region_types: np.ndarray,
    dosages: np.ndarray,
    labels: np.ndarray,
    model_type: str = 'hierarchical',
    n_folds: int = 5,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.1,
    patience: int = 15,
    device: str = 'auto',
    use_fp16: bool = False,
    use_bf16: bool = False,
    use_compile: bool = False
) -> Tuple[object, Dict]:
    """
    Train model with stratified k-fold cross-validation.

    Returns:
        Tuple of (best_model, metrics_dict)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Training on device: {device}")

    # Setup mixed precision training
    use_amp = (use_fp16 or use_bf16) and device == 'cuda'
    if use_amp:
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)  # BF16 doesn't need scaling
        logger.info(f"Using mixed precision training with {'BF16' if use_bf16 else 'FP16'}")
    else:
        amp_dtype = torch.float32
        scaler = None

    # Convert to tensors
    embeddings_t = torch.FloatTensor(embeddings)
    positions_t = torch.LongTensor(positions)
    region_types_t = torch.LongTensor(region_types)

    # Check for NaN/Inf in inputs
    if torch.isnan(embeddings_t).any():
        nan_count = torch.isnan(embeddings_t).sum().item()
        logger.warning(f"Found {nan_count} NaN values in embeddings, replacing with 0")
        embeddings_t = torch.nan_to_num(embeddings_t, nan=0.0)
    if torch.isinf(embeddings_t).any():
        inf_count = torch.isinf(embeddings_t).sum().item()
        logger.warning(f"Found {inf_count} Inf values in embeddings, clamping")
        embeddings_t = torch.clamp(embeddings_t, -1e6, 1e6)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Check for missing genotypes in dosages and report statistics
    missing_mask = np.isnan(dosages)
    n_missing = missing_mask.sum()
    if n_missing > 0:
        total_elements = dosages.size
        missing_pct = 100 * n_missing / total_elements

        # Per-variant missingness
        variant_missing_rate = missing_mask.mean(axis=0)
        max_variant_missing = variant_missing_rate.max() * 100
        variants_with_missing = (variant_missing_rate > 0).sum()

        # Per-sample missingness
        sample_missing_rate = missing_mask.mean(axis=1)
        max_sample_missing = sample_missing_rate.max() * 100
        samples_with_missing = (sample_missing_rate > 0).sum()

        logger.warning(f"Found {n_missing:,} missing genotypes ({missing_pct:.2f}% of matrix)")
        logger.warning(f"  Variants with missing data: {variants_with_missing:,}/{dosages.shape[1]:,}")
        logger.warning(f"  Max variant missingness: {max_variant_missing:.2f}%")
        logger.warning(f"  Samples with missing data: {samples_with_missing:,}/{dosages.shape[0]:,}")
        logger.warning(f"  Max sample missingness: {max_sample_missing:.2f}%")
        logger.info("Will use per-variant mean imputation (computed from training set each fold)")

    fold_metrics = []
    best_model_state = None
    best_auroc = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(dosages, labels)):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold + 1}/{n_folds}")
        logger.info(f"{'='*50}")

        # Split data
        train_dosages = torch.FloatTensor(dosages[train_idx])
        train_labels = torch.FloatTensor(labels[train_idx])
        val_dosages = torch.FloatTensor(dosages[val_idx])
        val_labels = torch.FloatTensor(labels[val_idx])

        # Per-variant mean imputation for missing genotypes
        # Use training set means to avoid data leakage
        if torch.isnan(train_dosages).any() or torch.isnan(val_dosages).any():
            # Compute per-variant mean from training data only
            # nanmean computes mean ignoring NaN values
            variant_means = torch.nanmean(train_dosages, dim=0)

            # Handle variants where ALL training samples are missing (rare edge case)
            # Use 0 (assumes rare variant with no observed alt alleles)
            all_missing_mask = torch.isnan(variant_means)
            if all_missing_mask.any():
                n_all_missing = all_missing_mask.sum().item()
                logger.warning(f"  {n_all_missing} variants have no observed genotypes in training, using 0")
                variant_means[all_missing_mask] = 0.0

            # Count imputed values
            train_nan_count = torch.isnan(train_dosages).sum().item()
            val_nan_count = torch.isnan(val_dosages).sum().item()

            # Impute training set: replace NaN with per-variant mean
            train_nan_mask = torch.isnan(train_dosages)
            train_dosages = torch.where(
                train_nan_mask,
                variant_means.unsqueeze(0).expand_as(train_dosages),
                train_dosages
            )

            # Impute validation set: use TRAINING means (no data leakage)
            val_nan_mask = torch.isnan(val_dosages)
            val_dosages = torch.where(
                val_nan_mask,
                variant_means.unsqueeze(0).expand_as(val_dosages),
                val_dosages
            )

            logger.info(f"  Imputed {train_nan_count:,} training + {val_nan_count:,} validation genotypes "
                       f"using per-variant mean from training set")

        # Create data loaders
        train_dataset = TensorDataset(train_dosages, train_labels)
        val_dataset = TensorDataset(val_dosages, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Build model
        embedding_dim = embeddings.shape[1]
        if model_type == 'hierarchical':
            model = build_hierarchical_model(embedding_dim=embedding_dim)
        else:
            model = build_baseline_model(embedding_dim=embedding_dim)
        model = model.to(device)

        # Apply torch.compile for faster training (PyTorch 2.0+)
        if use_compile and device == 'cuda' and fold == 0:  # Only compile once
            try:
                logger.info("Compiling model with torch.compile...")
                model = torch.compile(model, mode='reduce-overhead')
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        # Move shared tensors to device
        emb_device = embeddings_t.to(device)
        pos_device = positions_t.to(device)
        reg_device = region_types_t.to(device)

        # Class weights for imbalanced data
        n_pos = (labels[train_idx] == 1).sum()
        n_neg = (labels[train_idx] == 0).sum()
        class_weights = torch.FloatTensor([n_pos / len(train_idx), n_neg / len(train_idx)]).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Training loop
        best_fold_auroc = 0
        best_fold_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0

            for batch_dosages, batch_labels in train_loader:
                batch_dosages = batch_dosages.to(device)
                batch_labels = batch_labels.long().to(device)

                # Expand embeddings for batch
                batch_size_actual = batch_dosages.shape[0]
                batch_emb = emb_device.unsqueeze(0).expand(batch_size_actual, -1, -1)
                batch_pos = pos_device.unsqueeze(0).expand(batch_size_actual, -1)
                batch_reg = reg_device.unsqueeze(0).expand(batch_size_actual, -1)

                optimizer.zero_grad()

                # Mixed precision forward pass
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        logits = model(batch_emb, batch_pos, batch_reg, batch_dosages)
                        loss = criterion(logits, batch_labels)

                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                else:
                    logits = model(batch_emb, batch_pos, batch_reg, batch_dosages)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # Check for NaN loss
                loss_val = loss.item()
                if np.isnan(loss_val):
                    logger.warning(f"NaN loss detected at epoch {epoch}, resetting model")
                    # Reinitialize model weights
                    for layer in model.modules():
                        if hasattr(layer, 'reset_parameters'):
                            layer.reset_parameters()
                    break
                train_loss += loss_val

            scheduler.step()

            # Validation
            model.eval()
            val_probs = []
            val_true = []

            with torch.no_grad():
                for batch_dosages, batch_labels in val_loader:
                    batch_dosages = batch_dosages.to(device)
                    batch_size_actual = batch_dosages.shape[0]
                    batch_emb = emb_device.unsqueeze(0).expand(batch_size_actual, -1, -1)
                    batch_pos = pos_device.unsqueeze(0).expand(batch_size_actual, -1)
                    batch_reg = reg_device.unsqueeze(0).expand(batch_size_actual, -1)

                    if use_amp:
                        with torch.amp.autocast('cuda', dtype=amp_dtype):
                            logits = model(batch_emb, batch_pos, batch_reg, batch_dosages)
                    else:
                        logits = model(batch_emb, batch_pos, batch_reg, batch_dosages)
                    probs = torch.softmax(logits, dim=-1)[:, 1]

                    # Handle NaN in probabilities
                    probs_np = probs.float().cpu().numpy()
                    probs_np = np.nan_to_num(probs_np, nan=0.5)  # Replace NaN with 0.5
                    val_probs.extend(probs_np)
                    val_true.extend(batch_labels.numpy())

            # Check for NaN in validation probs
            val_probs = np.array(val_probs)
            if np.isnan(val_probs).any():
                logger.warning(f"NaN in validation probs at epoch {epoch}, replacing with 0.5")
                val_probs = np.nan_to_num(val_probs, nan=0.5)

            val_auroc = roc_auc_score(val_true, val_probs)

            if val_auroc > best_fold_auroc:
                best_fold_auroc = val_auroc
                best_fold_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
                           f"Val AUROC: {val_auroc:.4f} (best: {best_fold_auroc:.4f})")

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model for this fold
        model.load_state_dict(best_fold_state)

        # Final evaluation on validation set
        model.eval()
        val_probs = []
        val_true = []

        with torch.no_grad():
            for batch_dosages, batch_labels in val_loader:
                batch_dosages = batch_dosages.to(device)
                batch_size_actual = batch_dosages.shape[0]
                batch_emb = emb_device.unsqueeze(0).expand(batch_size_actual, -1, -1)
                batch_pos = pos_device.unsqueeze(0).expand(batch_size_actual, -1)
                batch_reg = reg_device.unsqueeze(0).expand(batch_size_actual, -1)

                logits = model(batch_emb, batch_pos, batch_reg, batch_dosages)
                probs = torch.softmax(logits, dim=-1)[:, 1]

                # Handle NaN in probabilities
                probs_np = probs.float().cpu().numpy()
                probs_np = np.nan_to_num(probs_np, nan=0.5)
                val_probs.extend(probs_np)
                val_true.extend(batch_labels.numpy())

        val_probs = np.array(val_probs)
        val_probs = np.nan_to_num(val_probs, nan=0.5)  # Final safety check
        val_true = np.array(val_true)
        val_pred = (val_probs > 0.5).astype(int)

        fold_metric = {
            'auroc': roc_auc_score(val_true, val_probs),
            'auprc': average_precision_score(val_true, val_probs),
            'f1': f1_score(val_true, val_pred),
            'precision': precision_score(val_true, val_pred, zero_division=0),
            'recall': recall_score(val_true, val_pred, zero_division=0),
        }
        fold_metrics.append(fold_metric)

        logger.info(f"Fold {fold + 1} - AUROC: {fold_metric['auroc']:.4f}, AUPRC: {fold_metric['auprc']:.4f}")

        # Track best model across folds
        if fold_metric['auroc'] > best_auroc:
            best_auroc = fold_metric['auroc']
            best_model_state = best_fold_state

    # Aggregate metrics
    aggregated = {}
    for key in fold_metrics[0].keys():
        values = [m[key] for m in fold_metrics]
        aggregated[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values
        }

    logger.info(f"\n{'='*50}")
    logger.info("Cross-validation Results:")
    logger.info(f"{'='*50}")
    for name, stats in aggregated.items():
        logger.info(f"  {name}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Build final model with best weights
    if model_type == 'hierarchical':
        final_model = build_hierarchical_model(embedding_dim=embeddings.shape[1])
    else:
        final_model = build_baseline_model(embedding_dim=embeddings.shape[1])
    final_model.load_state_dict(best_model_state)

    return final_model, aggregated


def train_model(args):
    """Main training function."""
    logger.info("=" * 50)
    logger.info("Phase 5: AIS Prediction Model Training")
    logger.info("=" * 50)

    # Load embeddings
    embeddings = load_embeddings(args.embeddings)

    # Use difference embeddings
    if 'diff' in embeddings:
        delta_embeddings = embeddings['diff']
    elif 'alt' in embeddings and 'ref' in embeddings:
        delta_embeddings = embeddings['alt'] - embeddings['ref']
    else:
        delta_embeddings = list(embeddings.values())[0]

    logger.info(f"Delta embeddings shape: {delta_embeddings.shape}")

    # Load annotations
    logger.info(f"Loading annotations from {args.annotations}")
    annotations = pd.read_parquet(args.annotations)
    logger.info(f"Loaded {len(annotations)} variants")

    # Get positions and region types
    positions = annotations['POS'].values if 'POS' in annotations.columns else np.arange(len(annotations))

    # Map consequences to region types
    if 'most_severe_consequence' in annotations.columns:
        region_types = np.array([
            get_region_type(c) for c in annotations['most_severe_consequence']
        ])
    elif 'Consequence' in annotations.columns:
        region_types = np.array([
            get_region_type(c.split(',')[0]) for c in annotations['Consequence']
        ])
    else:
        logger.warning("No consequence column found, using 'other' for all variants")
        region_types = np.full(len(annotations), REGION_TYPES.index('other'))

    # Region type distribution
    logger.info("Region type distribution:")
    for i, rtype in enumerate(REGION_TYPES):
        count = (region_types == i).sum()
        logger.info(f"  {rtype}: {count:,} ({100*count/len(region_types):.1f}%)")

    # Load dosages (supports both parquet and zarr formats)
    logger.info(f"Loading dosages from {args.dosages}")
    dosages_path = Path(args.dosages).resolve()  # Get absolute path
    logger.info(f"Resolved path: {dosages_path}")

    def find_zarr_store(path: Path) -> Path:
        """Find the actual zarr store, handling nested structures."""
        # Check if this path itself is a zarr store
        zarr_indicators = ['.zarray', '.zgroup', '.zattrs', 'zarr.json']
        for indicator in zarr_indicators:
            if (path / indicator).exists():
                return path

        # Check for nested zarr store with same name (dosages.zarr/dosages.zarr/)
        if path.is_dir():
            nested = path / path.name
            if nested.is_dir():
                for indicator in zarr_indicators:
                    if (nested / indicator).exists():
                        logger.info(f"Found nested zarr store at: {nested}")
                        return nested

            # Check for any .zarr subdirectory
            for subdir in path.iterdir():
                if subdir.is_dir() and subdir.suffix == '.zarr':
                    for indicator in zarr_indicators:
                        if (subdir / indicator).exists():
                            logger.info(f"Found zarr store in subdirectory: {subdir}")
                            return subdir

        return path

    # Check if it's a zarr store
    is_zarr = str(args.dosages).endswith('.zarr') or dosages_path.is_dir()

    if is_zarr:
        # Zarr format: (n_samples, n_variants) array
        try:
            import zarr

            # Find the actual zarr store (handles nested structures)
            actual_zarr_path = find_zarr_store(dosages_path)
            zarr_path_str = str(actual_zarr_path)
            logger.info(f"Opening zarr store at: {zarr_path_str}")
            logger.info(f"Path exists: {actual_zarr_path.exists()}")
            logger.info(f"Path contents: {list(actual_zarr_path.iterdir()) if actual_zarr_path.is_dir() else 'not a dir'}")

            # Check for zarr v3 format (zarr.json instead of .zarray)
            zarr_json_path = actual_zarr_path / 'zarr.json'
            is_zarr_v3 = zarr_json_path.exists()

            if is_zarr_v3:
                logger.info("Detected zarr v3 format (zarr.json found)")
                # Try to read zarr v3 format manually
                dosages = read_zarr_v3_array(actual_zarr_path)
                logger.info(f"Loaded zarr v3 dosages: {dosages.shape}")
            else:
                # Try different methods to open zarr v2 store
                store = None
                dosages = None

                # Method 1: Try zarr.open with explicit path string
                try:
                    store = zarr.open(zarr_path_str, mode='r')
                    logger.info(f"Opened with zarr.open, type: {type(store)}")
                except Exception as e1:
                    logger.warning(f"zarr.open failed: {e1}")

                    # Method 2: Try DirectoryStore (zarr v2)
                    try:
                        from zarr.storage import DirectoryStore
                        dir_store = DirectoryStore(zarr_path_str)
                        store = zarr.open(store=dir_store, mode='r')
                        logger.info(f"Opened with DirectoryStore, type: {type(store)}")
                    except Exception as e2:
                        logger.warning(f"DirectoryStore failed: {e2}")

                        # Method 3: Try zarr.open_array directly
                        try:
                            store = zarr.open_array(zarr_path_str, mode='r')
                            logger.info(f"Opened with zarr.open_array, type: {type(store)}")
                        except Exception as e3:
                            raise ValueError(f"Could not open zarr store. Tried multiple methods:\n"
                                           f"  zarr.open: {e1}\n"
                                           f"  DirectoryStore: {e2}\n"
                                           f"  open_array: {e3}\n"
                                           f"\nIf using zarr v3 format, ensure zarr.json exists or upgrade zarr: pip install 'zarr>=3.0'")

                # Handle both array and group formats
                if isinstance(store, zarr.Array):
                    dosages = np.array(store)
                elif hasattr(store, 'keys'):  # Group-like
                    # If it's a group, look for the dosage array inside
                    keys = list(store.keys())
                    logger.info(f"Zarr group keys: {keys}")
                    if 'dosages' in store:
                        dosages = np.array(store['dosages'])
                    elif 'data' in store:
                        dosages = np.array(store['data'])
                    elif len(keys) > 0:
                        # Use the first key
                        logger.info(f"Using first key: {keys[0]}")
                        dosages = np.array(store[keys[0]])
                    else:
                        raise ValueError(f"No arrays found in zarr group. Keys: {keys}")
                else:
                    dosages = np.array(store)

                logger.info(f"Loaded zarr v2 dosages: {dosages.shape}")

            # Try to load sample IDs from accompanying file (check multiple locations)
            sample_id_candidates = [
                dosages_path / 'sample_ids.txt',           # Inside the folder
                dosages_path.parent / 'sample_ids.txt',    # Next to the folder
                actual_zarr_path.parent / 'sample_ids.txt',  # Next to actual zarr
                Path(str(dosages_path).replace('.zarr', '_sample_ids.txt')),
                Path(str(dosages_path).replace('.zarr', '.sample_ids.txt')),
            ]

            sample_ids_path = None
            for candidate in sample_id_candidates:
                if candidate.exists():
                    sample_ids_path = candidate
                    break

            if sample_ids_path:
                with open(sample_ids_path, 'r') as f:
                    sample_ids = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(sample_ids)} sample IDs from {sample_ids_path}")
            else:
                sample_ids = [f"sample_{i}" for i in range(dosages.shape[0])]
                logger.warning(f"No sample_ids.txt found, using generic IDs")
        except ImportError:
            raise ImportError("zarr required for .zarr dosages. Install with: pip install zarr")
    else:
        # Parquet format: rows=variants, columns=samples
        dosages_df = pd.read_parquet(args.dosages)
        variant_cols = ['SNP', 'CHR', 'POS', 'A1', 'A2', 'REF', 'ALT', 'variant_id']
        sample_cols = [c for c in dosages_df.columns if c not in variant_cols]
        dosages = dosages_df[sample_cols].values.T  # (n_samples, n_variants)
        sample_ids = sample_cols

    logger.info(f"Dosages shape: {dosages.shape}")

    # ========== VARIANT SUBSETTING ==========
    # The dosage matrix may contain all variants (e.g., 546k) while we only have
    # embeddings for filtered variants (e.g., 92k). We need to subset the dosage
    # matrix to match the variants in our annotations.

    n_dosage_variants = dosages.shape[1]
    n_annotation_variants = len(annotations)

    if n_dosage_variants != n_annotation_variants:
        logger.info(f"Variant count mismatch: dosages has {n_dosage_variants}, annotations has {n_annotation_variants}")
        logger.info("Attempting to subset dosage matrix to match annotations...")

        # Try to load variant IDs for the dosage matrix
        variant_ids_dosage = None

        if is_zarr:
            # Look for variant_ids.txt file
            variant_id_candidates = [
                actual_zarr_path / 'variant_ids.txt',
                actual_zarr_path.parent / 'variant_ids.txt',
                dosages_path / 'variant_ids.txt',
                dosages_path.parent / 'variant_ids.txt',
                Path(str(dosages_path).replace('.zarr', '_variant_ids.txt')),
                Path(str(dosages_path).replace('.zarr', '.variant_ids.txt')),
            ]

            for candidate in variant_id_candidates:
                if candidate.exists():
                    with open(candidate, 'r') as f:
                        variant_ids_dosage = [line.strip() for line in f if line.strip()]
                    logger.info(f"Loaded {len(variant_ids_dosage)} variant IDs from {candidate}")
                    break

        # Construct variant IDs from annotations (chr:pos:ref:alt format)
        def make_variant_id(row):
            chrom = str(row.get('CHR', row.get('CHROM', row.get('#CHROM', ''))))
            pos = str(row.get('POS', ''))
            ref = str(row.get('REF', row.get('A2', '')))
            alt = str(row.get('ALT', row.get('A1', '')))
            return f"{chrom}:{pos}:{ref}:{alt}"

        annotation_variant_ids = annotations.apply(make_variant_id, axis=1).tolist()
        logger.info(f"Constructed {len(annotation_variant_ids)} variant IDs from annotations")
        logger.info(f"Sample annotation variant IDs: {annotation_variant_ids[:3]}")

        if variant_ids_dosage is not None:
            # Match by variant ID
            logger.info(f"Sample dosage variant IDs: {variant_ids_dosage[:3]}")

            # Create mapping from dosage variant ID to column index
            dosage_id_to_idx = {vid: idx for idx, vid in enumerate(variant_ids_dosage)}

            # Also create mapping with swapped alleles (REF/ALT vs A1/A2 order may differ)
            # and position-only mapping as fallback
            dosage_id_swapped = {}
            dosage_pos_to_idx = {}
            for idx, vid in enumerate(variant_ids_dosage):
                parts = vid.split(':')
                if len(parts) >= 4:
                    # Create swapped version: chr:pos:alt:ref -> chr:pos:ref:alt
                    swapped = f"{parts[0]}:{parts[1]}:{parts[3]}:{parts[2]}"
                    dosage_id_swapped[swapped] = idx
                    # Position-only key
                    pos_key = f"{parts[0]}:{parts[1]}"
                    if pos_key not in dosage_pos_to_idx:
                        dosage_pos_to_idx[pos_key] = idx

            # Find matching indices
            matched_variant_indices = []
            unmatched_count = 0
            match_by_swap = 0
            match_by_pos = 0
            for ann_id in annotation_variant_ids:
                if ann_id in dosage_id_to_idx:
                    matched_variant_indices.append(dosage_id_to_idx[ann_id])
                elif ann_id in dosage_id_swapped:
                    # Match with swapped alleles
                    matched_variant_indices.append(dosage_id_swapped[ann_id])
                    match_by_swap += 1
                else:
                    # Try alternative formats (with/without chr prefix)
                    alt_id = ann_id.replace('chr', '') if ann_id.startswith('chr') else f"chr{ann_id}"
                    if alt_id in dosage_id_to_idx:
                        matched_variant_indices.append(dosage_id_to_idx[alt_id])
                    elif alt_id in dosage_id_swapped:
                        matched_variant_indices.append(dosage_id_swapped[alt_id])
                        match_by_swap += 1
                    else:
                        # Try position-only match as last resort
                        parts = ann_id.split(':')
                        pos_key = f"{parts[0]}:{parts[1]}" if len(parts) >= 2 else None
                        if pos_key and pos_key in dosage_pos_to_idx:
                            matched_variant_indices.append(dosage_pos_to_idx[pos_key])
                            match_by_pos += 1
                        else:
                            matched_variant_indices.append(-1)  # Mark as unmatched
                            unmatched_count += 1

            if match_by_swap > 0:
                logger.info(f"Matched {match_by_swap} variants by swapped allele order (REF/ALT vs A1/A2)")
            if match_by_pos > 0:
                logger.info(f"Matched {match_by_pos} variants by position only")
            if unmatched_count > 0:
                logger.warning(f"{unmatched_count} variants in annotations not found in dosage matrix")

            # Filter out unmatched variants
            valid_indices = [i for i, idx in enumerate(matched_variant_indices) if idx >= 0]
            valid_dosage_indices = [matched_variant_indices[i] for i in valid_indices]

            if len(valid_indices) == 0:
                logger.error("No matching variants found between annotations and dosage matrix!")
                logger.error("Check that variant ID formats match (chr:pos:ref:alt)")
                return

            logger.info(f"Matched {len(valid_indices)} variants between annotations and dosages")

            # Subset both dosages and annotations/embeddings
            dosages = dosages[:, valid_dosage_indices]
            annotations = annotations.iloc[valid_indices].reset_index(drop=True)
            delta_embeddings = delta_embeddings[valid_indices]
            positions = positions[valid_indices]
            region_types = region_types[valid_indices]

            logger.info(f"Subsetted dosages shape: {dosages.shape}")
            logger.info(f"Subsetted embeddings shape: {delta_embeddings.shape}")
        else:
            # No variant IDs file - try to use original variant indices if stored
            # Check if annotations have an 'original_idx' or similar column
            if 'original_idx' in annotations.columns:
                variant_indices = annotations['original_idx'].values
                dosages = dosages[:, variant_indices]
                logger.info(f"Subsetted dosages using original_idx: {dosages.shape}")
            elif 'variant_idx' in annotations.columns:
                variant_indices = annotations['variant_idx'].values
                dosages = dosages[:, variant_indices]
                logger.info(f"Subsetted dosages using variant_idx: {dosages.shape}")
            else:
                logger.error("Cannot subset dosage matrix: no variant_ids.txt found and no index column in annotations")
                logger.error("Please provide a variant_ids.txt file in the dosages.zarr directory")
                logger.error("Each line should be a variant ID in format: chr:pos:ref:alt")
                logger.error(f"Expected {n_dosage_variants} lines (one per dosage column)")
                return

    # ========== END VARIANT SUBSETTING ==========

    # Load cohort
    logger.info(f"Loading cohort from {args.cohort}")
    cohort = pd.read_parquet(args.cohort)

    label_cols = ['is_case', 'case', 'label', 'AIS', 'phenotype']
    label_col = next((c for c in label_cols if c in cohort.columns), None)
    if label_col:
        cohort['label'] = cohort[label_col].astype(int)
    else:
        raise ValueError(f"No label column found. Expected one of: {label_cols}")

    # Match samples
    id_cols = ['eid', 'IID', 'sample_id', 'FID']
    id_col = next((c for c in id_cols if c in cohort.columns), None)

    if id_col:
        cohort_ids = set(cohort[id_col].astype(str))
        matched_indices = []
        matched_cohort_indices = []
        for i, sid in enumerate(sample_ids):
            if str(sid) in cohort_ids:
                matched_indices.append(i)
                matched_cohort_indices.append(
                    cohort[cohort[id_col].astype(str) == str(sid)].index[0]
                )
    else:
        matched_indices = list(range(min(len(sample_ids), len(cohort))))
        matched_cohort_indices = matched_indices

    logger.info(f"Matched {len(matched_indices)} samples")

    matched_dosages = dosages[matched_indices]
    labels = cohort.iloc[matched_cohort_indices]['label'].values

    n_cases = (labels == 1).sum()
    n_controls = (labels == 0).sum()
    logger.info(f"Cases: {n_cases}, Controls: {n_controls}")

    # Verify dimensions match
    if delta_embeddings.shape[0] != matched_dosages.shape[1]:
        logger.error(f"Dimension mismatch: embeddings {delta_embeddings.shape[0]} vs dosages {matched_dosages.shape[1]}")
        return

    # Train model
    model, cv_metrics = train_model_cv(
        embeddings=delta_embeddings,
        positions=positions,
        region_types=region_types,
        dosages=matched_dosages,
        labels=labels,
        model_type=args.model_type,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        use_fp16=args.fp16,
        use_bf16=args.bf16,
        use_compile=args.compile
    )

    # Save model and results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    import torch
    torch.save(model.state_dict(), output_path / 'model.pt')
    logger.info(f"Saved model to {output_path / 'model.pt'}")

    # Save model config
    config = {
        'model_type': args.model_type,
        'embedding_dim': int(delta_embeddings.shape[1]),
        'd_model': 64,
        'n_heads': 2,
        'n_regions': 8,
        'hidden_dim': 128,
        'dropout': 0.3,
        'n_variants': int(delta_embeddings.shape[0]),
        'n_samples': len(labels),
        'n_cases': int(n_cases),
        'n_controls': int(n_controls),
    }

    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save CV metrics
    results = {
        'cv_metrics': cv_metrics,
        'config': config,
    }

    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    # Save variant metadata for attribution
    variant_meta = {
        'positions': positions.tolist(),
        'region_types': region_types.tolist(),
        'region_type_names': REGION_TYPES,
    }
    with open(output_path / 'variant_metadata.json', 'w') as f:
        json.dump(variant_meta, f)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\nModel parameters: {n_params:,}")

    logger.info("")
    logger.info("=" * 50)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Parameters: {n_params:,}")
    logger.info(f"CV AUROC: {cv_metrics['auroc']['mean']:.4f} ± {cv_metrics['auroc']['std']:.4f}")
    logger.info(f"CV AUPRC: {cv_metrics['auprc']['mean']:.4f} ± {cv_metrics['auprc']['std']:.4f}")
    logger.info(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train hierarchical attention model for AIS prediction"
    )

    # Data inputs
    parser.add_argument('--embeddings', required=True,
                       help='Path prefix to HyenaDNA embeddings')
    parser.add_argument('--annotations', required=True,
                       help='Path to VEP-annotated variants parquet file')
    parser.add_argument('--dosages', required=True,
                       help='Path to genotype dosages parquet file')
    parser.add_argument('--cohort', required=True,
                       help='Path to cohort definition parquet file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for model and results')

    # Model configuration
    parser.add_argument('--model-type', default='hierarchical',
                       choices=['hierarchical', 'baseline'],
                       help='Model type (default: hierarchical)')

    # Training parameters
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Max training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                       help='Weight decay (default: 0.1)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device (default: auto)')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 mixed precision training (faster on H100/H200)')
    parser.add_argument('--bf16', action='store_true',
                       help='Use BF16 mixed precision training (recommended for H100/H200)')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile for faster training (PyTorch 2.0+)')

    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
