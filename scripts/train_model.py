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

                    # Mask out variants not in this region
                    scores = scores.masked_fill(~region_mask, float('-inf'))

                    # Softmax attention
                    attn = F.softmax(scores, dim=-1)  # (batch, n_variants)
                    attn = self.dropout(attn)

                    # Handle NaN from all -inf (no variants in region)
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

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

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

                train_loss += loss.item()

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

                    val_probs.extend(probs.cpu().numpy())
                    val_true.extend(batch_labels.numpy())

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

                val_probs.extend(probs.cpu().numpy())
                val_true.extend(batch_labels.numpy())

        val_probs = np.array(val_probs)
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

    # Check if it's a zarr store (folder with .zarray, .zgroup, or .zattrs)
    is_zarr = (
        str(args.dosages).endswith('.zarr') or
        (dosages_path.is_dir() and (
            (dosages_path / '.zarray').exists() or
            (dosages_path / '.zgroup').exists() or
            (dosages_path / '.zattrs').exists()
        ))
    )

    if is_zarr:
        # Zarr format: (n_samples, n_variants) array
        try:
            import zarr
            logger.info(f"Opening zarr store at: {dosages_path}")
            store = zarr.open(str(dosages_path), mode='r')

            # Handle both array and group formats
            if isinstance(store, zarr.Array):
                dosages = np.array(store)
            elif isinstance(store, zarr.Group):
                # If it's a group, look for the dosage array inside
                if 'dosages' in store:
                    dosages = np.array(store['dosages'])
                elif 'data' in store:
                    dosages = np.array(store['data'])
                else:
                    # Use the first array in the group
                    arrays = [k for k in store.keys() if isinstance(store[k], zarr.Array)]
                    if arrays:
                        logger.info(f"Found arrays in zarr group: {arrays}")
                        dosages = np.array(store[arrays[0]])
                    else:
                        raise ValueError(f"No arrays found in zarr group. Keys: {list(store.keys())}")
            else:
                dosages = np.array(store)

            logger.info(f"Loaded zarr dosages: {dosages.shape}")

            # Try to load sample IDs from accompanying file
            sample_ids_path = dosages_path.parent / 'sample_ids.txt'
            if not sample_ids_path.exists():
                sample_ids_path = Path(str(dosages_path).replace('.zarr', '_sample_ids.txt'))
            if not sample_ids_path.exists():
                sample_ids_path = Path(str(dosages_path).replace('.zarr', '.sample_ids.txt'))

            if sample_ids_path.exists():
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
