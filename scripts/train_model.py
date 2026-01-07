#!/usr/bin/env python3
"""
Phase 5: AIS Prediction Model Training

Trains a model to predict Acute Ischemic Stroke (AIS) risk using:
- HyenaDNA embeddings (variant sequence representations)
- VEP annotations (functional consequences)
- Genotype dosages (sample-level variant calls)

Architecture options:
1. Simple: Aggregated variant features → MLP classifier
2. Attention: Per-variant features → Attention pooling → MLP classifier
3. SetTransformer: Per-variant features → Set Transformer → MLP classifier

Usage:
    python scripts/train_model.py \
        --embeddings data/embeddings/variant_embeddings \
        --annotations data/variants/pruned/variants_vep_annotated.parquet \
        --dosages data/variants/pruned/chr6_pruned_dosages.parquet \
        --cohort data/cohorts/ais_cohort.parquet \
        --output models/ais_model \
        --model-type attention

"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_embeddings(embeddings_path: str) -> Dict[str, np.ndarray]:
    """
    Load HyenaDNA embeddings from zarr or numpy files.

    Returns:
        Dict with 'ref', 'alt', 'diff' embeddings
    """
    embeddings_path = Path(embeddings_path)
    embeddings = {}

    # Try zarr first, then numpy
    for key in ['ref', 'alt', 'diff']:
        zarr_path = embeddings_path.with_suffix(f'.{key}.zarr')
        npy_path = embeddings_path.with_suffix(f'.{key}.npy')

        if zarr_path.exists():
            try:
                import zarr
                embeddings[key] = np.array(zarr.open(str(zarr_path), mode='r'))
                logger.info(f"Loaded {key} embeddings from zarr: {embeddings[key].shape}")
            except ImportError:
                logger.warning("zarr not available, trying numpy")

        if key not in embeddings and npy_path.exists():
            embeddings[key] = np.load(npy_path)
            logger.info(f"Loaded {key} embeddings from numpy: {embeddings[key].shape}")

    if not embeddings:
        raise FileNotFoundError(f"No embedding files found at {embeddings_path}")

    return embeddings


def load_annotations(annotations_path: str) -> pd.DataFrame:
    """Load VEP annotations and encode categorical features."""
    logger.info(f"Loading annotations from {annotations_path}")
    df = pd.read_parquet(annotations_path)
    logger.info(f"Loaded {len(df):,} variants with annotations")
    return df


def encode_vep_features(df: pd.DataFrame) -> np.ndarray:
    """
    Encode VEP annotations into numerical features.

    Features extracted:
    - Consequence type (one-hot)
    - SIFT score (numeric, imputed)
    - PolyPhen score (numeric, imputed)
    - Impact category (ordinal)
    - CADD score if available
    """
    features = []

    # Consequence types to encode
    consequence_types = [
        'missense_variant', 'synonymous_variant', 'intron_variant',
        'upstream_gene_variant', 'downstream_gene_variant',
        '3_prime_UTR_variant', '5_prime_UTR_variant',
        'splice_region_variant', 'splice_donor_variant', 'splice_acceptor_variant',
        'stop_gained', 'stop_lost', 'start_lost', 'frameshift_variant',
        'inframe_insertion', 'inframe_deletion', 'regulatory_region_variant',
        'intergenic_variant'
    ]

    # One-hot encode consequence types
    if 'most_severe_consequence' in df.columns:
        for ctype in consequence_types:
            features.append(
                (df['most_severe_consequence'] == ctype).astype(float).values.reshape(-1, 1)
            )
    elif 'consequence_terms' in df.columns:
        # Handle list column
        for ctype in consequence_types:
            col = df['consequence_terms'].apply(
                lambda x: 1.0 if isinstance(x, list) and ctype in x else 0.0
            )
            features.append(col.values.reshape(-1, 1))

    # Impact category (ordinal encoding)
    impact_map = {'MODIFIER': 0, 'LOW': 1, 'MODERATE': 2, 'HIGH': 3}
    if 'impact' in df.columns:
        impact_encoded = df['impact'].map(impact_map).fillna(0).values.reshape(-1, 1)
        features.append(impact_encoded)

    # SIFT score (lower = more damaging)
    if 'sift_score' in df.columns:
        sift = df['sift_score'].fillna(1.0).values.reshape(-1, 1)  # 1.0 = tolerated
        features.append(sift)

    # PolyPhen score (higher = more damaging)
    if 'polyphen_score' in df.columns:
        polyphen = df['polyphen_score'].fillna(0.0).values.reshape(-1, 1)  # 0 = benign
        features.append(polyphen)

    # CADD score if available
    if 'cadd_phred' in df.columns:
        cadd = df['cadd_phred'].fillna(0.0).values.reshape(-1, 1)
        features.append(cadd)

    if not features:
        logger.warning("No VEP features found, using empty array")
        return np.zeros((len(df), 1))

    return np.hstack(features)


def load_dosages(dosages_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load genotype dosages.

    Returns:
        Tuple of (dosage_matrix, sample_ids, variant_ids)
        dosage_matrix shape: (n_samples, n_variants)
    """
    logger.info(f"Loading dosages from {dosages_path}")
    df = pd.read_parquet(dosages_path)

    # Identify sample columns (exclude variant ID columns)
    variant_cols = ['SNP', 'CHR', 'POS', 'A1', 'A2', 'REF', 'ALT', 'variant_id']
    sample_cols = [c for c in df.columns if c not in variant_cols]

    variant_ids = df['SNP'].tolist() if 'SNP' in df.columns else df.index.tolist()

    # Transpose: we want (samples, variants)
    dosage_matrix = df[sample_cols].values.T
    sample_ids = sample_cols

    logger.info(f"Dosage matrix shape: {dosage_matrix.shape} (samples × variants)")

    return dosage_matrix, sample_ids, variant_ids


def load_cohort(cohort_path: str) -> pd.DataFrame:
    """Load cohort definition with case/control labels."""
    logger.info(f"Loading cohort from {cohort_path}")
    df = pd.read_parquet(cohort_path)

    # Identify label column
    label_cols = ['is_case', 'case', 'label', 'AIS', 'phenotype']
    label_col = None
    for col in label_cols:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError(f"No label column found. Expected one of: {label_cols}")

    # Standardize column name
    df['label'] = df[label_col].astype(int)

    n_cases = df['label'].sum()
    n_controls = len(df) - n_cases
    logger.info(f"Cohort: {n_cases:,} cases, {n_controls:,} controls")

    return df


def create_sample_features(
    dosages: np.ndarray,
    embeddings: Dict[str, np.ndarray],
    vep_features: np.ndarray,
    aggregation: str = 'weighted'
) -> np.ndarray:
    """
    Create sample-level features by aggregating variant features.

    Args:
        dosages: (n_samples, n_variants) genotype dosages
        embeddings: Dict with variant embeddings
        vep_features: (n_variants, n_vep_features)
        aggregation: 'weighted', 'mean', or 'presence'

    Returns:
        Sample feature matrix (n_samples, feature_dim)
    """
    n_samples, n_variants = dosages.shape

    # Use difference embeddings (alt - ref) as they capture variant effect
    if 'diff' in embeddings:
        variant_emb = embeddings['diff']
    elif 'alt' in embeddings and 'ref' in embeddings:
        variant_emb = embeddings['alt'] - embeddings['ref']
    else:
        variant_emb = list(embeddings.values())[0]

    # Combine embeddings and VEP features
    variant_features = np.hstack([variant_emb, vep_features])
    logger.info(f"Variant feature dimension: {variant_features.shape[1]}")

    # Aggregate across variants for each sample
    if aggregation == 'weighted':
        # Weight by dosage: sum(dosage * features) / sum(dosage)
        # This gives more weight to variants the sample carries
        sample_features = []
        for i in range(n_samples):
            weights = dosages[i]
            if weights.sum() > 0:
                weighted_feat = (weights[:, np.newaxis] * variant_features).sum(axis=0) / weights.sum()
            else:
                weighted_feat = np.zeros(variant_features.shape[1])
            sample_features.append(weighted_feat)
        sample_features = np.array(sample_features)

    elif aggregation == 'mean':
        # Simple mean of features for variants present (dosage > 0.5)
        sample_features = []
        for i in range(n_samples):
            present = dosages[i] > 0.5
            if present.sum() > 0:
                mean_feat = variant_features[present].mean(axis=0)
            else:
                mean_feat = np.zeros(variant_features.shape[1])
            sample_features.append(mean_feat)
        sample_features = np.array(sample_features)

    elif aggregation == 'presence':
        # Binary presence/absence features
        sample_features = (dosages > 0.5).astype(float)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    logger.info(f"Sample feature matrix: {sample_features.shape}")
    return sample_features


def build_mlp_model(input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
    """Build a simple MLP classifier using PyTorch."""
    import torch
    import torch.nn as nn

    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, 1))

    return nn.Sequential(*layers)


def build_attention_model(
    embedding_dim: int,
    vep_dim: int,
    hidden_dim: int = 128,
    n_heads: int = 4,
    dropout: float = 0.3
):
    """
    Build attention-based model that processes per-variant features.

    Architecture:
    1. Project variant features to hidden_dim
    2. Multi-head self-attention over variants
    3. Attention pooling to aggregate
    4. MLP classifier
    """
    import torch
    import torch.nn as nn

    class AttentionAggregator(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_heads, dropout):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            self.pool_attention = nn.Linear(hidden_dim, 1)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x, mask=None):
            # x: (batch, n_variants, feature_dim)
            # mask: (batch, n_variants) - True for valid variants

            # Project to hidden dim
            h = self.input_proj(x)  # (batch, n_variants, hidden_dim)

            # Self-attention
            if mask is not None:
                # Convert to attention mask (True = ignore)
                attn_mask = ~mask
            else:
                attn_mask = None

            h, _ = self.attention(h, h, h, key_padding_mask=attn_mask)

            # Attention pooling
            pool_weights = self.pool_attention(h).squeeze(-1)  # (batch, n_variants)
            if mask is not None:
                pool_weights = pool_weights.masked_fill(~mask, float('-inf'))
            pool_weights = torch.softmax(pool_weights, dim=-1)

            # Weighted sum
            pooled = (pool_weights.unsqueeze(-1) * h).sum(dim=1)  # (batch, hidden_dim)

            # Classify
            return self.classifier(pooled)

    input_dim = embedding_dim + vep_dim
    return AttentionAggregator(input_dim, hidden_dim, n_heads, dropout)


class AISDataset:
    """Dataset for AIS prediction."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_ids: List[str] = None
    ):
        self.features = features
        self.labels = labels
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_sklearn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = 'xgboost'
) -> Tuple[object, Dict]:
    """
    Train a sklearn-compatible model.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_type: 'xgboost', 'lightgbm', 'rf', or 'logistic'

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    if model_type == 'xgboost':
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                random_state=42,
                use_label_encoder=False,
                eval_metric='auc'
            )
        except ImportError:
            logger.warning("XGBoost not available, falling back to Random Forest")
            model_type = 'rf'

    if model_type == 'lightgbm':
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42
            )
        except ImportError:
            logger.warning("LightGBM not available, falling back to Random Forest")
            model_type = 'rf'

    if model_type == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

    if model_type == 'logistic':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )

    # Train
    logger.info(f"Training {model_type} model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        'auroc': roc_auc_score(y_val, y_pred_proba),
        'auprc': average_precision_score(y_val, y_pred_proba),
        'f1': f1_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
    }

    logger.info(f"Validation metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    return model, metrics


def train_pytorch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = 'mlp',
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = 'auto'
) -> Tuple[object, Dict]:
    """
    Train a PyTorch model.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_type: 'mlp' or 'attention'
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: 'cuda', 'cpu', or 'auto'

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Training on device: {device}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    # Data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Build model
    input_dim = X_train.shape[1]
    if model_type == 'mlp':
        model = build_mlp_model(input_dim, [256, 128, 64])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # Loss with class weighting
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_auroc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t.to(device)).squeeze()
            val_loss = criterion(val_outputs, y_val_t.to(device)).item()
            val_proba = torch.sigmoid(val_outputs).cpu().numpy()

        val_auroc = roc_auc_score(y_val, val_proba)
        scheduler.step(val_loss)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}")

        if patience_counter >= 10:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_proba = torch.sigmoid(model(X_val_t.to(device)).squeeze()).cpu().numpy()

    y_pred = (val_proba > 0.5).astype(int)

    metrics = {
        'auroc': roc_auc_score(y_val, val_proba),
        'auprc': average_precision_score(y_val, val_proba),
        'f1': f1_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
    }

    logger.info(f"Final validation metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    return model, metrics


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    model_type: str = 'xgboost'
) -> Dict:
    """
    Perform stratified k-fold cross-validation.

    Returns:
        Dict with mean and std of metrics across folds
    """
    logger.info(f"Running {n_folds}-fold cross-validation...")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\nFold {fold + 1}/{n_folds}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Train model
        if model_type in ['xgboost', 'lightgbm', 'rf', 'logistic']:
            _, metrics = train_sklearn_model(X_train, y_train, X_val, y_val, model_type)
        else:
            _, metrics = train_pytorch_model(X_train, y_train, X_val, y_val, model_type)

        all_metrics.append(metrics)

    # Aggregate metrics
    aggregated = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }

    logger.info(f"\n{'='*50}")
    logger.info("Cross-validation results:")
    for name, stats in aggregated.items():
        logger.info(f"  {name}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    return aggregated


def train_model(args):
    """Main training function."""
    logger.info("=" * 50)
    logger.info("Phase 5: AIS Prediction Model Training")
    logger.info("=" * 50)

    # Load data
    embeddings = load_embeddings(args.embeddings)
    annotations = load_annotations(args.annotations)
    dosages, sample_ids, variant_ids = load_dosages(args.dosages)
    cohort = load_cohort(args.cohort)

    # Encode VEP features
    vep_features = encode_vep_features(annotations)
    logger.info(f"VEP feature dimension: {vep_features.shape[1]}")

    # Match samples between dosage matrix and cohort
    # Find sample ID column in cohort
    id_cols = ['eid', 'IID', 'sample_id', 'FID']
    id_col = None
    for col in id_cols:
        if col in cohort.columns:
            id_col = col
            break

    if id_col is None:
        logger.warning("No sample ID column found in cohort, assuming order matches")
        matched_indices = list(range(min(len(sample_ids), len(cohort))))
        matched_cohort_indices = matched_indices
    else:
        # Convert sample IDs to strings for matching
        cohort_ids = set(cohort[id_col].astype(str))
        matched_indices = []
        matched_cohort_indices = []

        for i, sid in enumerate(sample_ids):
            sid_str = str(sid)
            if sid_str in cohort_ids:
                matched_indices.append(i)
                matched_cohort_indices.append(
                    cohort[cohort[id_col].astype(str) == sid_str].index[0]
                )

    logger.info(f"Matched {len(matched_indices)} samples between dosages and cohort")

    if len(matched_indices) == 0:
        raise ValueError("No matching samples found between dosage and cohort files!")

    # Create sample features
    matched_dosages = dosages[matched_indices]
    sample_features = create_sample_features(
        matched_dosages, embeddings, vep_features, args.aggregation
    )

    # Get labels
    labels = cohort.iloc[matched_cohort_indices]['label'].values

    logger.info(f"Final dataset: {sample_features.shape[0]} samples, {sample_features.shape[1]} features")
    logger.info(f"Class distribution: {(labels == 1).sum()} cases, {(labels == 0).sum()} controls")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sample_features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, stratify=y_train, random_state=42
    )

    logger.info(f"Train: {len(y_train)} ({(y_train == 1).sum()} cases)")
    logger.info(f"Val: {len(y_val)} ({(y_val == 1).sum()} cases)")
    logger.info(f"Test: {len(y_test)} ({(y_test == 1).sum()} cases)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    if args.model_type in ['xgboost', 'lightgbm', 'rf', 'logistic']:
        model, val_metrics = train_sklearn_model(
            X_train_scaled, y_train, X_val_scaled, y_val, args.model_type
        )
    else:
        model, val_metrics = train_pytorch_model(
            X_train_scaled, y_train, X_val_scaled, y_val,
            args.model_type, args.epochs, args.batch_size, args.learning_rate, args.device
        )

    # Test set evaluation
    logger.info("\nTest set evaluation:")
    if args.model_type in ['xgboost', 'lightgbm', 'rf', 'logistic']:
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        import torch
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            X_test_t = torch.FloatTensor(X_test_scaled).to(device)
            y_test_proba = torch.sigmoid(model(X_test_t).squeeze()).cpu().numpy()

    y_test_pred = (y_test_proba > 0.5).astype(int)

    test_metrics = {
        'auroc': roc_auc_score(y_test, y_test_proba),
        'auprc': average_precision_score(y_test, y_test_proba),
        'f1': f1_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
    }

    for name, value in test_metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_test_pred, target_names=['Control', 'Case']))

    # Cross-validation if requested
    if args.cross_validate:
        cv_metrics = cross_validate(sample_features, labels, args.n_folds, args.model_type)
    else:
        cv_metrics = None

    # Save model and results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save sklearn model
    if args.model_type in ['xgboost', 'lightgbm', 'rf', 'logistic']:
        import joblib
        joblib.dump(model, output_path / 'model.joblib')
        joblib.dump(scaler, output_path / 'scaler.joblib')
        logger.info(f"Saved model to {output_path / 'model.joblib'}")
    else:
        import torch
        torch.save(model.state_dict(), output_path / 'model.pt')
        import joblib
        joblib.dump(scaler, output_path / 'scaler.joblib')
        logger.info(f"Saved model to {output_path / 'model.pt'}")

    # Save metrics
    results = {
        'model_type': args.model_type,
        'aggregation': args.aggregation,
        'n_samples': len(labels),
        'n_features': sample_features.shape[1],
        'n_variants': vep_features.shape[0],
        'embedding_dim': list(embeddings.values())[0].shape[1],
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }
    if cv_metrics:
        results['cv_metrics'] = cv_metrics

    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved results to {output_path / 'results.json'}")

    # Summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Test AUROC: {test_metrics['auroc']:.4f}")
    logger.info(f"Test AUPRC: {test_metrics['auprc']:.4f}")
    logger.info(f"Output directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train AIS prediction model using HyenaDNA embeddings"
    )

    # Data inputs
    parser.add_argument(
        '--embeddings', required=True,
        help='Path prefix to HyenaDNA embeddings (e.g., data/embeddings/variant_embeddings)'
    )
    parser.add_argument(
        '--annotations', required=True,
        help='Path to VEP-annotated variants parquet file'
    )
    parser.add_argument(
        '--dosages', required=True,
        help='Path to genotype dosages parquet file'
    )
    parser.add_argument(
        '--cohort', required=True,
        help='Path to cohort definition parquet file with case/control labels'
    )
    parser.add_argument(
        '--output', '-o', required=True,
        help='Output directory for model and results'
    )

    # Model configuration
    parser.add_argument(
        '--model-type', default='xgboost',
        choices=['xgboost', 'lightgbm', 'rf', 'logistic', 'mlp'],
        help='Model type to train (default: xgboost)'
    )
    parser.add_argument(
        '--aggregation', default='weighted',
        choices=['weighted', 'mean', 'presence'],
        help='How to aggregate variant features per sample (default: weighted)'
    )

    # Training parameters (for neural models)
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs for neural models (default: 50)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size for neural models (default: 64)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=1e-3,
        help='Learning rate for neural models (default: 1e-3)'
    )
    parser.add_argument(
        '--device', default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device for neural models (default: auto)'
    )

    # Cross-validation
    parser.add_argument(
        '--cross-validate', action='store_true',
        help='Perform k-fold cross-validation'
    )
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of folds for cross-validation (default: 5)'
    )

    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
