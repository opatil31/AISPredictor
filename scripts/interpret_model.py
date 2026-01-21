#!/usr/bin/env python3
"""
Phase 6: Model Interpretation & Attribution

Uses Layer Integrated Gradients (Captum) to compute per-patient attribution at:
1. Variant level - attribution per variant
2. Region level - aggregated by region type (promoter, exon, intron, etc.)
3. Gene level - aggregated across regions within each gene

As specified in the implementation plan:
- Method: Layer Integrated Gradients (Captum)
- Target Layer: Input projection (after dosage scaling)
- Baseline: Zero embedding (no variant effect)
- Steps: 50 interpolation steps

Usage:
    python scripts/interpret_model.py \
        --model models/ais_model/model.pt \
        --config models/ais_model/config.json \
        --embeddings data/embeddings/variant_embeddings \
        --annotations data/variants/pruned/variants_vep_annotated.parquet \
        --dosages data/variants/pruned/chr6_pruned_dosages.parquet \
        --cohort data/cohorts/ais_cohort.parquet \
        --output results/interpretation

"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# Region types (must match train_model.py)
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


def get_region_type(consequence: str) -> int:
    """Map VEP consequence to region type index."""
    region = REGION_TYPE_MAP.get(consequence, 'other')
    return REGION_TYPES.index(region)


def load_model_and_config(model_path: str, config_path: str):
    """Load trained model and configuration."""
    import torch

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Import model builder from train_model.py
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_model import build_hierarchical_model, build_baseline_model

    # Build model
    if config.get('model_type', 'hierarchical') == 'hierarchical':
        model = build_hierarchical_model(
            embedding_dim=config['embedding_dim'],
            d_model=config.get('d_model', 64),
            n_heads=config.get('n_heads', 2),
            n_regions=config.get('n_regions', 8),
            hidden_dim=config.get('hidden_dim', 128),
            dropout=config.get('dropout', 0.3)
        )
    else:
        model = build_baseline_model(
            embedding_dim=config['embedding_dim'],
            hidden_dim=config.get('hidden_dim', 128),
            dropout=config.get('dropout', 0.3)
        )

    # Load weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model type: {config.get('model_type', 'hierarchical')}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, config


def compute_integrated_gradients(
    model,
    embeddings: np.ndarray,
    positions: np.ndarray,
    region_types: np.ndarray,
    dosages: np.ndarray,
    target_class: int = 1,
    n_steps: int = 50,
    device: str = 'auto'
) -> np.ndarray:
    """
    Compute Layer Integrated Gradients using Captum.

    As specified in the implementation plan:
    - Target Layer: Input projection
    - Baseline: Zero embedding
    - Steps: 50 interpolation steps

    Args:
        model: Trained hierarchical model
        embeddings: (n_variants, embedding_dim) delta embeddings
        positions: (n_variants,) genomic positions
        region_types: (n_variants,) region type indices
        dosages: (n_samples, n_variants) genotype dosages
        target_class: Class to compute attribution for (1 = case)
        n_steps: Number of interpolation steps for IG
        device: Device to use

    Returns:
        (n_samples, n_variants) attribution scores per variant per patient
    """
    import torch
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError:
        raise ImportError("Captum required for attribution. Install with: pip install captum")

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.eval()

    # Convert to tensors
    embeddings_t = torch.FloatTensor(embeddings).to(device)
    positions_t = torch.LongTensor(positions).to(device)
    region_types_t = torch.LongTensor(region_types).to(device)

    n_samples = dosages.shape[0]
    n_variants = embeddings.shape[0]

    logger.info(f"Computing Integrated Gradients for {n_samples} samples...")
    logger.info(f"Baseline: Zero embedding, Steps: {n_steps}")

    # Create wrapper for Captum that takes dosage-scaled embeddings as input
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, positions, region_types):
            super().__init__()
            self.model = model
            self.positions = positions
            self.region_types = region_types

        def forward(self, scaled_embeddings, dosages):
            # scaled_embeddings: (batch, n_variants, embedding_dim)
            # We need to pass through input_proj and get logits
            batch_size = scaled_embeddings.shape[0]

            # Expand positions and region_types for batch
            pos_batch = self.positions.unsqueeze(0).expand(batch_size, -1)
            reg_batch = self.region_types.unsqueeze(0).expand(batch_size, -1)

            # Input projection
            x = self.model.input_proj(scaled_embeddings)

            # Add positional encoding
            pos_emb = self.model.pos_encoding(pos_batch)
            x = x + pos_emb

            # Region attention pooling
            region_emb = self.model.region_attention(x, reg_batch, mask=None)

            # Region combination
            region_flat = region_emb.view(batch_size, -1)
            patient_emb = self.model.region_combination(region_flat)

            # Classification
            logits = self.model.classifier(patient_emb)

            return logits

    wrapper = ModelWrapper(model, positions_t, region_types_t)
    wrapper.eval()

    # Layer Integrated Gradients targeting the input (dosage-scaled embeddings)
    # The baseline is zero embedding (no variant effect)
    lig = LayerIntegratedGradients(wrapper, wrapper.model.input_proj[0])  # Target first linear layer

    all_attributions = []

    # Process in batches
    batch_size = 32
    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)
        batch_dosages = torch.FloatTensor(dosages[start:end]).to(device)

        # Create scaled embeddings for this batch
        batch_size_actual = batch_dosages.shape[0]
        batch_emb = embeddings_t.unsqueeze(0).expand(batch_size_actual, -1, -1)
        scaled_emb = batch_emb * batch_dosages.unsqueeze(-1)

        # Baseline: zero embedding (no variant effect)
        baseline = torch.zeros_like(scaled_emb)

        # Compute attributions using Integrated Gradients
        # We compute attribution w.r.t. the scaled embeddings
        try:
            attributions = lig.attribute(
                inputs=scaled_emb,
                baselines=baseline,
                target=target_class,
                additional_forward_args=(batch_dosages,),
                n_steps=n_steps
            )

            # Sum over embedding dimension to get per-variant attribution
            # attributions shape: (batch, n_variants, embedding_dim)
            variant_attr = attributions.sum(dim=-1).cpu().numpy()  # (batch, n_variants)

        except Exception as e:
            logger.warning(f"Captum attribution failed for batch {batch_idx}: {e}")
            logger.info("Falling back to gradient-based attribution...")

            # Fallback: simple gradient-based attribution
            scaled_emb.requires_grad_(True)
            logits = wrapper(scaled_emb, batch_dosages)
            loss = logits[:, target_class].sum()
            loss.backward()

            # Gradient × input (approximate IG)
            variant_attr = (scaled_emb.grad * scaled_emb).sum(dim=-1).detach().cpu().numpy()

        all_attributions.append(variant_attr)

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            logger.info(f"Processed {end}/{n_samples} samples")

    attributions = np.concatenate(all_attributions, axis=0)
    logger.info(f"Attribution shape: {attributions.shape}")

    return attributions


def aggregate_to_regions(
    variant_attributions: np.ndarray,
    region_types: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Aggregate variant attributions to region level.

    Args:
        variant_attributions: (n_samples, n_variants) attribution scores
        region_types: (n_variants,) region type indices

    Returns:
        Dict mapping region name to (n_samples,) aggregated attributions
    """
    n_samples = variant_attributions.shape[0]
    region_attributions = {}

    for r_idx, r_name in enumerate(REGION_TYPES):
        mask = region_types == r_idx
        if mask.sum() > 0:
            # Sum absolute attributions in this region
            region_attr = np.abs(variant_attributions[:, mask]).sum(axis=1)
        else:
            region_attr = np.zeros(n_samples)
        region_attributions[r_name] = region_attr

    return region_attributions


def aggregate_to_genes(
    variant_attributions: np.ndarray,
    region_types: np.ndarray,
    variant_genes: List[str]
) -> pd.DataFrame:
    """
    Aggregate variant attributions to gene level.

    Args:
        variant_attributions: (n_samples, n_variants) attribution scores
        region_types: (n_variants,) region type indices
        variant_genes: (n_variants,) gene symbols for each variant

    Returns:
        DataFrame with gene-level attributions per sample
    """
    n_samples = variant_attributions.shape[0]

    # Get unique genes
    unique_genes = list(set(g for g in variant_genes if g and g != ''))

    gene_data = []

    for gene in unique_genes:
        # Get variants in this gene
        gene_mask = np.array([g == gene for g in variant_genes])

        if gene_mask.sum() == 0:
            continue

        # Aggregate attribution for each sample
        gene_attr = np.abs(variant_attributions[:, gene_mask]).sum(axis=1)

        # Also compute region breakdown for this gene
        region_breakdown = {}
        for r_idx, r_name in enumerate(REGION_TYPES):
            region_mask = (region_types == r_idx) & gene_mask
            if region_mask.sum() > 0:
                region_breakdown[r_name] = np.abs(variant_attributions[:, region_mask]).sum(axis=1).mean()
            else:
                region_breakdown[r_name] = 0.0

        gene_data.append({
            'gene': gene,
            'mean_attribution': gene_attr.mean(),
            'std_attribution': gene_attr.std(),
            'n_variants': gene_mask.sum(),
            **{f'{r}_attr': v for r, v in region_breakdown.items()}
        })

    df = pd.DataFrame(gene_data)
    df = df.sort_values('mean_attribution', ascending=False).reset_index(drop=True)

    return df


def generate_patient_reports(
    variant_attributions: np.ndarray,
    region_attributions: Dict[str, np.ndarray],
    gene_df: pd.DataFrame,
    sample_ids: List[str],
    predictions: np.ndarray,
    variant_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 10
):
    """
    Generate per-patient attribution reports as specified in the implementation plan.

    Report structure:
    - patient_id
    - prediction (probability, risk category)
    - top_genes with region breakdown
    - region_summary
    """
    logger.info(f"Generating patient reports...")

    reports_dir = output_path / 'patient_reports'
    reports_dir.mkdir(exist_ok=True)

    # Get gene list from variant annotations
    if 'SYMBOL' in variant_df.columns:
        variant_genes = variant_df['SYMBOL'].fillna('').tolist()
    elif 'gene' in variant_df.columns:
        variant_genes = variant_df['gene'].fillna('').tolist()
    else:
        variant_genes = [''] * len(variant_df)

    all_reports = []

    for i, sample_id in enumerate(sample_ids[:min(len(sample_ids), 100)]):  # Limit to first 100 for performance
        prob = predictions[i]
        risk_category = 'elevated' if prob > 0.5 else 'low'

        # Get top variants for this patient
        patient_attr = variant_attributions[i]
        top_var_idx = np.argsort(np.abs(patient_attr))[-top_n:][::-1]

        top_variants = []
        for idx in top_var_idx:
            var_info = {
                'rank': len(top_variants) + 1,
                'variant_idx': int(idx),
                'attribution': float(patient_attr[idx]),
            }
            if 'SNP' in variant_df.columns:
                var_info['variant_id'] = variant_df.iloc[idx]['SNP']
            if 'POS' in variant_df.columns:
                var_info['position'] = int(variant_df.iloc[idx]['POS'])
            if 'most_severe_consequence' in variant_df.columns:
                var_info['consequence'] = variant_df.iloc[idx]['most_severe_consequence']
            if len(variant_genes) > idx:
                var_info['gene'] = variant_genes[idx]
            top_variants.append(var_info)

        # Gene-level attribution for this patient
        gene_attr = {}
        unique_genes = list(set(g for g in variant_genes if g and g != ''))
        for gene in unique_genes:
            gene_mask = np.array([g == gene for g in variant_genes])
            if gene_mask.sum() > 0:
                gene_attr[gene] = float(np.abs(patient_attr[gene_mask]).sum())

        # Sort genes by attribution
        top_genes = sorted(gene_attr.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Region summary
        total_attr = sum(region_attributions[r][i] for r in REGION_TYPES)
        region_summary = {}
        for r in REGION_TYPES:
            if total_attr > 0:
                region_summary[f'{r}_contribution'] = float(region_attributions[r][i] / total_attr)
            else:
                region_summary[f'{r}_contribution'] = 0.0

        report = {
            'patient_id': str(sample_id),
            'prediction': {
                'ais_probability': float(prob),
                'risk_category': risk_category
            },
            'top_genes': [
                {'rank': j+1, 'gene_name': g, 'total_attribution': float(a)}
                for j, (g, a) in enumerate(top_genes)
            ],
            'top_variants': top_variants,
            'region_summary': region_summary
        }

        all_reports.append(report)

        # Save individual report
        with open(reports_dir / f'{sample_id}_report.json', 'w') as f:
            json.dump(report, f, indent=2)

    # Save summary of all reports
    with open(output_path / 'patient_reports_summary.json', 'w') as f:
        json.dump(all_reports, f, indent=2)

    logger.info(f"Generated {len(all_reports)} patient reports in {reports_dir}")

    return all_reports


def plot_gene_ranking(gene_df: pd.DataFrame, output_path: Path, top_n: int = 20):
    """Plot top genes by mean attribution."""
    plt.figure(figsize=(12, 8))

    top_genes = gene_df.head(top_n)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_genes)))

    plt.barh(range(len(top_genes)), top_genes['mean_attribution'], color=colors)
    plt.yticks(range(len(top_genes)), top_genes['gene'])
    plt.xlabel('Mean Attribution Score')
    plt.ylabel('Gene')
    plt.title(f'Top {top_n} Genes by Attribution')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path / 'gene_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved gene ranking plot")


def plot_region_breakdown(gene_df: pd.DataFrame, output_path: Path, top_n: int = 10):
    """Plot region breakdown for top genes."""
    top_genes = gene_df.head(top_n)

    # Prepare data for stacked bar chart
    region_cols = [f'{r}_attr' for r in REGION_TYPES if f'{r}_attr' in top_genes.columns]

    if not region_cols:
        logger.warning("No region attribution columns found")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create stacked bar chart
    bottom = np.zeros(len(top_genes))
    colors = plt.cm.Set3(np.linspace(0, 1, len(region_cols)))

    for col, color in zip(region_cols, colors):
        region_name = col.replace('_attr', '')
        values = top_genes[col].values
        ax.barh(range(len(top_genes)), values, left=bottom, label=region_name, color=color)
        bottom += values

    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes['gene'])
    ax.set_xlabel('Attribution Score')
    ax.set_ylabel('Gene')
    ax.set_title(f'Region Breakdown for Top {top_n} Genes')
    ax.legend(loc='lower right', title='Region Type')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path / 'region_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved region breakdown plot")


def plot_region_pie(region_attributions: Dict[str, np.ndarray], output_path: Path):
    """Plot overall region contribution pie chart."""
    plt.figure(figsize=(10, 8))

    # Sum across all samples
    region_totals = {r: np.abs(attr).sum() for r, attr in region_attributions.items()}
    total = sum(region_totals.values())

    if total == 0:
        logger.warning("No attributions to plot")
        return

    # Filter out very small regions
    significant = {r: v for r, v in region_totals.items() if v / total > 0.01}
    other = sum(v for r, v in region_totals.items() if v / total <= 0.01)
    if other > 0:
        significant['other_combined'] = other

    labels = list(significant.keys())
    sizes = list(significant.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Overall Attribution by Region Type')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path / 'region_pie.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved region pie chart")


def plot_variant_manhattan(
    variant_attributions: np.ndarray,
    positions: np.ndarray,
    variant_genes: List[str],
    output_path: Path
):
    """Create Manhattan-style plot of variant attributions."""
    plt.figure(figsize=(16, 6))

    # Mean absolute attribution across samples
    mean_attr = np.abs(variant_attributions).mean(axis=0)

    # Color by direction (positive vs negative mean attribution)
    mean_signed = variant_attributions.mean(axis=0)
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in mean_signed]

    plt.scatter(positions / 1e6, mean_attr, c=colors, alpha=0.5, s=10)

    # Highlight top variants
    top_idx = np.argsort(mean_attr)[-20:]
    for idx in top_idx:
        gene = variant_genes[idx] if idx < len(variant_genes) else ''
        if gene:
            plt.annotate(
                gene,
                (positions[idx] / 1e6, mean_attr[idx]),
                fontsize=7,
                alpha=0.8,
                xytext=(5, 5),
                textcoords='offset points'
            )

    plt.xlabel('Position (Mb)')
    plt.ylabel('Mean |Attribution|')
    plt.title('Chromosome 6 Variant Attribution (Manhattan Plot)')
    plt.tight_layout()
    plt.savefig(output_path / 'variant_manhattan.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Manhattan plot")


def plot_risk_distribution(predictions: np.ndarray, labels: np.ndarray, output_path: Path):
    """Plot risk score distribution for cases vs controls."""
    plt.figure(figsize=(10, 6))

    case_probs = predictions[labels == 1]
    control_probs = predictions[labels == 0]

    plt.hist(control_probs, bins=50, alpha=0.6, label=f'Controls (n={len(control_probs)})',
             color='#3498db', density=True)
    plt.hist(case_probs, bins=50, alpha=0.6, label=f'Cases (n={len(case_probs)})',
             color='#e74c3c', density=True)

    plt.xlabel('Predicted AIS Probability')
    plt.ylabel('Density')
    plt.title('Risk Score Distribution: Cases vs Controls')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'risk_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved risk distribution plot")


def validate_gpr126_signal(gene_df: pd.DataFrame, output_path: Path) -> Dict:
    """
    Validate recovery of known GPR126/ADGRG6 signal.

    As specified in the implementation plan:
    - GPR126 should be in top 5% of genes
    - Intronic signal should be >30% of GPR126 attribution
    """
    logger.info("Validating GPR126 signal...")

    # Find GPR126/ADGRG6
    gpr126_names = ['GPR126', 'ADGRG6']
    gpr126_row = None

    for name in gpr126_names:
        matches = gene_df[gene_df['gene'] == name]
        if len(matches) > 0:
            gpr126_row = matches.iloc[0]
            break

    validation = {
        'gpr126_found': gpr126_row is not None,
        'gpr126_rank': None,
        'gpr126_percentile': None,
        'intronic_fraction': None,
        'in_top_5_percent': False,
        'intronic_above_30_percent': False
    }

    if gpr126_row is not None:
        rank = gene_df[gene_df['mean_attribution'] >= gpr126_row['mean_attribution']].shape[0]
        total_genes = len(gene_df)
        percentile = 100 * (1 - rank / total_genes)

        validation['gpr126_rank'] = int(rank)
        validation['gpr126_percentile'] = float(percentile)
        validation['in_top_5_percent'] = percentile >= 95

        # Check intronic fraction
        total_attr = sum(gpr126_row.get(f'{r}_attr', 0) for r in REGION_TYPES)
        if total_attr > 0:
            intronic_attr = gpr126_row.get('intron_attr', 0)
            intronic_fraction = intronic_attr / total_attr
            validation['intronic_fraction'] = float(intronic_fraction)
            validation['intronic_above_30_percent'] = intronic_fraction > 0.3

        logger.info(f"GPR126 rank: {rank}/{total_genes} (top {100-percentile:.1f}%)")
        if validation['intronic_fraction']:
            logger.info(f"GPR126 intronic fraction: {validation['intronic_fraction']*100:.1f}%")
    else:
        logger.warning("GPR126/ADGRG6 not found in gene list")

    # Save validation results
    with open(output_path / 'gpr126_validation.json', 'w') as f:
        json.dump(validation, f, indent=2)

    return validation


def interpret_model(args):
    """Main interpretation function."""
    import torch

    logger.info("=" * 50)
    logger.info("Phase 6: Model Interpretation & Attribution")
    logger.info("=" * 50)
    logger.info("Method: Layer Integrated Gradients (Captum)")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, config = load_model_and_config(args.model, args.config)

    # Load embeddings
    embeddings_path = Path(args.embeddings)
    embeddings = {}
    for key in ['ref', 'alt', 'diff']:
        zarr_path = embeddings_path.with_suffix(f'.{key}.zarr')
        npy_path = embeddings_path.with_suffix(f'.{key}.npy')
        if zarr_path.exists():
            try:
                import zarr
                embeddings[key] = np.array(zarr.open(str(zarr_path), mode='r'))
            except ImportError:
                pass
        if key not in embeddings and npy_path.exists():
            embeddings[key] = np.load(npy_path)

    # Use delta embeddings
    if 'diff' in embeddings:
        delta_embeddings = embeddings['diff']
    elif 'alt' in embeddings and 'ref' in embeddings:
        delta_embeddings = embeddings['alt'] - embeddings['ref']
    else:
        delta_embeddings = list(embeddings.values())[0]

    logger.info(f"Loaded embeddings: {delta_embeddings.shape}")

    # Load annotations
    annotations = pd.read_parquet(args.annotations)
    logger.info(f"Loaded {len(annotations)} variant annotations")

    # Get positions and region types
    positions = annotations['POS'].values if 'POS' in annotations.columns else np.arange(len(annotations))

    if 'most_severe_consequence' in annotations.columns:
        region_types = np.array([
            get_region_type(c) for c in annotations['most_severe_consequence']
        ])
    elif 'Consequence' in annotations.columns:
        region_types = np.array([
            get_region_type(c.split(',')[0]) for c in annotations['Consequence']
        ])
    else:
        region_types = np.full(len(annotations), REGION_TYPES.index('other'))

    # Get variant genes
    if 'SYMBOL' in annotations.columns:
        variant_genes = annotations['SYMBOL'].fillna('').tolist()
    elif 'gene' in annotations.columns:
        variant_genes = annotations['gene'].fillna('').tolist()
    else:
        variant_genes = [''] * len(annotations)

    # Load dosages (support both parquet and zarr formats)
    dosages_path = Path(args.dosages)
    if dosages_path.suffix == '.zarr' or (dosages_path.is_dir() and (dosages_path / '.zarray').exists()):
        # Zarr format
        try:
            import zarr
            dosages = np.array(zarr.open(str(dosages_path), mode='r'))
            logger.info(f"Loaded dosages from zarr: {dosages.shape}")
            # For zarr, we need sample IDs from a separate file
            sample_ids_path = dosages_path.parent / 'sample_ids.txt'
            if sample_ids_path.exists():
                with open(sample_ids_path, 'r') as f:
                    sample_ids = [line.strip() for line in f if line.strip()]
            else:
                sample_ids = [str(i) for i in range(dosages.shape[0])]
        except ImportError:
            raise ImportError("zarr required for .zarr files. Install with: pip install zarr")
    else:
        # Parquet format
        dosages_df = pd.read_parquet(args.dosages)
        variant_cols = ['SNP', 'CHR', 'POS', 'A1', 'A2', 'REF', 'ALT', 'variant_id']
        sample_cols = [c for c in dosages_df.columns if c not in variant_cols]
        dosages = dosages_df[sample_cols].values.T
        sample_ids = sample_cols

    logger.info(f"Loaded dosages: {dosages.shape}")

    # Load cohort
    cohort = pd.read_parquet(args.cohort)
    label_cols = ['is_case', 'case', 'label', 'AIS', 'phenotype']
    label_col = next((c for c in label_cols if c in cohort.columns), None)
    cohort['label'] = cohort[label_col].astype(int) if label_col else 0

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
    matched_sample_ids = [sample_ids[i] for i in matched_indices]
    labels = cohort.iloc[matched_cohort_indices]['label'].values

    # Get predictions
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    embeddings_t = torch.FloatTensor(delta_embeddings).to(device)
    positions_t = torch.LongTensor(positions).to(device)
    region_types_t = torch.LongTensor(region_types).to(device)

    predictions = []
    batch_size = 32
    n_batches = (len(matched_dosages) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(matched_dosages))
            batch_dosages = torch.FloatTensor(matched_dosages[start:end]).to(device)

            batch_size_actual = batch_dosages.shape[0]
            batch_emb = embeddings_t.unsqueeze(0).expand(batch_size_actual, -1, -1)
            batch_pos = positions_t.unsqueeze(0).expand(batch_size_actual, -1)
            batch_reg = region_types_t.unsqueeze(0).expand(batch_size_actual, -1)

            logits = model(batch_emb, batch_pos, batch_reg, batch_dosages)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            predictions.extend(probs.cpu().numpy())

    predictions = np.array(predictions)

    # Compute Integrated Gradients
    logger.info("\n" + "=" * 50)
    logger.info("Computing Layer Integrated Gradients")
    logger.info("=" * 50)

    variant_attributions = compute_integrated_gradients(
        model=model,
        embeddings=delta_embeddings,
        positions=positions,
        region_types=region_types,
        dosages=matched_dosages,
        target_class=1,
        n_steps=args.n_steps,
        device=device
    )

    # Save variant-level attributions
    np.save(output_path / 'variant_attributions.npy', variant_attributions)
    logger.info(f"Saved variant attributions: {variant_attributions.shape}")

    # Aggregate to regions
    logger.info("\n" + "=" * 50)
    logger.info("Aggregating to Region Level")
    logger.info("=" * 50)

    region_attributions = aggregate_to_regions(variant_attributions, region_types)

    for r, attr in region_attributions.items():
        logger.info(f"  {r}: mean={attr.mean():.4f}, std={attr.std():.4f}")

    # Aggregate to genes
    logger.info("\n" + "=" * 50)
    logger.info("Aggregating to Gene Level")
    logger.info("=" * 50)

    gene_df = aggregate_to_genes(variant_attributions, region_types, variant_genes)
    gene_df.to_csv(output_path / 'gene_attributions.csv', index=False)

    logger.info(f"Top 10 genes by attribution:")
    for _, row in gene_df.head(10).iterrows():
        logger.info(f"  {row['gene']}: {row['mean_attribution']:.4f} ({row['n_variants']} variants)")

    # Generate patient reports
    logger.info("\n" + "=" * 50)
    logger.info("Generating Patient Reports")
    logger.info("=" * 50)

    patient_reports = generate_patient_reports(
        variant_attributions=variant_attributions,
        region_attributions=region_attributions,
        gene_df=gene_df,
        sample_ids=matched_sample_ids,
        predictions=predictions,
        variant_df=annotations,
        output_path=output_path
    )

    # Validate GPR126 signal
    logger.info("\n" + "=" * 50)
    logger.info("Validating Known Locus (GPR126)")
    logger.info("=" * 50)

    validation = validate_gpr126_signal(gene_df, output_path)

    # Generate visualizations
    logger.info("\n" + "=" * 50)
    logger.info("Generating Visualizations")
    logger.info("=" * 50)

    plot_gene_ranking(gene_df, output_path)
    plot_region_breakdown(gene_df, output_path)
    plot_region_pie(region_attributions, output_path)
    plot_variant_manhattan(variant_attributions, positions, variant_genes, output_path)
    plot_risk_distribution(predictions, labels, output_path)

    # Save summary
    summary = {
        'n_samples': len(matched_dosages),
        'n_variants': len(delta_embeddings),
        'n_genes': len(gene_df),
        'method': 'Layer Integrated Gradients',
        'n_steps': args.n_steps,
        'baseline': 'zero_embedding',
        'target_layer': 'input_projection',
        'validation': validation,
        'top_genes': gene_df.head(20).to_dict('records'),
    }

    with open(output_path / 'interpretation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("")
    logger.info("=" * 50)
    logger.info("INTERPRETATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Output directory: {output_path}")
    logger.info("Generated files:")
    for f in sorted(output_path.glob('*')):
        if f.is_file():
            logger.info(f"  - {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Interpret AIS model using Layer Integrated Gradients (Captum)"
    )

    parser.add_argument('--model', required=True,
                       help='Path to trained model (model.pt)')
    parser.add_argument('--config', required=True,
                       help='Path to model config (config.json)')
    parser.add_argument('--embeddings', required=True,
                       help='Path prefix to HyenaDNA embeddings')
    parser.add_argument('--annotations', required=True,
                       help='Path to VEP-annotated variants parquet file')
    parser.add_argument('--dosages', required=True,
                       help='Path to genotype dosages parquet file')
    parser.add_argument('--cohort', required=True,
                       help='Path to cohort definition parquet file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for interpretation results')
    parser.add_argument('--n-steps', type=int, default=50,
                       help='Number of interpolation steps for IG (default: 50)')
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device (default: auto)')

    args = parser.parse_args()
    interpret_model(args)


if __name__ == "__main__":
    main()
