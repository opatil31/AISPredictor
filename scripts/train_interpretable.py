#!/usr/bin/env python3
"""
Interpretable Model Training with SHAP Attribution

Trains XGBoost/LightGBM on raw dosages and computes SHAP values for
per-variant and per-gene attribution. This provides interpretable
feature importance without requiring embeddings.

Usage:
    python scripts/train_interpretable.py \
        --dosages data/variants_rare/dosages.zarr \
        --annotations data/variants_rare/variants_annotated.parquet \
        --cohort data/cohort.parquet \
        --output results/interpretable_model \
        --n-folds 5

Output:
    - Trained model with CV performance metrics
    - SHAP values per variant per sample
    - Gene-level attribution rankings
    - Visualizations (gene ranking, Manhattan plot, SHAP summary)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# Region type mapping (same as interpret_model.py)
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


def get_region_type(consequence: str) -> str:
    """Map VEP consequence to region type."""
    if pd.isna(consequence):
        return 'other'
    # Handle comma-separated consequences (take first)
    first_consequence = consequence.split(',')[0].strip()
    return REGION_TYPE_MAP.get(first_consequence, 'other')


def load_dosages(dosages_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load dosages from zarr or parquet format."""
    if dosages_path.suffix == '.zarr' or (dosages_path.is_dir() and (dosages_path / '.zarray').exists()):
        try:
            import zarr
            dosages = np.array(zarr.open(str(dosages_path), mode='r'))
            logger.info(f"Loaded dosages from zarr: {dosages.shape}")

            sample_ids_path = dosages_path.parent / 'sample_ids.txt'
            if sample_ids_path.exists():
                with open(sample_ids_path, 'r') as f:
                    sample_ids = [line.strip() for line in f if line.strip()]
            else:
                sample_ids = [str(i) for i in range(dosages.shape[0])]
            return dosages, sample_ids
        except ImportError:
            raise ImportError("zarr required. Install with: pip install zarr")
    else:
        dosages_df = pd.read_parquet(dosages_path)
        variant_cols = ['SNP', 'CHR', 'POS', 'A1', 'A2', 'REF', 'ALT', 'variant_id']
        sample_cols = [c for c in dosages_df.columns if c not in variant_cols]
        dosages = dosages_df[sample_cols].values.T
        return dosages, sample_cols


def train_xgboost_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    params: Optional[Dict] = None
) -> Tuple[List, np.ndarray, np.ndarray]:
    """
    Train XGBoost with cross-validation.

    Returns:
        models: List of trained models (one per fold)
        oof_predictions: Out-of-fold predictions
        oof_indices: Indices for each prediction
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost required. Install with: pip install xgboost")

    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'auc'
        }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    models = []
    oof_predictions = np.zeros(len(y))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Training fold {fold + 1}/{n_folds}...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = val_pred

        auroc = roc_auc_score(y_val, val_pred)
        auprc = average_precision_score(y_val, val_pred)

        logger.info(f"  Fold {fold + 1} - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")

        fold_metrics.append({'fold': fold + 1, 'auroc': auroc, 'auprc': auprc})
        models.append(model)

    # Overall metrics
    overall_auroc = roc_auc_score(y, oof_predictions)
    overall_auprc = average_precision_score(y, oof_predictions)

    logger.info(f"\nOverall CV - AUROC: {overall_auroc:.4f}, AUPRC: {overall_auprc:.4f}")

    return models, oof_predictions, fold_metrics


def compute_shap_values(
    models: List,
    X: np.ndarray,
    aggregate: str = 'mean'
) -> np.ndarray:
    """
    Compute SHAP values using trained models.

    Args:
        models: List of trained XGBoost models
        X: Feature matrix (n_samples, n_variants)
        aggregate: How to aggregate across folds ('mean' or 'first')

    Returns:
        shap_values: (n_samples, n_variants) SHAP values
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap required. Install with: pip install shap")

    logger.info("Computing SHAP values...")

    all_shap_values = []

    for i, model in enumerate(models):
        logger.info(f"  Computing SHAP for model {i + 1}/{len(models)}...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classification, shap_values might be a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 (case) SHAP values

        all_shap_values.append(shap_values)

    # Aggregate across models
    if aggregate == 'mean':
        final_shap = np.mean(all_shap_values, axis=0)
    else:
        final_shap = all_shap_values[0]

    logger.info(f"SHAP values shape: {final_shap.shape}")

    return final_shap


def aggregate_to_genes(
    shap_values: np.ndarray,
    variant_genes: List[str],
    variant_consequences: List[str]
) -> pd.DataFrame:
    """
    Aggregate SHAP values to gene level.

    Args:
        shap_values: (n_samples, n_variants) SHAP values
        variant_genes: Gene symbol for each variant
        variant_consequences: VEP consequence for each variant

    Returns:
        DataFrame with gene-level statistics
    """
    logger.info("Aggregating SHAP values to gene level...")

    unique_genes = list(set(g for g in variant_genes if g and g != '' and not pd.isna(g)))

    gene_data = []

    for gene in unique_genes:
        gene_mask = np.array([g == gene for g in variant_genes])
        n_variants = gene_mask.sum()

        if n_variants == 0:
            continue

        # Aggregate SHAP values for this gene
        gene_shap = shap_values[:, gene_mask]

        # Mean absolute SHAP across samples and variants
        mean_abs_shap = np.abs(gene_shap).mean()

        # Sum of absolute SHAP per sample, then mean across samples
        gene_burden = np.abs(gene_shap).sum(axis=1).mean()

        # Std across samples
        std_shap = np.abs(gene_shap).sum(axis=1).std()

        # Direction: positive = risk, negative = protective
        mean_signed = gene_shap.mean()
        direction = 'risk' if mean_signed > 0 else 'protective'

        # Region breakdown
        region_breakdown = {}
        gene_consequences = [variant_consequences[i] for i in range(len(variant_consequences)) if gene_mask[i]]
        for cons in gene_consequences:
            region = get_region_type(cons)
            region_breakdown[region] = region_breakdown.get(region, 0) + 1

        gene_data.append({
            'gene': gene,
            'mean_abs_shap': mean_abs_shap,
            'gene_burden_shap': gene_burden,
            'std_shap': std_shap,
            'direction': direction,
            'mean_signed_shap': mean_signed,
            'n_variants': n_variants,
            **{f'{r}_count': region_breakdown.get(r, 0) for r in REGION_TYPES}
        })

    df = pd.DataFrame(gene_data)
    df = df.sort_values('gene_burden_shap', ascending=False).reset_index(drop=True)

    logger.info(f"Aggregated to {len(df)} genes")

    return df


def aggregate_to_regions(
    shap_values: np.ndarray,
    variant_consequences: List[str]
) -> Dict[str, float]:
    """Aggregate SHAP values by region type."""
    region_shap = {}

    for region in REGION_TYPES:
        region_mask = np.array([get_region_type(c) == region for c in variant_consequences])
        if region_mask.sum() > 0:
            region_shap[region] = np.abs(shap_values[:, region_mask]).sum()
        else:
            region_shap[region] = 0.0

    return region_shap


def plot_gene_ranking(gene_df: pd.DataFrame, output_path: Path, top_n: int = 20):
    """Plot top genes by SHAP attribution."""
    plt.figure(figsize=(12, 8))

    top_genes = gene_df.head(top_n)

    colors = ['#e74c3c' if d == 'risk' else '#3498db'
              for d in top_genes['direction']]

    plt.barh(range(len(top_genes)), top_genes['gene_burden_shap'], color=colors)
    plt.yticks(range(len(top_genes)), top_genes['gene'])
    plt.xlabel('Mean |SHAP| Burden')
    plt.ylabel('Gene')
    plt.title(f'Top {top_n} Genes by SHAP Attribution\n(Red=Risk, Blue=Protective)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path / 'gene_ranking_shap.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved gene ranking plot")


def plot_shap_summary(shap_values: np.ndarray, variant_ids: List[str], output_path: Path, top_n: int = 30):
    """Create SHAP summary plot."""
    try:
        import shap

        plt.figure(figsize=(12, 10))

        # Get top variants by mean |SHAP|
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]

        # Create a DataFrame for the top variants
        top_shap = shap_values[:, top_idx]
        top_names = [variant_ids[i] if i < len(variant_ids) else f"var_{i}" for i in top_idx]

        shap.summary_plot(
            top_shap,
            feature_names=top_names,
            show=False,
            max_display=top_n
        )
        plt.tight_layout()
        plt.savefig(output_path / 'shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved SHAP summary plot")
    except Exception as e:
        logger.warning(f"Could not create SHAP summary plot: {e}")


def plot_manhattan(
    shap_values: np.ndarray,
    positions: np.ndarray,
    variant_genes: List[str],
    output_path: Path
):
    """Create Manhattan-style plot of variant SHAP values."""
    plt.figure(figsize=(16, 6))

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)

    colors = ['#e74c3c' if x > 0 else '#3498db' for x in mean_signed]

    plt.scatter(positions / 1e6, mean_abs_shap, c=colors, alpha=0.5, s=10)

    # Highlight top variants
    top_idx = np.argsort(mean_abs_shap)[-20:]
    for idx in top_idx:
        gene = variant_genes[idx] if idx < len(variant_genes) and variant_genes[idx] else ''
        if gene:
            plt.annotate(
                gene,
                (positions[idx] / 1e6, mean_abs_shap[idx]),
                fontsize=7,
                alpha=0.8,
                xytext=(5, 5),
                textcoords='offset points'
            )

    plt.xlabel('Position (Mb)')
    plt.ylabel('Mean |SHAP|')
    plt.title('Variant Attribution Manhattan Plot\n(Red=Risk increasing, Blue=Protective)')
    plt.tight_layout()
    plt.savefig(output_path / 'manhattan_shap.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved Manhattan plot")


def plot_region_pie(region_shap: Dict[str, float], output_path: Path):
    """Plot region contribution pie chart."""
    plt.figure(figsize=(10, 8))

    total = sum(region_shap.values())
    if total == 0:
        logger.warning("No SHAP values to plot")
        return

    # Filter small regions
    significant = {r: v for r, v in region_shap.items() if v / total > 0.01}
    other = sum(v for r, v in region_shap.items() if v / total <= 0.01)
    if other > 0:
        significant['other_combined'] = other

    labels = list(significant.keys())
    sizes = list(significant.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('SHAP Attribution by Region Type')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path / 'region_pie_shap.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved region pie chart")


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 8))

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)

    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved ROC curve")


def plot_feature_importance(models: List, variant_ids: List[str], output_path: Path, top_n: int = 30):
    """Plot XGBoost feature importance (averaged across folds)."""
    plt.figure(figsize=(12, 10))

    # Average importance across models
    all_importance = []
    for model in models:
        all_importance.append(model.feature_importances_)

    mean_importance = np.mean(all_importance, axis=0)

    # Top N features
    top_idx = np.argsort(mean_importance)[-top_n:][::-1]
    top_names = [variant_ids[i] if i < len(variant_ids) else f"var_{i}" for i in top_idx]
    top_importance = mean_importance[top_idx]

    plt.barh(range(len(top_names)), top_importance)
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel('Feature Importance (Gain)')
    plt.ylabel('Variant')
    plt.title(f'Top {top_n} Variants by XGBoost Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved feature importance plot")


def main():
    parser = argparse.ArgumentParser(
        description="Train interpretable XGBoost model with SHAP attribution"
    )

    parser.add_argument('--dosages', required=True,
                       help='Path to dosages (zarr or parquet)')
    parser.add_argument('--annotations', required=True,
                       help='Path to annotated variants parquet')
    parser.add_argument('--cohort', required=True,
                       help='Path to cohort parquet')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of XGBoost trees (default: 100)')
    parser.add_argument('--max-depth', type=int, default=4,
                       help='Max tree depth (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')

    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Interpretable Model Training (XGBoost + SHAP)")
    logger.info("=" * 50)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"\nLoading dosages from {args.dosages}")
    dosages, sample_ids = load_dosages(Path(args.dosages))
    logger.info(f"Dosages shape: {dosages.shape}")

    logger.info(f"\nLoading annotations from {args.annotations}")
    annotations = pd.read_parquet(args.annotations)
    logger.info(f"Loaded {len(annotations)} variants")

    # Get variant info
    if 'ID' in annotations.columns:
        variant_ids = annotations['ID'].tolist()
    elif 'SNP' in annotations.columns:
        variant_ids = annotations['SNP'].tolist()
    else:
        variant_ids = [f"var_{i}" for i in range(len(annotations))]

    if 'SYMBOL' in annotations.columns:
        variant_genes = annotations['SYMBOL'].fillna('').tolist()
    elif 'gene' in annotations.columns:
        variant_genes = annotations['gene'].fillna('').tolist()
    else:
        variant_genes = [''] * len(annotations)

    if 'most_severe_consequence' in annotations.columns:
        variant_consequences = annotations['most_severe_consequence'].fillna('').tolist()
    elif 'Consequence' in annotations.columns:
        variant_consequences = annotations['Consequence'].fillna('').tolist()
    else:
        variant_consequences = [''] * len(annotations)

    positions = annotations['POS'].values if 'POS' in annotations.columns else np.arange(len(annotations))

    # Load cohort and match samples
    logger.info(f"\nLoading cohort from {args.cohort}")
    cohort = pd.read_parquet(args.cohort)

    label_cols = ['is_case', 'case', 'label', 'AIS', 'phenotype']
    label_col = next((c for c in label_cols if c in cohort.columns), None)
    if label_col is None:
        raise ValueError(f"No label column found. Expected one of: {label_cols}")

    id_cols = ['eid', 'IID', 'sample_id', 'FID']
    id_col = next((c for c in id_cols if c in cohort.columns), None)

    # Match samples
    if id_col:
        cohort_ids = {str(eid): idx for idx, eid in enumerate(cohort[id_col])}
        matched_indices = []
        matched_labels = []

        for i, sid in enumerate(sample_ids):
            sid_str = str(sid)
            if sid_str in cohort_ids:
                matched_indices.append(i)
                matched_labels.append(cohort.iloc[cohort_ids[sid_str]][label_col])

        X = dosages[matched_indices]
        y = np.array(matched_labels).astype(int)
        matched_sample_ids = [sample_ids[i] for i in matched_indices]
    else:
        X = dosages
        y = cohort[label_col].values[:len(dosages)].astype(int)
        matched_sample_ids = sample_ids

    logger.info(f"Matched {len(X)} samples")
    logger.info(f"Cases: {y.sum()}, Controls: {len(y) - y.sum()}")

    # Handle missing values
    if np.isnan(X).any():
        logger.info("Imputing missing values with column means...")
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        X = np.where(nan_mask, np.tile(col_means, (X.shape[0], 1)), X)

    # Train XGBoost
    logger.info("\n" + "=" * 50)
    logger.info("Training XGBoost with Cross-Validation")
    logger.info("=" * 50)

    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc'
    }

    models, oof_predictions, fold_metrics = train_xgboost_cv(X, y, args.n_folds, params)

    # Compute SHAP values
    logger.info("\n" + "=" * 50)
    logger.info("Computing SHAP Values")
    logger.info("=" * 50)

    shap_values = compute_shap_values(models, X)

    # Save SHAP values
    np.save(output_path / 'shap_values.npy', shap_values)
    logger.info(f"Saved SHAP values: {shap_values.shape}")

    # Aggregate to genes
    logger.info("\n" + "=" * 50)
    logger.info("Aggregating to Gene Level")
    logger.info("=" * 50)

    gene_df = aggregate_to_genes(shap_values, variant_genes, variant_consequences)
    gene_df.to_csv(output_path / 'gene_attributions_shap.csv', index=False)

    logger.info(f"\nTop 10 genes by SHAP attribution:")
    for _, row in gene_df.head(10).iterrows():
        logger.info(f"  {row['gene']}: {row['gene_burden_shap']:.4f} ({row['n_variants']} variants, {row['direction']})")

    # Aggregate to regions
    region_shap = aggregate_to_regions(shap_values, variant_consequences)

    logger.info(f"\nRegion breakdown:")
    total_shap = sum(region_shap.values())
    for region, value in sorted(region_shap.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * value / total_shap if total_shap > 0 else 0
        logger.info(f"  {region}: {pct:.1f}%")

    # Per-variant attribution
    variant_df = pd.DataFrame({
        'variant_id': variant_ids,
        'gene': variant_genes,
        'consequence': variant_consequences,
        'position': positions,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'mean_signed_shap': shap_values.mean(axis=0),
        'std_shap': shap_values.std(axis=0)
    })
    variant_df = variant_df.sort_values('mean_abs_shap', ascending=False)
    variant_df.to_csv(output_path / 'variant_attributions_shap.csv', index=False)

    # Generate visualizations
    logger.info("\n" + "=" * 50)
    logger.info("Generating Visualizations")
    logger.info("=" * 50)

    plot_gene_ranking(gene_df, output_path)
    plot_manhattan(shap_values, positions, variant_genes, output_path)
    plot_region_pie(region_shap, output_path)
    plot_roc_curve(y, oof_predictions, output_path)
    plot_feature_importance(models, variant_ids, output_path)
    plot_shap_summary(shap_values, variant_ids, output_path)

    # Save summary
    overall_auroc = roc_auc_score(y, oof_predictions)
    overall_auprc = average_precision_score(y, oof_predictions)

    summary = {
        'n_samples': len(X),
        'n_variants': X.shape[1],
        'n_cases': int(y.sum()),
        'n_controls': int(len(y) - y.sum()),
        'n_folds': args.n_folds,
        'cv_auroc': float(overall_auroc),
        'cv_auprc': float(overall_auprc),
        'fold_metrics': fold_metrics,
        'model_params': params,
        'n_genes': len(gene_df),
        'top_genes': gene_df.head(20).to_dict('records'),
        'region_breakdown': {k: float(v) for k, v in region_shap.items()}
    }

    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save model
    import pickle
    with open(output_path / 'models.pkl', 'wb') as f:
        pickle.dump(models, f)

    logger.info("\n" + "=" * 50)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"CV AUROC: {overall_auroc:.4f}")
    logger.info(f"CV AUPRC: {overall_auprc:.4f}")
    logger.info(f"Output directory: {output_path}")
    logger.info("\nGenerated files:")
    for f in sorted(output_path.glob('*')):
        if f.is_file():
            logger.info(f"  - {f.name}")


if __name__ == "__main__":
    main()
