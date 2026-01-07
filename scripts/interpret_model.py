#!/usr/bin/env python3
"""
Phase 6: Model Interpretation & Attribution

Interprets the trained AIS prediction model to identify:
1. Most important variants for prediction
2. Feature contributions via SHAP values
3. Pathway enrichment of top variants
4. Risk score distributions

Usage:
    python scripts/interpret_model.py \
        --model models/ais_model/model.joblib \
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
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load trained model."""
    model_path = Path(model_path)

    if model_path.suffix == '.joblib':
        import joblib
        model = joblib.load(model_path)
        model_type = 'sklearn'
    elif model_path.suffix == '.pt':
        import torch
        model = torch.load(model_path)
        model_type = 'pytorch'
    else:
        raise ValueError(f"Unknown model format: {model_path.suffix}")

    logger.info(f"Loaded {model_type} model from {model_path}")
    return model, model_type


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: List[str],
    model_type: str = 'sklearn',
    n_background: int = 100
) -> Tuple[np.ndarray, object]:
    """
    Compute SHAP values for model interpretation.

    Returns:
        Tuple of (shap_values, explainer)
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Install with: pip install shap")
        return None, None

    logger.info("Computing SHAP values...")

    # Select background samples for SHAP
    if len(X) > n_background:
        background_idx = np.random.choice(len(X), n_background, replace=False)
        background = X[background_idx]
    else:
        background = X

    # Create appropriate explainer based on model type
    model_class = type(model).__name__

    if 'XGB' in model_class or 'LGBM' in model_class or 'RandomForest' in model_class:
        # Tree-based models - use TreeExplainer (fast)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class

    elif 'LogisticRegression' in model_class:
        # Linear model - use LinearExplainer
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(X)

    else:
        # Generic explainer using KernelSHAP (slow but works for any model)
        logger.info("Using KernelSHAP (may be slow)...")

        if model_type == 'sklearn':
            predict_fn = lambda x: model.predict_proba(x)[:, 1]
        else:
            # PyTorch model
            import torch
            def predict_fn(x):
                model.eval()
                with torch.no_grad():
                    x_t = torch.FloatTensor(x)
                    return torch.sigmoid(model(x_t)).numpy()

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X[:min(500, len(X))])  # Limit samples for speed

    logger.info(f"SHAP values shape: {shap_values.shape}")
    return shap_values, explainer


def get_feature_importance(
    model,
    feature_names: List[str],
    model_type: str = 'sklearn'
) -> pd.DataFrame:
    """
    Extract feature importance from model.

    Returns:
        DataFrame with feature names and importance scores
    """
    model_class = type(model).__name__

    if hasattr(model, 'feature_importances_'):
        # Tree-based models (XGBoost, LightGBM, Random Forest)
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importance = np.abs(model.coef_).flatten()
    else:
        logger.warning("Model doesn't have standard feature importance attribute")
        return None

    # Create DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)

    return df


def map_features_to_variants(
    feature_importance: pd.DataFrame,
    embedding_dim: int,
    vep_feature_names: List[str],
    variant_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Map aggregated feature importance back to variants.

    Since features are aggregated across variants, we need to
    identify which variant-level features contribute most.
    """
    # Features are: [embedding_0, ..., embedding_N, vep_0, ..., vep_M]
    n_embedding_features = embedding_dim
    n_vep_features = len(vep_feature_names)

    # Split importance into embedding and VEP components
    embedding_importance = feature_importance[
        feature_importance['feature'].str.startswith('emb_')
    ]['importance'].sum()

    vep_importance = feature_importance[
        feature_importance['feature'].str.startswith('vep_')
    ]['importance'].sum()

    logger.info(f"Embedding features total importance: {embedding_importance:.4f}")
    logger.info(f"VEP features total importance: {vep_importance:.4f}")

    # Get top VEP features
    vep_features = feature_importance[
        feature_importance['feature'].str.startswith('vep_')
    ].head(20)

    return vep_features


def compute_variant_attribution(
    model,
    dosages: np.ndarray,
    embeddings: Dict[str, np.ndarray],
    vep_features: np.ndarray,
    variant_df: pd.DataFrame,
    labels: np.ndarray,
    model_type: str = 'sklearn'
) -> pd.DataFrame:
    """
    Compute per-variant attribution scores.

    Uses permutation importance at the variant level to measure
    how much each variant contributes to predictions.
    """
    from sklearn.metrics import roc_auc_score

    logger.info("Computing variant-level attribution...")

    n_samples, n_variants = dosages.shape

    # Get baseline predictions
    if 'diff' in embeddings:
        variant_emb = embeddings['diff']
    else:
        variant_emb = embeddings['alt'] - embeddings['ref']

    # Create sample features
    variant_features = np.hstack([variant_emb, vep_features])

    def create_sample_features(dos):
        sample_features = []
        for i in range(len(dos)):
            weights = dos[i]
            if weights.sum() > 0:
                weighted_feat = (weights[:, np.newaxis] * variant_features).sum(axis=0) / weights.sum()
            else:
                weighted_feat = np.zeros(variant_features.shape[1])
            sample_features.append(weighted_feat)
        return np.array(sample_features)

    # Scale features
    X = create_sample_features(dosages)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Baseline score
    if model_type == 'sklearn':
        y_pred = model.predict_proba(X_scaled)[:, 1]
    else:
        import torch
        model.eval()
        with torch.no_grad():
            y_pred = torch.sigmoid(model(torch.FloatTensor(X_scaled))).numpy()

    baseline_auroc = roc_auc_score(labels, y_pred)
    logger.info(f"Baseline AUROC: {baseline_auroc:.4f}")

    # Permutation importance for each variant
    attribution_scores = []

    # Only test top variants by dosage variance (for speed)
    dosage_variance = dosages.var(axis=0)
    top_variant_idx = np.argsort(dosage_variance)[-500:]  # Top 500 most variable

    for i, var_idx in enumerate(top_variant_idx):
        # Permute this variant's dosages
        permuted_dosages = dosages.copy()
        np.random.shuffle(permuted_dosages[:, var_idx])

        # Recompute features and predict
        X_perm = create_sample_features(permuted_dosages)
        X_perm_scaled = scaler.transform(X_perm)

        if model_type == 'sklearn':
            y_pred_perm = model.predict_proba(X_perm_scaled)[:, 1]
        else:
            with torch.no_grad():
                y_pred_perm = torch.sigmoid(model(torch.FloatTensor(X_perm_scaled))).numpy()

        perm_auroc = roc_auc_score(labels, y_pred_perm)
        importance = baseline_auroc - perm_auroc

        attribution_scores.append({
            'variant_idx': var_idx,
            'importance': importance,
            'dosage_variance': dosage_variance[var_idx]
        })

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(top_variant_idx)} variants")

    # Create DataFrame with variant info
    attr_df = pd.DataFrame(attribution_scores)

    # Merge with variant annotations
    if 'SNP' in variant_df.columns:
        attr_df['SNP'] = variant_df.iloc[attr_df['variant_idx']]['SNP'].values
    if 'CHR' in variant_df.columns:
        attr_df['CHR'] = variant_df.iloc[attr_df['variant_idx']]['CHR'].values
    if 'POS' in variant_df.columns:
        attr_df['POS'] = variant_df.iloc[attr_df['variant_idx']]['POS'].values
    if 'SYMBOL' in variant_df.columns:
        attr_df['gene'] = variant_df.iloc[attr_df['variant_idx']]['SYMBOL'].values
    if 'most_severe_consequence' in variant_df.columns:
        attr_df['consequence'] = variant_df.iloc[attr_df['variant_idx']]['most_severe_consequence'].values

    attr_df = attr_df.sort_values('importance', ascending=False)

    return attr_df


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    output_path: Path,
    top_n: int = 30
):
    """Create SHAP summary plot."""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not available for plotting")
        return

    # Select top features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-top_n:]

    plt.figure(figsize=(10, 12))
    shap.summary_plot(
        shap_values[:, top_idx],
        X[:, top_idx],
        feature_names=[feature_names[i] for i in top_idx],
        show=False,
        max_display=top_n
    )
    plt.tight_layout()
    plt.savefig(output_path / 'shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP summary plot to {output_path / 'shap_summary.png'}")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 30
):
    """Create feature importance bar plot."""
    plt.figure(figsize=(10, 10))

    top_features = importance_df.head(top_n)

    sns.barplot(
        data=top_features,
        x='importance',
        y='feature',
        palette='viridis'
    )
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.savefig(output_path / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature importance plot to {output_path / 'feature_importance.png'}")


def plot_variant_attribution(
    attr_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 30
):
    """Create variant attribution plot."""
    plt.figure(figsize=(12, 10))

    top_variants = attr_df.head(top_n).copy()

    # Create label with gene and position
    if 'gene' in top_variants.columns and 'POS' in top_variants.columns:
        top_variants['label'] = top_variants.apply(
            lambda x: f"{x['gene'] if pd.notna(x['gene']) else 'intergenic'}:{x['POS']}",
            axis=1
        )
    elif 'SNP' in top_variants.columns:
        top_variants['label'] = top_variants['SNP']
    else:
        top_variants['label'] = [f"var_{i}" for i in range(len(top_variants))]

    colors = ['#e74c3c' if x > 0 else '#3498db' for x in top_variants['importance']]

    plt.barh(range(len(top_variants)), top_variants['importance'], color=colors)
    plt.yticks(range(len(top_variants)), top_variants['label'])
    plt.xlabel('Attribution Score (AUROC decrease when permuted)')
    plt.ylabel('Variant')
    plt.title(f'Top {top_n} Variants by Attribution Score')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path / 'variant_attribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved variant attribution plot to {output_path / 'variant_attribution.png'}")


def plot_risk_score_distribution(
    y_pred: np.ndarray,
    labels: np.ndarray,
    output_path: Path
):
    """Plot risk score distribution for cases vs controls."""
    plt.figure(figsize=(10, 6))

    # Separate by class
    case_scores = y_pred[labels == 1]
    control_scores = y_pred[labels == 0]

    # Plot distributions
    plt.hist(control_scores, bins=50, alpha=0.6, label=f'Controls (n={len(control_scores)})',
             color='#3498db', density=True)
    plt.hist(case_scores, bins=50, alpha=0.6, label=f'Cases (n={len(case_scores)})',
             color='#e74c3c', density=True)

    plt.xlabel('Risk Score')
    plt.ylabel('Density')
    plt.title('Risk Score Distribution: Cases vs Controls')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'risk_score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved risk score distribution to {output_path / 'risk_score_distribution.png'}")


def plot_manhattan(
    attr_df: pd.DataFrame,
    output_path: Path
):
    """Create Manhattan-style plot of variant attributions."""
    if 'POS' not in attr_df.columns:
        logger.warning("Position information not available for Manhattan plot")
        return

    plt.figure(figsize=(14, 6))

    # Use absolute importance for y-axis
    y_values = np.abs(attr_df['importance'])
    x_values = attr_df['POS']

    # Color by direction of effect
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in attr_df['importance']]

    plt.scatter(x_values / 1e6, y_values, c=colors, alpha=0.6, s=20)

    # Highlight top variants
    top_n = 10
    top_variants = attr_df.head(top_n)
    for _, row in top_variants.iterrows():
        plt.annotate(
            row.get('gene', row.get('SNP', '')),
            (row['POS'] / 1e6, np.abs(row['importance'])),
            fontsize=8,
            alpha=0.8
        )

    plt.xlabel('Position (Mb)')
    plt.ylabel('|Attribution Score|')
    plt.title('Chromosome 6 Variant Attribution (Manhattan Plot)')
    plt.tight_layout()
    plt.savefig(output_path / 'manhattan_attribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Manhattan plot to {output_path / 'manhattan_attribution.png'}")


def run_pathway_enrichment(
    top_variants: pd.DataFrame,
    output_path: Path,
    top_n_genes: int = 50
):
    """
    Run simple pathway/gene set enrichment analysis.

    Uses g:Profiler API for enrichment if available.
    """
    if 'gene' not in top_variants.columns and 'SYMBOL' not in top_variants.columns:
        logger.warning("Gene information not available for enrichment analysis")
        return None

    gene_col = 'gene' if 'gene' in top_variants.columns else 'SYMBOL'
    genes = top_variants[gene_col].dropna().unique()[:top_n_genes]

    if len(genes) < 5:
        logger.warning(f"Too few genes ({len(genes)}) for enrichment analysis")
        return None

    logger.info(f"Running enrichment analysis for {len(genes)} genes...")

    try:
        import requests

        # Use g:Profiler API
        url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"

        payload = {
            "organism": "hsapiens",
            "query": list(genes),
            "sources": ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"],
            "user_threshold": 0.05,
            "significance_threshold_method": "fdr"
        }

        response = requests.post(url, json=payload, timeout=60)

        if response.status_code == 200:
            results = response.json()

            if 'result' in results and results['result']:
                enrichment_results = []
                for r in results['result']:
                    enrichment_results.append({
                        'source': r.get('source', ''),
                        'term_name': r.get('name', ''),
                        'term_id': r.get('native', ''),
                        'p_value': r.get('p_value', 1.0),
                        'term_size': r.get('term_size', 0),
                        'intersection_size': r.get('intersection_size', 0),
                        'genes': ','.join(r.get('intersections', []))
                    })

                enrichment_df = pd.DataFrame(enrichment_results)
                enrichment_df = enrichment_df.sort_values('p_value')

                # Save results
                enrichment_df.to_csv(output_path / 'pathway_enrichment.csv', index=False)
                logger.info(f"Found {len(enrichment_df)} enriched terms")
                logger.info(f"Saved enrichment results to {output_path / 'pathway_enrichment.csv'}")

                # Plot top enriched terms
                if len(enrichment_df) > 0:
                    plot_enrichment(enrichment_df, output_path)

                return enrichment_df
            else:
                logger.info("No significant enrichment found")
                return None
        else:
            logger.warning(f"Enrichment API returned status {response.status_code}")
            return None

    except ImportError:
        logger.warning("requests library not available for enrichment analysis")
        return None
    except Exception as e:
        logger.warning(f"Enrichment analysis failed: {e}")
        return None


def plot_enrichment(
    enrichment_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20
):
    """Plot top enriched pathways."""
    plt.figure(figsize=(12, 8))

    top_terms = enrichment_df.head(top_n).copy()
    top_terms['-log10(p)'] = -np.log10(top_terms['p_value'].clip(1e-50))

    # Color by source
    source_colors = {
        'GO:BP': '#2ecc71',
        'GO:MF': '#3498db',
        'GO:CC': '#9b59b6',
        'KEGG': '#e74c3c',
        'REAC': '#f39c12'
    }
    colors = [source_colors.get(s, '#95a5a6') for s in top_terms['source']]

    plt.barh(range(len(top_terms)), top_terms['-log10(p)'], color=colors)
    plt.yticks(range(len(top_terms)), top_terms['term_name'])
    plt.xlabel('-log10(p-value)')
    plt.title('Top Enriched Pathways/Terms')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=s) for s, c in source_colors.items()
                      if s in top_terms['source'].values]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path / 'pathway_enrichment.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved enrichment plot to {output_path / 'pathway_enrichment.png'}")


def interpret_model(args):
    """Main interpretation function."""
    logger.info("=" * 50)
    logger.info("Phase 6: Model Interpretation & Attribution")
    logger.info("=" * 50)

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, model_type = load_model(args.model)

    # Load scaler if available
    scaler_path = Path(args.model).parent / 'scaler.joblib'
    if scaler_path.exists():
        import joblib
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
    else:
        scaler = None

    # Load data (same as train_model.py)
    logger.info("Loading data...")

    # Load embeddings
    from pathlib import Path as P
    embeddings_path = P(args.embeddings)
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

    logger.info(f"Loaded embeddings: {list(embeddings.keys())}")
    embedding_dim = list(embeddings.values())[0].shape[1]

    # Load annotations
    annotations = pd.read_parquet(args.annotations)
    logger.info(f"Loaded {len(annotations)} variant annotations")

    # Encode VEP features (same as train_model.py)
    consequence_types = [
        'missense_variant', 'synonymous_variant', 'intron_variant',
        'upstream_gene_variant', 'downstream_gene_variant',
        '3_prime_UTR_variant', '5_prime_UTR_variant',
        'splice_region_variant', 'splice_donor_variant', 'splice_acceptor_variant',
        'stop_gained', 'stop_lost', 'start_lost', 'frameshift_variant',
        'inframe_insertion', 'inframe_deletion', 'regulatory_region_variant',
        'intergenic_variant'
    ]

    vep_feature_names = []
    vep_features_list = []

    if 'most_severe_consequence' in annotations.columns:
        for ctype in consequence_types:
            vep_features_list.append(
                (annotations['most_severe_consequence'] == ctype).astype(float).values.reshape(-1, 1)
            )
            vep_feature_names.append(f'vep_{ctype}')

    if 'impact' in annotations.columns:
        impact_map = {'MODIFIER': 0, 'LOW': 1, 'MODERATE': 2, 'HIGH': 3}
        vep_features_list.append(
            annotations['impact'].map(impact_map).fillna(0).values.reshape(-1, 1)
        )
        vep_feature_names.append('vep_impact')

    if 'sift_score' in annotations.columns:
        vep_features_list.append(annotations['sift_score'].fillna(1.0).values.reshape(-1, 1))
        vep_feature_names.append('vep_sift')

    if 'polyphen_score' in annotations.columns:
        vep_features_list.append(annotations['polyphen_score'].fillna(0.0).values.reshape(-1, 1))
        vep_feature_names.append('vep_polyphen')

    if vep_features_list:
        vep_features = np.hstack(vep_features_list)
    else:
        vep_features = np.zeros((len(annotations), 1))
        vep_feature_names = ['vep_placeholder']

    # Load dosages
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
    if label_col:
        cohort['label'] = cohort[label_col].astype(int)

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

    # Create sample features
    matched_dosages = dosages[matched_indices]
    labels = cohort.iloc[matched_cohort_indices]['label'].values

    if 'diff' in embeddings:
        variant_emb = embeddings['diff']
    else:
        variant_emb = embeddings['alt'] - embeddings['ref']

    variant_features = np.hstack([variant_emb, vep_features])

    # Create feature names
    feature_names = [f'emb_{i}' for i in range(embedding_dim)] + vep_feature_names

    # Create sample features (weighted aggregation)
    sample_features = []
    for i in range(len(matched_dosages)):
        weights = matched_dosages[i]
        if weights.sum() > 0:
            weighted_feat = (weights[:, np.newaxis] * variant_features).sum(axis=0) / weights.sum()
        else:
            weighted_feat = np.zeros(variant_features.shape[1])
        sample_features.append(weighted_feat)
    X = np.array(sample_features)

    # Scale features
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    logger.info(f"Sample features: {X_scaled.shape}")

    # Get predictions for risk score analysis
    if model_type == 'sklearn':
        y_pred = model.predict_proba(X_scaled)[:, 1]
    else:
        import torch
        model.eval()
        with torch.no_grad():
            y_pred = torch.sigmoid(model(torch.FloatTensor(X_scaled))).numpy()

    # 1. Feature importance
    logger.info("\n" + "=" * 50)
    logger.info("Computing feature importance...")
    importance_df = get_feature_importance(model, feature_names, model_type)
    if importance_df is not None:
        importance_df.to_csv(output_path / 'feature_importance.csv', index=False)
        logger.info(f"Top 10 features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        plot_feature_importance(importance_df, output_path)

    # 2. SHAP values
    logger.info("\n" + "=" * 50)
    shap_values, explainer = compute_shap_values(model, X_scaled, feature_names, model_type)
    if shap_values is not None:
        # Save mean SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_shap
        }).sort_values('mean_abs_shap', ascending=False)
        shap_df.to_csv(output_path / 'shap_importance.csv', index=False)

        plot_shap_summary(shap_values, X_scaled, feature_names, output_path)

    # 3. Variant attribution
    logger.info("\n" + "=" * 50)
    attr_df = compute_variant_attribution(
        model, matched_dosages, embeddings, vep_features,
        annotations, labels, model_type
    )
    attr_df.to_csv(output_path / 'variant_attribution.csv', index=False)
    logger.info(f"Top 10 variants by attribution:")
    for _, row in attr_df.head(10).iterrows():
        gene = row.get('gene', 'N/A')
        pos = row.get('POS', 'N/A')
        logger.info(f"  {gene}:{pos} - importance: {row['importance']:.6f}")

    plot_variant_attribution(attr_df, output_path)
    plot_manhattan(attr_df, output_path)

    # 4. Risk score distribution
    logger.info("\n" + "=" * 50)
    logger.info("Plotting risk score distribution...")
    plot_risk_score_distribution(y_pred, labels, output_path)

    # 5. Pathway enrichment
    logger.info("\n" + "=" * 50)
    enrichment_df = run_pathway_enrichment(attr_df, output_path)

    # Save summary
    summary = {
        'n_samples': len(labels),
        'n_cases': int((labels == 1).sum()),
        'n_controls': int((labels == 0).sum()),
        'n_variants': len(annotations),
        'n_features': X_scaled.shape[1],
        'embedding_dim': embedding_dim,
        'n_vep_features': len(vep_feature_names),
        'top_features': importance_df.head(20).to_dict('records') if importance_df is not None else [],
        'top_variants': attr_df.head(20).to_dict('records'),
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
        logger.info(f"  - {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Interpret AIS prediction model and identify important variants"
    )

    parser.add_argument(
        '--model', required=True,
        help='Path to trained model (model.joblib or model.pt)'
    )
    parser.add_argument(
        '--embeddings', required=True,
        help='Path prefix to HyenaDNA embeddings'
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
        help='Path to cohort definition parquet file'
    )
    parser.add_argument(
        '--output', '-o', required=True,
        help='Output directory for interpretation results'
    )

    args = parser.parse_args()
    interpret_model(args)


if __name__ == "__main__":
    main()
