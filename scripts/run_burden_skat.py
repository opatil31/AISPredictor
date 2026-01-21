#!/usr/bin/env python3
"""
Burden Test and SKAT Analysis for Rare Variants

Implements gene-level rare variant association tests:
1. Burden Test - Tests if cumulative rare variant burden differs between cases/controls
2. SKAT - Sequence Kernel Association Test (allows bi-directional effects)
3. SKAT-O - Optimal combination of burden and SKAT

These are gold-standard methods for rare variant analysis in genetics.

Usage:
    python scripts/run_burden_skat.py \
        --dosages data/variants_rare/dosages.zarr \
        --annotations data/variants_rare/variants_annotated.parquet \
        --cohort data/cohort.parquet \
        --output results/burden_skat \
        --method all

Output:
    - Gene-level p-values and effect sizes
    - QQ plots for p-value calibration
    - Manhattan-style plots
    - Comparison of methods
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.special import comb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress convergence warnings for logistic regression
warnings.filterwarnings('ignore', category=UserWarning)

# Region type mapping
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


def compute_burden_score(genotypes: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute burden score for each individual.

    Args:
        genotypes: (n_samples, n_variants) genotype matrix
        weights: Optional variant weights (e.g., based on MAF or function)

    Returns:
        (n_samples,) burden scores
    """
    if weights is None:
        weights = np.ones(genotypes.shape[1])

    # Handle missing values
    genotypes_filled = np.nan_to_num(genotypes, nan=0)

    return np.dot(genotypes_filled, weights)


def burden_test(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Perform burden test using logistic regression.

    Tests H0: burden score is not associated with phenotype.

    Args:
        genotypes: (n_samples, n_variants) genotype matrix for a gene
        phenotypes: (n_samples,) binary phenotype (0/1)
        covariates: Optional (n_samples, n_covariates) covariate matrix
        weights: Optional variant weights

    Returns:
        Dict with p-value, beta, se, and other statistics
    """
    n_samples, n_variants = genotypes.shape

    if n_variants == 0:
        return {'p_value': np.nan, 'beta': np.nan, 'se': np.nan, 'z_score': np.nan}

    # Compute burden score
    burden = compute_burden_score(genotypes, weights)

    # Check for variation
    if burden.std() == 0:
        return {'p_value': np.nan, 'beta': np.nan, 'se': np.nan, 'z_score': np.nan}

    # Standardize burden
    burden_scaled = (burden - burden.mean()) / burden.std()

    # Prepare design matrix
    if covariates is not None:
        X = np.column_stack([burden_scaled, covariates])
    else:
        X = burden_scaled.reshape(-1, 1)

    try:
        # Fit logistic regression
        model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        model.fit(X, phenotypes)

        # Get coefficient for burden
        beta = model.coef_[0, 0]

        # Compute standard error using Hessian
        pred_probs = model.predict_proba(X)[:, 1]
        W = np.diag(pred_probs * (1 - pred_probs))

        # Information matrix
        try:
            info_matrix = X.T @ W @ X
            cov_matrix = np.linalg.inv(info_matrix)
            se = np.sqrt(cov_matrix[0, 0])
        except np.linalg.LinAlgError:
            se = np.nan

        # Wald test
        if se > 0 and not np.isnan(se):
            z_score = beta / se
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
        else:
            z_score = np.nan
            p_value = np.nan

        return {
            'p_value': p_value,
            'beta': beta,
            'se': se,
            'z_score': z_score,
            'burden_mean_cases': burden[phenotypes == 1].mean(),
            'burden_mean_controls': burden[phenotypes == 0].mean()
        }

    except Exception as e:
        logger.debug(f"Burden test failed: {e}")
        return {'p_value': np.nan, 'beta': np.nan, 'se': np.nan, 'z_score': np.nan}


def skat_test(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    kernel: str = 'linear'
) -> Dict:
    """
    Perform SKAT (Sequence Kernel Association Test).

    SKAT is a variance-component test that aggregates variant effects
    using a kernel and tests for association. It's powerful when variants
    have different effect directions.

    Args:
        genotypes: (n_samples, n_variants) genotype matrix for a gene
        phenotypes: (n_samples,) binary phenotype (0/1)
        covariates: Optional covariate matrix
        weights: Optional variant weights (default: Beta(1,25) based on MAF)
        kernel: 'linear' or 'quadratic'

    Returns:
        Dict with p-value and test statistic
    """
    n_samples, n_variants = genotypes.shape

    if n_variants == 0:
        return {'p_value': np.nan, 'Q_stat': np.nan}

    # Handle missing values
    G = np.nan_to_num(genotypes, nan=0)
    y = phenotypes.astype(float)

    # Compute MAF-based weights if not provided
    if weights is None:
        maf = G.mean(axis=0) / 2
        maf = np.clip(maf, 0.001, 0.999)  # Avoid edge cases
        # Beta(1, 25) weights - upweight rare variants
        weights = stats.beta.pdf(maf, 1, 25)
        weights = weights / weights.sum() * n_variants  # Normalize

    # Weight the genotypes
    G_weighted = G * np.sqrt(weights)

    # Fit null model (phenotype ~ covariates)
    if covariates is not None:
        X0 = np.column_stack([np.ones(n_samples), covariates])
    else:
        X0 = np.ones((n_samples, 1))

    try:
        # Fit null logistic regression
        null_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        if covariates is not None:
            null_model.fit(covariates, y)
            mu = null_model.predict_proba(covariates)[:, 1]
        else:
            # Just use prevalence
            mu = np.full(n_samples, y.mean())

        # Residuals under null
        residuals = y - mu

        # Compute kernel matrix
        if kernel == 'linear':
            K = G_weighted @ G_weighted.T
        else:  # quadratic
            K_linear = G_weighted @ G_weighted.T
            K = (1 + K_linear) ** 2

        # SKAT test statistic: Q = (y - mu)' K (y - mu) / 2
        Q = residuals @ K @ residuals / 2

        # Compute p-value using Davies' method approximation
        # For simplicity, use a scaled chi-square approximation

        # Variance weights
        V = np.diag(mu * (1 - mu))

        # P0 = V - V X0 (X0' V X0)^-1 X0' V  (projection matrix)
        try:
            VX0 = V @ X0
            inv_term = np.linalg.inv(X0.T @ V @ X0)
            P0 = V - VX0 @ inv_term @ VX0.T
        except np.linalg.LinAlgError:
            P0 = V

        # Expected value and variance of Q under null
        # Using moment-matching to chi-square
        PKP = P0 @ K @ P0

        E_Q = np.trace(PKP) / 2

        # Variance approximation
        Var_Q = np.trace(PKP @ PKP)

        if Var_Q > 0 and E_Q > 0:
            # Match to scaled chi-square: a * chi^2(df)
            # E[Q] = a * df, Var[Q] = 2 * a^2 * df
            df = 2 * E_Q ** 2 / Var_Q
            scale = Var_Q / (2 * E_Q)

            # P-value
            p_value = 1 - stats.chi2.cdf(Q / scale, df)
        else:
            p_value = np.nan

        return {
            'p_value': p_value,
            'Q_stat': Q,
            'E_Q': E_Q,
            'df': df if Var_Q > 0 else np.nan
        }

    except Exception as e:
        logger.debug(f"SKAT test failed: {e}")
        return {'p_value': np.nan, 'Q_stat': np.nan}


def skat_o_test(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    rho_values: Optional[List[float]] = None
) -> Dict:
    """
    Perform SKAT-O (Optimal SKAT).

    SKAT-O combines burden and SKAT tests optimally by searching over
    a grid of rho values (rho=0 is SKAT, rho=1 is burden).

    Args:
        genotypes: (n_samples, n_variants) genotype matrix
        phenotypes: (n_samples,) binary phenotype
        covariates: Optional covariate matrix
        weights: Optional variant weights
        rho_values: Grid of rho values to search (default: [0, 0.25, 0.5, 0.75, 1])

    Returns:
        Dict with optimal p-value, best rho, and component results
    """
    if rho_values is None:
        rho_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    n_samples, n_variants = genotypes.shape

    if n_variants == 0:
        return {'p_value': np.nan, 'best_rho': np.nan}

    # Run burden test
    burden_result = burden_test(genotypes, phenotypes, covariates, weights)

    # Run SKAT test
    skat_result = skat_test(genotypes, phenotypes, covariates, weights)

    # Simple SKAT-O: take minimum p-value with Bonferroni-like correction
    # (Full SKAT-O uses more sophisticated combination)
    p_values = []

    for rho in rho_values:
        if rho == 0:
            p = skat_result['p_value']
        elif rho == 1:
            p = burden_result['p_value']
        else:
            # Interpolate (simplified)
            p_skat = skat_result['p_value'] if not np.isnan(skat_result['p_value']) else 1.0
            p_burden = burden_result['p_value'] if not np.isnan(burden_result['p_value']) else 1.0
            p = (1 - rho) * p_skat + rho * p_burden

        p_values.append(p if not np.isnan(p) else 1.0)

    # Find optimal rho
    min_idx = np.argmin(p_values)
    min_p = p_values[min_idx]
    best_rho = rho_values[min_idx]

    # Apply correction for multiple testing across rho values
    # Using simple Bonferroni (conservative)
    p_corrected = min(min_p * len(rho_values), 1.0)

    return {
        'p_value': p_corrected,
        'p_value_uncorrected': min_p,
        'best_rho': best_rho,
        'burden_p': burden_result['p_value'],
        'skat_p': skat_result['p_value'],
        'burden_beta': burden_result.get('beta', np.nan)
    }


def run_gene_level_tests(
    dosages: np.ndarray,
    phenotypes: np.ndarray,
    variant_genes: List[str],
    variant_consequences: List[str],
    methods: List[str] = ['burden', 'skat', 'skat_o'],
    min_variants: int = 2
) -> pd.DataFrame:
    """
    Run association tests for all genes.

    Args:
        dosages: (n_samples, n_variants) genotype matrix
        phenotypes: (n_samples,) binary phenotype
        variant_genes: Gene symbol for each variant
        variant_consequences: VEP consequence for each variant
        methods: Which tests to run
        min_variants: Minimum variants per gene

    Returns:
        DataFrame with gene-level results
    """
    # Get unique genes
    unique_genes = list(set(g for g in variant_genes if g and g != '' and not pd.isna(g)))
    logger.info(f"Testing {len(unique_genes)} genes...")

    results = []

    for i, gene in enumerate(unique_genes):
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(unique_genes)} genes...")

        # Get variants in this gene
        gene_mask = np.array([g == gene for g in variant_genes])
        n_variants = gene_mask.sum()

        if n_variants < min_variants:
            continue

        gene_genotypes = dosages[:, gene_mask]

        # Get region breakdown
        gene_consequences = [variant_consequences[j] for j in range(len(variant_consequences)) if gene_mask[j]]
        region_counts = {}
        for cons in gene_consequences:
            region = get_region_type(cons)
            region_counts[region] = region_counts.get(region, 0) + 1

        result = {
            'gene': gene,
            'n_variants': n_variants,
            'n_carriers': (gene_genotypes.sum(axis=1) > 0).sum(),
            'carrier_rate': (gene_genotypes.sum(axis=1) > 0).mean(),
            **{f'{r}_count': region_counts.get(r, 0) for r in REGION_TYPES}
        }

        # Run selected tests
        if 'burden' in methods:
            burden_result = burden_test(gene_genotypes, phenotypes)
            result['burden_p'] = burden_result['p_value']
            result['burden_beta'] = burden_result.get('beta', np.nan)
            result['burden_z'] = burden_result.get('z_score', np.nan)

        if 'skat' in methods:
            skat_result = skat_test(gene_genotypes, phenotypes)
            result['skat_p'] = skat_result['p_value']
            result['skat_Q'] = skat_result.get('Q_stat', np.nan)

        if 'skat_o' in methods:
            skato_result = skat_o_test(gene_genotypes, phenotypes)
            result['skato_p'] = skato_result['p_value']
            result['skato_rho'] = skato_result.get('best_rho', np.nan)

        results.append(result)

    df = pd.DataFrame(results)

    # Sort by best p-value
    p_cols = [c for c in df.columns if c.endswith('_p')]
    if p_cols:
        df['min_p'] = df[p_cols].min(axis=1)
        df = df.sort_values('min_p').reset_index(drop=True)

    return df


def compute_genomic_inflation(p_values: np.ndarray) -> float:
    """Compute genomic inflation factor (lambda)."""
    p_valid = p_values[~np.isnan(p_values) & (p_values > 0) & (p_values < 1)]
    if len(p_valid) == 0:
        return np.nan

    chi2_obs = stats.chi2.ppf(1 - p_valid, df=1)
    lambda_gc = np.median(chi2_obs) / stats.chi2.ppf(0.5, df=1)
    return lambda_gc


def plot_qq(p_values: np.ndarray, method_name: str, output_path: Path):
    """Create QQ plot for p-values."""
    plt.figure(figsize=(8, 8))

    p_valid = p_values[~np.isnan(p_values) & (p_values > 0) & (p_values < 1)]

    if len(p_valid) == 0:
        logger.warning(f"No valid p-values for {method_name} QQ plot")
        return

    # Sort p-values
    p_sorted = np.sort(p_valid)
    n = len(p_sorted)

    # Expected p-values under null
    expected = np.arange(1, n + 1) / (n + 1)

    # -log10 transform
    obs_log = -np.log10(p_sorted)
    exp_log = -np.log10(expected)

    # Compute lambda
    lambda_gc = compute_genomic_inflation(p_valid)

    # Plot
    plt.scatter(exp_log, obs_log, alpha=0.6, s=20)

    # Diagonal line
    max_val = max(exp_log.max(), obs_log.max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=1)

    # Confidence interval
    ci_upper = -np.log10(stats.beta.ppf(0.025, np.arange(1, n + 1), np.arange(n, 0, -1)))
    ci_lower = -np.log10(stats.beta.ppf(0.975, np.arange(1, n + 1), np.arange(n, 0, -1)))
    plt.fill_between(exp_log, ci_lower, ci_upper, alpha=0.2, color='gray')

    plt.xlabel('Expected -log10(p)')
    plt.ylabel('Observed -log10(p)')
    plt.title(f'{method_name} QQ Plot\n(λ = {lambda_gc:.3f})')
    plt.tight_layout()
    plt.savefig(output_path / f'qq_{method_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {method_name} QQ plot (λ = {lambda_gc:.3f})")


def plot_manhattan_genes(
    gene_df: pd.DataFrame,
    p_col: str,
    method_name: str,
    output_path: Path,
    top_n_label: int = 10
):
    """Create Manhattan-style plot for gene-level results."""
    plt.figure(figsize=(14, 6))

    df = gene_df.dropna(subset=[p_col]).copy()

    if len(df) == 0:
        logger.warning(f"No valid p-values for {method_name} Manhattan plot")
        return

    # -log10 p-values
    df['neg_log_p'] = -np.log10(df[p_col].clip(lower=1e-300))

    # Sort by gene name for consistent ordering
    df = df.sort_values('gene').reset_index(drop=True)
    df['x'] = range(len(df))

    # Alternating colors
    colors = ['#1f77b4' if i % 2 == 0 else '#2ca02c' for i in range(len(df))]

    plt.scatter(df['x'], df['neg_log_p'], c=colors, alpha=0.6, s=20)

    # Significance threshold
    bonf_threshold = -np.log10(0.05 / len(df))
    plt.axhline(bonf_threshold, color='red', linestyle='--', linewidth=1,
                label=f'Bonferroni (p={0.05/len(df):.2e})')

    # Label top genes
    top_genes = df.nlargest(top_n_label, 'neg_log_p')
    for _, row in top_genes.iterrows():
        plt.annotate(
            row['gene'],
            (row['x'], row['neg_log_p']),
            fontsize=8,
            alpha=0.8,
            xytext=(5, 5),
            textcoords='offset points'
        )

    plt.xlabel('Gene')
    plt.ylabel('-log10(p-value)')
    plt.title(f'{method_name} Gene-Level Association')
    plt.legend(loc='upper right')
    plt.xticks([])  # Too many genes to label
    plt.tight_layout()
    plt.savefig(output_path / f'manhattan_{method_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {method_name} Manhattan plot")


def plot_method_comparison(gene_df: pd.DataFrame, output_path: Path):
    """Compare p-values across methods."""
    methods = []
    if 'burden_p' in gene_df.columns:
        methods.append(('burden_p', 'Burden'))
    if 'skat_p' in gene_df.columns:
        methods.append(('skat_p', 'SKAT'))
    if 'skato_p' in gene_df.columns:
        methods.append(('skato_p', 'SKAT-O'))

    if len(methods) < 2:
        return

    n_plots = len(methods) * (len(methods) - 1) // 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    plot_idx = 0
    for i, (col1, name1) in enumerate(methods):
        for j, (col2, name2) in enumerate(methods[i+1:], i+1):
            ax = axes[plot_idx]

            valid = gene_df[[col1, col2]].dropna()
            x = -np.log10(valid[col1].clip(lower=1e-300))
            y = -np.log10(valid[col2].clip(lower=1e-300))

            ax.scatter(x, y, alpha=0.5, s=10)

            # Diagonal
            max_val = max(x.max(), y.max())
            ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1)

            # Correlation
            corr = np.corrcoef(x, y)[0, 1]
            ax.set_xlabel(f'-log10(p) {name1}')
            ax.set_ylabel(f'-log10(p) {name2}')
            ax.set_title(f'{name1} vs {name2}\n(r = {corr:.3f})')

            plot_idx += 1

    plt.tight_layout()
    plt.savefig(output_path / 'method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved method comparison plot")


def plot_top_genes(gene_df: pd.DataFrame, output_path: Path, top_n: int = 20):
    """Bar plot of top genes by significance."""
    p_cols = [c for c in gene_df.columns if c.endswith('_p')]

    if 'min_p' not in gene_df.columns and p_cols:
        gene_df['min_p'] = gene_df[p_cols].min(axis=1)

    if 'min_p' not in gene_df.columns:
        return

    top = gene_df.nsmallest(top_n, 'min_p').copy()
    top['neg_log_p'] = -np.log10(top['min_p'].clip(lower=1e-300))

    plt.figure(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top)))

    plt.barh(range(len(top)), top['neg_log_p'], color=colors)
    plt.yticks(range(len(top)), top['gene'])
    plt.xlabel('-log10(p-value)')
    plt.ylabel('Gene')
    plt.title(f'Top {top_n} Genes by Association Significance')

    # Add Bonferroni line
    bonf = -np.log10(0.05 / len(gene_df))
    plt.axvline(bonf, color='red', linestyle='--', label=f'Bonferroni threshold')
    plt.legend()

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path / 'top_genes.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved top genes plot")


def main():
    parser = argparse.ArgumentParser(
        description="Run burden test and SKAT for rare variant analysis"
    )

    parser.add_argument('--dosages', required=True,
                       help='Path to dosages (zarr or parquet)')
    parser.add_argument('--annotations', required=True,
                       help='Path to annotated variants parquet')
    parser.add_argument('--cohort', required=True,
                       help='Path to cohort parquet')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory')
    parser.add_argument('--method', default='all',
                       choices=['burden', 'skat', 'skat_o', 'all'],
                       help='Which test(s) to run (default: all)')
    parser.add_argument('--min-variants', type=int, default=2,
                       help='Minimum variants per gene (default: 2)')
    parser.add_argument('--functional-only', action='store_true',
                       help='Only test genes with functional variants (missense, splice, stop, frameshift)')
    parser.add_argument('--burden-only', action='store_true',
                       help='Run only burden test (fastest, skip SKAT)')

    args = parser.parse_args()

    # Override method if burden-only specified
    if args.burden_only:
        args.method = 'burden'

    logger.info("=" * 50)
    logger.info("Burden Test / SKAT Analysis")
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

    # Filter to functional variants if requested
    if args.functional_only:
        functional_consequences = {
            'missense_variant', 'stop_gained', 'stop_lost', 'start_lost',
            'frameshift_variant', 'inframe_insertion', 'inframe_deletion',
            'splice_acceptor_variant', 'splice_donor_variant', 'splice_region_variant',
            'protein_altering_variant', 'coding_sequence_variant'
        }

        functional_mask = []
        for cons in variant_consequences:
            is_functional = False
            if cons:
                for c in cons.split(','):
                    if c.strip() in functional_consequences:
                        is_functional = True
                        break
            functional_mask.append(is_functional)

        functional_mask = np.array(functional_mask)
        n_functional = functional_mask.sum()

        if n_functional == 0:
            logger.warning("No functional variants found! Running on all variants instead.")
        else:
            logger.info(f"Filtering to {n_functional} functional variants (from {len(functional_mask)})")
            dosages = dosages[:, functional_mask]
            variant_genes = [g for g, m in zip(variant_genes, functional_mask) if m]
            variant_consequences = [c for c, m in zip(variant_consequences, functional_mask) if m]

    # Load cohort
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
    else:
        X = dosages
        y = cohort[label_col].values[:len(dosages)].astype(int)

    logger.info(f"Matched {len(X)} samples")
    logger.info(f"Cases: {y.sum()}, Controls: {len(y) - y.sum()}")

    # Handle missing values
    if np.isnan(X).any():
        logger.info("Imputing missing values with 0 (no minor allele)...")
        X = np.nan_to_num(X, nan=0)

    # Determine methods
    if args.method == 'all':
        methods = ['burden', 'skat', 'skat_o']
    else:
        methods = [args.method]

    # Run tests
    logger.info("\n" + "=" * 50)
    logger.info(f"Running Gene-Level Tests: {', '.join(methods)}")
    logger.info("=" * 50)

    gene_df = run_gene_level_tests(
        X, y, variant_genes, variant_consequences,
        methods=methods,
        min_variants=args.min_variants
    )

    # Save results
    gene_df.to_csv(output_path / 'gene_results.csv', index=False)
    logger.info(f"\nSaved results for {len(gene_df)} genes")

    # Summary statistics
    logger.info("\n" + "=" * 50)
    logger.info("Results Summary")
    logger.info("=" * 50)

    for method in methods:
        p_col = f'{method}_p'
        if p_col in gene_df.columns:
            valid_p = gene_df[p_col].dropna()
            n_sig_nominal = (valid_p < 0.05).sum()
            n_sig_bonf = (valid_p < 0.05 / len(valid_p)).sum()
            lambda_gc = compute_genomic_inflation(valid_p.values)

            logger.info(f"\n{method.upper()}:")
            logger.info(f"  Genes tested: {len(valid_p)}")
            logger.info(f"  Significant (p<0.05): {n_sig_nominal}")
            logger.info(f"  Significant (Bonferroni): {n_sig_bonf}")
            logger.info(f"  Genomic inflation (λ): {lambda_gc:.3f}")

    # Top genes
    logger.info("\nTop 10 genes by significance:")
    for _, row in gene_df.head(10).iterrows():
        p_vals = []
        for method in methods:
            p_col = f'{method}_p'
            if p_col in row and not np.isnan(row[p_col]):
                p_vals.append(f"{method}={row[p_col]:.2e}")
        logger.info(f"  {row['gene']}: {', '.join(p_vals)} ({row['n_variants']} variants)")

    # Generate visualizations
    logger.info("\n" + "=" * 50)
    logger.info("Generating Visualizations")
    logger.info("=" * 50)

    for method in methods:
        p_col = f'{method}_p'
        if p_col in gene_df.columns:
            plot_qq(gene_df[p_col].values, method.upper(), output_path)
            plot_manhattan_genes(gene_df, p_col, method.upper(), output_path)

    plot_method_comparison(gene_df, output_path)
    plot_top_genes(gene_df, output_path)

    # Save summary
    summary = {
        'n_samples': len(X),
        'n_cases': int(y.sum()),
        'n_controls': int(len(y) - y.sum()),
        'n_variants': X.shape[1],
        'n_genes_tested': len(gene_df),
        'methods': methods,
        'min_variants_per_gene': args.min_variants,
        'top_genes': gene_df.head(20).to_dict('records')
    }

    for method in methods:
        p_col = f'{method}_p'
        if p_col in gene_df.columns:
            valid_p = gene_df[p_col].dropna()
            summary[f'{method}_lambda'] = float(compute_genomic_inflation(valid_p.values))
            summary[f'{method}_n_sig_nominal'] = int((valid_p < 0.05).sum())
            summary[f'{method}_n_sig_bonferroni'] = int((valid_p < 0.05 / len(valid_p)).sum())

    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n" + "=" * 50)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Output directory: {output_path}")
    logger.info("\nGenerated files:")
    for f in sorted(output_path.glob('*')):
        if f.is_file():
            logger.info(f"  - {f.name}")


if __name__ == "__main__":
    main()
