#!/usr/bin/env python3
"""
Mock Attribution Visualization

Generates visually appealing mock-up plots showing what the attribution
results might look like if the model successfully recovers the GPR126/ADGRG6
locus and intronic signal for Adolescent Idiopathic Scoliosis (AIS).

Based on known AIS genetics literature:
- GPR126/ADGRG6 at 6q24.1 is the most replicated chr6 signal
- rs6570507 is the lead SNP (intronic)
- Intronic variants suggest regulatory mechanism
- HLA region variants also implicated

Usage:
    python scripts/visualize_mock_attributions.py --output figures/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)


# =============================================================================
# Mock Data Generation (Based on AIS Genetics Literature)
# =============================================================================

# Known/hypothetical AIS-associated genes on chromosome 6
# GPR126/ADGRG6 is the most replicated signal
MOCK_GENES = {
    # Gene: (position_start, position_end, expected_attribution, primary_region)
    'ADGRG6': (142_500_000, 142_800_000, 0.35, 'intron'),      # GPR126 - main signal
    'ESR1': (151_650_000, 152_100_000, 0.12, 'intron'),         # Estrogen receptor
    'FOXP4': (41_500_000, 41_600_000, 0.08, 'promoter'),        # Forkhead TF
    'COL9A1': (70_900_000, 71_100_000, 0.07, 'exon'),           # Collagen
    'HLA-DQB1': (32_600_000, 32_650_000, 0.06, 'exon'),         # HLA region
    'RUNX2': (45_300_000, 45_500_000, 0.05, 'intron'),          # Bone development
    'GDF5': (137_800_000, 137_850_000, 0.04, 'utr3'),           # Growth factor
    'BMP6': (7_700_000, 7_900_000, 0.04, 'intron'),             # Bone morphogenetic
    'CITED2': (139_600_000, 139_700_000, 0.03, 'promoter'),     # CBP/p300 interacting
    'SOX4': (21_590_000, 21_600_000, 0.03, 'utr5'),             # SRY-box TF
    'CDKN1A': (36_640_000, 36_660_000, 0.02, 'exon'),           # Cell cycle
    'VEGFA': (43_700_000, 43_750_000, 0.02, 'intron'),          # Angiogenesis
    'HLA-B': (31_350_000, 31_360_000, 0.02, 'exon'),            # HLA region
    'TFAP2B': (50_750_000, 50_800_000, 0.015, 'intron'),        # TF AP-2 beta
    'MEF2C': (87_700_000, 88_200_000, 0.015, 'intron'),         # Myocyte enhancer
    'PARK2': (161_700_000, 163_100_000, 0.01, 'intron'),        # Parkinson gene
    'HMGA1': (34_200_000, 34_250_000, 0.01, 'promoter'),        # High mobility group
    'IGF2R': (160_350_000, 160_500_000, 0.01, 'intron'),        # IGF receptor
    'CNPY3': (42_850_000, 42_900_000, 0.008, 'exon'),           # Canopy FGF signaling
    'BCKDHB': (80_100_000, 80_250_000, 0.007, 'splice'),        # BCAA metabolism
}

REGION_TYPES = ['promoter', 'utr5', 'exon', 'intron', 'utr3', 'splice', 'downstream', 'other']
REGION_COLORS = {
    'promoter': '#E74C3C',    # Red
    'utr5': '#E67E22',        # Orange
    'exon': '#F1C40F',        # Yellow
    'intron': '#27AE60',      # Green (dominant for GPR126)
    'utr3': '#3498DB',        # Blue
    'splice': '#9B59B6',      # Purple
    'downstream': '#1ABC9C',  # Teal
    'other': '#95A5A6',       # Gray
}


def generate_mock_variant_attributions(n_variants: int = 5000) -> pd.DataFrame:
    """Generate mock variant-level attribution data."""

    variants = []

    for gene, (start, end, gene_attr, primary_region) in MOCK_GENES.items():
        # Number of variants per gene (proportional to size and importance)
        gene_size = end - start
        n_gene_variants = max(10, int(n_variants * gene_attr * 2))

        for i in range(n_gene_variants):
            pos = np.random.randint(start, end)

            # Assign region type (weighted toward primary region)
            if np.random.random() < 0.6:
                region = primary_region
            else:
                region = np.random.choice(REGION_TYPES, p=[0.1, 0.05, 0.15, 0.4, 0.1, 0.05, 0.1, 0.05])

            # Attribution score (log-normal distribution, higher for key genes)
            base_attr = gene_attr / n_gene_variants * np.random.lognormal(0, 0.5)

            # Boost intronic variants for GPR126 (the known biology)
            if gene == 'ADGRG6' and region == 'intron':
                base_attr *= 1.5

            variants.append({
                'variant_id': f'chr6_{pos}_{np.random.choice(["A", "C", "G", "T"])}_{np.random.choice(["A", "C", "G", "T"])}',
                'CHR': 6,
                'POS': pos,
                'gene': gene,
                'region_type': region,
                'attribution': base_attr,
                'abs_attribution': abs(base_attr),
            })

    # Add some intergenic variants with low attribution
    for _ in range(n_variants // 5):
        pos = np.random.randint(1_000_000, 170_000_000)
        variants.append({
            'variant_id': f'chr6_{pos}_intergenic',
            'CHR': 6,
            'POS': pos,
            'gene': 'intergenic',
            'region_type': 'other',
            'attribution': np.random.exponential(0.001),
            'abs_attribution': np.random.exponential(0.001),
        })

    df = pd.DataFrame(variants)

    # Normalize attributions to sum to 1
    df['attribution'] = df['attribution'] / df['attribution'].sum()
    df['abs_attribution'] = df['abs_attribution'] / df['abs_attribution'].sum()

    return df


def generate_mock_patient_data(n_patients: int = 100) -> pd.DataFrame:
    """Generate mock patient-level attribution summaries."""

    patients = []

    for i in range(n_patients):
        is_case = i < n_patients // 2  # First half are cases

        # Cases should have higher GPR126 attribution on average
        gpr126_attr = np.random.beta(5, 2) if is_case else np.random.beta(2, 5)

        # Generate gene attributions
        gene_attrs = {}
        for gene, (_, _, expected_attr, _) in MOCK_GENES.items():
            if gene == 'ADGRG6':
                gene_attrs[gene] = gpr126_attr * 0.4
            else:
                gene_attrs[gene] = np.random.exponential(expected_attr)

        # Normalize
        total = sum(gene_attrs.values())
        gene_attrs = {k: v/total for k, v in gene_attrs.items()}

        patients.append({
            'patient_id': f'UKB_{1000000 + i}',
            'is_case': is_case,
            'risk_score': 0.7 + 0.2 * np.random.randn() if is_case else 0.3 + 0.2 * np.random.randn(),
            'top_gene': max(gene_attrs, key=gene_attrs.get),
            'ADGRG6_attribution': gene_attrs['ADGRG6'],
            **{f'{g}_attr': v for g, v in gene_attrs.items()}
        })

    return pd.DataFrame(patients)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_gene_ranking(variant_df: pd.DataFrame, output_path: Path, top_n: int = 20):
    """Create bar chart of top genes by attribution."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Aggregate by gene
    gene_attrs = variant_df.groupby('gene')['abs_attribution'].sum().sort_values(ascending=False)
    gene_attrs = gene_attrs[gene_attrs.index != 'intergenic'].head(top_n)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color bars by whether they're the known signal
    colors = ['#E74C3C' if g == 'ADGRG6' else '#3498DB' for g in gene_attrs.index]

    bars = ax.barh(range(len(gene_attrs)), gene_attrs.values, color=colors, edgecolor='white', linewidth=0.5)

    # Customize
    ax.set_yticks(range(len(gene_attrs)))
    ax.set_yticklabels(gene_attrs.index, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Aggregated Attribution Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Genes by Attribution Score\n(Chromosome 6 - AIS Prediction)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, gene_attrs.values)):
        ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=9)

    # Highlight GPR126/ADGRG6
    ax.annotate('Known AIS\nsusceptibility locus',
                xy=(gene_attrs['ADGRG6'], 0),
                xytext=(gene_attrs['ADGRG6'] + 0.08, 2),
                fontsize=10, color='#E74C3C',
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', edgecolor='#E74C3C'))

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', label='Validated AIS gene (ADGRG6/GPR126)'),
        Patch(facecolor='#3498DB', label='Other contributing genes')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, gene_attrs.max() * 1.3)

    plt.tight_layout()
    plt.savefig(output_path / 'gene_ranking_bar.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path / 'gene_ranking_bar.pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved gene ranking plot to {output_path / 'gene_ranking_bar.png'}")


def plot_region_pie_chart(variant_df: pd.DataFrame, output_path: Path):
    """Create pie chart of attribution by region type."""
    import matplotlib.pyplot as plt

    # Aggregate by region
    region_attrs = variant_df.groupby('region_type')['abs_attribution'].sum()
    region_attrs = region_attrs.reindex(REGION_TYPES).fillna(0)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Overall pie chart
    colors = [REGION_COLORS[r] for r in region_attrs.index]
    explode = [0.05 if r == 'intron' else 0 for r in region_attrs.index]

    wedges, texts, autotexts = ax1.pie(
        region_attrs.values,
        labels=region_attrs.index,
        colors=colors,
        explode=explode,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        pctdistance=0.75,
        startangle=90,
        wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2)
    )

    ax1.set_title('Overall Attribution by Region Type', fontsize=13, fontweight='bold', pad=20)

    # Add center annotation
    ax1.annotate('Intronic\ndominance\n(regulatory)', xy=(0, 0), fontsize=10,
                ha='center', va='center', fontweight='bold', color='#27AE60')

    # GPR126-specific pie chart
    gpr126_df = variant_df[variant_df['gene'] == 'ADGRG6']
    gpr126_regions = gpr126_df.groupby('region_type')['abs_attribution'].sum()
    gpr126_regions = gpr126_regions.reindex(REGION_TYPES).fillna(0)

    colors2 = [REGION_COLORS[r] for r in gpr126_regions.index]
    explode2 = [0.08 if r == 'intron' else 0 for r in gpr126_regions.index]

    wedges2, texts2, autotexts2 = ax2.pie(
        gpr126_regions.values,
        labels=gpr126_regions.index,
        colors=colors2,
        explode=explode2,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        pctdistance=0.75,
        startangle=90,
        wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2)
    )

    ax2.set_title('ADGRG6 (GPR126) Attribution by Region', fontsize=13, fontweight='bold', pad=20)

    # Highlight intronic signal
    intron_pct = gpr126_regions['intron'] / gpr126_regions.sum() * 100
    ax2.annotate(f'{intron_pct:.0f}% intronic\n(validates known\nbiology)',
                xy=(0, 0), fontsize=10, ha='center', va='center',
                fontweight='bold', color='#27AE60')

    plt.tight_layout()
    plt.savefig(output_path / 'region_pie_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path / 'region_pie_chart.pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved region pie chart to {output_path / 'region_pie_chart.png'}")


def plot_manhattan(variant_df: pd.DataFrame, output_path: Path):
    """Create Manhattan-style plot of variant attributions."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(16, 6))

    # Sort by position
    df = variant_df.sort_values('POS').copy()
    df['log_attr'] = -np.log10(df['abs_attribution'].clip(lower=1e-10))

    # Color by region type
    colors = df['region_type'].map(REGION_COLORS)

    # Plot all variants
    scatter = ax.scatter(
        df['POS'] / 1e6,
        df['log_attr'],
        c=colors,
        s=15,
        alpha=0.6,
        edgecolors='none'
    )

    # Highlight top variants
    top_variants = df.nlargest(20, 'abs_attribution')
    ax.scatter(
        top_variants['POS'] / 1e6,
        top_variants['log_attr'],
        c='red',
        s=80,
        marker='*',
        edgecolors='black',
        linewidths=0.5,
        zorder=5,
        label='Top 20 variants'
    )

    # Add gene labels for top regions
    for gene, (start, end, attr, _) in list(MOCK_GENES.items())[:8]:
        gene_variants = df[(df['POS'] >= start) & (df['POS'] <= end)]
        if len(gene_variants) > 0:
            max_var = gene_variants.loc[gene_variants['log_attr'].idxmax()]
            ax.annotate(
                gene,
                xy=(max_var['POS'] / 1e6, max_var['log_attr']),
                xytext=(0, 10),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold' if gene == 'ADGRG6' else 'normal',
                color='#E74C3C' if gene == 'ADGRG6' else '#2C3E50',
                ha='center'
            )

    # Highlight GPR126 region
    gpr126_start, gpr126_end = 142.5, 142.8
    ax.axvspan(gpr126_start, gpr126_end, alpha=0.2, color='#E74C3C',
               label='ADGRG6 locus')

    # Customize
    ax.set_xlabel('Chromosome 6 Position (Mb)', fontsize=12, fontweight='bold')
    ax.set_ylabel('-log₁₀(Attribution Score)', fontsize=12, fontweight='bold')
    ax.set_title('Variant Attribution Manhattan Plot - Chromosome 6\n(AIS Case vs Control Prediction)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add significance threshold line
    threshold = -np.log10(0.001)
    ax.axhline(y=threshold, color='#E74C3C', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(170, threshold + 0.1, 'High attribution threshold', fontsize=9, color='#E74C3C')

    # Legend for region types
    legend_patches = [mpatches.Patch(color=color, label=region)
                     for region, color in REGION_COLORS.items()]
    legend_patches.append(mpatches.Patch(color='#E74C3C', alpha=0.2, label='ADGRG6 locus'))

    ax.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=3)

    ax.set_xlim(0, 171)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path / 'manhattan_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path / 'manhattan_plot.pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved Manhattan plot to {output_path / 'manhattan_plot.png'}")


def plot_patient_report(patient_df: pd.DataFrame, variant_df: pd.DataFrame, output_path: Path):
    """Create example patient-specific attribution report."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Select a high-risk case patient
    case_patients = patient_df[patient_df['is_case'] == True].nlargest(1, 'ADGRG6_attribution')
    patient = case_patients.iloc[0]

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(f'Patient Attribution Report: {patient["patient_id"]}\n'
                 f'Predicted Risk Score: {patient["risk_score"]:.2f} | Status: {"Case" if patient["is_case"] else "Control"}',
                 fontsize=14, fontweight='bold', y=0.98)

    # 1. Top genes for this patient
    ax1 = fig.add_subplot(gs[0, 0])
    gene_cols = [c for c in patient.index if c.endswith('_attr') and c != 'ADGRG6_attribution']
    gene_attrs = patient[gene_cols].sort_values(ascending=True).tail(10)
    gene_names = [c.replace('_attr', '') for c in gene_attrs.index]

    colors = ['#E74C3C' if 'ADGRG6' in g else '#3498DB' for g in gene_names]
    ax1.barh(range(len(gene_attrs)), gene_attrs.values, color=colors)
    ax1.set_yticks(range(len(gene_attrs)))
    ax1.set_yticklabels(gene_names, fontsize=10)
    ax1.set_xlabel('Attribution Score', fontsize=10)
    ax1.set_title('Top Contributing Genes', fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 2. Region breakdown
    ax2 = fig.add_subplot(gs[0, 1])

    # Mock patient-specific region breakdown
    region_attrs = {
        'intron': 0.45,
        'exon': 0.18,
        'promoter': 0.15,
        'utr5': 0.08,
        'utr3': 0.07,
        'splice': 0.04,
        'downstream': 0.02,
        'other': 0.01
    }

    colors = [REGION_COLORS[r] for r in region_attrs.keys()]
    wedges, texts, autotexts = ax2.pie(
        region_attrs.values(),
        labels=region_attrs.keys(),
        colors=colors,
        autopct='%1.0f%%',
        startangle=90,
        wedgeprops=dict(width=0.6, edgecolor='white')
    )
    ax2.set_title('Attribution by Region Type', fontsize=12, fontweight='bold')

    # 3. Key findings text box
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    findings_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════╗
    ║                              KEY FINDINGS FOR {patient["patient_id"]}                              ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                          ║
    ║  🧬 PRIMARY GENETIC DRIVER:                                                               ║
    ║     • ADGRG6 (GPR126) contributes {patient["ADGRG6_attribution"]*100:.1f}% of total attribution                           ║
    ║     • Predominantly intronic variants (45%) suggesting regulatory mechanism               ║
    ║     • Consistent with known AIS susceptibility locus at 6q24.1                           ║
    ║                                                                                          ║
    ║  📊 RISK ASSESSMENT:                                                                      ║
    ║     • Predicted probability: {patient["risk_score"]*100:.1f}%                                                       ║
    ║     • Risk category: {"HIGH" if patient["risk_score"] > 0.6 else "MODERATE" if patient["risk_score"] > 0.4 else "LOW"}                                                                   ║
    ║                                                                                          ║
    ║  🔬 SECONDARY CONTRIBUTORS:                                                               ║
    ║     • ESR1 (estrogen receptor): Supports hormonal influence hypothesis                   ║
    ║     • COL9A1 (collagen): Structural component of spinal disc                             ║
    ║     • HLA-DQB1: Immune/inflammatory pathway involvement                                  ║
    ║                                                                                          ║
    ║  💡 CLINICAL RELEVANCE:                                                                   ║
    ║     • Strong intronic signal suggests potential for therapeutic targeting                ║
    ║     • Consider enhanced monitoring given genetic risk profile                            ║
    ║                                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════╝
    """

    ax3.text(0.5, 0.5, findings_text, transform=ax3.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#2C3E50', linewidth=2))

    plt.savefig(output_path / 'patient_report_example.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path / 'patient_report_example.pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved patient report to {output_path / 'patient_report_example.png'}")


def plot_case_control_comparison(patient_df: pd.DataFrame, output_path: Path):
    """Create case vs control attribution comparison."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. GPR126 attribution distribution
    ax1 = axes[0]
    cases = patient_df[patient_df['is_case'] == True]['ADGRG6_attribution']
    controls = patient_df[patient_df['is_case'] == False]['ADGRG6_attribution']

    ax1.hist(controls, bins=20, alpha=0.7, color='#3498DB', label='Controls', density=True)
    ax1.hist(cases, bins=20, alpha=0.7, color='#E74C3C', label='Cases', density=True)
    ax1.axvline(cases.mean(), color='#E74C3C', linestyle='--', linewidth=2, label=f'Case mean: {cases.mean():.3f}')
    ax1.axvline(controls.mean(), color='#3498DB', linestyle='--', linewidth=2, label=f'Control mean: {controls.mean():.3f}')

    ax1.set_xlabel('ADGRG6 Attribution', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('ADGRG6 Attribution: Case vs Control', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 2. Risk score distribution
    ax2 = axes[1]
    case_scores = patient_df[patient_df['is_case'] == True]['risk_score']
    control_scores = patient_df[patient_df['is_case'] == False]['risk_score']

    ax2.hist(control_scores, bins=20, alpha=0.7, color='#3498DB', label='Controls', density=True)
    ax2.hist(case_scores, bins=20, alpha=0.7, color='#E74C3C', label='Cases', density=True)

    ax2.set_xlabel('Predicted Risk Score', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Risk Score Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 3. Top gene comparison heatmap
    ax3 = axes[2]

    # Calculate mean attribution per gene for cases and controls
    gene_cols = [c for c in patient_df.columns if c.endswith('_attr')]
    case_means = patient_df[patient_df['is_case'] == True][gene_cols].mean()
    control_means = patient_df[patient_df['is_case'] == False][gene_cols].mean()

    comparison_df = pd.DataFrame({
        'Cases': case_means,
        'Controls': control_means
    })
    comparison_df.index = [c.replace('_attr', '') for c in comparison_df.index]
    comparison_df = comparison_df.sort_values('Cases', ascending=False).head(10)

    sns.heatmap(comparison_df, annot=True, fmt='.3f', cmap='RdYlBu_r',
                ax=ax3, cbar_kws={'label': 'Mean Attribution'})
    ax3.set_title('Top Genes: Case vs Control', fontsize=12, fontweight='bold')
    ax3.set_ylabel('')

    plt.tight_layout()
    plt.savefig(output_path / 'case_control_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path / 'case_control_comparison.pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved case-control comparison to {output_path / 'case_control_comparison.png'}")


def plot_gpr126_zoom(variant_df: pd.DataFrame, output_path: Path):
    """Create detailed zoom view of GPR126/ADGRG6 locus."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])

    # Filter to GPR126 region
    gpr126_df = variant_df[(variant_df['POS'] >= 142_400_000) & (variant_df['POS'] <= 142_900_000)].copy()
    gpr126_df['log_attr'] = -np.log10(gpr126_df['abs_attribution'].clip(lower=1e-10))

    # Top panel: variant attributions
    colors = gpr126_df['region_type'].map(REGION_COLORS)
    ax1.scatter(gpr126_df['POS'] / 1e6, gpr126_df['log_attr'], c=colors, s=50, alpha=0.7, edgecolors='white', linewidth=0.5)

    # Highlight lead SNP (rs6570507 position approximation)
    lead_snp_pos = 142.665  # Approximate position
    ax1.axvline(x=lead_snp_pos, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.7)
    ax1.annotate('rs6570507\n(Lead SNP)', xy=(lead_snp_pos, ax1.get_ylim()[1] * 0.9),
                fontsize=10, color='#E74C3C', fontweight='bold', ha='center')

    ax1.set_ylabel('-log₁₀(Attribution)', fontsize=11, fontweight='bold')
    ax1.set_title('ADGRG6 (GPR126) Locus Detail - 6q24.1\nValidated AIS Susceptibility Gene',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=color, label=region) for region, color in REGION_COLORS.items()]
    ax1.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=4)

    # Bottom panel: gene structure
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(0, 1)

    # Draw gene body
    gene_start, gene_end = 142.50, 142.80
    ax2.add_patch(plt.Rectangle((gene_start, 0.4), gene_end - gene_start, 0.2,
                                 facecolor='#2C3E50', edgecolor='black', linewidth=1))

    # Draw exons (mock positions)
    exon_positions = [
        (142.52, 142.53), (142.55, 142.56), (142.60, 142.62),
        (142.66, 142.67), (142.70, 142.72), (142.75, 142.77)
    ]
    for start, end in exon_positions:
        ax2.add_patch(plt.Rectangle((start, 0.3), end - start, 0.4,
                                     facecolor='#F1C40F', edgecolor='black', linewidth=1))

    # Add arrow for transcription direction
    ax2.annotate('', xy=(gene_end + 0.02, 0.5), xytext=(gene_end - 0.05, 0.5),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))

    ax2.text((gene_start + gene_end) / 2, 0.15, 'ADGRG6 (GPR126)',
            ha='center', fontsize=12, fontweight='bold', color='#2C3E50')
    ax2.text((gene_start + gene_end) / 2, 0.02, 'Exons shown in yellow; intronic regions contain regulatory variants',
            ha='center', fontsize=9, style='italic', color='#7F8C8D')

    ax2.set_xlabel('Chromosome 6 Position (Mb)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Gene Structure', fontsize=11)
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path / 'gpr126_zoom.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path / 'gpr126_zoom.pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved GPR126 zoom plot to {output_path / 'gpr126_zoom.png'}")


def generate_bed_file(variant_df: pd.DataFrame, output_path: Path):
    """Generate BED file for genome browser visualization."""

    # Select top variants
    top_variants = variant_df.nlargest(1000, 'abs_attribution').copy()

    # Create BED format
    bed_lines = ['track name="AIS_Attribution" description="Variant Attribution Scores" useScore=1']

    for _, row in top_variants.iterrows():
        # BED format: chrom, start, end, name, score (0-1000)
        score = int(min(1000, row['abs_attribution'] * 10000))
        bed_lines.append(f"chr6\t{int(row['POS'])-1}\t{int(row['POS'])}\t{row['gene']}_{row['region_type']}\t{score}")

    bed_path = output_path / 'variant_attributions.bed'
    with open(bed_path, 'w') as f:
        f.write('\n'.join(bed_lines))

    logger.info(f"Saved BED file to {bed_path}")


def generate_summary_json(variant_df: pd.DataFrame, patient_df: pd.DataFrame, output_path: Path):
    """Generate summary statistics JSON."""

    # Gene-level summary
    gene_summary = variant_df.groupby('gene').agg({
        'abs_attribution': ['sum', 'mean', 'count'],
        'region_type': lambda x: x.mode().iloc[0] if len(x) > 0 else 'unknown'
    }).round(4)
    gene_summary.columns = ['total_attribution', 'mean_attribution', 'n_variants', 'primary_region']
    gene_summary = gene_summary.sort_values('total_attribution', ascending=False).head(20)

    # Region summary
    region_summary = variant_df.groupby('region_type')['abs_attribution'].sum()
    region_summary = (region_summary / region_summary.sum() * 100).round(1)

    # Validation metrics
    gpr126_attr = gene_summary.loc['ADGRG6', 'total_attribution'] if 'ADGRG6' in gene_summary.index else 0
    gpr126_rank = list(gene_summary.index).index('ADGRG6') + 1 if 'ADGRG6' in gene_summary.index else -1
    intronic_pct = region_summary.get('intron', 0)

    summary = {
        'validation_metrics': {
            'gpr126_in_top_5_percent': gpr126_rank <= 1,
            'gpr126_rank': gpr126_rank,
            'gpr126_total_attribution': float(gpr126_attr),
            'intronic_signal_percent': float(intronic_pct),
            'intronic_above_30_percent': intronic_pct > 30
        },
        'top_genes': gene_summary.to_dict('index'),
        'region_breakdown_percent': region_summary.to_dict(),
        'patient_summary': {
            'n_cases': int(patient_df['is_case'].sum()),
            'n_controls': int((~patient_df['is_case']).sum()),
            'mean_case_gpr126_attr': float(patient_df[patient_df['is_case']]['ADGRG6_attribution'].mean()),
            'mean_control_gpr126_attr': float(patient_df[~patient_df['is_case']]['ADGRG6_attribution'].mean())
        }
    }

    json_path = output_path / 'attribution_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved summary JSON to {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate mock attribution visualizations")
    parser.add_argument('--output', '-o', default='figures', help='Output directory for figures')
    parser.add_argument('--n-variants', type=int, default=5000, help='Number of mock variants')
    parser.add_argument('--n-patients', type=int, default=100, help='Number of mock patients')

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Generating Mock Attribution Visualizations")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_path}")

    # Generate mock data
    logger.info("\nGenerating mock data...")
    variant_df = generate_mock_variant_attributions(args.n_variants)
    patient_df = generate_mock_patient_data(args.n_patients)

    logger.info(f"Generated {len(variant_df):,} variant attributions")
    logger.info(f"Generated {len(patient_df):,} patient records")

    # Save mock data
    variant_df.to_parquet(output_path / 'mock_variant_attributions.parquet', index=False)
    patient_df.to_parquet(output_path / 'mock_patient_attributions.parquet', index=False)

    # Generate visualizations
    logger.info("\nGenerating visualizations...")

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        plot_gene_ranking(variant_df, output_path)
        plot_region_pie_chart(variant_df, output_path)
        plot_manhattan(variant_df, output_path)
        plot_patient_report(patient_df, variant_df, output_path)
        plot_case_control_comparison(patient_df, output_path)
        plot_gpr126_zoom(variant_df, output_path)
        generate_bed_file(variant_df, output_path)

    except ImportError as e:
        logger.warning(f"Matplotlib/seaborn not available: {e}")
        logger.warning("Skipping visualizations, generating data files only")

    # Generate summary
    summary = generate_summary_json(variant_df, patient_df, output_path)

    # Print validation summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"GPR126/ADGRG6 rank: #{summary['validation_metrics']['gpr126_rank']}")
    logger.info(f"GPR126 in top 5%: {summary['validation_metrics']['gpr126_in_top_5_percent']}")
    logger.info(f"Intronic signal: {summary['validation_metrics']['intronic_signal_percent']:.1f}%")
    logger.info(f"Intronic > 30%: {summary['validation_metrics']['intronic_above_30_percent']}")
    logger.info("=" * 60)

    logger.info(f"\nAll outputs saved to: {output_path}/")
    logger.info("Files generated:")
    for f in sorted(output_path.iterdir()):
        logger.info(f"  - {f.name}")


if __name__ == "__main__":
    main()
