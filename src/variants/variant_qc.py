"""
Variant quality control filtering for UK Biobank chromosome 6 data.

Filters:
- Minor Allele Frequency (MAF) >= 0.01
- Imputation INFO score >= 0.8 (if available)
- SNPs only (no indels)
- Hardy-Weinberg Equilibrium p > 1e-6
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QCThresholds:
    """Quality control thresholds for variant filtering."""
    maf_min: float = 0.01
    info_min: float = 0.8
    hwe_p_min: float = 1e-6
    snps_only: bool = True


class VariantQC:
    """
    Variant quality control for PLINK2 format files.

    Reads .pvar file and applies QC filters.
    """

    def __init__(self, pvar_path: str, thresholds: Optional[QCThresholds] = None):
        """
        Initialize VariantQC.

        Args:
            pvar_path: Path to .pvar file
            thresholds: QC thresholds (uses defaults if None)
        """
        self.pvar_path = Path(pvar_path)
        self.thresholds = thresholds or QCThresholds()
        self.variants_df: Optional[pd.DataFrame] = None
        self.qc_stats: dict = {}

    def load_variants(self) -> pd.DataFrame:
        """Load variant information from .pvar file."""
        logger.info(f"Loading variants from {self.pvar_path}")

        # Read .pvar file (tab-separated, may have header starting with #)
        # Standard columns: #CHROM, POS, ID, REF, ALT (+ optional INFO columns)

        # First, check if file has header
        with open(self.pvar_path, 'r') as f:
            first_line = f.readline()

        if first_line.startswith('#CHROM') or first_line.startswith('##'):
            # Has header - find the column header line
            skip_rows = 0
            with open(self.pvar_path, 'r') as f:
                for i, line in enumerate(f):
                    if line.startswith('#CHROM'):
                        skip_rows = i
                        break
                    elif line.startswith('##'):
                        continue
                    else:
                        break

            self.variants_df = pd.read_csv(
                self.pvar_path,
                sep='\t',
                skiprows=skip_rows,
                dtype={'#CHROM': str, 'POS': np.int64, 'ID': str, 'REF': str, 'ALT': str}
            )
            # Rename #CHROM to CHROM
            if '#CHROM' in self.variants_df.columns:
                self.variants_df = self.variants_df.rename(columns={'#CHROM': 'CHROM'})
        else:
            # No header - assume standard PLINK2 format
            self.variants_df = pd.read_csv(
                self.pvar_path,
                sep='\t',
                header=None,
                names=['CHROM', 'POS', 'ID', 'REF', 'ALT'],
                dtype={'CHROM': str, 'POS': np.int64, 'ID': str, 'REF': str, 'ALT': str}
            )

        n_variants = len(self.variants_df)
        logger.info(f"Loaded {n_variants:,} variants")
        self.qc_stats['n_total'] = n_variants

        return self.variants_df

    def filter_snps_only(self) -> pd.DataFrame:
        """Filter to SNPs only (single nucleotide variants)."""
        if self.variants_df is None:
            raise ValueError("Must call load_variants() first")

        n_before = len(self.variants_df)

        # SNPs have single-character REF and ALT alleles (A, C, G, T)
        is_snp = (
            self.variants_df['REF'].str.len() == 1
        ) & (
            self.variants_df['ALT'].str.len() == 1
        ) & (
            self.variants_df['REF'].isin(['A', 'C', 'G', 'T'])
        ) & (
            self.variants_df['ALT'].isin(['A', 'C', 'G', 'T'])
        )

        self.variants_df = self.variants_df[is_snp].copy()
        n_after = len(self.variants_df)

        logger.info(f"SNP filter: {n_before:,} -> {n_after:,} ({n_before - n_after:,} removed)")
        self.qc_stats['n_after_snp_filter'] = n_after
        self.qc_stats['n_indels_removed'] = n_before - n_after

        return self.variants_df

    def compute_allele_frequencies(self, dosages: np.ndarray) -> np.ndarray:
        """
        Compute allele frequencies from dosage matrix.

        Args:
            dosages: (n_samples, n_variants) dosage matrix

        Returns:
            Array of allele frequencies
        """
        # Allele frequency = mean dosage / 2
        af = np.nanmean(dosages, axis=0) / 2
        return af

    def filter_maf(self, dosages: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Filter variants by minor allele frequency.

        Args:
            dosages: (n_samples, n_variants) dosage matrix

        Returns:
            Filtered variants dataframe and dosage matrix
        """
        if self.variants_df is None:
            raise ValueError("Must call load_variants() first")

        n_before = len(self.variants_df)

        # Compute allele frequencies
        af = self.compute_allele_frequencies(dosages)

        # Minor allele frequency
        maf = np.minimum(af, 1 - af)

        # Store MAF in dataframe
        self.variants_df['AF'] = af
        self.variants_df['MAF'] = maf

        # Filter
        keep_mask = maf >= self.thresholds.maf_min

        self.variants_df = self.variants_df[keep_mask].copy()
        dosages = dosages[:, keep_mask]

        n_after = len(self.variants_df)

        logger.info(f"MAF filter (>= {self.thresholds.maf_min}): {n_before:,} -> {n_after:,} ({n_before - n_after:,} removed)")
        self.qc_stats['n_after_maf_filter'] = n_after
        self.qc_stats['n_low_maf_removed'] = n_before - n_after

        return self.variants_df, dosages

    def compute_hwe_pvalues(self, dosages: np.ndarray) -> np.ndarray:
        """
        Compute Hardy-Weinberg equilibrium p-values.

        Uses chi-square test comparing observed vs expected genotype frequencies.

        Args:
            dosages: (n_samples, n_variants) dosage matrix (rounded to 0/1/2)

        Returns:
            Array of HWE p-values
        """
        from scipy.stats import chisquare

        n_samples = dosages.shape[0]
        n_variants = dosages.shape[1]

        # Round dosages to hard calls
        hard_calls = np.round(dosages).astype(int)
        hard_calls = np.clip(hard_calls, 0, 2)

        hwe_pvals = np.zeros(n_variants)

        for i in range(n_variants):
            geno = hard_calls[:, i]
            valid = ~np.isnan(dosages[:, i])
            geno = geno[valid]
            n = len(geno)

            if n < 10:
                hwe_pvals[i] = np.nan
                continue

            # Observed genotype counts
            n_aa = np.sum(geno == 0)
            n_ab = np.sum(geno == 1)
            n_bb = np.sum(geno == 2)

            # Allele frequency
            p = (2 * n_aa + n_ab) / (2 * n)
            q = 1 - p

            # Expected counts under HWE
            exp_aa = n * p * p
            exp_ab = n * 2 * p * q
            exp_bb = n * q * q

            # Chi-square test
            observed = np.array([n_aa, n_ab, n_bb])
            expected = np.array([exp_aa, exp_ab, exp_bb])

            # Avoid division by zero
            if np.any(expected < 1):
                hwe_pvals[i] = np.nan
            else:
                try:
                    _, pval = chisquare(observed, expected)
                    hwe_pvals[i] = pval
                except:
                    hwe_pvals[i] = np.nan

        return hwe_pvals

    def filter_hwe(self, dosages: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Filter variants by Hardy-Weinberg equilibrium.

        Args:
            dosages: (n_samples, n_variants) dosage matrix

        Returns:
            Filtered variants dataframe and dosage matrix
        """
        if self.variants_df is None:
            raise ValueError("Must call load_variants() first")

        n_before = len(self.variants_df)

        logger.info("Computing HWE p-values...")
        hwe_pvals = self.compute_hwe_pvalues(dosages)

        self.variants_df['HWE_P'] = hwe_pvals

        # Filter: keep variants with p > threshold (not violating HWE)
        keep_mask = (hwe_pvals > self.thresholds.hwe_p_min) | np.isnan(hwe_pvals)

        self.variants_df = self.variants_df[keep_mask].copy()
        dosages = dosages[:, keep_mask]

        n_after = len(self.variants_df)

        logger.info(f"HWE filter (p > {self.thresholds.hwe_p_min}): {n_before:,} -> {n_after:,} ({n_before - n_after:,} removed)")
        self.qc_stats['n_after_hwe_filter'] = n_after
        self.qc_stats['n_hwe_removed'] = n_before - n_after

        return self.variants_df, dosages

    def get_qc_report(self) -> dict:
        """Return QC statistics."""
        return self.qc_stats.copy()
