#!/usr/bin/env python3
"""
Phase 1: Variant Extraction Pipeline

Extracts and QC-filters chromosome 6 variants for the AIS cohort.

Inputs:
    - PLINK2 files (.pgen/.pvar/.psam)
    - Cohort file (cohort.parquet)

Outputs:
    - variants.parquet: Variant metadata
    - dosages.zarr: Dosage matrix (n_samples, n_variants)
    - sample_ids.txt: Sample order in dosage matrix

Usage:
    python scripts/extract_variants.py \\
        --pfile /path/to/chr6 \\
        --cohort data/cohort/cohort.parquet \\
        --output-dir data/variants
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from variants.variant_qc import VariantQC, QCThresholds
from variants.dosage_extractor import DosageExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and QC-filter variants for AIS cohort"
    )
    parser.add_argument(
        "--pfile",
        required=True,
        help="PLINK2 file prefix (e.g., /path/to/chr6 for chr6.pgen/pvar/psam)"
    )
    parser.add_argument(
        "--cohort",
        required=True,
        help="Path to cohort.parquet file"
    )
    parser.add_argument(
        "--output-dir",
        default="data/variants",
        help="Output directory"
    )
    parser.add_argument(
        "--maf-min",
        type=float,
        default=0.01,
        help="Minimum minor allele frequency (default: 0.01)"
    )
    parser.add_argument(
        "--hwe-p-min",
        type=float,
        default=1e-6,
        help="Minimum HWE p-value (default: 1e-6)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Variants per batch for reading (default: 10000)"
    )
    parser.add_argument(
        "--skip-hwe",
        action="store_true",
        help="Skip HWE filtering (faster)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cohort
    logger.info(f"Loading cohort from {args.cohort}")
    cohort = pd.read_parquet(args.cohort)
    cohort_ids = cohort['eid'].astype(str).tolist()
    logger.info(f"Cohort has {len(cohort_ids)} samples")

    # Initialize extractor
    logger.info(f"Initializing dosage extractor for {args.pfile}")
    extractor = DosageExtractor(args.pfile)
    extractor.load_samples()

    # Initialize QC
    thresholds = QCThresholds(
        maf_min=args.maf_min,
        hwe_p_min=args.hwe_p_min,
        snps_only=True
    )
    qc = VariantQC(f"{args.pfile}.pvar", thresholds)

    # Load variants
    variants_df = qc.load_variants()

    # Filter to SNPs
    variants_df = qc.filter_snps_only()

    # Get variant indices after SNP filter
    snp_indices = variants_df.index.values

    # Extract dosages for cohort (SNPs only)
    logger.info("Extracting dosages for cohort...")
    sample_indices = extractor.get_sample_indices(cohort_ids)

    # Read dosages
    dosages, ordered_ids = extractor.extract_for_cohort(
        cohort_ids,
        batch_size=args.batch_size
    )

    # We need to subset to SNP indices only
    # First, reload full variant set to get mapping
    full_qc = VariantQC(f"{args.pfile}.pvar")
    full_variants = full_qc.load_variants()
    n_total_variants = len(full_variants)

    # The dosages array has all variants, subset to SNPs
    # Create SNP mask based on the original indices
    snp_mask = np.zeros(n_total_variants, dtype=bool)
    snp_mask[snp_indices] = True

    # Extract only SNP dosages
    dosages = dosages[:, snp_mask]
    logger.info(f"After SNP filter: {dosages.shape[1]} variants")

    # Reset variants_df index
    variants_df = variants_df.reset_index(drop=True)

    # Filter by MAF
    variants_df, dosages = qc.filter_maf(dosages)

    # Filter by HWE (optional)
    if not args.skip_hwe:
        variants_df, dosages = qc.filter_hwe(dosages)
    else:
        logger.info("Skipping HWE filter (--skip-hwe)")

    # Reset index for clean output
    variants_df = variants_df.reset_index(drop=True)

    # Add variant index
    variants_df['variant_idx'] = np.arange(len(variants_df))

    # Get QC stats
    qc_stats = qc.get_qc_report()

    # Save outputs
    logger.info("Saving outputs...")

    # 1. Variants parquet
    variants_path = output_dir / "variants.parquet"
    variants_df.to_parquet(variants_path, index=False)
    logger.info(f"Saved {len(variants_df):,} variants to {variants_path}")

    # 2. Sample IDs
    sample_path = output_dir / "sample_ids.txt"
    with open(sample_path, 'w') as f:
        for sid in ordered_ids:
            f.write(f"{sid}\n")
    logger.info(f"Saved {len(ordered_ids)} sample IDs to {sample_path}")

    # 3. Dosages as zarr
    zarr_path = output_dir / "dosages.zarr"
    zarr_array = zarr.open(
        str(zarr_path),
        mode='w',
        shape=dosages.shape,
        chunks=(min(100, dosages.shape[0]), min(10000, dosages.shape[1])),
        dtype='float32'
    )
    zarr_array[:] = dosages
    logger.info(f"Saved dosages {dosages.shape} to {zarr_path}")

    # 4. QC report
    report_path = output_dir / "qc_report.txt"
    with open(report_path, 'w') as f:
        f.write("Variant QC Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input file: {args.pfile}\n")
        f.write(f"Cohort samples: {len(cohort_ids)}\n")
        f.write(f"Samples found: {len(ordered_ids)}\n\n")
        f.write("QC Thresholds:\n")
        f.write(f"  MAF minimum: {args.maf_min}\n")
        f.write(f"  HWE p-value minimum: {args.hwe_p_min}\n")
        f.write(f"  SNPs only: True\n\n")
        f.write("Filter Results:\n")
        for key, value in qc_stats.items():
            f.write(f"  {key}: {value:,}\n")
        f.write(f"\nFinal variant count: {len(variants_df):,}\n")
        f.write(f"Final dosage matrix: {dosages.shape}\n")
    logger.info(f"Saved QC report to {report_path}")

    # Summary
    logger.info("=" * 50)
    logger.info("VARIANT EXTRACTION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Samples: {len(ordered_ids):,}")
    logger.info(f"Variants: {len(variants_df):,}")
    logger.info(f"Dosage matrix: {dosages.shape}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
