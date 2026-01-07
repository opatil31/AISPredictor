#!/usr/bin/env python3
"""
Phase 1: Variant Extraction Pipeline (PLINK2 export version)

This version uses PLINK2 command-line tool to filter and export variants,
avoiding the need for pgenlib (which has Windows compatibility issues).

STEP 1: Create cohort ID file
    python scripts/extract_variants_plink2.py prepare-ids --cohort data/cohort/cohort.parquet

STEP 2: Run PLINK2 (in terminal)
    plink2 --pfile /path/to/chr6 \\
        --keep data/variants/cohort_ids.txt \\
        --maf 0.01 \\
        --snps-only \\
        --hwe 1e-6 \\
        --export A \\
        --out data/variants/chr6_filtered

STEP 3: Convert to project format
    python scripts/extract_variants_plink2.py convert \\
        --raw data/variants/chr6_filtered.raw \\
        --output-dir data/variants
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_ids(args):
    """Prepare cohort IDs file for PLINK2 --keep."""
    logger.info(f"Loading cohort from {args.cohort}")
    cohort = pd.read_parquet(args.cohort)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # PLINK2 --keep expects FID IID format (one per line, tab or space separated)
    # For UK Biobank, FID = IID = eid
    ids_path = output_dir / "cohort_ids.txt"

    with open(ids_path, 'w') as f:
        for eid in cohort['eid']:
            # Write as "FID IID" format
            f.write(f"{eid}\t{eid}\n")

    logger.info(f"Wrote {len(cohort)} sample IDs to {ids_path}")
    logger.info("")
    logger.info("Next step: Run PLINK2 with the following command:")
    logger.info("")
    logger.info(f"  plink2 --pfile /path/to/chr6 \\")
    logger.info(f"      --keep {ids_path} \\")
    logger.info(f"      --maf 0.01 \\")
    logger.info(f"      --snps-only \\")
    logger.info(f"      --hwe 1e-6 \\")
    logger.info(f"      --export A \\")
    logger.info(f"      --out {output_dir}/chr6_filtered")
    logger.info("")
    logger.info("Then run: python scripts/extract_variants_plink2.py convert ...")


def convert_raw(args):
    """Convert PLINK2 .raw export to project format."""
    raw_path = Path(args.raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading PLINK2 export from {raw_path}")

    # .raw format: FID IID PAT MAT SEX PHENOTYPE <variant columns>
    # Variant columns are named: <rsid>_<allele>
    # Values are dosages (0, 1, 2, or NA)

    # Read header first to get variant names
    with open(raw_path, 'r') as f:
        header = f.readline().strip().split()

    # First 6 columns are metadata
    meta_cols = header[:6]
    variant_cols = header[6:]

    logger.info(f"Found {len(variant_cols)} variants")

    # Read full file
    logger.info("Reading dosage data...")
    df = pd.read_csv(raw_path, sep=r'\s+', na_values='NA')

    # Extract sample IDs
    sample_ids = df['IID'].astype(str).tolist()
    logger.info(f"Loaded {len(sample_ids)} samples")

    # Extract dosage matrix
    dosages = df[variant_cols].values.astype(np.float32)
    logger.info(f"Dosage matrix shape: {dosages.shape}")

    # Parse variant information from column names
    # Format: rsid_allele (e.g., rs123456_A)
    variants_data = []
    for col in variant_cols:
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            var_id, counted_allele = parts
        else:
            var_id = col
            counted_allele = 'Unknown'

        variants_data.append({
            'ID': var_id,
            'COUNTED_ALLELE': counted_allele,
            'variant_idx': len(variants_data)
        })

    variants_df = pd.DataFrame(variants_data)

    # Calculate MAF from dosages
    af = np.nanmean(dosages, axis=0) / 2
    maf = np.minimum(af, 1 - af)
    variants_df['AF'] = af
    variants_df['MAF'] = maf

    # Save outputs
    logger.info("Saving outputs...")

    # 1. Variants parquet
    variants_path = output_dir / "variants.parquet"
    variants_df.to_parquet(variants_path, index=False)
    logger.info(f"Saved {len(variants_df):,} variants to {variants_path}")

    # 2. Sample IDs
    sample_path = output_dir / "sample_ids.txt"
    with open(sample_path, 'w') as f:
        for sid in sample_ids:
            f.write(f"{sid}\n")
    logger.info(f"Saved {len(sample_ids)} sample IDs to {sample_path}")

    # 3. Dosages as zarr (or numpy if zarr not available)
    try:
        import zarr
        zarr_path = output_dir / "dosages.zarr"
        zarr_array = zarr.open(
            str(zarr_path),
            mode='w',
            shape=dosages.shape,
            chunks=(min(100, dosages.shape[0]), min(10000, dosages.shape[1])),
            dtype='float32'
        )
        zarr_array[:] = dosages
        logger.info(f"Saved dosages to {zarr_path}")
    except ImportError:
        # Fall back to numpy
        npy_path = output_dir / "dosages.npy"
        np.save(npy_path, dosages)
        logger.info(f"Saved dosages to {npy_path} (zarr not available)")

    # 4. QC report
    report_path = output_dir / "qc_report.txt"
    with open(report_path, 'w') as f:
        f.write("Variant Extraction Report (PLINK2 method)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input file: {raw_path}\n")
        f.write(f"Samples: {len(sample_ids)}\n")
        f.write(f"Variants: {len(variants_df)}\n\n")
        f.write("Note: QC filtering was performed by PLINK2\n")
        f.write("  --maf 0.01\n")
        f.write("  --snps-only\n")
        f.write("  --hwe 1e-6\n\n")
        f.write(f"MAF range: {maf.min():.4f} - {maf.max():.4f}\n")
        f.write(f"Missing rate: {np.isnan(dosages).mean() * 100:.2f}%\n")
    logger.info(f"Saved QC report to {report_path}")

    # Summary
    logger.info("=" * 50)
    logger.info("VARIANT EXTRACTION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Samples: {len(sample_ids):,}")
    logger.info(f"Variants: {len(variants_df):,}")
    logger.info(f"Dosage matrix: {dosages.shape}")
    logger.info(f"Output directory: {output_dir}")


def convert_raw_chunked(args):
    """Convert large PLINK2 .raw export using chunked reading."""
    raw_path = Path(args.raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading PLINK2 export from {raw_path} (chunked mode)")

    # Read header first
    with open(raw_path, 'r') as f:
        header = f.readline().strip().split()

    meta_cols = header[:6]
    variant_cols = header[6:]
    n_variants = len(variant_cols)

    logger.info(f"Found {n_variants} variants")

    # Count lines
    logger.info("Counting samples...")
    with open(raw_path, 'r') as f:
        n_samples = sum(1 for _ in f) - 1  # Subtract header

    logger.info(f"Found {n_samples} samples")

    # Initialize zarr array
    try:
        import zarr
        zarr_path = output_dir / "dosages.zarr"
        zarr_array = zarr.open(
            str(zarr_path),
            mode='w',
            shape=(n_samples, n_variants),
            chunks=(min(100, n_samples), min(10000, n_variants)),
            dtype='float32'
        )
    except ImportError:
        logger.error("zarr is required for chunked mode. Install with: pip install zarr")
        return

    # Read in chunks
    chunk_size = args.chunk_size
    sample_ids = []

    logger.info(f"Reading in chunks of {chunk_size}...")

    for i, chunk in enumerate(pd.read_csv(raw_path, sep=r'\s+', na_values='NA', chunksize=chunk_size)):
        start_idx = i * chunk_size
        end_idx = start_idx + len(chunk)

        sample_ids.extend(chunk['IID'].astype(str).tolist())
        zarr_array[start_idx:end_idx, :] = chunk[variant_cols].values.astype(np.float32)

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {end_idx:,} samples...")

    logger.info(f"Loaded {len(sample_ids)} samples")

    # Parse variant information
    variants_data = []
    for col in variant_cols:
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            var_id, counted_allele = parts
        else:
            var_id = col
            counted_allele = 'Unknown'
        variants_data.append({
            'ID': var_id,
            'COUNTED_ALLELE': counted_allele,
            'variant_idx': len(variants_data)
        })

    variants_df = pd.DataFrame(variants_data)

    # Calculate MAF from zarr (sample to avoid loading all data)
    logger.info("Calculating allele frequencies...")
    sample_size = min(1000, n_samples)
    sample_dosages = zarr_array[:sample_size, :]
    af = np.nanmean(sample_dosages, axis=0) / 2
    maf = np.minimum(af, 1 - af)
    variants_df['AF'] = af
    variants_df['MAF'] = maf

    # Save outputs
    variants_path = output_dir / "variants.parquet"
    variants_df.to_parquet(variants_path, index=False)
    logger.info(f"Saved {len(variants_df):,} variants to {variants_path}")

    sample_path = output_dir / "sample_ids.txt"
    with open(sample_path, 'w') as f:
        for sid in sample_ids:
            f.write(f"{sid}\n")
    logger.info(f"Saved {len(sample_ids)} sample IDs to {sample_path}")

    logger.info("=" * 50)
    logger.info("VARIANT EXTRACTION COMPLETE")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Extract variants using PLINK2 export"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # prepare-ids subcommand
    prep_parser = subparsers.add_parser(
        'prepare-ids',
        help='Prepare cohort ID file for PLINK2 --keep'
    )
    prep_parser.add_argument(
        '--cohort',
        required=True,
        help='Path to cohort.parquet'
    )
    prep_parser.add_argument(
        '--output-dir',
        default='data/variants',
        help='Output directory'
    )

    # convert subcommand
    conv_parser = subparsers.add_parser(
        'convert',
        help='Convert PLINK2 .raw export to project format'
    )
    conv_parser.add_argument(
        '--raw',
        required=True,
        help='Path to PLINK2 .raw export file'
    )
    conv_parser.add_argument(
        '--output-dir',
        default='data/variants',
        help='Output directory'
    )
    conv_parser.add_argument(
        '--chunked',
        action='store_true',
        help='Use chunked reading for large files'
    )
    conv_parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Samples per chunk (default: 1000)'
    )

    args = parser.parse_args()

    if args.command == 'prepare-ids':
        prepare_ids(args)
    elif args.command == 'convert':
        if args.chunked:
            convert_raw_chunked(args)
        else:
            convert_raw(args)


if __name__ == "__main__":
    main()
