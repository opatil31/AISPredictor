#!/usr/bin/env python3
"""
Phase 1: Variant Extraction Pipeline (PLINK2 export version)

This version uses PLINK2 command-line tool to filter and export variants,
avoiding the need for pgenlib (which has Windows compatibility issues).

Usage:
    python scripts/extract_variants_plink2.py \\
        --pfile /path/to/chr6 \\
        --cohort data/cohort/cohort.parquet \\
        --output-dir data/variants

The script will:
1. Find PLINK2 on your system
2. Create cohort ID file
3. Run PLINK2 with QC filters
4. Convert output to project format
"""

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def find_plink2() -> str:
    """
    Find PLINK2 executable on the system.

    Returns:
        Path to plink2 executable

    Raises:
        FileNotFoundError if plink2 cannot be found
    """
    # Common executable names
    exe_names = ['plink2', 'plink2.exe']

    # First, check if it's in PATH
    for name in exe_names:
        path = shutil.which(name)
        if path:
            logger.info(f"Found PLINK2 in PATH: {path}")
            return path

    # Common installation locations on Windows
    if platform.system() == 'Windows':
        common_paths = [
            Path.home() / 'plink2' / 'plink2.exe',
            Path.home() / 'plink' / 'plink2.exe',
            Path.home() / 'Downloads' / 'plink2.exe',
            Path.home() / 'Downloads' / 'plink2_win64' / 'plink2.exe',
            Path.home() / 'Downloads' / 'plink2_win_avx2' / 'plink2.exe',
            Path.home() / 'Desktop' / 'plink2.exe',
            Path('C:/plink2/plink2.exe'),
            Path('C:/plink/plink2.exe'),
            Path('C:/Program Files/plink2/plink2.exe'),
            Path('C:/Program Files (x86)/plink2/plink2.exe'),
            Path.home() / 'AppData' / 'Local' / 'plink2' / 'plink2.exe',
        ]

        # Also search Documents folder
        docs = Path.home() / 'Documents'
        if docs.exists():
            common_paths.extend([
                docs / 'plink2.exe',
                docs / 'plink2' / 'plink2.exe',
            ])
            # Search subdirectories of Documents
            for subdir in docs.iterdir():
                if subdir.is_dir():
                    common_paths.append(subdir / 'plink2.exe')
    else:
        # Linux/macOS paths
        common_paths = [
            Path.home() / 'plink2' / 'plink2',
            Path.home() / 'bin' / 'plink2',
            Path('/usr/local/bin/plink2'),
            Path('/usr/bin/plink2'),
            Path('/opt/plink2/plink2'),
        ]

    for path in common_paths:
        if path.exists():
            logger.info(f"Found PLINK2 at: {path}")
            return str(path)

    # Try searching the entire home directory (limited depth)
    logger.info("Searching home directory for plink2...")
    home = Path.home()
    for root, dirs, files in os.walk(home):
        # Limit search depth
        depth = len(Path(root).relative_to(home).parts)
        if depth > 4:
            dirs.clear()  # Don't descend further
            continue

        # Skip hidden directories and common large directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in
                   ['node_modules', 'venv', '.venv', '__pycache__', 'AppData', '.git']]

        for name in exe_names:
            if name in files:
                path = Path(root) / name
                logger.info(f"Found PLINK2 at: {path}")
                return str(path)

    raise FileNotFoundError(
        "Could not find PLINK2. Please either:\n"
        "1. Add plink2 to your PATH\n"
        "2. Specify the path with --plink2-path\n"
        "3. Download from https://www.cog-genomics.org/plink/2.0/"
    )


def run_plink2(plink2_path: str, pfile: str, keep_file: str, output_prefix: str,
               maf: float = 0.01, hwe_p: float = 1e-6, export_format: str = 'bed') -> bool:
    """
    Run PLINK2 to filter variants and export.

    Args:
        plink2_path: Path to plink2 executable
        pfile: Path prefix to .pgen/.pvar/.psam files
        keep_file: Path to file with sample IDs to keep
        output_prefix: Output file prefix
        maf: Minimum minor allele frequency
        hwe_p: Minimum HWE p-value
        export_format: 'bed' for binary (fast) or 'raw' for text (slow)

    Returns:
        True if successful
    """
    if export_format == 'bed':
        # Binary format - much faster and smaller
        cmd = [
            plink2_path,
            '--pfile', pfile,
            '--keep', keep_file,
            '--maf', str(maf),
            '--snps-only',
            '--hwe', str(hwe_p),
            '--make-bed',
            '--out', output_prefix
        ]
    else:
        # Text format (legacy)
        cmd = [
            plink2_path,
            '--pfile', pfile,
            '--keep', keep_file,
            '--maf', str(maf),
            '--snps-only',
            '--hwe', str(hwe_p),
            '--export', 'A',
            '--out', output_prefix
        ]

    logger.info(f"Running PLINK2 command:")
    logger.info(f"  {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("PLINK2 completed successfully")
        if result.stdout:
            # Print last few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:
                logger.info(f"  {line}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"PLINK2 failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"Error output:\n{e.stderr}")
        if e.stdout:
            logger.error(f"Standard output:\n{e.stdout}")
        return False
    except FileNotFoundError:
        logger.error(f"PLINK2 executable not found at: {plink2_path}")
        return False


def convert_bed(args):
    """Convert PLINK1 binary format (.bed/.bim/.fam) to project format - FAST!"""
    bed_prefix = Path(args.bed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bed_path = bed_prefix.with_suffix('.bed')
    bim_path = bed_prefix.with_suffix('.bim')
    fam_path = bed_prefix.with_suffix('.fam')

    logger.info(f"Loading PLINK binary files from {bed_prefix}")

    # Check files exist
    for p in [bed_path, bim_path, fam_path]:
        if not p.exists():
            logger.error(f"File not found: {p}")
            return

    # Read .bim (variant info): chr, id, cm, pos, a1, a2
    logger.info("Reading variant info (.bim)...")
    bim_df = pd.read_csv(bim_path, sep='\t', header=None,
                         names=['CHR', 'ID', 'CM', 'POS', 'A1', 'A2'])
    n_variants = len(bim_df)
    logger.info(f"Found {n_variants:,} variants")

    # Read .fam (sample info): fid, iid, father, mother, sex, pheno
    logger.info("Reading sample info (.fam)...")
    fam_df = pd.read_csv(fam_path, sep=r'\s+', header=None,
                         names=['FID', 'IID', 'FATHER', 'MOTHER', 'SEX', 'PHENO'])
    n_samples = len(fam_df)
    sample_ids = fam_df['IID'].astype(str).tolist()
    logger.info(f"Found {n_samples:,} samples")

    # Read .bed (genotype data) - binary format
    logger.info("Reading genotype data (.bed)...")

    # .bed file format:
    # - First 3 bytes: magic number (0x6c, 0x1b, 0x01 for SNP-major)
    # - Then: ceil(n_samples/4) bytes per variant
    # - Each byte encodes 4 samples (2 bits each): 00=hom_a1, 01=het, 10=missing, 11=hom_a2

    bytes_per_variant = (n_samples + 3) // 4

    with open(bed_path, 'rb') as f:
        # Check magic number
        magic = f.read(3)
        if magic != b'\x6c\x1b\x01':
            logger.error(f"Invalid .bed file format (not SNP-major)")
            return

        # Initialize dosage array
        try:
            import zarr
            zarr_path = output_dir / "dosages.zarr"
            dosages = zarr.open(
                str(zarr_path),
                mode='w',
                shape=(n_samples, n_variants),
                chunks=(min(500, n_samples), min(50000, n_variants)),
                dtype='float32'
            )
        except ImportError:
            logger.info("zarr not available, using numpy array (may use more memory)")
            dosages = np.zeros((n_samples, n_variants), dtype=np.float32)
            zarr_path = None

        # Decode table: 2-bit encoding -> dosage (count of A1 allele)
        # PLINK BED format specification:
        # 00 (index 0) -> 2 (homozygous first/A1 allele)
        # 01 (index 1) -> NaN (missing genotype)
        # 10 (index 2) -> 1 (heterozygous)
        # 11 (index 3) -> 0 (homozygous second/A2 allele)
        decode_table = np.array([2, np.nan, 1, 0], dtype=np.float32)

        # Process in batches of variants for efficiency
        BATCH_SIZE = 10000
        n_batches = (n_variants + BATCH_SIZE - 1) // BATCH_SIZE

        logger.info(f"Processing {n_batches} batches of {BATCH_SIZE} variants (vectorized)...")

        # Precompute number of samples rounded up to multiple of 4
        n_samples_padded = bytes_per_variant * 4

        for batch_idx in range(n_batches):
            start_var = batch_idx * BATCH_SIZE
            end_var = min(start_var + BATCH_SIZE, n_variants)
            batch_n_vars = end_var - start_var

            # Read bytes for this batch
            batch_bytes = np.frombuffer(f.read(bytes_per_variant * batch_n_vars), dtype=np.uint8)
            batch_bytes = batch_bytes.reshape(batch_n_vars, bytes_per_variant)

            # VECTORIZED decoding - extract all 4 genotypes from each byte using bitwise ops
            # Each byte encodes 4 samples (2 bits each)
            geno_0 = batch_bytes & 0x03          # bits 0-1
            geno_1 = (batch_bytes >> 2) & 0x03   # bits 2-3
            geno_2 = (batch_bytes >> 4) & 0x03   # bits 4-5
            geno_3 = (batch_bytes >> 6) & 0x03   # bits 6-7

            # Interleave to get proper sample order: shape (n_vars, bytes_per_var, 4)
            geno_interleaved = np.stack([geno_0, geno_1, geno_2, geno_3], axis=-1)
            # Reshape to (n_vars, n_samples_padded)
            geno_codes = geno_interleaved.reshape(batch_n_vars, n_samples_padded)
            # Trim to actual number of samples and transpose to (n_samples, n_vars)
            geno_codes = geno_codes[:, :n_samples].T

            # Convert genotype codes to dosages using lookup table
            batch_dosages = decode_table[geno_codes]

            # Write to output
            dosages[:, start_var:end_var] = batch_dosages

            if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
                logger.info(f"Processed {end_var:,}/{n_variants:,} variants ({100*end_var/n_variants:.1f}%)")

    logger.info(f"Loaded dosage matrix: {n_samples} x {n_variants}")

    # ========== MISSINGNESS QC FILTERING ==========
    # Get missingness thresholds from args (with defaults)
    max_variant_missing = getattr(args, 'max_variant_missing', 0.05)  # Default 5%
    max_sample_missing = getattr(args, 'max_sample_missing', 0.10)    # Default 10%
    impute_missing = getattr(args, 'impute_missing', False)

    logger.info("=" * 50)
    logger.info("MISSINGNESS QC")
    logger.info("=" * 50)

    # Calculate missingness rates
    # For zarr arrays, we need to load into memory for this calculation
    if zarr_path:
        dosages_np = np.array(dosages)
    else:
        dosages_np = dosages

    missing_mask = np.isnan(dosages_np)
    total_missing = missing_mask.sum()
    total_elements = dosages_np.size

    if total_missing > 0:
        missing_pct = 100 * total_missing / total_elements
        logger.info(f"Total missing genotypes: {total_missing:,} ({missing_pct:.3f}%)")

        # Per-variant missingness
        variant_missing_rate = missing_mask.mean(axis=0)
        logger.info(f"Variant missingness: min={variant_missing_rate.min()*100:.3f}%, "
                   f"max={variant_missing_rate.max()*100:.3f}%, "
                   f"mean={variant_missing_rate.mean()*100:.3f}%")

        # Per-sample missingness
        sample_missing_rate = missing_mask.mean(axis=1)
        logger.info(f"Sample missingness: min={sample_missing_rate.min()*100:.3f}%, "
                   f"max={sample_missing_rate.max()*100:.3f}%, "
                   f"mean={sample_missing_rate.mean()*100:.3f}%")

        # Filter samples with high missingness
        samples_to_keep = sample_missing_rate <= max_sample_missing
        n_samples_removed = (~samples_to_keep).sum()
        if n_samples_removed > 0:
            logger.warning(f"Removing {n_samples_removed:,} samples with >{max_sample_missing*100:.1f}% missing genotypes")
            dosages_np = dosages_np[samples_to_keep, :]
            sample_ids = [sid for sid, keep in zip(sample_ids, samples_to_keep) if keep]
            fam_df = fam_df[samples_to_keep].reset_index(drop=True)
            n_samples = len(sample_ids)
            # Recalculate variant missingness after sample filtering
            missing_mask = np.isnan(dosages_np)
            variant_missing_rate = missing_mask.mean(axis=0)

        # Filter variants with high missingness
        variants_to_keep = variant_missing_rate <= max_variant_missing
        n_variants_removed = (~variants_to_keep).sum()
        if n_variants_removed > 0:
            logger.warning(f"Removing {n_variants_removed:,} variants with >{max_variant_missing*100:.1f}% missing genotypes")
            dosages_np = dosages_np[:, variants_to_keep]
            bim_df = bim_df[variants_to_keep].reset_index(drop=True)
            n_variants = len(bim_df)

        # Report remaining missingness
        missing_mask = np.isnan(dosages_np)
        remaining_missing = missing_mask.sum()
        if remaining_missing > 0:
            remaining_pct = 100 * remaining_missing / dosages_np.size
            logger.info(f"Remaining missing after filtering: {remaining_missing:,} ({remaining_pct:.3f}%)")

            # Optionally impute remaining missing values using per-variant mean
            if impute_missing:
                logger.info("Imputing remaining missing genotypes with per-variant mean...")
                variant_means = np.nanmean(dosages_np, axis=0)
                # Handle variants with all missing (shouldn't happen after filtering, but safety check)
                all_missing_variants = np.isnan(variant_means)
                if all_missing_variants.any():
                    variant_means[all_missing_variants] = 0.0

                # Apply imputation
                for j in range(dosages_np.shape[1]):
                    col_missing = np.isnan(dosages_np[:, j])
                    if col_missing.any():
                        dosages_np[col_missing, j] = variant_means[j]

                logger.info(f"Imputed {remaining_missing:,} genotypes")
        else:
            logger.info("No missing genotypes remaining after filtering")

        # Update dosages array
        if zarr_path:
            # Recreate zarr array with new shape
            dosages = zarr.open(
                str(zarr_path),
                mode='w',
                shape=dosages_np.shape,
                chunks=(min(500, n_samples), min(50000, n_variants)),
                dtype='float32'
            )
            dosages[:] = dosages_np
        else:
            dosages = dosages_np

        logger.info(f"Final matrix shape: {n_samples:,} samples x {n_variants:,} variants")

        # Check if all data was filtered out
        if n_samples == 0 or n_variants == 0:
            logger.error("=" * 50)
            logger.error("CRITICAL: All data was filtered out!")
            logger.error("=" * 50)
            logger.error("This indicates a data quality issue. Possible causes:")
            logger.error("  1. Source PLINK files may not be imputed (raw genotypes have higher missingness)")
            logger.error("  2. QC thresholds may be too strict for this dataset")
            logger.error("  3. There may be an issue with the genotype encoding")
            logger.error("")
            logger.error("Recommendations:")
            logger.error("  1. Check your source PLINK files with: plink2 --bfile <prefix> --missing")
            logger.error("  2. Try more lenient thresholds: --max-sample-missing 0.25 --max-variant-missing 0.10")
            logger.error("  3. If data is not imputed, consider imputation with a reference panel")
            logger.error("")
            logger.error("No output files will be saved.")
            return

    else:
        logger.info("No missing genotypes found - no filtering needed")

    logger.info("=" * 50)
    # ========== END MISSINGNESS QC ==========

    # Save outputs
    logger.info("Saving outputs...")

    # Variant info
    bim_df['variant_idx'] = np.arange(len(bim_df))
    # Calculate MAF
    if zarr_path:
        # Sample for MAF calculation
        sample_dosages = dosages[:min(1000, n_samples), :]
        af = np.nanmean(sample_dosages, axis=0) / 2
    else:
        af = np.nanmean(dosages, axis=0) / 2
    maf = np.minimum(af, 1 - af)
    bim_df['AF'] = af
    bim_df['MAF'] = maf

    variants_path = output_dir / "variants.parquet"
    bim_df.to_parquet(variants_path, index=False)
    logger.info(f"Saved {n_variants:,} variants to {variants_path}")

    # Sample IDs
    sample_path = output_dir / "sample_ids.txt"
    with open(sample_path, 'w') as f:
        for sid in sample_ids:
            f.write(f"{sid}\n")
    logger.info(f"Saved {n_samples:,} sample IDs to {sample_path}")

    # If we used numpy array, save as zarr now
    if zarr_path is None:
        try:
            import zarr
            zarr_path = output_dir / "dosages.zarr"
            zarr_array = zarr.open(
                str(zarr_path), mode='w',
                shape=dosages.shape,
                chunks=(min(500, n_samples), min(50000, n_variants)),
                dtype='float32'
            )
            zarr_array[:] = dosages
            logger.info(f"Saved dosages to {zarr_path}")
        except ImportError:
            npy_path = output_dir / "dosages.npy"
            np.save(npy_path, dosages)
            logger.info(f"Saved dosages to {npy_path}")
    else:
        logger.info(f"Dosages already saved to {zarr_path}")

    # QC report
    report_path = output_dir / "qc_report.txt"

    # Calculate final missingness for report
    if zarr_path:
        final_dosages = np.array(dosages)
    else:
        final_dosages = dosages if isinstance(dosages, np.ndarray) else dosages_np
    final_missing = np.isnan(final_dosages).sum()
    final_missing_pct = 100 * final_missing / final_dosages.size if final_dosages.size > 0 else 0

    with open(report_path, 'w') as f:
        f.write("Variant Extraction Report (PLINK binary method)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Final Samples: {n_samples}\n")
        f.write(f"Final Variants: {n_variants}\n\n")
        f.write(f"MAF range: {np.nanmin(maf):.4f} - {np.nanmax(maf):.4f}\n\n")
        f.write("Missingness QC:\n")
        f.write(f"  Max variant missingness threshold: {max_variant_missing*100:.1f}%\n")
        f.write(f"  Max sample missingness threshold: {max_sample_missing*100:.1f}%\n")
        f.write(f"  Impute remaining missing: {impute_missing}\n")
        f.write(f"  Final missing genotypes: {final_missing:,} ({final_missing_pct:.3f}%)\n")

    logger.info("=" * 50)
    logger.info("VARIANT EXTRACTION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Samples: {n_samples:,}")
    logger.info(f"Variants: {n_variants:,}")
    logger.info(f"Output directory: {output_dir}")


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
    """Convert large PLINK2 .raw export using optimized batched reading."""
    raw_path = Path(args.raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading PLINK2 export from {raw_path} (optimized batched mode)")

    # Read header first
    with open(raw_path, 'r') as f:
        header = f.readline().strip().split()

    meta_cols = header[:6]
    variant_cols = header[6:]
    n_variants = len(variant_cols)

    logger.info(f"Found {n_variants} variants")

    # Count lines efficiently
    logger.info("Counting samples...")
    with open(raw_path, 'rb') as f:
        n_samples = sum(1 for _ in f) - 1  # Subtract header (binary mode is faster)

    logger.info(f"Found {n_samples} samples")

    # Initialize zarr array
    try:
        import zarr
        zarr_path = output_dir / "dosages.zarr"
        zarr_array = zarr.open(
            str(zarr_path),
            mode='w',
            shape=(n_samples, n_variants),
            chunks=(min(500, n_samples), min(50000, n_variants)),  # Larger chunks for efficiency
            dtype='float32'
        )
    except ImportError:
        logger.error("zarr is required for chunked mode. Install with: pip install zarr")
        return

    # Batch size for zarr writes (KEY OPTIMIZATION: batch writes instead of per-row)
    BATCH_SIZE = 500
    sample_ids = []
    batch_dosages = []
    batch_start = 0

    logger.info(f"Reading samples in batches of {BATCH_SIZE}...")

    with open(raw_path, 'r') as f:
        # Skip header
        f.readline()

        for i, line in enumerate(f):
            # Split only first 6 fields to get metadata
            parts = line.split(maxsplit=6)
            sample_ids.append(parts[1])  # IID

            # Parse dosages using numpy
            dosage_str = parts[6].replace('NA', 'nan')
            dosages = np.fromstring(dosage_str, sep=' ', dtype=np.float32)
            batch_dosages.append(dosages)

            # Write batch to zarr when full
            if len(batch_dosages) >= BATCH_SIZE:
                batch_array = np.vstack(batch_dosages)
                zarr_array[batch_start:batch_start + len(batch_dosages), :] = batch_array
                batch_start += len(batch_dosages)
                batch_dosages = []

                logger.info(f"Processed {batch_start:,}/{n_samples:,} samples ({100*batch_start/n_samples:.1f}%)")

        # Write remaining samples
        if batch_dosages:
            batch_array = np.vstack(batch_dosages)
            zarr_array[batch_start:batch_start + len(batch_dosages), :] = batch_array
            logger.info(f"Processed {n_samples:,}/{n_samples:,} samples (100.0%)")

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


def run_full_pipeline(args):
    """Run the full extraction pipeline: prepare IDs, run PLINK2, convert output."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine export format - binary is MUCH faster (default)
    use_text_format = getattr(args, 'use_text_format', False)
    export_format = 'raw' if use_text_format else 'bed'

    # Step 1: Find PLINK2
    logger.info("=" * 50)
    logger.info("STEP 1: Finding PLINK2")
    logger.info("=" * 50)

    if args.plink2_path:
        plink2_path = args.plink2_path
        if not Path(plink2_path).exists():
            logger.error(f"Specified PLINK2 path does not exist: {plink2_path}")
            return
        logger.info(f"Using specified PLINK2: {plink2_path}")
    else:
        try:
            plink2_path = find_plink2()
        except FileNotFoundError as e:
            logger.error(str(e))
            return

    # Step 2: Prepare cohort IDs
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 2: Preparing cohort IDs")
    logger.info("=" * 50)

    logger.info(f"Loading cohort from {args.cohort}")
    cohort = pd.read_parquet(args.cohort)

    ids_path = output_dir / "cohort_ids.txt"
    with open(ids_path, 'w') as f:
        for eid in cohort['eid']:
            f.write(f"{eid}\t{eid}\n")

    logger.info(f"Wrote {len(cohort)} sample IDs to {ids_path}")

    # Step 3: Run PLINK2
    logger.info("")
    logger.info("=" * 50)
    logger.info(f"STEP 3: Running PLINK2 QC and export (format: {export_format})")
    logger.info("=" * 50)

    output_prefix = str(output_dir / "chr6_filtered")

    success = run_plink2(
        plink2_path=plink2_path,
        pfile=args.pfile,
        keep_file=str(ids_path),
        output_prefix=output_prefix,
        maf=args.maf_min,
        hwe_p=args.hwe_p_min,
        export_format=export_format
    )

    if not success:
        logger.error("PLINK2 failed. Check the error messages above.")
        return

    # Step 4: Convert to project format
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 4: Converting to project format")
    logger.info("=" * 50)

    # Create args-like object for convert function
    class ConvertArgs:
        pass

    convert_args = ConvertArgs()
    convert_args.output_dir = str(output_dir)

    if export_format == 'bed':
        # Binary format - FAST
        bed_path = Path(f"{output_prefix}.bed")
        if not bed_path.exists():
            logger.error(f"Expected output file not found: {bed_path}")
            return
        convert_args.bed = output_prefix
        convert_bed(convert_args)
    else:
        # Text format - slow (legacy)
        raw_path = Path(f"{output_prefix}.raw")
        if not raw_path.exists():
            logger.error(f"Expected output file not found: {raw_path}")
            return
        convert_args.raw = str(raw_path)
        convert_args.chunked = getattr(args, 'chunked', False)
        convert_args.chunk_size = getattr(args, 'chunk_size', 1000)
        if convert_args.chunked:
            convert_raw_chunked(convert_args)
        else:
            convert_raw(convert_args)

    logger.info("")
    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Extract variants using PLINK2 export"
    )
    subparsers = parser.add_subparsers(dest='command')

    # Main run command (default)
    run_parser = subparsers.add_parser(
        'run',
        help='Run full extraction pipeline'
    )
    run_parser.add_argument(
        '--pfile',
        required=True,
        help='PLINK2 file prefix (e.g., /path/to/chr6)'
    )
    run_parser.add_argument(
        '--cohort',
        required=True,
        help='Path to cohort.parquet'
    )
    run_parser.add_argument(
        '--output-dir',
        default='data/variants',
        help='Output directory'
    )
    run_parser.add_argument(
        '--plink2-path',
        help='Path to plink2 executable (auto-detected if not specified)'
    )
    run_parser.add_argument(
        '--maf-min',
        type=float,
        default=0.01,
        help='Minimum MAF (default: 0.01)'
    )
    run_parser.add_argument(
        '--hwe-p-min',
        type=float,
        default=1e-6,
        help='Minimum HWE p-value (default: 1e-6)'
    )
    run_parser.add_argument(
        '--chunked',
        action='store_true',
        help='Use chunked reading for large files (only for text format)'
    )
    run_parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Samples per chunk for text format (default: 1000)'
    )
    run_parser.add_argument(
        '--use-text-format',
        action='store_true',
        dest='use_text_format',
        help='Use slow text format (.raw) instead of fast binary format (.bed)'
    )
    run_parser.add_argument(
        '--max-variant-missing',
        type=float,
        default=0.05,
        help='Maximum variant missingness rate (default: 0.05 = 5%%)'
    )
    run_parser.add_argument(
        '--max-sample-missing',
        type=float,
        default=0.10,
        help='Maximum sample missingness rate (default: 0.10 = 10%%)'
    )
    run_parser.add_argument(
        '--impute-missing',
        action='store_true',
        help='Impute remaining missing genotypes with per-variant mean after filtering'
    )

    # prepare-ids subcommand (for manual workflow)
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

    # convert subcommand (for manual workflow)
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

    # convert-bed subcommand (for binary format - FAST)
    bed_parser = subparsers.add_parser(
        'convert-bed',
        help='Convert PLINK binary (.bed/.bim/.fam) to project format - FAST!'
    )
    bed_parser.add_argument(
        '--bed',
        required=True,
        help='Path prefix to PLINK binary files (e.g., /path/to/chr6 for chr6.bed/bim/fam)'
    )
    bed_parser.add_argument(
        '--output-dir',
        default='data/variants',
        help='Output directory'
    )
    bed_parser.add_argument(
        '--max-variant-missing',
        type=float,
        default=0.05,
        help='Maximum variant missingness rate (default: 0.05 = 5%%)'
    )
    bed_parser.add_argument(
        '--max-sample-missing',
        type=float,
        default=0.10,
        help='Maximum sample missingness rate (default: 0.10 = 10%%)'
    )
    bed_parser.add_argument(
        '--impute-missing',
        action='store_true',
        help='Impute remaining missing genotypes with per-variant mean after filtering'
    )

    # Also allow running without subcommand (same as 'run')
    parser.add_argument(
        '--pfile',
        help='PLINK2 file prefix (e.g., /path/to/chr6)'
    )
    parser.add_argument(
        '--cohort',
        help='Path to cohort.parquet'
    )
    parser.add_argument(
        '--output-dir',
        default='data/variants',
        help='Output directory'
    )
    parser.add_argument(
        '--plink2-path',
        help='Path to plink2 executable (auto-detected if not specified)'
    )
    parser.add_argument(
        '--maf-min',
        type=float,
        default=0.01,
        help='Minimum MAF (default: 0.01)'
    )
    parser.add_argument(
        '--hwe-p-min',
        type=float,
        default=1e-6,
        help='Minimum HWE p-value (default: 1e-6)'
    )
    parser.add_argument(
        '--chunked',
        action='store_true',
        help='Use chunked reading for large files'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Samples per chunk (default: 1000)'
    )

    args = parser.parse_args()

    # Handle command routing
    if args.command == 'run':
        run_full_pipeline(args)
    elif args.command == 'prepare-ids':
        prepare_ids(args)
    elif args.command == 'convert':
        if args.chunked:
            convert_raw_chunked(args)
        else:
            convert_raw(args)
    elif args.command == 'convert-bed':
        convert_bed(args)
    elif args.pfile and args.cohort:
        # No subcommand but --pfile and --cohort provided: run full pipeline
        run_full_pipeline(args)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Full pipeline (uses fast binary format by default):")
        print("  python scripts/extract_variants_plink2.py --pfile /path/to/chr6 --cohort data/cohort/cohort.parquet")
        print("")
        print("  # Convert existing PLINK binary files directly:")
        print("  python scripts/extract_variants_plink2.py convert-bed --bed /path/to/chr6_filtered")
        print("")
        print("  # Use slower text format (not recommended):")
        print("  python scripts/extract_variants_plink2.py run --pfile /path/to/chr6 --cohort data/cohort/cohort.parquet --use-text-format")


if __name__ == "__main__":
    main()
