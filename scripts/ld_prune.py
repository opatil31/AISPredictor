#!/usr/bin/env python3
"""
LD Pruning Script for Variant Data

Performs linkage disequilibrium (LD) pruning to reduce redundant variants
while retaining tag SNPs that capture most of the genetic variation.

Parameters (from implementation plan):
- Window: 500kb
- Step: 1 variant
- r² threshold: 0.8

Expected reduction: ~300K variants -> ~50-100K tag variants

Usage:
    # Method 1: Use PLINK2 (recommended - much faster)
    python scripts/ld_prune.py plink2 \\
        --pfile /path/to/chr6 \\
        --cohort data/cohort/cohort.parquet \\
        --output-dir data/variants

    # Method 2: Prune existing dosage data (slower, no PLINK2 needed)
    python scripts/ld_prune.py prune \\
        --variants data/variants/variants.parquet \\
        --dosages data/variants/dosages.zarr \\
        --output-dir data/variants/pruned
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Set

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# LD Pruning parameters from implementation plan
DEFAULT_WINDOW_KB = 500  # 500kb window
DEFAULT_STEP = 1         # 1 variant step
DEFAULT_R2_THRESHOLD = 0.8  # r² threshold


def find_plink2() -> Optional[str]:
    """Find PLINK2 executable."""
    for name in ['plink2', 'plink2.exe']:
        path = shutil.which(name)
        if path:
            return path
    return None


def run_plink2_ld_prune(
    plink2_path: str,
    pfile: str,
    keep_file: str,
    output_prefix: str,
    window_kb: int = DEFAULT_WINDOW_KB,
    step: int = DEFAULT_STEP,
    r2_threshold: float = DEFAULT_R2_THRESHOLD,
    maf_min: float = 0.01,
    maf_max: float = None,
    hwe_p_min: float = 1e-6,
) -> Optional[List[str]]:
    """
    Run PLINK2 to perform LD pruning and export pruned variants.

    Args:
        plink2_path: Path to PLINK2 executable
        pfile: PLINK2 file prefix
        keep_file: File with sample IDs to keep
        output_prefix: Output file prefix
        window_kb: LD window size in kb
        step: Step size in variants
        r2_threshold: r² threshold for pruning
        maf_min: Minimum MAF (set to 0 for rare variant analysis)
        maf_max: Maximum MAF (for rare variant analysis, e.g., 0.01)

    Returns:
        List of variant IDs that passed pruning, or None if failed
    """
    # Step 1: Calculate LD and get pruned variant list
    logger.info("Step 1: Running PLINK2 LD pruning...")

    prune_cmd = [
        plink2_path,
        '--pfile', pfile,
        '--keep', keep_file,
        '--maf', str(maf_min),
        '--snps-only',
        '--hwe', str(hwe_p_min),
        '--indep-pairwise', str(window_kb), 'kb', str(step), str(r2_threshold),
        '--out', output_prefix
    ]

    # Add max-maf filter for rare variant analysis
    if maf_max is not None:
        prune_cmd.extend(['--max-maf', str(maf_max)])

    logger.info(f"Command: {' '.join(prune_cmd)}")

    try:
        result = subprocess.run(
            prune_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("LD pruning completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"PLINK2 LD pruning failed: {e.stderr}")
        return None

    # Read pruned variant list
    prune_in_file = Path(f"{output_prefix}.prune.in")
    if not prune_in_file.exists():
        logger.error(f"Pruned variant file not found: {prune_in_file}")
        return None

    with open(prune_in_file, 'r') as f:
        pruned_variants = [line.strip() for line in f if line.strip()]

    logger.info(f"Retained {len(pruned_variants):,} variants after LD pruning")

    # Step 2: Export pruned dosages in binary format (FAST)
    logger.info("Step 2: Exporting pruned genotypes (binary format)...")

    export_cmd = [
        plink2_path,
        '--pfile', pfile,
        '--keep', keep_file,
        '--extract', str(prune_in_file),
        '--make-bed',
        '--out', f"{output_prefix}_pruned"
    ]

    logger.info(f"Command: {' '.join(export_cmd)}")

    try:
        result = subprocess.run(
            export_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("Export completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"PLINK2 export failed: {e.stderr}")
        return None

    return pruned_variants


def compute_ld_matrix_chunk(dosages: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    Compute LD (r²) between variants in a chunk.

    Args:
        dosages: (n_samples, n_variants) dosage matrix
        start: Start variant index
        end: End variant index

    Returns:
        Correlation matrix for the chunk
    """
    chunk = dosages[:, start:end]

    # Standardize (handle missing values)
    chunk_centered = chunk - np.nanmean(chunk, axis=0)
    chunk_std = np.nanstd(chunk, axis=0)
    chunk_std[chunk_std == 0] = 1  # Avoid division by zero
    chunk_normalized = chunk_centered / chunk_std

    # Replace NaN with 0 for correlation calculation
    chunk_normalized = np.nan_to_num(chunk_normalized, nan=0)

    # Compute correlation matrix
    n_samples = chunk_normalized.shape[0]
    corr = np.dot(chunk_normalized.T, chunk_normalized) / n_samples

    return corr ** 2  # Return r²


def ld_prune_python(
    dosages: np.ndarray,
    positions: np.ndarray,
    window_kb: int = DEFAULT_WINDOW_KB,
    r2_threshold: float = DEFAULT_R2_THRESHOLD,
) -> np.ndarray:
    """
    Perform LD pruning using Python (slower than PLINK2 but self-contained).

    Uses a greedy algorithm: iterate through variants, keep variant if not
    in high LD (r² > threshold) with any previously kept variant in window.

    Args:
        dosages: (n_samples, n_variants) dosage matrix
        positions: Array of variant positions (bp)
        window_kb: Window size in kb
        r2_threshold: r² threshold

    Returns:
        Boolean mask of variants to keep
    """
    n_variants = dosages.shape[1]
    window_bp = window_kb * 1000

    keep_mask = np.ones(n_variants, dtype=bool)

    logger.info(f"LD pruning {n_variants:,} variants (window={window_kb}kb, r²={r2_threshold})")

    for i in range(n_variants):
        if not keep_mask[i]:
            continue

        if (i + 1) % 10000 == 0:
            n_kept = keep_mask[:i+1].sum()
            logger.info(f"Processed {i+1:,}/{n_variants:,} variants, kept {n_kept:,}")

        # Find variants in window ahead of current variant
        pos_i = positions[i]
        window_end = pos_i + window_bp

        # Check variants ahead in window
        for j in range(i + 1, n_variants):
            if positions[j] > window_end:
                break

            if not keep_mask[j]:
                continue

            # Compute r² between variants i and j
            geno_i = dosages[:, i]
            geno_j = dosages[:, j]

            # Handle missing values
            valid = ~(np.isnan(geno_i) | np.isnan(geno_j))
            if valid.sum() < 10:
                continue

            geno_i_valid = geno_i[valid]
            geno_j_valid = geno_j[valid]

            # Compute correlation
            corr = np.corrcoef(geno_i_valid, geno_j_valid)[0, 1]
            r2 = corr ** 2

            if r2 > r2_threshold:
                keep_mask[j] = False

    n_kept = keep_mask.sum()
    logger.info(f"LD pruning complete: {n_variants:,} -> {n_kept:,} variants")

    return keep_mask


def prune_existing_data(args):
    """Prune existing variant/dosage data using Python."""
    logger.info("=" * 50)
    logger.info("LD Pruning (Python method)")
    logger.info("=" * 50)

    # Load variants
    logger.info(f"Loading variants from {args.variants}")
    variants_df = pd.read_parquet(args.variants)
    logger.info(f"Loaded {len(variants_df):,} variants")

    # Load dosages
    logger.info(f"Loading dosages from {args.dosages}")
    try:
        import zarr
        dosages = zarr.open(args.dosages, mode='r')[:]
    except:
        dosages = np.load(args.dosages)
    logger.info(f"Dosage matrix shape: {dosages.shape}")

    # Check if we have position information
    if 'POS' not in variants_df.columns:
        logger.warning("No position column found. Using variant index as proxy.")
        positions = np.arange(len(variants_df)) * 1000  # Assume 1kb spacing
    else:
        positions = variants_df['POS'].values

    # Run LD pruning
    keep_mask = ld_prune_python(
        dosages,
        positions,
        window_kb=args.window_kb,
        r2_threshold=args.r2_threshold
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save pruned variants
    pruned_variants = variants_df[keep_mask].copy()
    pruned_variants = pruned_variants.reset_index(drop=True)
    pruned_variants['variant_idx'] = np.arange(len(pruned_variants))

    variants_path = output_dir / "variants_pruned.parquet"
    pruned_variants.to_parquet(variants_path, index=False)
    logger.info(f"Saved {len(pruned_variants):,} pruned variants to {variants_path}")

    # Save pruned dosages
    pruned_dosages = dosages[:, keep_mask]

    try:
        import zarr
        zarr_path = output_dir / "dosages_pruned.zarr"
        zarr_array = zarr.open(
            str(zarr_path),
            mode='w',
            shape=pruned_dosages.shape,
            chunks=(min(100, pruned_dosages.shape[0]), min(10000, pruned_dosages.shape[1])),
            dtype='float32'
        )
        zarr_array[:] = pruned_dosages
        logger.info(f"Saved pruned dosages to {zarr_path}")
    except ImportError:
        npy_path = output_dir / "dosages_pruned.npy"
        np.save(npy_path, pruned_dosages)
        logger.info(f"Saved pruned dosages to {npy_path}")

    # Save pruning report
    report_path = output_dir / "ld_prune_report.txt"
    with open(report_path, 'w') as f:
        f.write("LD Pruning Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Method: Python\n")
        f.write(f"Window size: {args.window_kb} kb\n")
        f.write(f"r² threshold: {args.r2_threshold}\n\n")
        f.write(f"Input variants: {len(variants_df):,}\n")
        f.write(f"Output variants: {len(pruned_variants):,}\n")
        f.write(f"Reduction: {100 * (1 - len(pruned_variants)/len(variants_df)):.1f}%\n")
    logger.info(f"Saved report to {report_path}")

    logger.info("=" * 50)
    logger.info("LD PRUNING COMPLETE")
    logger.info("=" * 50)


def prune_with_plink2(args):
    """Run full LD pruning pipeline using PLINK2."""
    logger.info("=" * 50)
    logger.info("LD Pruning (PLINK2 method)")
    logger.info("=" * 50)

    # Find PLINK2
    if args.plink2_path:
        plink2_path = args.plink2_path
    else:
        plink2_path = find_plink2()
        if not plink2_path:
            logger.error("PLINK2 not found. Please specify --plink2-path or add to PATH.")
            return

    logger.info(f"Using PLINK2: {plink2_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create cohort ID file
    logger.info(f"Loading cohort from {args.cohort}")
    cohort = pd.read_parquet(args.cohort)

    keep_file = output_dir / "cohort_ids.txt"
    with open(keep_file, 'w') as f:
        for eid in cohort['eid']:
            f.write(f"{eid}\t{eid}\n")
    logger.info(f"Wrote {len(cohort)} sample IDs")

    # Run PLINK2 LD pruning
    output_prefix = str(output_dir / "chr6")

    pruned_variants = run_plink2_ld_prune(
        plink2_path=plink2_path,
        pfile=args.pfile,
        keep_file=str(keep_file),
        output_prefix=output_prefix,
        window_kb=args.window_kb,
        step=args.step,
        r2_threshold=args.r2_threshold,
        maf_min=args.maf_min,
        maf_max=getattr(args, 'maf_max', None),
        hwe_p_min=args.hwe_p_min,
    )

    if pruned_variants is None:
        logger.error("LD pruning failed")
        return

    # Convert pruned .bed to project format (FAST binary conversion)
    bed_path = Path(f"{output_prefix}_pruned.bed")
    if not bed_path.exists():
        logger.error(f"Pruned .bed file not found: {bed_path}")
        return

    logger.info("Converting pruned data to project format (binary)...")

    # Use the convert function from extract_variants_plink2
    sys.path.insert(0, str(Path(__file__).parent))
    from extract_variants_plink2 import convert_bed

    class ConvertArgs:
        pass

    convert_args = ConvertArgs()
    convert_args.bed = f"{output_prefix}_pruned"
    convert_args.output_dir = str(output_dir)

    convert_bed(convert_args)

    # Rename output files to indicate they're pruned
    for old_name, new_name in [
        ("variants.parquet", "variants_pruned.parquet"),
        ("dosages.zarr", "dosages_pruned.zarr"),
        ("sample_ids.txt", "sample_ids_pruned.txt"),
    ]:
        old_path = output_dir / old_name
        new_path = output_dir / new_name
        if old_path.exists():
            old_path.rename(new_path)

    # Save pruning report
    report_path = output_dir / "ld_prune_report.txt"
    with open(report_path, 'w') as f:
        f.write("LD Pruning Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Method: PLINK2\n")
        f.write(f"Window size: {args.window_kb} kb\n")
        f.write(f"Step: {args.step} variant(s)\n")
        f.write(f"r² threshold: {args.r2_threshold}\n")
        f.write(f"MAF threshold: {args.maf_min}\n")
        f.write(f"HWE p-value threshold: {args.hwe_p_min}\n\n")
        f.write(f"Variants retained: {len(pruned_variants):,}\n")
    logger.info(f"Saved report to {report_path}")

    logger.info("=" * 50)
    logger.info("LD PRUNING COMPLETE")
    logger.info("=" * 50)


def prune_from_bed(args):
    """Run LD pruning on existing PLINK1 binary (.bed) files."""
    logger.info("=" * 50)
    logger.info("LD Pruning from .bed files (PLINK2 method)")
    logger.info("=" * 50)

    # Find PLINK2
    if args.plink2_path:
        plink2_path = args.plink2_path
    else:
        plink2_path = find_plink2()
        if not plink2_path:
            logger.error("PLINK2 not found. Please specify --plink2-path or add to PATH.")
            return

    logger.info(f"Using PLINK2: {plink2_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check input files exist
    bed_prefix = Path(args.bfile)
    for ext in ['.bed', '.bim', '.fam']:
        if not bed_prefix.with_suffix(ext).exists():
            # Try without suffix
            test_path = Path(str(bed_prefix) + ext)
            if not test_path.exists():
                logger.error(f"File not found: {bed_prefix}{ext}")
                return

    # Step 1: Run LD pruning
    logger.info("Step 1: Running PLINK2 LD pruning...")
    output_prefix = str(output_dir / "chr6")

    prune_cmd = [
        plink2_path,
        '--bfile', str(bed_prefix),
        '--indep-pairwise', str(args.window_kb), 'kb', str(args.step), str(args.r2_threshold),
        '--out', output_prefix
    ]

    logger.info(f"Command: {' '.join(prune_cmd)}")

    try:
        result = subprocess.run(prune_cmd, capture_output=True, text=True, check=True)
        logger.info("LD pruning completed successfully")
        # Print relevant output
        for line in result.stdout.split('\n')[-15:]:
            if line.strip():
                logger.info(f"  {line}")
    except subprocess.CalledProcessError as e:
        logger.error(f"PLINK2 LD pruning failed: {e.stderr}")
        return

    # Read pruned variant list
    prune_in_file = Path(f"{output_prefix}.prune.in")
    if not prune_in_file.exists():
        logger.error(f"Pruned variant file not found: {prune_in_file}")
        return

    with open(prune_in_file, 'r') as f:
        pruned_variants = [line.strip() for line in f if line.strip()]

    logger.info(f"Retained {len(pruned_variants):,} variants after LD pruning")

    # Step 2: Extract pruned variants to new .bed file
    logger.info("Step 2: Extracting pruned variants (binary format)...")

    extract_cmd = [
        plink2_path,
        '--bfile', str(bed_prefix),
        '--extract', str(prune_in_file),
        '--make-bed',
        '--out', f"{output_prefix}_pruned"
    ]

    logger.info(f"Command: {' '.join(extract_cmd)}")

    try:
        result = subprocess.run(extract_cmd, capture_output=True, text=True, check=True)
        logger.info("Extraction completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"PLINK2 extraction failed: {e.stderr}")
        return

    # Step 3: Convert to project format
    bed_path = Path(f"{output_prefix}_pruned.bed")
    if not bed_path.exists():
        logger.error(f"Pruned .bed file not found: {bed_path}")
        return

    logger.info("Step 3: Converting to project format...")

    sys.path.insert(0, str(Path(__file__).parent))
    from extract_variants_plink2 import convert_bed

    class ConvertArgs:
        pass

    convert_args = ConvertArgs()
    convert_args.bed = f"{output_prefix}_pruned"
    convert_args.output_dir = str(output_dir)

    convert_bed(convert_args)

    # Rename output files
    for old_name, new_name in [
        ("variants.parquet", "variants_pruned.parquet"),
        ("dosages.zarr", "dosages_pruned.zarr"),
        ("sample_ids.txt", "sample_ids_pruned.txt"),
    ]:
        old_path = output_dir / old_name
        new_path = output_dir / new_name
        if old_path.exists():
            if new_path.exists():
                shutil.rmtree(new_path) if new_path.is_dir() else new_path.unlink()
            old_path.rename(new_path)

    # Save pruning report
    report_path = output_dir / "ld_prune_report.txt"
    with open(report_path, 'w') as f:
        f.write("LD Pruning Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Method: PLINK2 (from .bed)\n")
        f.write(f"Input: {bed_prefix}\n")
        f.write(f"Window size: {args.window_kb} kb\n")
        f.write(f"Step: {args.step} variant(s)\n")
        f.write(f"r² threshold: {args.r2_threshold}\n\n")
        f.write(f"Variants retained: {len(pruned_variants):,}\n")
    logger.info(f"Saved report to {report_path}")

    logger.info("=" * 50)
    logger.info("LD PRUNING COMPLETE")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Pruned variants: {len(pruned_variants):,}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="LD pruning for variant data"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # PLINK2 method
    plink2_parser = subparsers.add_parser(
        'plink2',
        help='Use PLINK2 for LD pruning (recommended)'
    )
    plink2_parser.add_argument(
        '--pfile', required=True,
        help='PLINK2 file prefix'
    )
    plink2_parser.add_argument(
        '--cohort', required=True,
        help='Path to cohort.parquet'
    )
    plink2_parser.add_argument(
        '--output-dir', default='data/variants/pruned',
        help='Output directory'
    )
    plink2_parser.add_argument(
        '--plink2-path',
        help='Path to PLINK2 executable'
    )
    plink2_parser.add_argument(
        '--window-kb', type=int, default=DEFAULT_WINDOW_KB,
        help=f'LD window size in kb (default: {DEFAULT_WINDOW_KB})'
    )
    plink2_parser.add_argument(
        '--step', type=int, default=DEFAULT_STEP,
        help=f'Step size in variants (default: {DEFAULT_STEP})'
    )
    plink2_parser.add_argument(
        '--r2-threshold', type=float, default=DEFAULT_R2_THRESHOLD,
        help=f'r² threshold (default: {DEFAULT_R2_THRESHOLD})'
    )
    plink2_parser.add_argument(
        '--maf-min', type=float, default=0.01,
        help='Minimum MAF (default: 0.01). Set to 0 for rare variant analysis.'
    )
    plink2_parser.add_argument(
        '--maf-max', type=float, default=None,
        help='Maximum MAF for rare variant analysis (e.g., 0.01 for <1%%). '
             'When set, filters OUT common variants and keeps only rare variants.'
    )
    plink2_parser.add_argument(
        '--hwe-p-min', type=float, default=1e-6,
        help='Minimum HWE p-value (default: 1e-6)'
    )

    # From existing .bed files (RECOMMENDED for already-extracted data)
    bed_parser = subparsers.add_parser(
        'from-bed',
        help='LD prune from existing .bed files (recommended if you already extracted variants)'
    )
    bed_parser.add_argument(
        '--bfile', required=True,
        help='PLINK1 binary file prefix (e.g., data/variants/chr6_filtered)'
    )
    bed_parser.add_argument(
        '--output-dir', default='data/variants/pruned',
        help='Output directory'
    )
    bed_parser.add_argument(
        '--plink2-path',
        help='Path to PLINK2 executable'
    )
    bed_parser.add_argument(
        '--window-kb', type=int, default=DEFAULT_WINDOW_KB,
        help=f'LD window size in kb (default: {DEFAULT_WINDOW_KB})'
    )
    bed_parser.add_argument(
        '--step', type=int, default=DEFAULT_STEP,
        help=f'Step size in variants (default: {DEFAULT_STEP})'
    )
    bed_parser.add_argument(
        '--r2-threshold', type=float, default=DEFAULT_R2_THRESHOLD,
        help=f'r² threshold (default: {DEFAULT_R2_THRESHOLD})'
    )

    # Python method (prune existing data)
    prune_parser = subparsers.add_parser(
        'prune',
        help='Prune existing dosage data using Python (slower, no PLINK2 needed)'
    )
    prune_parser.add_argument(
        '--variants', required=True,
        help='Path to variants.parquet'
    )
    prune_parser.add_argument(
        '--dosages', required=True,
        help='Path to dosages.zarr or dosages.npy'
    )
    prune_parser.add_argument(
        '--output-dir', default='data/variants/pruned',
        help='Output directory'
    )
    prune_parser.add_argument(
        '--window-kb', type=int, default=DEFAULT_WINDOW_KB,
        help=f'LD window size in kb (default: {DEFAULT_WINDOW_KB})'
    )
    prune_parser.add_argument(
        '--r2-threshold', type=float, default=DEFAULT_R2_THRESHOLD,
        help=f'r² threshold (default: {DEFAULT_R2_THRESHOLD})'
    )

    args = parser.parse_args()

    if args.command == 'plink2':
        prune_with_plink2(args)
    elif args.command == 'from-bed':
        prune_from_bed(args)
    elif args.command == 'prune':
        prune_existing_data(args)


if __name__ == "__main__":
    main()
