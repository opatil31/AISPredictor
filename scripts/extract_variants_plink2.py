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
               maf: float = 0.01, hwe_p: float = 1e-6) -> bool:
    """
    Run PLINK2 to filter variants and export dosages.

    Args:
        plink2_path: Path to plink2 executable
        pfile: Path prefix to .pgen/.pvar/.psam files
        keep_file: Path to file with sample IDs to keep
        output_prefix: Output file prefix
        maf: Minimum minor allele frequency
        hwe_p: Minimum HWE p-value

    Returns:
        True if successful
    """
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
    logger.info("STEP 3: Running PLINK2 QC and export")
    logger.info("=" * 50)

    output_prefix = str(output_dir / "chr6_filtered")

    success = run_plink2(
        plink2_path=plink2_path,
        pfile=args.pfile,
        keep_file=str(ids_path),
        output_prefix=output_prefix,
        maf=args.maf_min,
        hwe_p=args.hwe_p_min
    )

    if not success:
        logger.error("PLINK2 failed. Check the error messages above.")
        return

    # Step 4: Convert to project format
    logger.info("")
    logger.info("=" * 50)
    logger.info("STEP 4: Converting to project format")
    logger.info("=" * 50)

    raw_path = Path(f"{output_prefix}.raw")
    if not raw_path.exists():
        logger.error(f"Expected output file not found: {raw_path}")
        return

    # Create args-like object for convert function
    class ConvertArgs:
        pass

    convert_args = ConvertArgs()
    convert_args.raw = str(raw_path)
    convert_args.output_dir = str(output_dir)
    convert_args.chunked = args.chunked
    convert_args.chunk_size = args.chunk_size

    if args.chunked:
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
        help='Use chunked reading for large files'
    )
    run_parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Samples per chunk (default: 1000)'
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
    elif args.pfile and args.cohort:
        # No subcommand but --pfile and --cohort provided: run full pipeline
        run_full_pipeline(args)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/extract_variants_plink2.py --pfile /path/to/chr6 --cohort data/cohort/cohort.parquet")
        print("  python scripts/extract_variants_plink2.py run --pfile /path/to/chr6 --cohort data/cohort/cohort.parquet")


if __name__ == "__main__":
    main()
