#!/usr/bin/env python3
"""
Phase 2: VEP Annotation Pipeline

Annotates variants with functional consequences using Ensembl VEP.

Workflow:
1. Convert PLINK binary (.bed) to VCF format
2. Run VEP (locally or submit to Ensembl)
3. Parse VEP output and integrate with variant data

Usage:
    # Step 1: Convert to VCF
    python scripts/run_vep.py convert-vcf \
        --bfile data/variants/pruned/chr6_pruned \
        --output data/variants/pruned/chr6_pruned.vcf

    # Step 2: Run VEP (see instructions printed by script)

    # Step 3: Parse VEP output
    python scripts/run_vep.py parse \
        --vep-output data/variants/pruned/chr6_vep.txt \
        --variants data/variants/pruned/variants_pruned.parquet \
        --output data/variants/pruned/variants_annotated.parquet
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def find_plink2() -> Optional[str]:
    """Find PLINK2 executable."""
    for name in ['plink2', 'plink2.exe']:
        path = shutil.which(name)
        if path:
            return path
    return None


def convert_bed_to_vcf(args):
    """Convert PLINK binary files to VCF format for VEP input."""
    logger.info("=" * 50)
    logger.info("Converting PLINK binary to VCF format")
    logger.info("=" * 50)

    # Find PLINK2
    if hasattr(args, 'plink2_path') and args.plink2_path:
        plink2_path = args.plink2_path
    else:
        plink2_path = find_plink2()
        if not plink2_path:
            logger.error("PLINK2 not found. Please specify --plink2-path or add to PATH.")
            return

    logger.info(f"Using PLINK2: {plink2_path}")

    # Check input files exist
    bed_prefix = Path(args.bfile)
    for ext in ['.bed', '.bim', '.fam']:
        test_path = Path(str(bed_prefix) + ext)
        if not test_path.exists():
            # Try with suffix replacement
            test_path = bed_prefix.with_suffix(ext)
            if not test_path.exists():
                logger.error(f"File not found: {bed_prefix}{ext}")
                return

    # Determine output path
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix != '.vcf':
            output_path = output_path.with_suffix('.vcf')
        output_prefix = str(output_path.with_suffix(''))
    else:
        output_prefix = str(bed_prefix)

    # Run PLINK2 to export VCF
    cmd = [
        plink2_path,
        '--bfile', str(bed_prefix),
        '--export', 'vcf',
        '--out', output_prefix
    ]

    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("VCF export completed successfully")
        for line in result.stdout.split('\n')[-10:]:
            if line.strip():
                logger.info(f"  {line}")
    except subprocess.CalledProcessError as e:
        logger.error(f"PLINK2 VCF export failed: {e.stderr}")
        return

    vcf_path = Path(f"{output_prefix}.vcf")
    if vcf_path.exists():
        # Count variants
        with open(vcf_path, 'r') as f:
            n_variants = sum(1 for line in f if not line.startswith('#'))

        file_size = vcf_path.stat().st_size / (1024 * 1024)  # MB

        logger.info("")
        logger.info("=" * 50)
        logger.info("VCF FILE CREATED")
        logger.info("=" * 50)
        logger.info(f"Output: {vcf_path}")
        logger.info(f"Variants: {n_variants:,}")
        logger.info(f"File size: {file_size:.1f} MB")
        logger.info("")
        logger.info("=" * 50)
        logger.info("NEXT STEPS: Run VEP")
        logger.info("=" * 50)
        print_vep_instructions(vcf_path)
    else:
        logger.error(f"VCF file not found: {vcf_path}")


def print_vep_instructions(vcf_path: Path):
    """Print instructions for running VEP."""
    logger.info("")
    logger.info("Option 1: Ensembl VEP Web Interface (easiest)")
    logger.info("-" * 50)
    logger.info("1. Go to: https://www.ensembl.org/Tools/VEP")
    logger.info("2. Upload your VCF file or paste variants")
    logger.info(f"   File: {vcf_path}")
    logger.info("3. Settings:")
    logger.info("   - Species: Human (GRCh38)")
    logger.info("   - Transcript database: Ensembl/GENCODE transcripts")
    logger.info("   - Check: SIFT, PolyPhen, Regulatory")
    logger.info("4. Submit and download results as TXT (tab-delimited)")
    logger.info("")
    logger.info("Option 2: Command-line VEP (if installed)")
    logger.info("-" * 50)
    logger.info(f"""
vep -i {vcf_path} \\
    -o {vcf_path.with_suffix('.vep.txt')} \\
    --cache \\
    --assembly GRCh38 \\
    --sift b \\
    --polyphen b \\
    --regulatory \\
    --canonical \\
    --symbol \\
    --tab \\
    --fields "Uploaded_variation,Location,Allele,Gene,Feature,Feature_type,Consequence,cDNA_position,CDS_position,Protein_position,Amino_acids,Codons,IMPACT,SYMBOL,SIFT,PolyPhen,CANONICAL"
""")
    logger.info("")
    logger.info("Option 3: VEP REST API (programmatic)")
    logger.info("-" * 50)
    logger.info("For large files, use the VEP REST API in batches:")
    logger.info("https://rest.ensembl.org/documentation/info/vep_region_post")
    logger.info("")
    logger.info("After running VEP, parse the output with:")
    logger.info(f"  python scripts/run_vep.py parse \\")
    logger.info(f"      --vep-output {vcf_path.with_suffix('.vep.txt')} \\")
    logger.info(f"      --variants data/variants/pruned/variants_pruned.parquet \\")
    logger.info(f"      --output data/variants/pruned/variants_annotated.parquet")


def parse_vep_output(args):
    """Parse VEP output and integrate with variant data."""
    logger.info("=" * 50)
    logger.info("Parsing VEP Output")
    logger.info("=" * 50)

    vep_path = Path(args.vep_output)
    variants_path = Path(args.variants)
    output_path = Path(args.output)

    if not vep_path.exists():
        logger.error(f"VEP output file not found: {vep_path}")
        return

    if not variants_path.exists():
        logger.error(f"Variants file not found: {variants_path}")
        return

    # Load existing variant data
    logger.info(f"Loading variants from {variants_path}")
    variants_df = pd.read_parquet(variants_path)
    logger.info(f"Loaded {len(variants_df):,} variants")

    # Parse VEP output
    logger.info(f"Parsing VEP output from {vep_path}")
    vep_df = parse_vep_file(vep_path)

    if vep_df is None or len(vep_df) == 0:
        logger.error("Failed to parse VEP output or no annotations found")
        return

    logger.info(f"Parsed {len(vep_df):,} VEP annotations")

    # Merge VEP annotations with variant data
    logger.info("Merging annotations with variant data...")
    annotated_df = merge_vep_annotations(variants_df, vep_df)

    # Save annotated variants
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_df.to_parquet(output_path, index=False)
    logger.info(f"Saved annotated variants to {output_path}")

    # Print summary
    print_annotation_summary(annotated_df)


def parse_vep_file(vep_path: Path) -> Optional[pd.DataFrame]:
    """
    Parse VEP tab-delimited output file.

    VEP output format varies based on options used. This handles common formats.
    """
    # Read file and find header line
    header_line = None
    data_start = 0

    with open(vep_path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('#Uploaded_variation') or line.startswith('#CHROM'):
                header_line = line.lstrip('#').strip()
                data_start = i + 1
                break
            elif line.startswith('##'):
                continue
            elif line.startswith('#'):
                header_line = line.lstrip('#').strip()
                data_start = i + 1
                break

    if header_line is None:
        # Try reading as simple tab-delimited
        logger.warning("No header found, attempting to read as tab-delimited")
        try:
            df = pd.read_csv(vep_path, sep='\t', comment='#')
            return df
        except Exception as e:
            logger.error(f"Failed to parse VEP file: {e}")
            return None

    # Parse with identified header
    try:
        df = pd.read_csv(
            vep_path,
            sep='\t',
            skiprows=data_start,
            names=header_line.split('\t'),
            na_values=['-', '.', ''],
            low_memory=False
        )
        return df
    except Exception as e:
        logger.error(f"Failed to parse VEP file: {e}")
        return None


def merge_vep_annotations(variants_df: pd.DataFrame, vep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge VEP annotations with variant data.

    Handles different VEP output formats and column names.
    """
    # Identify variant ID column in VEP output
    vep_id_col = None
    for col in ['Uploaded_variation', '#Uploaded_variation', 'ID', 'Variant']:
        if col in vep_df.columns:
            vep_id_col = col
            break

    # Identify variant ID column in variants data
    var_id_col = None
    for col in ['ID', 'variant_id', 'rsid', 'SNP']:
        if col in variants_df.columns:
            var_id_col = col
            break

    if vep_id_col is None or var_id_col is None:
        logger.warning("Could not identify matching ID columns. Attempting positional merge.")
        return merge_by_position(variants_df, vep_df)

    # Aggregate VEP annotations per variant (take most severe consequence)
    vep_agg = aggregate_vep_annotations(vep_df, vep_id_col)

    # Merge
    annotated_df = variants_df.merge(
        vep_agg,
        left_on=var_id_col,
        right_on=vep_id_col,
        how='left'
    )

    # Drop duplicate ID column if present
    if vep_id_col in annotated_df.columns and vep_id_col != var_id_col:
        annotated_df = annotated_df.drop(columns=[vep_id_col])

    return annotated_df


def merge_by_position(variants_df: pd.DataFrame, vep_df: pd.DataFrame) -> pd.DataFrame:
    """Merge by genomic position when IDs don't match."""
    # Extract position from VEP Location column if available
    if 'Location' in vep_df.columns:
        # Format: chr:pos or chr:start-end
        vep_df = vep_df.copy()
        loc_parts = vep_df['Location'].str.split(':', expand=True)
        vep_df['CHR_VEP'] = loc_parts[0]
        vep_df['POS_VEP'] = loc_parts[1].str.split('-').str[0].astype(int)

        # Aggregate by position
        vep_agg = vep_df.groupby(['CHR_VEP', 'POS_VEP']).first().reset_index()

        # Merge with variants
        variants_df = variants_df.copy()
        if 'CHR' in variants_df.columns and 'POS' in variants_df.columns:
            variants_df['CHR_str'] = variants_df['CHR'].astype(str)
            annotated_df = variants_df.merge(
                vep_agg,
                left_on=['CHR_str', 'POS'],
                right_on=['CHR_VEP', 'POS_VEP'],
                how='left'
            )
            annotated_df = annotated_df.drop(columns=['CHR_str', 'CHR_VEP', 'POS_VEP'], errors='ignore')
            return annotated_df

    logger.warning("Could not merge by position. Returning original variants with empty annotations.")
    return variants_df


def aggregate_vep_annotations(vep_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Aggregate VEP annotations per variant, keeping most severe consequence.
    """
    # Consequence severity ranking (higher = more severe)
    consequence_severity = {
        'transcript_ablation': 100,
        'splice_acceptor_variant': 95,
        'splice_donor_variant': 95,
        'stop_gained': 90,
        'frameshift_variant': 90,
        'stop_lost': 85,
        'start_lost': 85,
        'transcript_amplification': 80,
        'inframe_insertion': 75,
        'inframe_deletion': 75,
        'missense_variant': 70,
        'protein_altering_variant': 65,
        'splice_region_variant': 60,
        'incomplete_terminal_codon_variant': 55,
        'start_retained_variant': 50,
        'stop_retained_variant': 50,
        'synonymous_variant': 45,
        'coding_sequence_variant': 40,
        'mature_miRNA_variant': 35,
        '5_prime_UTR_variant': 30,
        '3_prime_UTR_variant': 30,
        'non_coding_transcript_exon_variant': 25,
        'intron_variant': 20,
        'NMD_transcript_variant': 15,
        'non_coding_transcript_variant': 15,
        'upstream_gene_variant': 10,
        'downstream_gene_variant': 10,
        'TFBS_ablation': 8,
        'TFBS_amplification': 8,
        'TF_binding_site_variant': 7,
        'regulatory_region_ablation': 6,
        'regulatory_region_amplification': 6,
        'regulatory_region_variant': 5,
        'feature_elongation': 4,
        'feature_truncation': 4,
        'intergenic_variant': 1,
    }

    # Find consequence column
    conseq_col = None
    for col in ['Consequence', 'CONSEQUENCE', 'consequence']:
        if col in vep_df.columns:
            conseq_col = col
            break

    if conseq_col is None:
        # No consequence column, just take first entry per variant
        return vep_df.groupby(id_col).first().reset_index()

    # Calculate severity score for each annotation
    def get_severity(consequence_str):
        if pd.isna(consequence_str):
            return 0
        consequences = str(consequence_str).split(',')
        max_severity = 0
        for c in consequences:
            c = c.strip()
            if c in consequence_severity:
                max_severity = max(max_severity, consequence_severity[c])
        return max_severity

    vep_df = vep_df.copy()
    vep_df['_severity'] = vep_df[conseq_col].apply(get_severity)

    # Sort by severity (descending) and take first per variant
    vep_df = vep_df.sort_values('_severity', ascending=False)
    vep_agg = vep_df.groupby(id_col).first().reset_index()
    vep_agg = vep_agg.drop(columns=['_severity'])

    return vep_agg


def print_annotation_summary(df: pd.DataFrame):
    """Print summary of VEP annotations."""
    logger.info("")
    logger.info("=" * 50)
    logger.info("ANNOTATION SUMMARY")
    logger.info("=" * 50)

    # Find consequence column
    conseq_col = None
    for col in ['Consequence', 'CONSEQUENCE', 'consequence']:
        if col in df.columns:
            conseq_col = col
            break

    if conseq_col:
        # Count consequence types
        conseq_counts = df[conseq_col].value_counts().head(15)
        logger.info("")
        logger.info("Top consequence types:")
        for conseq, count in conseq_counts.items():
            logger.info(f"  {conseq}: {count:,}")

    # Find impact column
    impact_col = None
    for col in ['IMPACT', 'Impact', 'impact']:
        if col in df.columns:
            impact_col = col
            break

    if impact_col:
        impact_counts = df[impact_col].value_counts()
        logger.info("")
        logger.info("Impact distribution:")
        for impact, count in impact_counts.items():
            logger.info(f"  {impact}: {count:,} ({100*count/len(df):.1f}%)")

    # Check annotation coverage
    non_null_count = df[conseq_col].notna().sum() if conseq_col else 0
    logger.info("")
    logger.info(f"Total variants: {len(df):,}")
    logger.info(f"Annotated variants: {non_null_count:,} ({100*non_null_count/len(df):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="VEP annotation pipeline for variant data"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Convert to VCF subcommand
    vcf_parser = subparsers.add_parser(
        'convert-vcf',
        help='Convert PLINK binary to VCF format for VEP input'
    )
    vcf_parser.add_argument(
        '--bfile', required=True,
        help='PLINK1 binary file prefix'
    )
    vcf_parser.add_argument(
        '--output', '-o',
        help='Output VCF file path (default: same as input with .vcf extension)'
    )
    vcf_parser.add_argument(
        '--plink2-path',
        help='Path to PLINK2 executable'
    )

    # Parse VEP output subcommand
    parse_parser = subparsers.add_parser(
        'parse',
        help='Parse VEP output and merge with variant data'
    )
    parse_parser.add_argument(
        '--vep-output', required=True,
        help='Path to VEP output file (tab-delimited)'
    )
    parse_parser.add_argument(
        '--variants', required=True,
        help='Path to variants parquet file'
    )
    parse_parser.add_argument(
        '--output', '-o', required=True,
        help='Output path for annotated variants parquet'
    )

    # Full pipeline subcommand (for local VEP)
    full_parser = subparsers.add_parser(
        'run',
        help='Run full VEP pipeline (requires local VEP installation)'
    )
    full_parser.add_argument(
        '--bfile', required=True,
        help='PLINK1 binary file prefix'
    )
    full_parser.add_argument(
        '--variants', required=True,
        help='Path to variants parquet file'
    )
    full_parser.add_argument(
        '--output-dir', default='data/variants/annotated',
        help='Output directory'
    )
    full_parser.add_argument(
        '--vep-path',
        help='Path to VEP executable'
    )
    full_parser.add_argument(
        '--cache-dir',
        help='VEP cache directory'
    )

    args = parser.parse_args()

    if args.command == 'convert-vcf':
        convert_bed_to_vcf(args)
    elif args.command == 'parse':
        parse_vep_output(args)
    elif args.command == 'run':
        logger.error("Full pipeline not yet implemented. Please use convert-vcf and parse separately.")
        logger.info("See instructions with: python scripts/run_vep.py convert-vcf --help")


if __name__ == "__main__":
    main()
