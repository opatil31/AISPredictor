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
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

# For REST API
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

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


def run_vep_rest_api(args):
    """Run VEP annotation using the Ensembl REST API in batches."""
    if not HAS_REQUESTS:
        logger.error("requests library required. Install with: pip install requests")
        return

    logger.info("=" * 50)
    logger.info("VEP Annotation via REST API")
    logger.info("=" * 50)

    vcf_path = Path(args.vcf)
    if not vcf_path.exists():
        logger.error(f"VCF file not found: {vcf_path}")
        return

    output_path = Path(args.output) if args.output else vcf_path.with_suffix('.vep.json')

    # Parse VCF to get variants
    logger.info(f"Reading variants from {vcf_path}")
    variants = parse_vcf_for_api(vcf_path)
    logger.info(f"Found {len(variants):,} variants to annotate")

    # VEP REST API settings
    server = "https://rest.ensembl.org"
    endpoint = "/vep/homo_sapiens/region"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Batch settings
    batch_size = args.batch_size  # Max 200 per request
    n_batches = (len(variants) + batch_size - 1) // batch_size

    logger.info(f"Processing in {n_batches} batches of {batch_size} variants")
    logger.info(f"Estimated time: {n_batches * 1.5 / 60:.1f} - {n_batches * 3 / 60:.1f} minutes")

    all_results = []
    failed_batches = []
    start_time = time.time()

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(variants))
        batch_variants = variants[start:end]

        # Prepare request payload
        payload = {
            "variants": batch_variants,
            "canonical": 1,
            "SIFT": "b",
            "PolyPhen": "b",
            "regulatory": 1,
        }

        # Retry logic with exponential backoff
        max_retries = 4
        for retry in range(max_retries):
            try:
                response = requests.post(
                    server + endpoint,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=60
                )

                if response.status_code == 200:
                    results = response.json()
                    all_results.extend(results)
                    break
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** (retry + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Batch {batch_idx+1}: HTTP {response.status_code}")
                    if retry < max_retries - 1:
                        time.sleep(2 ** retry)
                    else:
                        failed_batches.append(batch_idx)
            except requests.exceptions.Timeout:
                logger.warning(f"Batch {batch_idx+1}: Timeout, retrying...")
                if retry == max_retries - 1:
                    failed_batches.append(batch_idx)
            except Exception as e:
                logger.warning(f"Batch {batch_idx+1}: Error {e}")
                if retry == max_retries - 1:
                    failed_batches.append(batch_idx)

        # Rate limiting - be nice to the API
        time.sleep(0.1)  # 10 requests per second max

        # Progress update
        if (batch_idx + 1) % 50 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - start_time
            rate = (batch_idx + 1) / elapsed
            remaining = (n_batches - batch_idx - 1) / rate if rate > 0 else 0
            logger.info(
                f"Processed {batch_idx+1}/{n_batches} batches "
                f"({end:,}/{len(variants):,} variants) - "
                f"ETA: {remaining/60:.1f} min"
            )

    elapsed_total = time.time() - start_time
    logger.info(f"Completed in {elapsed_total/60:.1f} minutes")

    if failed_batches:
        logger.warning(f"Failed batches: {len(failed_batches)} (will have missing annotations)")

    # Save raw JSON results
    logger.info(f"Saving {len(all_results):,} annotations to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(all_results, f)

    # Also convert to tab-delimited format for easier parsing
    txt_path = output_path.with_suffix('.vep.txt')
    convert_json_to_tabular(all_results, txt_path)

    logger.info("=" * 50)
    logger.info("VEP ANNOTATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"JSON output: {output_path}")
    logger.info(f"Tabular output: {txt_path}")
    logger.info("")
    logger.info("Next step - parse and merge with variants:")
    logger.info(f"  python scripts/run_vep.py parse \\")
    logger.info(f"      --vep-output {txt_path} \\")
    logger.info(f"      --variants data/variants/pruned/variants_pruned.parquet \\")
    logger.info(f"      --output data/variants/pruned/variants_annotated.parquet")


def parse_vcf_for_api(vcf_path: Path) -> List[str]:
    """Parse VCF file and convert to VEP REST API format."""
    variants = []

    with open(vcf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            chrom = parts[0].replace('chr', '')
            pos = parts[1]
            ref = parts[3]
            alt = parts[4]

            # Handle multiple alts
            for a in alt.split(','):
                # Format: "chr pos ref alt"
                # For SNPs: "6 26092913 A G"
                # For indels: "6 26092913 AT A"
                variant_str = f"{chrom} {pos} {ref} {a}"
                variants.append(variant_str)

    return variants


def convert_json_to_tabular(results: List[Dict], output_path: Path):
    """Convert VEP JSON results to tab-delimited format."""
    rows = []

    for result in results:
        input_var = result.get('input', '')

        # Get most severe consequence
        most_severe = result.get('most_severe_consequence', '')

        # Get transcript consequences
        transcript_consequences = result.get('transcript_consequences', [])

        if transcript_consequences:
            for tc in transcript_consequences:
                row = {
                    'Uploaded_variation': input_var,
                    'Location': f"{result.get('seq_region_name', '')}:{result.get('start', '')}",
                    'Allele': result.get('allele_string', '').split('/')[-1] if '/' in result.get('allele_string', '') else '',
                    'Gene': tc.get('gene_id', ''),
                    'Feature': tc.get('transcript_id', ''),
                    'Feature_type': 'Transcript',
                    'Consequence': ','.join(tc.get('consequence_terms', [])),
                    'IMPACT': tc.get('impact', ''),
                    'SYMBOL': tc.get('gene_symbol', ''),
                    'SIFT': f"{tc.get('sift_prediction', '')}({tc.get('sift_score', '')})" if tc.get('sift_prediction') else '',
                    'PolyPhen': f"{tc.get('polyphen_prediction', '')}({tc.get('polyphen_score', '')})" if tc.get('polyphen_prediction') else '',
                    'CANONICAL': 'YES' if tc.get('canonical') else '',
                    'Amino_acids': tc.get('amino_acids', ''),
                    'Codons': tc.get('codons', ''),
                    'Protein_position': tc.get('protein_start', ''),
                }
                rows.append(row)
        else:
            # No transcript consequences - intergenic or regulatory
            row = {
                'Uploaded_variation': input_var,
                'Location': f"{result.get('seq_region_name', '')}:{result.get('start', '')}",
                'Allele': result.get('allele_string', '').split('/')[-1] if '/' in result.get('allele_string', '') else '',
                'Gene': '',
                'Feature': '',
                'Feature_type': '',
                'Consequence': most_severe,
                'IMPACT': '',
                'SYMBOL': '',
                'SIFT': '',
                'PolyPhen': '',
                'CANONICAL': '',
                'Amino_acids': '',
                'Codons': '',
                'Protein_position': '',
            }
            rows.append(row)

    # Write to file
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)
    logger.info(f"Wrote {len(rows):,} annotation rows to {output_path}")


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

    # REST API subcommand (recommended for ~100K variants)
    api_parser = subparsers.add_parser(
        'rest-api',
        help='Run VEP via Ensembl REST API (recommended for ~100K variants)'
    )
    api_parser.add_argument(
        '--vcf', required=True,
        help='Path to VCF file (from convert-vcf step)'
    )
    api_parser.add_argument(
        '--output', '-o',
        help='Output file path (default: input.vep.json)'
    )
    api_parser.add_argument(
        '--batch-size', type=int, default=200,
        help='Variants per API request (default: 200, max: 200)'
    )

    args = parser.parse_args()

    if args.command == 'convert-vcf':
        convert_bed_to_vcf(args)
    elif args.command == 'parse':
        parse_vep_output(args)
    elif args.command == 'rest-api':
        run_vep_rest_api(args)
    elif args.command == 'run':
        logger.error("Full pipeline not yet implemented. Please use convert-vcf and parse separately.")
        logger.info("See instructions with: python scripts/run_vep.py convert-vcf --help")


if __name__ == "__main__":
    main()
