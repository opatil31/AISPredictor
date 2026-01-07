#!/usr/bin/env python3
"""
Phase 3: Sequence Construction

Constructs DNA sequences around each variant for HyenaDNA embedding.

For each variant, we create:
1. Reference sequence: context window around the variant with reference allele
2. Alternate sequence: context window around the variant with alternate allele

Usage:
    python scripts/construct_sequences.py \
        --variants data/variants/pruned/variants_annotated.parquet \
        --fasta /path/to/GRCh38.fa \
        --output data/sequences/variant_sequences.parquet \
        --context-size 512

The context size determines how much flanking sequence to include.
HyenaDNA supports up to 1M tokens, but 512-2048 is typical for variant analysis.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# Try to import pysam (preferred) or pyfaidx for FASTA reading
try:
    import pysam
    HAS_PYSAM = True
except ImportError:
    HAS_PYSAM = False

try:
    from pyfaidx import Fasta
    HAS_PYFAIDX = True
except ImportError:
    HAS_PYFAIDX = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


class FastaReader:
    """Wrapper for reading sequences from FASTA files."""

    def __init__(self, fasta_path: str):
        self.fasta_path = fasta_path

        if HAS_PYSAM:
            self.backend = 'pysam'
            self.fasta = pysam.FastaFile(fasta_path)
            self.chroms = set(self.fasta.references)
        elif HAS_PYFAIDX:
            self.backend = 'pyfaidx'
            self.fasta = Fasta(fasta_path)
            self.chroms = set(self.fasta.keys())
        else:
            raise ImportError(
                "Neither pysam nor pyfaidx is available. "
                "Install with: pip install pysam or pip install pyfaidx"
            )

        logger.info(f"Loaded FASTA with {len(self.chroms)} chromosomes using {self.backend}")

    def get_sequence(self, chrom: str, start: int, end: int) -> str:
        """
        Get sequence from FASTA.

        Args:
            chrom: Chromosome name (with or without 'chr' prefix)
            start: Start position (0-based)
            end: End position (exclusive)

        Returns:
            DNA sequence as uppercase string
        """
        # Handle chromosome naming (chr6 vs 6)
        if chrom not in self.chroms:
            if chrom.startswith('chr'):
                chrom = chrom[3:]
            else:
                chrom = 'chr' + chrom

        if chrom not in self.chroms:
            return ''

        if self.backend == 'pysam':
            return self.fasta.fetch(chrom, start, end).upper()
        else:
            return str(self.fasta[chrom][start:end]).upper()

    def close(self):
        if self.backend == 'pysam':
            self.fasta.close()


def construct_variant_sequence(
    fasta: FastaReader,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    context_size: int = 512
) -> Tuple[str, str]:
    """
    Construct reference and alternate sequences around a variant.

    Args:
        fasta: FastaReader instance
        chrom: Chromosome
        pos: 1-based position
        ref: Reference allele
        alt: Alternate allele
        context_size: Total sequence length (variant will be centered)

    Returns:
        Tuple of (reference_sequence, alternate_sequence)
    """
    # Convert to 0-based coordinates
    var_start = pos - 1
    var_end = var_start + len(ref)

    # Calculate flanking region sizes
    # We want the variant roughly centered in the context window
    left_flank = (context_size - len(ref)) // 2
    right_flank = context_size - len(ref) - left_flank

    # Get flanking sequences
    left_seq = fasta.get_sequence(chrom, max(0, var_start - left_flank), var_start)
    right_seq = fasta.get_sequence(chrom, var_end, var_end + right_flank)

    # Pad if we hit chromosome boundaries
    if len(left_seq) < left_flank:
        left_seq = 'N' * (left_flank - len(left_seq)) + left_seq
    if len(right_seq) < right_flank:
        right_seq = right_seq + 'N' * (right_flank - len(right_seq))

    # Construct full sequences
    ref_seq = left_seq + ref + right_seq
    alt_seq = left_seq + alt + right_seq

    return ref_seq, alt_seq


def construct_sequences(args):
    """Main function to construct sequences for all variants."""
    logger.info("=" * 50)
    logger.info("Phase 3: Sequence Construction")
    logger.info("=" * 50)

    # Load variants
    logger.info(f"Loading variants from {args.variants}")
    variants_df = pd.read_parquet(args.variants)
    logger.info(f"Loaded {len(variants_df):,} variants")

    # Check required columns
    required_cols = ['CHR', 'POS', 'A1', 'A2']  # A1=ref, A2=alt in PLINK convention
    alt_cols = ['REF', 'ALT']  # Alternative column names

    # Map column names
    col_map = {}
    if 'CHR' in variants_df.columns:
        col_map['chrom'] = 'CHR'
    elif 'CHROM' in variants_df.columns:
        col_map['chrom'] = 'CHROM'
    elif '#CHROM' in variants_df.columns:
        col_map['chrom'] = '#CHROM'

    if 'POS' in variants_df.columns:
        col_map['pos'] = 'POS'

    if 'A1' in variants_df.columns and 'A2' in variants_df.columns:
        col_map['ref'] = 'A1'
        col_map['alt'] = 'A2'
    elif 'REF' in variants_df.columns and 'ALT' in variants_df.columns:
        col_map['ref'] = 'REF'
        col_map['alt'] = 'ALT'

    if len(col_map) < 4:
        logger.error(f"Missing required columns. Found: {variants_df.columns.tolist()}")
        logger.error("Need: CHR/CHROM, POS, and either A1/A2 or REF/ALT")
        return

    # Open FASTA
    logger.info(f"Opening FASTA: {args.fasta}")
    fasta = FastaReader(args.fasta)

    # Construct sequences
    logger.info(f"Constructing sequences with context size {args.context_size}...")

    ref_sequences = []
    alt_sequences = []
    failed_count = 0

    for idx, row in variants_df.iterrows():
        chrom = str(row[col_map['chrom']])
        pos = int(row[col_map['pos']])
        ref = str(row[col_map['ref']])
        alt = str(row[col_map['alt']])

        try:
            ref_seq, alt_seq = construct_variant_sequence(
                fasta, chrom, pos, ref, alt, args.context_size
            )

            # Validate sequences
            if len(ref_seq) == 0 or 'N' * 50 in ref_seq:
                ref_seq = ''
                alt_seq = ''
                failed_count += 1

            ref_sequences.append(ref_seq)
            alt_sequences.append(alt_seq)

        except Exception as e:
            logger.warning(f"Failed for variant at {chrom}:{pos}: {e}")
            ref_sequences.append('')
            alt_sequences.append('')
            failed_count += 1

        if (idx + 1) % 10000 == 0:
            logger.info(f"Processed {idx + 1:,}/{len(variants_df):,} variants")

    fasta.close()

    # Add sequences to dataframe
    variants_df['ref_sequence'] = ref_sequences
    variants_df['alt_sequence'] = alt_sequences
    variants_df['sequence_length'] = [len(s) for s in ref_sequences]

    # Filter out failed variants if requested
    if args.drop_failed:
        n_before = len(variants_df)
        variants_df = variants_df[variants_df['ref_sequence'] != ''].reset_index(drop=True)
        logger.info(f"Dropped {n_before - len(variants_df)} variants with missing sequences")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variants_df.to_parquet(output_path, index=False)

    logger.info("")
    logger.info("=" * 50)
    logger.info("SEQUENCE CONSTRUCTION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total variants: {len(variants_df):,}")
    logger.info(f"Failed/missing: {failed_count:,}")
    logger.info(f"Context size: {args.context_size}")
    logger.info(f"Output: {output_path}")

    # Print sample
    if len(variants_df) > 0:
        sample = variants_df.iloc[0]
        logger.info("")
        logger.info("Sample sequence (first variant):")
        logger.info(f"  Position: {sample[col_map['chrom']]}:{sample[col_map['pos']]}")
        logger.info(f"  Ref allele: {sample[col_map['ref']]}")
        logger.info(f"  Alt allele: {sample[col_map['alt']]}")
        if sample['ref_sequence']:
            logger.info(f"  Ref seq (first 50bp): {sample['ref_sequence'][:50]}...")
            logger.info(f"  Alt seq (first 50bp): {sample['alt_sequence'][:50]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Construct DNA sequences around variants for HyenaDNA"
    )
    parser.add_argument(
        '--variants', required=True,
        help='Path to variants parquet file (with annotations)'
    )
    parser.add_argument(
        '--fasta', required=True,
        help='Path to reference genome FASTA file (GRCh38)'
    )
    parser.add_argument(
        '--output', '-o', required=True,
        help='Output parquet file with sequences'
    )
    parser.add_argument(
        '--context-size', type=int, default=512,
        help='Total sequence length around variant (default: 512)'
    )
    parser.add_argument(
        '--drop-failed', action='store_true',
        help='Drop variants where sequence extraction failed'
    )

    args = parser.parse_args()
    construct_sequences(args)


if __name__ == "__main__":
    main()
