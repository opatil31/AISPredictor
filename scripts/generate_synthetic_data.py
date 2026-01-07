#!/usr/bin/env python3
"""
Generate Synthetic UK Biobank-like Data for Testing

Creates realistic synthetic data to test the cohort building pipeline
without requiring actual UK Biobank access.

This generates:
1. Phenotype file with ICD-10 diagnoses, age, sex, ethnicity
2. Genetic PCs file
3. Optional kinship file

The synthetic data mimics UK Biobank structure and field naming conventions.

Usage:
    python scripts/generate_synthetic_data.py \
        --n-samples 50000 \
        --n-cases 3000 \
        --output data/synthetic
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


def generate_phenotypes(
    n_samples: int,
    n_cases: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic phenotype data.

    Args:
        n_samples: Total number of samples.
        n_cases: Number of AIS cases.
        seed: Random seed.

    Returns:
        DataFrame with phenotype data.
    """
    np.random.seed(seed)

    # Generate sample IDs
    eids = [f"UKB{i:07d}" for i in range(1, n_samples + 1)]

    # Age at recruitment (typically 40-70 for UK Biobank)
    ages = np.random.normal(55, 8, n_samples).clip(40, 70).astype(int)

    # Sex (0 = female, 1 = male) - AIS has female predominance
    # Cases: ~70% female, Controls: ~50% female
    case_indices = set(np.random.choice(n_samples, n_cases, replace=False))

    sex = []
    for i in range(n_samples):
        if i in case_indices:
            # Cases: 70% female
            sex.append(0 if np.random.random() < 0.7 else 1)
        else:
            # Controls: 50% female
            sex.append(0 if np.random.random() < 0.5 else 1)

    # Ethnicity (field 21000)
    # 1 = White, 1001 = British, 1002 = Irish, 1003 = Any other white
    # We want ~90% European for realistic UK Biobank
    european_codes = [1, 1001, 1002, 1003]
    other_codes = [2, 3, 4, 5, 6]  # Other ancestries

    ethnicity = []
    for _ in range(n_samples):
        if np.random.random() < 0.90:
            ethnicity.append(np.random.choice(european_codes, p=[0.1, 0.7, 0.1, 0.1]))
        else:
            ethnicity.append(np.random.choice(other_codes))

    # Genetic ethnicity (field 22006) - 1 = Caucasian
    genetic_ethnicity = [
        1 if eth in european_codes else np.random.choice([2, 3, 4])
        for eth in ethnicity
    ]

    # ICD-10 diagnoses - generate as array columns
    # UK Biobank has multiple diagnosis instances (41270-0.0 to 41270-0.N)
    n_diag_cols = 20  # Number of diagnosis columns

    # Initialize diagnosis arrays
    diagnoses = {f"41270-0.{i}": [""] * n_samples for i in range(n_diag_cols)}

    # AIS ICD-10 codes
    ais_codes = ["M41.1", "M41.2"]
    other_scoliosis = ["M41.0", "M41.3", "M41.4", "M41.5", "M41.8", "M41.9"]

    # Common ICD-10 codes for background (non-scoliosis)
    background_codes = [
        "I10", "E11", "J45", "M54", "K21", "F32", "G43", "N39",
        "E78", "M79", "J30", "K29", "M17", "H52", "I25", "E03",
    ]

    # Assign diagnoses
    for i in range(n_samples):
        diag_list = []

        if i in case_indices:
            # AIS cases - assign AIS code
            if np.random.random() < 0.8:
                diag_list.append("M41.2")  # ~80% M41.2
            else:
                diag_list.append("M41.1")  # ~20% M41.1

        # Add random background diagnoses
        n_background = np.random.poisson(3)
        diag_list.extend(np.random.choice(background_codes, min(n_background, 10), replace=False))

        # Also add some non-AIS scoliosis to random controls (for exclusion testing)
        if i not in case_indices and np.random.random() < 0.005:
            diag_list.append(np.random.choice(other_scoliosis))

        # Assign to columns
        for j, diag in enumerate(diag_list[:n_diag_cols]):
            diagnoses[f"41270-0.{j}"][i] = diag

    # Create DataFrame
    df = pd.DataFrame({
        "eid": eids,
        "21003-0.0": ages,  # Age at recruitment
        "31-0.0": sex,      # Sex
        "21000-0.0": ethnicity,  # Self-reported ethnicity
        "22006-0.0": genetic_ethnicity,  # Genetic ethnicity
        **diagnoses,
    })

    logger.info(f"Generated phenotype data for {n_samples} samples")
    logger.info(f"  AIS cases (M41.1/M41.2): {n_cases}")

    return df


def generate_pcs(
    phenotype_df: pd.DataFrame,
    n_pcs: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic genetic PCs.

    European samples cluster together, with outliers for non-Europeans.

    Args:
        phenotype_df: Phenotype DataFrame with eid and ethnicity.
        n_pcs: Number of PCs to generate.
        seed: Random seed.

    Returns:
        DataFrame with genetic PCs.
    """
    np.random.seed(seed)

    n_samples = len(phenotype_df)
    eids = phenotype_df["eid"].values

    # Get ethnicity
    eth_col = "21000-0.0" if "21000-0.0" in phenotype_df.columns else "ethnicity"
    ethnicity = phenotype_df[eth_col].values

    # European codes
    european_codes = [1, 1001, 1002, 1003]
    is_european = [eth in european_codes for eth in ethnicity]

    # Generate PCs
    pcs = np.zeros((n_samples, n_pcs))

    # PC1-4 show ancestry structure
    for i in range(n_samples):
        if is_european[i]:
            # European cluster - tight distribution
            pcs[i, 0] = np.random.normal(0, 1)
            pcs[i, 1] = np.random.normal(0, 1)
            pcs[i, 2] = np.random.normal(0, 0.8)
            pcs[i, 3] = np.random.normal(0, 0.8)
        else:
            # Non-European - offset from European cluster
            offset = np.random.uniform(5, 15)
            angle = np.random.uniform(0, 2 * np.pi)
            pcs[i, 0] = offset * np.cos(angle) + np.random.normal(0, 1)
            pcs[i, 1] = offset * np.sin(angle) + np.random.normal(0, 1)
            pcs[i, 2] = np.random.normal(3, 2)
            pcs[i, 3] = np.random.normal(3, 2)

    # Remaining PCs are just noise
    for j in range(4, n_pcs):
        pcs[:, j] = np.random.normal(0, 0.5, n_samples)

    # Add a few European outliers (for QC testing)
    n_outliers = int(0.005 * sum(is_european))  # 0.5% outliers
    european_indices = [i for i, e in enumerate(is_european) if e]
    outlier_indices = np.random.choice(european_indices, n_outliers, replace=False)
    for idx in outlier_indices:
        pcs[idx, 0] += np.random.choice([-1, 1]) * np.random.uniform(7, 10)
        pcs[idx, 1] += np.random.choice([-1, 1]) * np.random.uniform(7, 10)

    # Create DataFrame
    pc_columns = {f"22009-0.{i+1}": pcs[:, i] for i in range(n_pcs)}
    df = pd.DataFrame({"eid": eids, **pc_columns})

    logger.info(f"Generated {n_pcs} genetic PCs for {n_samples} samples")
    logger.info(f"  European samples: {sum(is_european)}")
    logger.info(f"  Outliers added: {n_outliers}")

    return df


def generate_kinship(
    eids: list[str],
    n_related_pairs: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic kinship data.

    Args:
        eids: List of sample IDs.
        n_related_pairs: Number of related pairs to generate.
        seed: Random seed.

    Returns:
        DataFrame with kinship coefficients.
    """
    np.random.seed(seed)

    pairs = []
    used_ids = set()

    for _ in range(n_related_pairs):
        # Pick two random samples
        available = [e for e in eids if e not in used_ids]
        if len(available) < 2:
            break

        id1, id2 = np.random.choice(available, 2, replace=False)
        used_ids.add(id1)
        used_ids.add(id2)

        # Generate kinship coefficient
        # 0.5 = identical twins/duplicates
        # 0.25 = parent-child, full siblings
        # 0.125 = half-siblings, grandparent
        # 0.0625 = first cousins
        kinship = np.random.choice(
            [0.5, 0.25, 0.125, 0.0625, 0.1, 0.09],
            p=[0.01, 0.1, 0.2, 0.3, 0.2, 0.19],
        )

        pairs.append({
            "ID1": id1,
            "ID2": id2,
            "KINSHIP": kinship,
        })

    df = pd.DataFrame(pairs)

    logger.info(f"Generated kinship data with {len(pairs)} related pairs")

    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic UK Biobank-like data for testing"
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=50000,
        help="Total number of samples (default: 50000)",
    )

    parser.add_argument(
        "--n-cases",
        type=int,
        default=3000,
        help="Number of AIS cases (default: 3000)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic"),
        help="Output directory (default: data/synthetic)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (default: parquet)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Generate data
    logger.info("Generating synthetic phenotype data...")
    phenotypes = generate_phenotypes(args.n_samples, args.n_cases, args.seed)

    logger.info("Generating synthetic genetic PCs...")
    pcs = generate_pcs(phenotypes, seed=args.seed)

    logger.info("Generating synthetic kinship data...")
    kinship = generate_kinship(phenotypes["eid"].tolist(), seed=args.seed)

    # Save data
    if args.format == "parquet":
        phenotypes.to_parquet(args.output / "phenotypes.parquet", index=False)
        pcs.to_parquet(args.output / "genetic_pcs.parquet", index=False)
        kinship.to_parquet(args.output / "kinship.parquet", index=False)
    else:
        phenotypes.to_csv(args.output / "phenotypes.csv", index=False)
        pcs.to_csv(args.output / "genetic_pcs.csv", index=False)
        kinship.to_csv(args.output / "kinship.tsv", index=False, sep="\t")

    logger.info(f"\nSynthetic data saved to {args.output}/")
    logger.info(f"  phenotypes.{args.format}")
    logger.info(f"  genetic_pcs.{args.format}")
    logger.info(f"  kinship.{args.format}")

    # Print summary
    logger.info("\nData Summary:")
    logger.info(f"  Total samples: {args.n_samples}")
    logger.info(f"  AIS cases: {args.n_cases}")
    logger.info(f"  Expected controls after matching: ~{args.n_cases * 4}")

    logger.info("\nTo build cohort with this data, run:")
    logger.info(f"  python scripts/build_cohort.py \\")
    logger.info(f"    --phenotype {args.output}/phenotypes.{args.format} \\")
    logger.info(f"    --pcs {args.output}/genetic_pcs.{args.format} \\")
    logger.info(f"    --kinship {args.output}/kinship.{args.format}")


if __name__ == "__main__":
    main()
