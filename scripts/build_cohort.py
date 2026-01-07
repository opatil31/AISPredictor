#!/usr/bin/env python3
"""
Build AIS Case-Control Cohort (Phase 0)

This script orchestrates the cohort building pipeline:
1. Identify AIS cases from ICD-10 diagnoses
2. Apply ancestry QC (European + PCA outliers)
3. Exclude related samples
4. Match controls to cases (4:1 ratio)
5. Save final cohort files

Usage:
    python scripts/build_cohort.py \
        --phenotype data/ukb_phenotypes.parquet \
        --pcs data/ukb_pcs.parquet \
        --kinship data/ukb_kinship.tsv \
        --config configs/cohort_config.yaml \
        --output data/cohort

Output Files:
    - cohort.parquet: Final matched cohort with labels
    - matching_info.parquet: Case-control pair mappings
    - ancestry_pcs.parquet: Genetic PCs for matched samples
    - cohort_qc_report.yaml: QC statistics and balance metrics
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cohort import CohortBuilder
from src.cohort.cohort_builder import CohortConfig


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build AIS case-control cohort from UK Biobank data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--phenotype",
        type=Path,
        required=True,
        help="Path to phenotype file (parquet or CSV) with ICD-10, age, sex, ethnicity",
    )

    parser.add_argument(
        "--pcs",
        type=Path,
        default=None,
        help="Path to genetic PCs file (optional, can be in phenotype file)",
    )

    parser.add_argument(
        "--kinship",
        type=Path,
        default=None,
        help="Path to KING kinship file for relatedness exclusion",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "cohort_config.yaml",
        help="Path to configuration YAML file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "cohort",
        help="Output directory for cohort files",
    )

    parser.add_argument(
        "--controls-per-case",
        type=int,
        default=None,
        help="Override control:case ratio (default: from config)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Validate inputs
    if not args.phenotype.exists():
        logger.error(f"Phenotype file not found: {args.phenotype}")
        return 1

    if args.pcs and not args.pcs.exists():
        logger.error(f"PCs file not found: {args.pcs}")
        return 1

    if args.kinship and not args.kinship.exists():
        logger.warning(f"Kinship file not found: {args.kinship}")
        args.kinship = None

    # Load configuration
    if args.config.exists():
        logger.info(f"Loading configuration from {args.config}")
        config = CohortConfig.from_yaml(args.config)
    else:
        logger.info("Using default configuration")
        config = CohortConfig()

    # Override config with command line arguments
    config.output_dir = str(args.output)
    if args.controls_per_case is not None:
        config.controls_per_case = args.controls_per_case

    # Build cohort
    builder = CohortBuilder(config)

    try:
        results = builder.build_cohort(
            phenotype_file=args.phenotype,
            pc_file=args.pcs,
            kinship_file=args.kinship,
        )

        cohort = results["cohort"]
        n_cases = (cohort["label"] == 1).sum()
        n_controls = (cohort["label"] == 0).sum()

        logger.info("\n" + "=" * 60)
        logger.info("SUCCESS: Cohort building complete!")
        logger.info(f"  Total samples: {len(cohort)}")
        logger.info(f"  Cases: {n_cases}")
        logger.info(f"  Controls: {n_controls}")
        logger.info(f"  Output: {args.output}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Cohort building failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
