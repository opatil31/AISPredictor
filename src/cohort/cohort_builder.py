"""
Cohort Builder - Main orchestration module for Phase 0

Coordinates case identification, ancestry QC, and control matching
to build the final AIS case-control cohort.

Outputs:
    - cohort.parquet: Final matched cohort with labels and covariates
    - matching_info.parquet: Case-control pair mappings
    - ancestry_pcs.parquet: Genetic PCs for matched samples
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from .case_identification import CaseIdentifier
from .ancestry_qc import AncestryQC
from .control_matching import ControlMatcher, exclude_related_samples

logger = logging.getLogger(__name__)


@dataclass
class CohortConfig:
    """Configuration for cohort building."""

    # Default values as class constants
    _DEFAULT_CASE_CODES = ["M41.1", "M41.2"]
    _DEFAULT_EXCLUSION_CODES = [
        "M41", "M41.0", "M41.1", "M41.2", "M41.3",
        "M41.4", "M41.5", "M41.8", "M41.9"
    ]
    _DEFAULT_EUROPEAN_CODES = [1, 1001, 1002, 1003]
    _DEFAULT_MATCHING_VARS = ["age", "sex", "pc1", "pc2", "pc3", "pc4"]

    # Case identification
    case_icd10_codes: list[str] = field(
        default_factory=lambda: ["M41.1", "M41.2"]
    )
    exclusion_icd10_codes: list[str] = field(
        default_factory=lambda: ["M41", "M41.0", "M41.1", "M41.2", "M41.3",
                                  "M41.4", "M41.5", "M41.8", "M41.9"]
    )

    # Ancestry QC
    european_codes: list[int] = field(
        default_factory=lambda: [1, 1001, 1002, 1003]
    )
    n_pcs_for_outlier: int = 4
    sd_threshold: float = 6.0
    use_genetic_ethnicity: bool = True

    # Control matching
    controls_per_case: int = 4
    matching_variables: list[str] = field(
        default_factory=lambda: ["age", "sex", "pc1", "pc2", "pc3", "pc4"]
    )
    matching_replacement: bool = False
    matching_caliper: Optional[float] = None

    # Relatedness
    exclude_related: bool = True
    kinship_threshold: float = 0.0884

    # Output
    output_dir: str = "data/cohort"
    save_intermediate: bool = True

    @classmethod
    def from_yaml(cls, config_file: Path) -> "CohortConfig":
        """Load configuration from YAML file."""
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Extract relevant sections
        case_cfg = config.get("case_identification", {})
        ancestry_cfg = config.get("ancestry_qc", {})
        matching_cfg = config.get("control_matching", {})
        output_cfg = config.get("output", {})

        return cls(
            case_icd10_codes=case_cfg.get("icd10_codes", cls._DEFAULT_CASE_CODES),
            exclusion_icd10_codes=case_cfg.get(
                "scoliosis_exclusion_codes", cls._DEFAULT_EXCLUSION_CODES
            ),
            european_codes=ancestry_cfg.get("european_codes", cls._DEFAULT_EUROPEAN_CODES),
            n_pcs_for_outlier=ancestry_cfg.get(
                "pca_outlier_removal", {}
            ).get("n_pcs", 4),
            sd_threshold=ancestry_cfg.get(
                "pca_outlier_removal", {}
            ).get("sd_threshold", 6.0),
            use_genetic_ethnicity=ancestry_cfg.get(
                "use_genetic_ethnicity", True
            ),
            controls_per_case=matching_cfg.get(
                "controls_per_case", 4
            ),
            matching_variables=matching_cfg.get(
                "matching_variables", cls._DEFAULT_MATCHING_VARS
            ),
            matching_replacement=matching_cfg.get(
                "algorithm", {}
            ).get("replacement", False),
            matching_caliper=matching_cfg.get(
                "algorithm", {}
            ).get("caliper"),
            exclude_related=matching_cfg.get(
                "kinship", {}
            ).get("exclude_related", True),
            kinship_threshold=matching_cfg.get(
                "kinship", {}
            ).get("king_threshold", 0.0884),
            output_dir=output_cfg.get("directory", "data/cohort"),
            save_intermediate=output_cfg.get("save_intermediate", True),
        )


class CohortBuilder:
    """
    Main class for building the AIS case-control cohort.

    This class orchestrates the full Phase 0 pipeline:
    1. Load and validate input data
    2. Identify AIS cases from ICD-10 diagnoses
    3. Apply ancestry QC filters
    4. Exclude related samples
    5. Match controls to cases
    6. Save final cohort files
    """

    def __init__(self, config: Optional[CohortConfig] = None):
        """
        Initialize cohort builder.

        Args:
            config: Cohort configuration. Uses defaults if not provided.
        """
        self.config = config or CohortConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize component modules
        self.case_identifier = CaseIdentifier(
            case_icd10_codes=self.config.case_icd10_codes,
            exclusion_icd10_codes=self.config.exclusion_icd10_codes,
        )

        self.ancestry_qc = AncestryQC(
            european_codes=self.config.european_codes,
            n_pcs_for_outlier=self.config.n_pcs_for_outlier,
            sd_threshold=self.config.sd_threshold,
            use_genetic_ethnicity=self.config.use_genetic_ethnicity,
        )

        self.control_matcher = ControlMatcher(
            controls_per_case=self.config.controls_per_case,
            matching_variables=self.config.matching_variables,
            replacement=self.config.matching_replacement,
            caliper=self.config.matching_caliper,
        )

        logger.info("CohortBuilder initialized with configuration:")
        logger.info(f"  Output directory: {self.output_dir}")

    def build_cohort(
        self,
        phenotype_file: Path,
        pc_file: Optional[Path] = None,
        kinship_file: Optional[Path] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Build the complete AIS case-control cohort.

        Args:
            phenotype_file: Path to UK Biobank phenotype file containing
                ICD-10 diagnoses, age, sex, and ethnicity.
            pc_file: Optional path to genetic PCs file.
            kinship_file: Optional path to KING kinship file.

        Returns:
            Dictionary with DataFrames:
                - cohort: Final matched cohort
                - matching_info: Case-control pair mappings
                - ancestry_pcs: Genetic PCs for matched samples
        """
        logger.info("=" * 60)
        logger.info("Starting cohort building pipeline")
        logger.info("=" * 60)

        # Step 1: Identify cases and potential controls
        logger.info("\n[Step 1/5] Identifying AIS cases...")
        case_labels = self.case_identifier.get_case_control_labels(phenotype_file)

        if self.config.save_intermediate:
            case_labels.to_parquet(self.output_dir / "case_labels_raw.parquet")

        n_cases_raw = (case_labels["label"] == 1).sum()
        n_controls_raw = (case_labels["label"] == 0).sum()
        n_excluded = (case_labels["label"] == -1).sum()
        logger.info(f"  Raw cases: {n_cases_raw}")
        logger.info(f"  Potential controls: {n_controls_raw}")
        logger.info(f"  Excluded (other scoliosis): {n_excluded}")

        # Step 2: Apply ancestry QC
        logger.info("\n[Step 2/5] Applying ancestry QC...")
        ancestry_df = self.ancestry_qc.run_qc(phenotype_file, pc_file)

        if self.config.save_intermediate:
            ancestry_df.to_parquet(self.output_dir / "ancestry_qc_passed.parquet")

        # Filter case labels to ancestry-QC-passed samples
        case_labels = case_labels[
            case_labels["eid"].isin(ancestry_df["eid"])
        ].copy()

        n_cases_qc = (case_labels["label"] == 1).sum()
        n_controls_qc = (case_labels["label"] == 0).sum()
        logger.info(f"  Cases after ancestry QC: {n_cases_qc}")
        logger.info(f"  Potential controls after ancestry QC: {n_controls_qc}")

        # Step 3: Exclude related samples
        if self.config.exclude_related and kinship_file is not None:
            logger.info("\n[Step 3/5] Excluding related samples...")
            # Merge labels with ancestry for exclusion
            merged = case_labels.merge(ancestry_df, on="eid", how="inner")
            merged = exclude_related_samples(
                merged,
                kinship_file,
                threshold=self.config.kinship_threshold,
            )
            case_labels = case_labels[
                case_labels["eid"].isin(merged["eid"])
            ].copy()
            ancestry_df = ancestry_df[
                ancestry_df["eid"].isin(merged["eid"])
            ].copy()
        else:
            logger.info("\n[Step 3/5] Skipping relatedness exclusion (no kinship file)")

        n_cases_unrelated = (case_labels["label"] == 1).sum()
        n_controls_unrelated = (case_labels["label"] == 0).sum()
        logger.info(f"  Cases after relatedness: {n_cases_unrelated}")
        logger.info(f"  Potential controls after relatedness: {n_controls_unrelated}")

        # Step 4: Prepare data for matching
        logger.info("\n[Step 4/5] Preparing data for matching...")

        # Load phenotype data for matching variables
        phenotype_df = self._load_phenotype_data(phenotype_file)

        matching_df = self.control_matcher.prepare_matching_data(
            phenotype_df,
            case_labels,
            ancestry_df,
        )

        # Step 5: Match controls to cases
        logger.info("\n[Step 5/5] Matching controls to cases...")
        matched_cohort, matching_info = self.control_matcher.match_controls(
            matching_df
        )

        # Assess balance
        balance = self.control_matcher.assess_balance(matched_cohort, matching_info)

        # Prepare final outputs
        logger.info("\n" + "=" * 60)
        logger.info("Preparing final outputs...")

        # Get ancestry PCs for matched samples
        matched_ids = set(matched_cohort["eid"])
        ancestry_pcs = ancestry_df[ancestry_df["eid"].isin(matched_ids)].copy()

        # Add additional phenotype columns to cohort
        final_cohort = self._finalize_cohort(matched_cohort, phenotype_df, ancestry_df)

        # Save outputs
        self._save_outputs(final_cohort, matching_info, ancestry_pcs)

        # Generate QC report
        self._generate_qc_report(final_cohort, matching_info, balance)

        logger.info("=" * 60)
        logger.info("Cohort building complete!")
        logger.info("=" * 60)

        return {
            "cohort": final_cohort,
            "matching_info": matching_info,
            "ancestry_pcs": ancestry_pcs,
        }

    def _load_phenotype_data(self, phenotype_file: Path) -> pd.DataFrame:
        """Load phenotype data with age and sex."""
        if phenotype_file.suffix == ".parquet":
            df = pd.read_parquet(phenotype_file)
        else:
            df = pd.read_csv(phenotype_file)

        # Find ID column
        id_col = None
        for col in ["eid", "IID", "FID", "sample_id", "id"]:
            if col in df.columns:
                id_col = col
                break
        if id_col is None:
            id_col = df.columns[0]

        # Standardize column names
        result = pd.DataFrame({"eid": df[id_col]})

        # Find and add age column
        age_col = None
        for col in df.columns:
            if "age" in col.lower() or "21003" in str(col):
                age_col = col
                break
        if age_col:
            result["age"] = df[age_col].values

        # Find and add sex column
        sex_col = None
        for col in df.columns:
            if col.lower() == "sex" or "31-" in str(col) or col == "31":
                sex_col = col
                break
        if sex_col:
            result["sex"] = df[sex_col].values

        return result

    def _finalize_cohort(
        self,
        matched_cohort: pd.DataFrame,
        phenotype_df: pd.DataFrame,
        ancestry_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare final cohort DataFrame with all relevant columns."""
        # Start with matched cohort
        final = matched_cohort[["eid", "label"]].copy()

        # Add is_case column for clarity
        final["is_case"] = final["label"] == 1

        # Add phenotype data
        if "age" in phenotype_df.columns:
            age_map = dict(zip(phenotype_df["eid"], phenotype_df["age"]))
            final["age"] = final["eid"].map(age_map)

        if "sex" in phenotype_df.columns:
            sex_map = dict(zip(phenotype_df["eid"], phenotype_df["sex"]))
            final["sex"] = final["eid"].map(sex_map)

        # Add key PCs for reference
        pc_cols = ["pc1", "pc2", "pc3", "pc4"]
        for pc in pc_cols:
            if pc in ancestry_df.columns:
                pc_map = dict(zip(ancestry_df["eid"], ancestry_df[pc]))
                final[pc] = final["eid"].map(pc_map)

        return final

    def _save_outputs(
        self,
        cohort: pd.DataFrame,
        matching_info: pd.DataFrame,
        ancestry_pcs: pd.DataFrame,
    ) -> None:
        """Save output files."""
        cohort_path = self.output_dir / "cohort.parquet"
        matching_path = self.output_dir / "matching_info.parquet"
        pcs_path = self.output_dir / "ancestry_pcs.parquet"

        cohort.to_parquet(cohort_path, index=False)
        matching_info.to_parquet(matching_path, index=False)
        ancestry_pcs.to_parquet(pcs_path, index=False)

        logger.info(f"Saved cohort to {cohort_path}")
        logger.info(f"Saved matching info to {matching_path}")
        logger.info(f"Saved ancestry PCs to {pcs_path}")

    def _generate_qc_report(
        self,
        cohort: pd.DataFrame,
        matching_info: pd.DataFrame,
        balance: dict,
    ) -> None:
        """Generate and save QC report."""
        import numpy as np

        def to_python(val):
            """Convert numpy types to Python native types."""
            if isinstance(val, (np.integer, np.int64, np.int32)):
                return int(val)
            elif isinstance(val, (np.floating, np.float64, np.float32)):
                return float(val)
            elif isinstance(val, (np.bool_, np.bool)):
                return bool(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            return val

        n_cases = (cohort["label"] == 1).sum()
        n_controls = (cohort["label"] == 0).sum()
        ratio = n_controls / n_cases if n_cases > 0 else 0

        report = {
            "cohort_summary": {
                "total_samples": int(len(cohort)),
                "n_cases": int(n_cases),
                "n_controls": int(n_controls),
                "control_case_ratio": float(round(ratio, 2)),
            },
            "matching_summary": {
                "mean_distance": float(round(matching_info["distance"].mean(), 4)),
                "max_distance": float(round(matching_info["distance"].max(), 4)),
                "min_distance": float(round(matching_info["distance"].min(), 4)),
            },
            "covariate_balance": {},
        }

        # Add balance statistics
        for var, stats in balance.items():
            if var == "overall_balanced":
                report["covariate_balance"]["overall_balanced"] = bool(stats)
            elif isinstance(stats, dict):
                report["covariate_balance"][var] = {
                    "standardized_mean_diff": float(round(to_python(stats["standardized_mean_diff"]), 4)),
                    "balanced": bool(stats["balanced"]),
                }

        # Add demographic summary
        if "age" in cohort.columns:
            cases = cohort[cohort["label"] == 1]
            controls = cohort[cohort["label"] == 0]
            report["demographics"] = {
                "case_mean_age": float(round(cases["age"].mean(), 1)),
                "control_mean_age": float(round(controls["age"].mean(), 1)),
            }

        if "sex" in cohort.columns:
            cases = cohort[cohort["label"] == 1]
            controls = cohort[cohort["label"] == 0]
            report["demographics"]["case_female_pct"] = float(round(
                100 * (cases["sex"] == 0).mean(), 1
            ))
            report["demographics"]["control_female_pct"] = float(round(
                100 * (controls["sex"] == 0).mean(), 1
            ))

        # Save report
        report_path = self.output_dir / "cohort_qc_report.yaml"
        with open(report_path, "w") as f:
            yaml.dump(report, f, default_flow_style=False)

        logger.info(f"Saved QC report to {report_path}")

        # Print summary
        logger.info("\nCohort Summary:")
        logger.info(f"  Total samples: {len(cohort)}")
        logger.info(f"  Cases: {n_cases}")
        logger.info(f"  Controls: {n_controls}")
        logger.info(f"  Ratio: {ratio:.2f}:1")


def build_cohort_from_config(
    config_file: Path,
    phenotype_file: Path,
    pc_file: Optional[Path] = None,
    kinship_file: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """
    Build cohort using configuration file.

    Args:
        config_file: Path to YAML configuration file.
        phenotype_file: Path to phenotype file.
        pc_file: Optional path to genetic PCs file.
        kinship_file: Optional path to kinship file.

    Returns:
        Dictionary with cohort, matching_info, and ancestry_pcs DataFrames.
    """
    config = CohortConfig.from_yaml(config_file)
    builder = CohortBuilder(config)
    return builder.build_cohort(phenotype_file, pc_file, kinship_file)
