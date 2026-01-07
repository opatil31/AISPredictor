"""
Case Identification Module for AIS Cohort Definition

Identifies Adolescent Idiopathic Scoliosis (AIS) cases from UK Biobank
hospital inpatient records using ICD-10 diagnosis codes.

ICD-10 Codes:
    - M41.1: Juvenile idiopathic scoliosis
    - M41.2: Other idiopathic scoliosis
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CaseIdentifier:
    """Identifies AIS cases and excludes scoliosis from control pool."""

    # Default ICD-10 codes for AIS cases
    DEFAULT_CASE_CODES = ["M41.1", "M41.2"]

    # All scoliosis codes for control exclusion
    DEFAULT_EXCLUSION_CODES = [
        "M41", "M41.0", "M41.1", "M41.2", "M41.3",
        "M41.4", "M41.5", "M41.8", "M41.9"
    ]

    def __init__(
        self,
        case_icd10_codes: Optional[list[str]] = None,
        exclusion_icd10_codes: Optional[list[str]] = None,
        diagnosis_field: str = "41270",
    ):
        """
        Initialize case identifier.

        Args:
            case_icd10_codes: ICD-10 codes defining AIS cases.
            exclusion_icd10_codes: ICD-10 codes to exclude from controls.
            diagnosis_field: UK Biobank field ID for ICD-10 diagnoses.
        """
        self.case_codes = case_icd10_codes or self.DEFAULT_CASE_CODES
        self.exclusion_codes = exclusion_icd10_codes or self.DEFAULT_EXCLUSION_CODES
        self.diagnosis_field = diagnosis_field

        logger.info(f"Case identification codes: {self.case_codes}")
        logger.info(f"Control exclusion codes: {self.exclusion_codes}")

    def _load_diagnoses(
        self,
        phenotype_file: Path,
        sample_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Load ICD-10 diagnoses from UK Biobank phenotype file.

        UK Biobank field 41270 contains ICD-10 diagnoses from hospital
        inpatient records. The field has multiple instances (array columns)
        as patients may have multiple diagnoses.

        Args:
            phenotype_file: Path to UK Biobank phenotype file (.csv or .parquet).
            sample_ids: Optional list of sample IDs to filter to.

        Returns:
            DataFrame with columns: eid, diagnoses (list of ICD-10 codes)
        """
        file_path = Path(phenotype_file)

        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Identify diagnosis columns (field 41270 with array indices)
        diag_cols = [
            col for col in df.columns
            if col.startswith(f"{self.diagnosis_field}-")
            or col == self.diagnosis_field
            or col.startswith("p41270")  # Alternative naming convention
        ]

        if not diag_cols:
            # Try to find columns containing ICD codes directly
            logger.warning(
                f"No columns found for field {self.diagnosis_field}. "
                "Looking for columns containing 'icd' or 'diag'."
            )
            diag_cols = [
                col for col in df.columns
                if "icd" in col.lower() or "diag" in col.lower()
            ]

        if not diag_cols:
            raise ValueError(
                f"Could not find diagnosis columns in {phenotype_file}. "
                f"Expected columns starting with '{self.diagnosis_field}-'"
            )

        logger.info(f"Found {len(diag_cols)} diagnosis columns")

        # Ensure we have an ID column
        id_col = self._find_id_column(df)

        # Extract relevant columns
        df = df[[id_col] + diag_cols].copy()
        df = df.rename(columns={id_col: "eid"})

        # Filter to specific samples if provided
        if sample_ids is not None:
            df = df[df["eid"].isin(sample_ids)]

        # Combine all diagnosis columns into a single list per patient
        def collect_diagnoses(row):
            diagnoses = []
            for col in diag_cols:
                val = row[col]
                if pd.notna(val) and val != "":
                    diagnoses.append(str(val).strip())
            return diagnoses

        df["diagnoses"] = df.apply(collect_diagnoses, axis=1)
        df = df[["eid", "diagnoses"]]

        logger.info(f"Loaded diagnoses for {len(df)} samples")

        return df

    def _find_id_column(self, df: pd.DataFrame) -> str:
        """Find the sample ID column in the dataframe."""
        possible_id_cols = ["eid", "IID", "FID", "sample_id", "id", "ID"]
        for col in possible_id_cols:
            if col in df.columns:
                return col
        # Use first column if no standard ID found
        return df.columns[0]

    def _has_diagnosis(self, diagnoses: list[str], codes: list[str]) -> bool:
        """Check if any diagnosis matches the given codes."""
        for diag in diagnoses:
            for code in codes:
                # Match exact code or prefix (e.g., "M41" matches "M41.1")
                if diag == code or diag.startswith(code + ".") or diag.startswith(code):
                    return True
        return False

    def identify_cases(
        self,
        phenotype_file: Path,
        sample_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Identify AIS cases from UK Biobank diagnoses.

        Args:
            phenotype_file: Path to phenotype file with ICD-10 diagnoses.
            sample_ids: Optional list of sample IDs to consider.

        Returns:
            DataFrame with columns:
                - eid: Sample ID
                - is_case: True for AIS cases
                - has_any_scoliosis: True for any scoliosis diagnosis
                - case_codes_matched: List of matched AIS codes
        """
        logger.info("Identifying AIS cases from ICD-10 diagnoses...")

        # Load diagnoses
        diag_df = self._load_diagnoses(phenotype_file, sample_ids)

        # Identify cases (have AIS diagnosis)
        diag_df["is_case"] = diag_df["diagnoses"].apply(
            lambda x: self._has_diagnosis(x, self.case_codes)
        )

        # Identify any scoliosis (for control exclusion)
        diag_df["has_any_scoliosis"] = diag_df["diagnoses"].apply(
            lambda x: self._has_diagnosis(x, self.exclusion_codes)
        )

        # Record which case codes matched
        def get_matched_codes(diagnoses):
            matched = []
            for diag in diagnoses:
                for code in self.case_codes:
                    if diag == code or diag.startswith(code):
                        matched.append(diag)
            return matched

        diag_df["case_codes_matched"] = diag_df["diagnoses"].apply(get_matched_codes)

        # Summary statistics
        n_cases = diag_df["is_case"].sum()
        n_any_scoliosis = diag_df["has_any_scoliosis"].sum()

        logger.info(f"Identified {n_cases} AIS cases")
        logger.info(f"Identified {n_any_scoliosis} samples with any scoliosis diagnosis")

        # Detailed breakdown by code
        for code in self.case_codes:
            n_with_code = diag_df["diagnoses"].apply(
                lambda x: self._has_diagnosis(x, [code])
            ).sum()
            logger.info(f"  {code}: {n_with_code} samples")

        return diag_df[["eid", "is_case", "has_any_scoliosis", "case_codes_matched"]]

    def get_case_control_labels(
        self,
        phenotype_file: Path,
        sample_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Get case/control labels for cohort construction.

        Cases: Have M41.1 or M41.2 diagnosis
        Potential Controls: Do not have any M41.x diagnosis
        Excluded: Have other scoliosis (M41.0, M41.3-M41.9) but not AIS

        Args:
            phenotype_file: Path to phenotype file.
            sample_ids: Optional sample ID filter.

        Returns:
            DataFrame with eid and label columns:
                - label: 1 = case, 0 = potential control, -1 = excluded
        """
        case_df = self.identify_cases(phenotype_file, sample_ids)

        # Assign labels
        def assign_label(row):
            if row["is_case"]:
                return 1  # Case
            elif row["has_any_scoliosis"]:
                return -1  # Excluded (has scoliosis but not AIS)
            else:
                return 0  # Potential control

        case_df["label"] = case_df.apply(assign_label, axis=1)

        n_cases = (case_df["label"] == 1).sum()
        n_controls = (case_df["label"] == 0).sum()
        n_excluded = (case_df["label"] == -1).sum()

        logger.info(
            f"Labels assigned: {n_cases} cases, {n_controls} potential controls, "
            f"{n_excluded} excluded"
        )

        return case_df[["eid", "label", "case_codes_matched"]]


def identify_cases_from_file(
    phenotype_file: str | Path,
    case_codes: Optional[list[str]] = None,
    exclusion_codes: Optional[list[str]] = None,
    output_file: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Convenience function to identify cases from a phenotype file.

    Args:
        phenotype_file: Path to UK Biobank phenotype file.
        case_codes: ICD-10 codes for cases (default: M41.1, M41.2).
        exclusion_codes: ICD-10 codes to exclude from controls.
        output_file: Optional path to save results.

    Returns:
        DataFrame with case/control labels.
    """
    identifier = CaseIdentifier(
        case_icd10_codes=case_codes,
        exclusion_icd10_codes=exclusion_codes,
    )

    result = identifier.get_case_control_labels(Path(phenotype_file))

    if output_file:
        output_path = Path(output_file)
        if output_path.suffix == ".parquet":
            result.to_parquet(output_path, index=False)
        else:
            result.to_csv(output_path, index=False)
        logger.info(f"Saved case labels to {output_path}")

    return result
