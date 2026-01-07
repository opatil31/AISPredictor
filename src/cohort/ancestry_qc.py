"""
Ancestry Quality Control Module for AIS Cohort Definition

Filters UK Biobank samples to European ancestry and removes
PCA-based outliers to ensure genetic homogeneity.

Filters applied:
    1. Self-reported European ancestry (field 21000)
    2. Optional: Genetic ethnic grouping (field 22006)
    3. PCA outlier removal (>6 SD from mean on PC1-PC4)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class AncestryQC:
    """Performs ancestry-based quality control for UK Biobank samples."""

    # Default European ancestry codes for field 21000
    DEFAULT_EUROPEAN_CODES = [1, 1001, 1002, 1003]
    # 1 = White, 1001 = British, 1002 = Irish, 1003 = Any other white

    # Default genetic European code for field 22006
    DEFAULT_GENETIC_EUROPEAN_CODE = 1  # Caucasian

    def __init__(
        self,
        european_codes: Optional[list[int]] = None,
        genetic_european_code: int = 1,
        n_pcs_for_outlier: int = 4,
        sd_threshold: float = 6.0,
        use_genetic_ethnicity: bool = True,
    ):
        """
        Initialize ancestry QC.

        Args:
            european_codes: Self-reported European ancestry codes (field 21000).
            genetic_european_code: Genetic ethnicity code for European (field 22006).
            n_pcs_for_outlier: Number of PCs to use for outlier detection.
            sd_threshold: SD threshold for outlier removal.
            use_genetic_ethnicity: Whether to also filter on genetic ethnicity.
        """
        self.european_codes = european_codes or self.DEFAULT_EUROPEAN_CODES
        self.genetic_european_code = genetic_european_code
        self.n_pcs = n_pcs_for_outlier
        self.sd_threshold = sd_threshold
        self.use_genetic_ethnicity = use_genetic_ethnicity

        logger.info(f"Ancestry QC initialized:")
        logger.info(f"  European codes (field 21000): {self.european_codes}")
        logger.info(f"  PCA outlier detection: {self.n_pcs} PCs, {self.sd_threshold} SD")
        logger.info(f"  Use genetic ethnicity: {self.use_genetic_ethnicity}")

    def load_ancestry_data(
        self,
        phenotype_file: Path,
        pc_file: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Load ancestry-related fields from UK Biobank.

        Args:
            phenotype_file: Path to phenotype file with ethnicity fields.
            pc_file: Optional separate file with genetic PCs.

        Returns:
            DataFrame with eid, ethnicity, genetic_ethnicity, and PC columns.
        """
        file_path = Path(phenotype_file)

        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)

        # Find ID column
        id_col = self._find_id_column(df)
        result = df[[id_col]].copy()
        result = result.rename(columns={id_col: "eid"})

        # Find ethnicity column (field 21000)
        eth_col = self._find_field_column(df, "21000", ["ethnicity", "ethnic"])
        if eth_col:
            result["ethnicity"] = df[eth_col].values
        else:
            logger.warning("Ethnicity field (21000) not found")
            result["ethnicity"] = np.nan

        # Find genetic ethnicity column (field 22006)
        gen_eth_col = self._find_field_column(
            df, "22006", ["genetic_ethnic", "genetic_ethnicity"]
        )
        if gen_eth_col:
            result["genetic_ethnicity"] = df[gen_eth_col].values
        else:
            logger.warning("Genetic ethnicity field (22006) not found")
            result["genetic_ethnicity"] = np.nan

        # Load PCs (field 22009)
        if pc_file:
            pc_df = self._load_pcs(pc_file)
            result = result.merge(pc_df, on="eid", how="left")
        else:
            # Try to find PC columns in the same file
            for i in range(1, 41):  # UK Biobank provides 40 PCs
                pc_col = self._find_field_column(
                    df, f"22009-0.{i}", [f"pc{i}", f"PC{i}"]
                )
                if pc_col:
                    result[f"pc{i}"] = df[pc_col].values

        # Check if we have the required PCs
        pc_cols = [f"pc{i}" for i in range(1, self.n_pcs + 1)]
        missing_pcs = [col for col in pc_cols if col not in result.columns]
        if missing_pcs:
            logger.warning(f"Missing PC columns: {missing_pcs}")

        logger.info(f"Loaded ancestry data for {len(result)} samples")

        return result

    def _find_id_column(self, df: pd.DataFrame) -> str:
        """Find the sample ID column."""
        for col in ["eid", "IID", "FID", "sample_id", "id", "ID"]:
            if col in df.columns:
                return col
        return df.columns[0]

    def _find_field_column(
        self,
        df: pd.DataFrame,
        field_id: str,
        alt_names: list[str],
    ) -> Optional[str]:
        """Find a UK Biobank field column by ID or alternative names."""
        # Try exact field ID match
        if field_id in df.columns:
            return field_id

        # Try field ID with prefix (p21000, 21000-0.0, etc.)
        for col in df.columns:
            if field_id in col:
                return col

        # Try alternative names
        for name in alt_names:
            for col in df.columns:
                if name.lower() in col.lower():
                    return col

        return None

    def _load_pcs(self, pc_file: Path) -> pd.DataFrame:
        """Load genetic PCs from a separate file."""
        if pc_file.suffix == ".parquet":
            df = pd.read_parquet(pc_file)
        else:
            df = pd.read_csv(pc_file)

        id_col = self._find_id_column(df)
        result = df.rename(columns={id_col: "eid"})

        # Standardize PC column names
        rename_map = {}
        for col in result.columns:
            if col == "eid":
                continue

            # Try UK Biobank format: 22009-0.1, 22009-0.2, etc.
            if "22009" in col:
                import re
                match = re.search(r"22009[.-]0?\.?(\d+)", col)
                if match:
                    pc_num = int(match.group(1))
                    rename_map[col] = f"pc{pc_num}"
                    continue

            # Try standard PC naming: pc1, PC1, pc_1, etc.
            col_lower = col.lower()
            for i in range(1, 41):
                if f"pc{i}" == col_lower or f"pc_{i}" == col_lower:
                    rename_map[col] = f"pc{i}"
                    break

        if rename_map:
            result = result.rename(columns=rename_map)
            logger.info(f"Renamed {len(rename_map)} PC columns")

        return result

    def filter_european(self, ancestry_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to self-reported European ancestry.

        Args:
            ancestry_df: DataFrame with ethnicity column.

        Returns:
            Filtered DataFrame with only European samples.
        """
        n_before = len(ancestry_df)

        # Filter by self-reported ethnicity
        mask = ancestry_df["ethnicity"].isin(self.european_codes)
        result = ancestry_df[mask].copy()

        n_after = len(result)
        logger.info(
            f"European ancestry filter: {n_before} -> {n_after} "
            f"({n_before - n_after} removed)"
        )

        # Optionally also filter by genetic ethnicity
        if self.use_genetic_ethnicity and "genetic_ethnicity" in result.columns:
            n_before_gen = len(result)
            gen_mask = result["genetic_ethnicity"] == self.genetic_european_code
            # Allow missing genetic ethnicity
            gen_mask = gen_mask | result["genetic_ethnicity"].isna()
            result = result[gen_mask].copy()
            n_after_gen = len(result)
            logger.info(
                f"Genetic ethnicity filter: {n_before_gen} -> {n_after_gen} "
                f"({n_before_gen - n_after_gen} removed)"
            )

        return result

    def remove_pca_outliers(self, ancestry_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove PCA-based ancestry outliers.

        Samples with any of PC1-PC{n_pcs} more than {sd_threshold} SD
        from the mean are removed.

        Args:
            ancestry_df: DataFrame with PC columns.

        Returns:
            DataFrame with outliers removed.
        """
        n_before = len(ancestry_df)

        pc_cols = [f"pc{i}" for i in range(1, self.n_pcs + 1)]
        available_pcs = [col for col in pc_cols if col in ancestry_df.columns]

        if not available_pcs:
            logger.warning("No PC columns available for outlier removal")
            return ancestry_df

        logger.info(f"Using PCs for outlier detection: {available_pcs}")

        # Calculate mean and SD for each PC
        outlier_mask = pd.Series(False, index=ancestry_df.index)

        for pc in available_pcs:
            pc_values = ancestry_df[pc].dropna()
            mean = pc_values.mean()
            std = pc_values.std()

            # Flag samples outside threshold
            pc_outliers = (
                (ancestry_df[pc] - mean).abs() > self.sd_threshold * std
            ) & ancestry_df[pc].notna()

            n_outliers = pc_outliers.sum()
            if n_outliers > 0:
                logger.info(f"  {pc}: {n_outliers} outliers (>{self.sd_threshold} SD)")

            outlier_mask = outlier_mask | pc_outliers

        # Remove outliers
        result = ancestry_df[~outlier_mask].copy()
        n_after = len(result)

        logger.info(
            f"PCA outlier removal: {n_before} -> {n_after} "
            f"({n_before - n_after} removed)"
        )

        return result

    def run_qc(
        self,
        phenotype_file: Path,
        pc_file: Optional[Path] = None,
        sample_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Run full ancestry QC pipeline.

        Args:
            phenotype_file: Path to phenotype file.
            pc_file: Optional path to PC file.
            sample_ids: Optional list of sample IDs to consider.

        Returns:
            DataFrame with samples passing QC, including:
                - eid: Sample ID
                - ethnicity: Self-reported ethnicity code
                - pc1-pc{n}: Genetic principal components
                - passed_ancestry_qc: True for all (retained samples)
        """
        logger.info("Running ancestry QC pipeline...")

        # Load data
        ancestry_df = self.load_ancestry_data(phenotype_file, pc_file)

        # Filter to specified samples if provided
        if sample_ids is not None:
            ancestry_df = ancestry_df[ancestry_df["eid"].isin(sample_ids)]
            logger.info(f"Filtered to {len(ancestry_df)} specified samples")

        n_initial = len(ancestry_df)

        # Step 1: Filter to European ancestry
        ancestry_df = self.filter_european(ancestry_df)

        # Step 2: Remove PCA outliers
        ancestry_df = self.remove_pca_outliers(ancestry_df)

        # Add QC flag
        ancestry_df["passed_ancestry_qc"] = True

        n_final = len(ancestry_df)
        logger.info(
            f"Ancestry QC complete: {n_initial} -> {n_final} "
            f"({100 * n_final / n_initial:.1f}% retained)"
        )

        return ancestry_df

    def get_qc_summary(self, ancestry_df: pd.DataFrame) -> dict:
        """
        Generate summary statistics for ancestry QC.

        Args:
            ancestry_df: DataFrame after QC.

        Returns:
            Dictionary with QC statistics.
        """
        pc_cols = [col for col in ancestry_df.columns if col.startswith("pc")]

        summary = {
            "n_samples": len(ancestry_df),
            "ethnicity_distribution": ancestry_df["ethnicity"].value_counts().to_dict(),
        }

        # PC statistics
        for pc in pc_cols[:self.n_pcs]:
            if pc in ancestry_df.columns:
                summary[f"{pc}_mean"] = ancestry_df[pc].mean()
                summary[f"{pc}_std"] = ancestry_df[pc].std()

        return summary


def run_ancestry_qc(
    phenotype_file: str | Path,
    pc_file: Optional[str | Path] = None,
    output_file: Optional[str | Path] = None,
    european_codes: Optional[list[int]] = None,
    n_pcs: int = 4,
    sd_threshold: float = 6.0,
) -> pd.DataFrame:
    """
    Convenience function to run ancestry QC.

    Args:
        phenotype_file: Path to UK Biobank phenotype file.
        pc_file: Optional path to genetic PCs file.
        output_file: Optional path to save results.
        european_codes: Self-reported European ancestry codes.
        n_pcs: Number of PCs for outlier detection.
        sd_threshold: SD threshold for outliers.

    Returns:
        DataFrame with samples passing ancestry QC.
    """
    qc = AncestryQC(
        european_codes=european_codes,
        n_pcs_for_outlier=n_pcs,
        sd_threshold=sd_threshold,
    )

    result = qc.run_qc(
        Path(phenotype_file),
        Path(pc_file) if pc_file else None,
    )

    if output_file:
        output_path = Path(output_file)
        if output_path.suffix == ".parquet":
            result.to_parquet(output_path, index=False)
        else:
            result.to_csv(output_path, index=False)
        logger.info(f"Saved ancestry QC results to {output_path}")

    return result
