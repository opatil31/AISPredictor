"""
Control Matching Module for AIS Cohort Definition

Performs nearest-neighbor matching to select controls for AIS cases.
Matching is performed on age, sex, and genetic principal components
to minimize confounding.

Matching Strategy:
    - 4:1 control-to-case ratio
    - Matching without replacement
    - Variables: age, sex, PC1-PC4
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ControlMatcher:
    """Performs nearest-neighbor matching of controls to cases."""

    DEFAULT_MATCHING_VARS = ["age", "sex", "pc1", "pc2", "pc3", "pc4"]

    def __init__(
        self,
        controls_per_case: int = 4,
        matching_variables: Optional[list[str]] = None,
        replacement: bool = False,
        caliper: Optional[float] = None,
        random_seed: int = 42,
    ):
        """
        Initialize control matcher.

        Args:
            controls_per_case: Number of controls to match per case.
            matching_variables: Variables to match on.
            replacement: Whether to sample with replacement.
            caliper: Maximum allowed distance (in SD units) for matches.
            random_seed: Random seed for reproducibility.
        """
        self.n_controls = controls_per_case
        self.matching_vars = matching_variables or self.DEFAULT_MATCHING_VARS
        self.replacement = replacement
        self.caliper = caliper
        self.random_seed = random_seed

        logger.info(f"Control matcher initialized:")
        logger.info(f"  Controls per case: {self.n_controls}")
        logger.info(f"  Matching variables: {self.matching_vars}")
        logger.info(f"  Replacement: {self.replacement}")
        logger.info(f"  Caliper: {self.caliper}")

    def prepare_matching_data(
        self,
        phenotype_df: pd.DataFrame,
        case_labels: pd.DataFrame,
        ancestry_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare data for matching by merging phenotypes, labels, and ancestry.

        Args:
            phenotype_df: DataFrame with age and sex.
            case_labels: DataFrame with eid and label (1=case, 0=control, -1=excluded).
            ancestry_df: DataFrame with eid and PC columns.

        Returns:
            Merged DataFrame ready for matching.
        """
        # Start with case labels
        df = case_labels[case_labels["label"] >= 0].copy()  # Exclude -1 labels

        # Merge with phenotype data
        if "age" in phenotype_df.columns or "sex" in phenotype_df.columns:
            id_col = self._find_id_column(phenotype_df)
            pheno_cols = ["age", "sex"]
            available_cols = [id_col] + [c for c in pheno_cols if c in phenotype_df.columns]
            df = df.merge(
                phenotype_df[available_cols].rename(columns={id_col: "eid"}),
                on="eid",
                how="left",
            )

        # Merge with ancestry data
        if ancestry_df is not None:
            pc_cols = [col for col in ancestry_df.columns if col.startswith("pc")]
            merge_cols = ["eid"] + pc_cols
            available_merge = [c for c in merge_cols if c in ancestry_df.columns]
            df = df.merge(ancestry_df[available_merge], on="eid", how="left")

        # Log data availability
        available_vars = [v for v in self.matching_vars if v in df.columns]
        missing_vars = [v for v in self.matching_vars if v not in df.columns]

        if missing_vars:
            logger.warning(f"Missing matching variables: {missing_vars}")

        logger.info(f"Available matching variables: {available_vars}")
        logger.info(f"Prepared data for {len(df)} samples")

        return df

    def _find_id_column(self, df: pd.DataFrame) -> str:
        """Find the sample ID column."""
        for col in ["eid", "IID", "FID", "sample_id", "id", "ID"]:
            if col in df.columns:
                return col
        return df.columns[0]

    def _standardize_features(
        self,
        cases_df: pd.DataFrame,
        controls_df: pd.DataFrame,
        variables: list[str],
    ) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Standardize matching features.

        Args:
            cases_df: DataFrame with case samples.
            controls_df: DataFrame with control samples.
            variables: List of variables to standardize.

        Returns:
            Tuple of (case_features, control_features, scaler).
        """
        # Extract features
        case_features = cases_df[variables].values
        control_features = controls_df[variables].values

        # Handle missing values (impute with mean)
        for i, var in enumerate(variables):
            all_values = np.concatenate([case_features[:, i], control_features[:, i]])
            mean_val = np.nanmean(all_values)
            case_features[:, i] = np.where(
                np.isnan(case_features[:, i]), mean_val, case_features[:, i]
            )
            control_features[:, i] = np.where(
                np.isnan(control_features[:, i]), mean_val, control_features[:, i]
            )

        # Standardize using all samples
        scaler = StandardScaler()
        all_features = np.vstack([case_features, control_features])
        scaler.fit(all_features)

        case_features_scaled = scaler.transform(case_features)
        control_features_scaled = scaler.transform(control_features)

        return case_features_scaled, control_features_scaled, scaler

    def match_controls(
        self,
        matching_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform nearest-neighbor matching.

        Args:
            matching_df: DataFrame with labels and matching variables.

        Returns:
            Tuple of (matched_cohort, matching_info).
                - matched_cohort: DataFrame with matched cases and controls
                - matching_info: DataFrame with case-control pair information
        """
        np.random.seed(self.random_seed)

        # Split into cases and controls
        cases_df = matching_df[matching_df["label"] == 1].copy()
        controls_df = matching_df[matching_df["label"] == 0].copy()

        n_cases = len(cases_df)
        n_controls = len(controls_df)

        logger.info(f"Matching {n_cases} cases to {n_controls} potential controls")
        logger.info(f"Target: {self.n_controls} controls per case")

        # Check if we have enough controls
        required_controls = n_cases * self.n_controls
        if not self.replacement and n_controls < required_controls:
            logger.warning(
                f"Insufficient controls ({n_controls}) for {required_controls} "
                f"required matches. Will match {n_controls // n_cases} per case."
            )
            effective_ratio = n_controls // n_cases
        else:
            effective_ratio = self.n_controls

        # Get available matching variables
        available_vars = [v for v in self.matching_vars if v in matching_df.columns]
        if not available_vars:
            raise ValueError("No matching variables available in data")

        # Standardize features
        case_features, control_features, scaler = self._standardize_features(
            cases_df, controls_df, available_vars
        )

        # Compute pairwise distances
        logger.info("Computing pairwise distances...")
        distances = cdist(case_features, control_features, metric="euclidean")

        # Perform matching
        matched_control_indices = []
        matching_records = []

        available_controls = set(range(n_controls))
        case_ids = cases_df["eid"].values
        control_ids = controls_df["eid"].values

        for i, case_id in enumerate(case_ids):
            if not available_controls:
                logger.warning(f"Ran out of controls at case {i}/{n_cases}")
                break

            # Get distances to available controls
            available_list = sorted(list(available_controls))
            case_distances = distances[i, available_list]

            # Find nearest controls
            n_to_match = min(effective_ratio, len(available_list))
            nearest_indices = np.argsort(case_distances)[:n_to_match]

            # Apply caliper if specified
            if self.caliper is not None:
                nearest_indices = [
                    idx for idx in nearest_indices
                    if case_distances[idx] <= self.caliper
                ]

            # Record matches
            for rank, local_idx in enumerate(nearest_indices):
                global_idx = available_list[local_idx]
                control_id = control_ids[global_idx]
                distance = case_distances[local_idx]

                matched_control_indices.append(global_idx)
                matching_records.append({
                    "case_eid": case_id,
                    "control_eid": control_id,
                    "match_rank": rank + 1,
                    "distance": distance,
                })

                # Remove from available if not using replacement
                if not self.replacement:
                    available_controls.discard(global_idx)

        # Create matching info DataFrame
        matching_info = pd.DataFrame(matching_records)

        # Create matched cohort
        matched_control_ids = set(matching_info["control_eid"])
        matched_cases = cases_df.copy()
        matched_controls = controls_df[
            controls_df["eid"].isin(matched_control_ids)
        ].copy()

        matched_cohort = pd.concat([matched_cases, matched_controls], ignore_index=True)

        # Summary statistics
        n_matched_cases = len(matched_cases)
        n_matched_controls = len(matched_controls)
        actual_ratio = n_matched_controls / n_matched_cases if n_matched_cases > 0 else 0

        logger.info(f"Matching complete:")
        logger.info(f"  Matched cases: {n_matched_cases}")
        logger.info(f"  Matched controls: {n_matched_controls}")
        logger.info(f"  Actual ratio: {actual_ratio:.2f}:1")
        logger.info(f"  Mean distance: {matching_info['distance'].mean():.3f}")
        logger.info(f"  Max distance: {matching_info['distance'].max():.3f}")

        return matched_cohort, matching_info

    def assess_balance(
        self,
        matched_cohort: pd.DataFrame,
        matching_info: pd.DataFrame,
    ) -> dict:
        """
        Assess covariate balance after matching.

        Args:
            matched_cohort: DataFrame with matched samples.
            matching_info: DataFrame with matching pair info.

        Returns:
            Dictionary with balance statistics.
        """
        cases = matched_cohort[matched_cohort["label"] == 1]
        controls = matched_cohort[matched_cohort["label"] == 0]

        balance = {}

        for var in self.matching_vars:
            if var not in matched_cohort.columns:
                continue

            case_mean = cases[var].mean()
            control_mean = controls[var].mean()
            case_std = cases[var].std()
            control_std = controls[var].std()

            # Standardized mean difference
            pooled_std = np.sqrt((case_std**2 + control_std**2) / 2)
            smd = (case_mean - control_mean) / pooled_std if pooled_std > 0 else 0

            balance[var] = {
                "case_mean": case_mean,
                "control_mean": control_mean,
                "case_std": case_std,
                "control_std": control_std,
                "standardized_mean_diff": smd,
                "balanced": abs(smd) < 0.1,  # Common threshold
            }

        # Overall balance assessment
        all_balanced = all(b["balanced"] for b in balance.values())
        balance["overall_balanced"] = all_balanced

        # Log balance summary
        logger.info("Covariate balance assessment:")
        for var, stats in balance.items():
            if var == "overall_balanced":
                continue
            status = "OK" if stats["balanced"] else "IMBALANCED"
            logger.info(f"  {var}: SMD = {stats['standardized_mean_diff']:.3f} [{status}]")

        return balance


def exclude_related_samples(
    cohort_df: pd.DataFrame,
    kinship_file: Path,
    threshold: float = 0.0884,
) -> pd.DataFrame:
    """
    Exclude related samples based on KING kinship coefficients.

    For related pairs, preferentially removes controls over cases.

    Args:
        cohort_df: DataFrame with eid and label columns.
        kinship_file: Path to KING kinship file.
        threshold: Kinship threshold (0.0884 = 2nd degree relatives).

    Returns:
        DataFrame with related samples removed.
    """
    logger.info(f"Excluding related samples (kinship > {threshold})...")

    # Load kinship data
    if kinship_file.suffix == ".parquet":
        kinship = pd.read_parquet(kinship_file)
    else:
        kinship = pd.read_csv(kinship_file, sep="\t")

    # Identify kinship columns
    id_cols = [col for col in kinship.columns if "id" in col.lower()]
    kinship_col = [col for col in kinship.columns if "kinship" in col.lower() or "king" in col.lower()]

    if len(id_cols) < 2 or len(kinship_col) < 1:
        logger.warning("Could not identify kinship file columns, skipping exclusion")
        return cohort_df

    id1_col, id2_col = id_cols[:2]
    kinship_val_col = kinship_col[0]

    # Filter to related pairs
    related = kinship[kinship[kinship_val_col] > threshold]
    logger.info(f"Found {len(related)} related pairs above threshold")

    # Get samples in cohort
    cohort_ids = set(cohort_df["eid"])
    label_map = dict(zip(cohort_df["eid"], cohort_df["label"]))

    # Identify samples to remove
    to_remove = set()
    for _, row in related.iterrows():
        id1 = row[id1_col]
        id2 = row[id2_col]

        if id1 not in cohort_ids or id2 not in cohort_ids:
            continue

        if id1 in to_remove or id2 in to_remove:
            continue

        # Prefer to remove controls over cases
        label1 = label_map.get(id1, 0)
        label2 = label_map.get(id2, 0)

        if label1 == 1 and label2 == 0:
            to_remove.add(id2)  # Remove control
        elif label1 == 0 and label2 == 1:
            to_remove.add(id1)  # Remove control
        else:
            # Both same type, remove one randomly
            to_remove.add(id1)

    result = cohort_df[~cohort_df["eid"].isin(to_remove)].copy()

    logger.info(f"Removed {len(to_remove)} related samples")
    logger.info(f"Remaining: {len(result)} samples")

    return result


def match_controls(
    cases_df: pd.DataFrame,
    controls_df: pd.DataFrame,
    controls_per_case: int = 4,
    matching_variables: Optional[list[str]] = None,
    output_file: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for control matching.

    Args:
        cases_df: DataFrame with case samples.
        controls_df: DataFrame with potential control samples.
        controls_per_case: Number of controls per case.
        matching_variables: Variables to match on.
        output_file: Optional path to save matched cohort.

    Returns:
        Tuple of (matched_cohort, matching_info).
    """
    # Combine into single DataFrame with labels
    cases_df = cases_df.copy()
    controls_df = controls_df.copy()
    cases_df["label"] = 1
    controls_df["label"] = 0

    combined = pd.concat([cases_df, controls_df], ignore_index=True)

    matcher = ControlMatcher(
        controls_per_case=controls_per_case,
        matching_variables=matching_variables,
    )

    matched_cohort, matching_info = matcher.match_controls(combined)

    if output_file:
        output_path = Path(output_file)
        if output_path.suffix == ".parquet":
            matched_cohort.to_parquet(output_path, index=False)
        else:
            matched_cohort.to_csv(output_path, index=False)
        logger.info(f"Saved matched cohort to {output_path}")

    return matched_cohort, matching_info
