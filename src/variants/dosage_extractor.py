"""
Dosage extraction from PLINK2 format files (.pgen/.pvar/.psam).

Uses pgenlib for efficient reading of genotype dosages.
"""

import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DosageExtractor:
    """
    Extract genotype dosages from PLINK2 format files.

    Supports filtering to specific samples (e.g., cohort members only).
    """

    def __init__(self, pfile_prefix: str):
        """
        Initialize DosageExtractor.

        Args:
            pfile_prefix: Path prefix for .pgen/.pvar/.psam files
                         (e.g., "data/chr6" for chr6.pgen, chr6.pvar, chr6.psam)
        """
        self.prefix = Path(pfile_prefix)
        self.pgen_path = self.prefix.with_suffix('.pgen')
        self.pvar_path = self.prefix.with_suffix('.pvar')
        self.psam_path = self.prefix.with_suffix('.psam')

        # Verify files exist
        for path in [self.pgen_path, self.pvar_path, self.psam_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

        self.sample_df: Optional[pd.DataFrame] = None
        self.n_samples: int = 0
        self.n_variants: int = 0
        self._pgen_reader = None

    def load_samples(self) -> pd.DataFrame:
        """Load sample information from .psam file."""
        logger.info(f"Loading samples from {self.psam_path}")

        # .psam format: #FID IID SEX (or #IID for PLINK2 single-ID mode)
        with open(self.psam_path, 'r') as f:
            first_line = f.readline().strip()

        if first_line.startswith('#IID'):
            # Single-ID mode
            self.sample_df = pd.read_csv(
                self.psam_path,
                sep='\t',
                dtype={'#IID': str}
            )
            self.sample_df = self.sample_df.rename(columns={'#IID': 'IID'})
        elif first_line.startswith('#FID'):
            # Dual-ID mode
            self.sample_df = pd.read_csv(
                self.psam_path,
                sep='\t',
                dtype={'#FID': str, 'IID': str}
            )
            self.sample_df = self.sample_df.rename(columns={'#FID': 'FID'})
        else:
            # No header - assume FID IID format
            self.sample_df = pd.read_csv(
                self.psam_path,
                sep='\t',
                header=None,
                names=['FID', 'IID'],
                dtype={'FID': str, 'IID': str}
            )

        self.n_samples = len(self.sample_df)
        logger.info(f"Loaded {self.n_samples:,} samples")

        return self.sample_df

    def get_sample_indices(self, sample_ids: List[str]) -> np.ndarray:
        """
        Get indices of specified samples in the .pgen file.

        Args:
            sample_ids: List of sample IDs to find

        Returns:
            Array of indices into the .pgen file
        """
        if self.sample_df is None:
            self.load_samples()

        # Match by IID
        sample_set = set(str(s) for s in sample_ids)

        # Create ID column for matching (prefer IID, fall back to FID_IID)
        if 'IID' in self.sample_df.columns:
            self.sample_df['match_id'] = self.sample_df['IID'].astype(str)
        else:
            self.sample_df['match_id'] = self.sample_df.iloc[:, 0].astype(str)

        # Find matching indices
        mask = self.sample_df['match_id'].isin(sample_set)
        indices = np.where(mask)[0]

        n_found = len(indices)
        n_requested = len(sample_ids)
        logger.info(f"Found {n_found:,}/{n_requested:,} requested samples in .pgen file")

        if n_found < n_requested:
            found_ids = set(self.sample_df.loc[mask, 'match_id'])
            missing = sample_set - found_ids
            logger.warning(f"Missing {len(missing)} samples. First 5: {list(missing)[:5]}")

        return indices

    def extract_dosages(
        self,
        sample_indices: Optional[np.ndarray] = None,
        variant_indices: Optional[np.ndarray] = None,
        batch_size: int = 10000
    ) -> np.ndarray:
        """
        Extract dosages from .pgen file.

        Args:
            sample_indices: Indices of samples to extract (None = all)
            variant_indices: Indices of variants to extract (None = all)
            batch_size: Number of variants to read at once

        Returns:
            Dosage matrix of shape (n_samples, n_variants)
        """
        try:
            import pgenlib
        except ImportError:
            raise ImportError(
                "pgenlib is required for reading .pgen files. "
                "Install with: pip install pgenlib"
            )

        logger.info(f"Opening {self.pgen_path}")

        # Open pgen file
        pgen = pgenlib.PgenReader(bytes(str(self.pgen_path), encoding='utf-8'))

        n_samples_file = pgen.get_raw_sample_ct()
        n_variants_file = pgen.get_variant_ct()
        logger.info(f"File contains {n_samples_file:,} samples, {n_variants_file:,} variants")

        # Determine samples to read
        if sample_indices is None:
            sample_indices = np.arange(n_samples_file)
        n_samples = len(sample_indices)

        # Determine variants to read
        if variant_indices is None:
            variant_indices = np.arange(n_variants_file)
        n_variants = len(variant_indices)

        logger.info(f"Extracting {n_samples:,} samples x {n_variants:,} variants")

        # Allocate output array
        dosages = np.zeros((n_samples, n_variants), dtype=np.float32)

        # Read in batches
        n_batches = (n_variants + batch_size - 1) // batch_size
        logger.info(f"Reading in {n_batches} batches of {batch_size} variants")

        # Buffer for reading
        buf = np.zeros(n_samples_file, dtype=np.float64)

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_variants)

            if batch_idx % 10 == 0:
                logger.info(f"Processing batch {batch_idx + 1}/{n_batches}")

            for i, var_idx in enumerate(variant_indices[start:end]):
                # Read dosages for this variant
                pgen.read_dosages(var_idx, buf)

                # Extract only requested samples
                dosages[:, start + i] = buf[sample_indices]

        pgen.close()
        logger.info(f"Extraction complete. Dosage matrix shape: {dosages.shape}")

        return dosages

    def extract_for_cohort(
        self,
        cohort_ids: List[str],
        batch_size: int = 10000
    ) -> tuple[np.ndarray, List[str]]:
        """
        Extract dosages for a specific cohort.

        Args:
            cohort_ids: List of sample IDs in the cohort
            batch_size: Number of variants per batch

        Returns:
            Tuple of (dosage matrix, ordered sample IDs that were found)
        """
        if self.sample_df is None:
            self.load_samples()

        # Get indices of cohort samples
        sample_indices = self.get_sample_indices(cohort_ids)

        # Get the IDs in file order
        if 'IID' in self.sample_df.columns:
            id_col = 'IID'
        else:
            id_col = self.sample_df.columns[0]

        ordered_ids = self.sample_df.iloc[sample_indices][id_col].astype(str).tolist()

        # Extract dosages
        dosages = self.extract_dosages(sample_indices=sample_indices, batch_size=batch_size)

        return dosages, ordered_ids
