"""
I/O module for reading and writing braidz files.

Braidz files are ZIP archives containing trajectory data from the Strand-Braid
tracking system. They typically contain:
    - kalman_estimates.csv.gz: Main trajectory data with position and velocity
    - opto.csv: Optogenetic stimulation events (optional)
    - stim.csv: Visual stimulus events (optional)
"""

import gzip
import io
import logging
import os
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import polars as pl
from tqdm import tqdm

# =============================================================================
# Logging Configuration
# =============================================================================

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

logger = logging.getLogger(__name__)


# =============================================================================
# Optional Dependencies
# =============================================================================

PYARROW_AVAILABLE = find_spec("pyarrow") is not None
if PYARROW_AVAILABLE:
    import pyarrow as pa
    import pyarrow.csv as pv


# =============================================================================
# Custom Exceptions
# =============================================================================


class EmptyKalmanError(Exception):
    """Raised when kalman_estimates.csv.gz is empty or missing."""

    pass


class InvalidBraidzError(Exception):
    """Raised when a braidz file is invalid or corrupted."""

    pass


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class CSVFiles:
    """Standard CSV filenames within braidz archives."""

    KALMAN: str = "kalman_estimates.csv.gz"
    OPTO: str = "opto.csv"
    STIM: str = "stim.csv"


@dataclass
class BraidzData:
    """
    Container for data loaded from braidz file(s).

    Attributes:
        trajectories: DataFrame containing Kalman filter estimates with columns:
            - obj_id: Unique object identifier
            - frame: Frame number
            - timestamp: Time in seconds
            - x, y, z: Position in meters
            - xvel, yvel, zvel: Velocity in m/s
            - exp_num: Experiment number (when loading multiple files)

        opto: DataFrame of optogenetic events, or None if not present.
            Typically contains: obj_id, frame, timestamp, intensity, duration, sham

        stim: DataFrame of stimulus events, or None if not present.
            Typically contains: obj_id, frame, timestamp, stimulus parameters

        other_csvs: Dictionary of any additional CSV files found in the archive.

        source_files: List of source file paths that were loaded.
    """

    trajectories: pl.DataFrame
    opto: Optional[pl.DataFrame] = None
    stim: Optional[pl.DataFrame] = None
    other_csvs: Optional[Dict[str, pl.DataFrame]] = None
    source_files: Optional[List[str]] = None

    def __repr__(self) -> str:
        n_traj = self.trajectories["obj_id"].n_unique() if len(self.trajectories) > 0 else 0
        n_frames = len(self.trajectories)
        opto_str = f", {len(self.opto)} opto events" if self.opto is not None else ""
        stim_str = f", {len(self.stim)} stim events" if self.stim is not None else ""
        return f"BraidzData({n_traj} trajectories, {n_frames} frames{opto_str}{stim_str})"

    @property
    def has_opto(self) -> bool:
        """Check if optogenetic data is available."""
        return self.opto is not None and len(self.opto) > 0

    @property
    def has_stim(self) -> bool:
        """Check if stimulus data is available."""
        return self.stim is not None and len(self.stim) > 0


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _open_file_or_url(path: str) -> io.BufferedReader:
    """
    Open a file from either a local path or URL.

    Args:
        path: Local file path or URL.

    Returns:
        File-like object with read and seek capability.
    """
    parsed = urllib.parse.urlparse(path)
    is_windows_drive = len(parsed.scheme) == 1  # e.g., "C:"

    if is_windows_drive or parsed.scheme == "":
        # Local file
        return open(path, mode="rb")

    # URL - download to memory
    req = urllib.request.Request(path, headers={"User-Agent": "braidz_analysis"})
    response = urllib.request.urlopen(req)
    return io.BytesIO(response.read())


def _normalize_file_list(
    files: Union[str, Path, List[Union[str, Path]]],
    base_folder: Optional[Union[str, Path]] = None,
) -> List[str]:
    """
    Normalize file input to a list of absolute paths.

    Handles:
        - Single file path
        - List of file paths
        - Comma-separated string of paths
        - Optional base folder prefix
    """
    # Handle comma-separated string
    if isinstance(files, str) and "," in files:
        files = [f.strip() for f in files.split(",")]
    elif isinstance(files, (str, Path)):
        files = [files]

    # Convert to strings and apply base folder
    result = []
    for f in files:
        f_str = str(f)
        if base_folder is not None:
            f_str = os.path.join(str(base_folder), f_str)
        result.append(f_str)

    return result


def _read_csv_from_archive(
    archive: zipfile.ZipFile,
    filename: str,
    exp_num: int,
    use_pyarrow: bool = True,
) -> Optional[pl.DataFrame]:
    """
    Read a CSV file from a ZIP archive with error handling.

    Args:
        archive: Open ZipFile object.
        filename: Name of CSV file within the archive.
        exp_num: Experiment number to add as a column.
        use_pyarrow: Whether to use PyArrow for faster parsing.

    Returns:
        DataFrame if successful, None if file doesn't exist or is empty.

    Raises:
        EmptyKalmanError: If kalman_estimates.csv.gz is empty.
    """
    if filename not in archive.namelist():
        logger.debug(f"{filename} not found in archive")
        return None

    try:
        is_kalman = filename == CSVFiles.KALMAN

        if use_pyarrow and PYARROW_AVAILABLE and is_kalman:
            # Use PyArrow for faster parsing of large Kalman files
            with gzip.open(archive.open(filename), "rt") as f:
                first_line = f.readline()
                # Skip comment lines (metadata)
                if first_line.startswith("#"):
                    content = f.read()
                else:
                    content = first_line + f.read()

            try:
                arrow_table = pv.read_csv(
                    pa.py_buffer(content.encode("utf-8")),
                    read_options=pv.ReadOptions(),
                )
                df = pl.from_arrow(arrow_table)
            except pa.ArrowInvalid as e:
                if "Empty CSV file" in str(e):
                    raise EmptyKalmanError(f"Empty {filename}")
                raise

        else:
            # Use polars for other files or when PyArrow unavailable
            file_bytes = archive.read(filename)
            if filename.endswith(".gz"):
                file_bytes = gzip.decompress(file_bytes)

            # Handle comment lines for kalman files
            if is_kalman:
                content = file_bytes.decode("utf-8")
                lines = content.split("\n")
                # Skip comment lines
                filtered_lines = [line for line in lines if not line.startswith("#")]
                content = "\n".join(filtered_lines)
                file_bytes = content.encode("utf-8")

            if len(file_bytes.strip()) == 0:
                if is_kalman:
                    raise EmptyKalmanError(f"Empty {filename}")
                return None

            df = pl.read_csv(io.BytesIO(file_bytes))

        if len(df) == 0:
            if is_kalman:
                raise EmptyKalmanError(f"Empty {filename}")
            return None

        # Add experiment number for multi-file loading
        df = df.with_columns(pl.lit(exp_num).alias("exp_num"))
        return df

    except pl.exceptions.NoDataError:
        if filename == CSVFiles.KALMAN:
            raise EmptyKalmanError(f"Empty {filename}")
        logger.debug(f"Empty {filename} in archive")
        return None


def _filter_short_trajectories(
    df: pl.DataFrame,
    min_frames: int = 100,
) -> pl.DataFrame:
    """
    Remove trajectories with fewer than min_frames data points.

    This pre-filtering step reduces memory usage when loading many files.
    """
    group_cols = ["exp_num", "obj_id"] if "exp_num" in df.columns else ["obj_id"]
    return df.filter(pl.len().over(group_cols) >= min_frames)


# =============================================================================
# Public API
# =============================================================================


def read_braidz(
    files: Union[str, Path, List[Union[str, Path]]],
    base_folder: Optional[Union[str, Path]] = None,
    engine: Literal["pandas", "pyarrow", "auto"] = "auto",
    pre_filter: bool = True,
    min_frames: int = 100,
    log_level: str = "warning",
    progressbar: bool = False,
) -> BraidzData:
    """
    Read trajectory data from one or more braidz files.

    Args:
        files: Single file path, list of paths, or comma-separated string of paths.
            Can be local files or URLs.

        base_folder: Optional base folder to prepend to all file paths.

        engine: CSV parsing engine.
            - "auto": Use PyArrow if available, otherwise polars
            - "pyarrow": Force PyArrow (faster for large files)
            - "pandas": Force polars (legacy option name, now uses polars)

        pre_filter: If True, remove trajectories shorter than min_frames
            during loading. Reduces memory usage for large datasets.

        min_frames: Minimum trajectory length to keep when pre_filter=True.

        log_level: Logging verbosity ("debug", "info", "warning", "error").

        progressbar: Show progress bar when loading multiple files.

    Returns:
        BraidzData object containing trajectories and optional event data.

    Raises:
        FileNotFoundError: If any specified file doesn't exist.
        ValueError: If no valid data found in any file.

    Example:
        >>> # Load single file
        >>> data = read_braidz("experiment.braidz")

        >>> # Load multiple files
        >>> data = read_braidz(["exp1.braidz", "exp2.braidz"])

        >>> # Load from folder
        >>> data = read_braidz("20230101.braidz", base_folder="/data/experiments")

        >>> # Load from URL
        >>> data = read_braidz("https://example.com/data.braidz")
    """
    # Configure logging
    log_level = log_level.lower()
    if log_level not in LOG_LEVELS:
        raise ValueError(f"Invalid log_level: {log_level}. Use: {list(LOG_LEVELS.keys())}")
    logger.setLevel(LOG_LEVELS[log_level])

    # Normalize file list
    files = _normalize_file_list(files, base_folder)

    # Validate files exist
    for f in files:
        if not f.startswith(("http://", "https://")) and not os.path.exists(f):
            raise FileNotFoundError(f"File not found: {f}")

    # Determine parsing engine
    if engine == "auto":
        use_pyarrow = PYARROW_AVAILABLE
    else:
        use_pyarrow = engine == "pyarrow"

    if use_pyarrow and not PYARROW_AVAILABLE:
        logger.warning("PyArrow requested but not available, falling back to polars")
        use_pyarrow = False

    # Containers for combined data
    all_trajectories: List[pl.DataFrame] = []
    all_opto: List[pl.DataFrame] = []
    all_stim: List[pl.DataFrame] = []
    all_other_csvs: Dict[str, List[pl.DataFrame]] = {}
    skipped_files: List[str] = []

    # Process each file
    file_iterator = tqdm(enumerate(files), total=len(files), disable=not progressbar)
    for exp_num, filepath in file_iterator:
        logger.debug(f"Reading: {filepath}")

        try:
            with zipfile.ZipFile(_open_file_or_url(filepath), "r") as archive:
                # Read required Kalman estimates
                try:
                    kalman_df = _read_csv_from_archive(
                        archive, CSVFiles.KALMAN, exp_num, use_pyarrow
                    )
                    if kalman_df is None:
                        raise InvalidBraidzError(f"No {CSVFiles.KALMAN} in {filepath}")

                    if pre_filter:
                        original_count = kalman_df["obj_id"].n_unique()
                        kalman_df = _filter_short_trajectories(kalman_df, min_frames)
                        filtered_count = kalman_df["obj_id"].n_unique() if len(kalman_df) > 0 else 0
                        if filtered_count == 0:
                            logger.warning(
                                f"Skipping {filepath}: no trajectories >= {min_frames} frames "
                                f"(had {original_count} shorter trajectories)"
                            )
                            skipped_files.append(filepath)
                            continue
                        elif filtered_count < original_count:
                            logger.debug(
                                f"{filepath}: kept {filtered_count}/{original_count} trajectories "
                                f"(filtered by min_frames={min_frames})"
                            )

                    all_trajectories.append(kalman_df)

                except EmptyKalmanError:
                    logger.warning(f"Skipping {filepath}: empty Kalman estimates")
                    skipped_files.append(filepath)
                    continue

                # Read optional event files
                opto_df = _read_csv_from_archive(archive, CSVFiles.OPTO, exp_num, False)
                if opto_df is not None:
                    all_opto.append(opto_df)

                stim_df = _read_csv_from_archive(archive, CSVFiles.STIM, exp_num, False)
                if stim_df is not None:
                    all_stim.append(stim_df)

                # Read any other CSV files
                excluded = {CSVFiles.KALMAN, CSVFiles.OPTO, CSVFiles.STIM}
                for name in archive.namelist():
                    if name.endswith(".csv") and name not in excluded:
                        try:
                            file_bytes = archive.read(name)
                            if len(file_bytes.strip()) > 0:
                                other_df = pl.read_csv(io.BytesIO(file_bytes))
                                other_df = other_df.with_columns(pl.lit(exp_num).alias("exp_num"))
                                all_other_csvs.setdefault(name, []).append(other_df)
                        except pl.exceptions.NoDataError:
                            logger.debug(f"Empty {name} in archive")
                        except pl.exceptions.ComputeError as e:
                            logger.warning(f"Failed to parse {name} in {filepath}: {e}")
                        except Exception as e:
                            logger.warning(f"Unexpected error reading {name} in {filepath}: {e}")

        except zipfile.BadZipFile:
            logger.error(f"Invalid ZIP file: {filepath}")
            skipped_files.append(filepath)
            continue
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            skipped_files.append(filepath)
            continue

    # Validate we got some data
    if not all_trajectories:
        raise ValueError("No valid trajectory data found in any of the provided files")

    # Combine all data
    trajectories = pl.concat(all_trajectories)
    opto = pl.concat(all_opto) if all_opto else None
    stim = pl.concat(all_stim) if all_stim else None

    # Combine other CSVs
    other_csvs = None
    if all_other_csvs:
        other_csvs = {name: pl.concat(dfs) for name, dfs in all_other_csvs.items()}

    if skipped_files:
        logger.warning(f"Skipped {len(skipped_files)} file(s) due to errors")

    return BraidzData(
        trajectories=trajectories,
        opto=opto,
        stim=stim,
        other_csvs=other_csvs,
        source_files=files,
    )


# Backwards compatibility alias
read = read_braidz
