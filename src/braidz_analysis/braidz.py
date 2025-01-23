import io
import logging
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Dict, List, Literal, Optional, Union

import pandas as pd
from tqdm import tqdm
from typing_extensions import TypedDict


class EmptyKalmanError(Exception):
    """Raised when kalman_estimates.csv.gz is empty"""

    pass


# Configure logging and constants
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

logger = logging.getLogger(__name__)

# Check if pyarrow is available
PYARROW_AVAILABLE = find_spec("pyarrow") is not None
if PYARROW_AVAILABLE:
    import pyarrow as pa
    import pyarrow.csv as pv
else:
    logger.info("PyArrow not found, defaulting to pandas for CSV parsing")


class BraidzData(TypedDict):
    """Type definition for the return value of read_braidz"""

    df: pd.DataFrame  # kalman estimates
    opto: Optional[pd.DataFrame]  # optogenetics data
    stim: Optional[pd.DataFrame]  # stimulus data
    other_csvs: Dict[str, List[pd.DataFrame]]  # other CSV files


@dataclass
class CSVFiles:
    """Container for CSV file names in braidz archives"""

    KALMAN = "kalman_estimates.csv.gz"
    OPTO = "opto.csv"
    STIM = "stim.csv"


def _open_filename_or_url(filename_or_url: str) -> io.BufferedReader:
    """
    Open a file from either a local path or URL.

    Args:
        filename_or_url: Local file path or URL to open

    Returns:
        File object with seek capability
    """
    parsed = urllib.parse.urlparse(filename_or_url)
    is_windows_drive = len(parsed.scheme) == 1

    if is_windows_drive or parsed.scheme == "":
        return open(filename_or_url, mode="rb")

    # Handle URL case
    req = urllib.request.Request(filename_or_url, headers={"User-Agent": "IPYNB"})
    fileobj = urllib.request.urlopen(req)
    return io.BytesIO(fileobj.read())


def _read_csv_safe(
    archive: zipfile.ZipFile,
    filename: str,
    exp_num: int,
    engine: str = "pandas",
    **kwargs,
) -> Optional[pd.DataFrame]:
    """
    Safely read a CSV file from a zip archive with error handling.

    Args:
        archive: Zip archive containing the CSV
        filename: Name of the CSV file in the archive
        exp_num: Experiment number to add as a column
        engine: CSV parsing engine to use ('pandas' or 'pyarrow')
        **kwargs: Additional arguments passed to the CSV reader

    Returns:
        DataFrame if successful, None if file is empty or doesn't exist

    Raises:
        EmptyKalmanError: If kalman_estimates.csv.gz is empty
    """
    if filename not in archive.namelist():
        logger.debug(f"{filename} not found in archive")
        return None

    try:
        if engine == "pyarrow" and filename == CSVFiles.KALMAN:
            import gzip

            # Read and decompress the gzipped content first
            with gzip.open(archive.open(filename), "rt") as f:
                # Skip the first line if it's a comment
                first_line = f.readline()
                if first_line.startswith("#"):
                    content = f.read()
                else:
                    content = first_line + f.read()

            # Use PyArrow to read the decompressed content
            try:
                df = pv.read_csv(
                    pa.py_buffer(content.encode("utf-8")), read_options=pv.ReadOptions()
                ).to_pandas()
            except pa.ArrowInvalid as e:
                if "Empty CSV file" in str(e) and filename == CSVFiles.KALMAN:
                    raise EmptyKalmanError(f"Empty {filename}")
                logger.debug(f"Empty {filename} in archive")
                return None
        else:
            compression = "gzip" if filename.endswith(".gz") else None
            df = pd.read_csv(
                archive.open(filename),
                compression=compression,
                comment="#" if filename == CSVFiles.KALMAN else None,
                **kwargs,
            )

        df["exp_num"] = exp_num
        return df

    except pd.errors.EmptyDataError:
        if filename == CSVFiles.KALMAN:
            raise EmptyKalmanError(f"Empty {filename}")
        logger.debug(f"Empty {filename} in archive")
        return None


def _process_other_csvs(
    archive: zipfile.ZipFile, excluded_files: List[str]
) -> Dict[str, List[pd.DataFrame]]:
    """
    Process all CSV files in the archive except those in excluded_files.

    Args:
        archive: Zip archive containing the CSV files
        excluded_files: List of CSV files to exclude from processing

    Returns:
        Dictionary mapping CSV filenames to lists of DataFrames
    """
    other_csvs = {}

    for name in archive.namelist():
        if name.endswith(".csv") and name not in excluded_files:
            try:
                df = pd.read_csv(archive.open(name))
                other_csvs.setdefault(name, []).append(df)
            except pd.errors.EmptyDataError:
                logger.debug(f"Empty {name} in archive")

    return other_csvs


def _convert_to_list(files):
    if isinstance(files, str):
        # Check if it's a comma-separated string
        if "," in files:
            # Split by comma and strip whitespace
            return [f.strip() for f in files.split(",")]
        # Single file string
        return [files]
    # Already a list
    return files


def _filter_df(df: pd.DataFrame, min_length: int = 100) -> pd.DataFrame:
    """
    Filter a DataFrame to remove groups with less than min_length rows.

    Args:
        df: Input DataFrame
        min_length: Minimum number of rows to keep in each group

    Returns:
        Filtered DataFrame
    """
    groups = ["exp_num", "obj_id"] if "exp_num" in df.columns else ["obj_id"]
    return (
        df.groupby(groups).filter(lambda x: len(x) >= min_length).reset_index(drop=True)
    )


def read_braidz(
    files: Union[str, List[str]],
    base_folder: Optional[str] = None,
    engine: Literal["pandas", "pyarrow", "auto"] = "auto",
    log_level: str = "info",
    progressbar: bool = False,
    pre_filter: bool = True,
) -> BraidzData:
    """
    Read data from one or more .braidz files.

    Args:
        files: Single file path/URL or list of file paths/URLs
        base_folder: Optional base folder to prepend to file paths
        engine: CSV parsing engine to use ('pandas', 'pyarrow', or 'auto')
        log_level: Logging level to use (default: 'info'). Must be one of:
                  'debug', 'info', 'warning', 'error', or 'critical'
        progressbar: Whether to show a progress bar (default: False)
        pre_filter: Whether to filter out objects with less than 100 rows. This should improve memory usage when loading many files. (default: True)

    Returns:
        Dictionary containing combined data from all files:
        - 'df': Kalman estimates DataFrame
        - 'opto': Optogenetics DataFrame (if present)
        - 'stim': Stimulus DataFrame (if present)
        - 'other_csvs': Dictionary of other CSV files found

    Raises:
        FileNotFoundError: If any specified file doesn't exist
        ValueError: If no valid kalman_estimates.csv.gz found in any file
                   or if invalid log_level provided
    """
    # Set logging level
    log_level = log_level.lower()
    if log_level not in LOG_LEVELS:
        raise ValueError(
            f"Invalid log_level: {log_level}. Must be one of: {', '.join(LOG_LEVELS.keys())}"
        )
    logger.setLevel(LOG_LEVELS[log_level])

    # Handle input validation and normalization
    files = _convert_to_list(files)

    if base_folder is not None:
        files = [os.path.join(base_folder, f) for f in files]

    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File not found: {f}")

    # Determine engine to use
    if engine == "auto":
        engine = "pyarrow" if PYARROW_AVAILABLE else "pandas"

    # Initialize data containers
    kalman_dfs = []
    optos = []
    stims = []
    all_other_csvs = {}

    # Process each file
    skipped_files = []
    for exp_num, filepath in tqdm(
        enumerate(files), total=len(files), disable=not progressbar
    ):
        logger.debug(f"Reading file: {filepath}")

        try:
            with zipfile.ZipFile(_open_filename_or_url(filepath), "r") as archive:
                # Read required kalman estimates
                try:
                    kalman_df = _read_csv_safe(
                        archive, CSVFiles.KALMAN, exp_num, engine
                    )
                    if kalman_df is None:
                        raise FileNotFoundError(
                            f"{CSVFiles.KALMAN} not found in {filepath}"
                        )

                    kalman_dfs.append(
                        _filter_df(kalman_df) if pre_filter else kalman_df
                    )
                except EmptyKalmanError:
                    logger.warning(f"Skipping {filepath} due to empty kalman estimates")
                    skipped_files.append(filepath)
                    continue

                # Read optional files
                opto_df = _read_csv_safe(archive, CSVFiles.OPTO, exp_num)
                if opto_df is not None:
                    optos.append(opto_df)

                stim_df = _read_csv_safe(archive, CSVFiles.STIM, exp_num)
                if stim_df is not None:
                    stims.append(stim_df)

                # Process other CSVs
                excluded_files = [CSVFiles.KALMAN, CSVFiles.OPTO, CSVFiles.STIM]
                other_csvs = _process_other_csvs(archive, excluded_files)

                # Merge other_csvs dictionaries
                for name, dfs in other_csvs.items():
                    all_other_csvs.setdefault(name, []).extend(dfs)
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            skipped_files.append(filepath)
            continue

    if not kalman_dfs:
        raise ValueError("No valid kalman_estimates.csv.gz found in any of the files")

    # Combine results
    try:
        return {
            "df": pd.concat(kalman_dfs),
            "opto": pd.concat(optos) if optos else None,
            "stim": pd.concat(stims) if stims else None,
            "other_csvs": all_other_csvs,
        }
    except ValueError:
        raise ValueError("No valid kalman_estimates.csv.gz found in any of the files")
