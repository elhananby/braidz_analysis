"""
braidz: A module for reading and processing .braidz files and related data.

This module provides utilities for reading single or multiple .braidz files,
which are zip archives containing CSV and gzipped CSV data. It supports both
direct reading from archives and extraction to folders, with options for
different CSV parsing engines.

Features:
- Single and multi-file reading capabilities
- Support for both pandas and pyarrow parsing engines with automatic fallback
- Robust error handling and validation
- Progress tracking for long operations
"""

import os
import zipfile
import gzip
import logging
from typing import Dict, Optional, Literal, Union, List
from dataclasses import dataclass
from importlib.util import find_spec

import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# Set log level to INFO
logger.setLevel(logging.INFO)

# Check if pyarrow is available
PYARROW_AVAILABLE = find_spec("pyarrow") is not None
if PYARROW_AVAILABLE:
    import pyarrow as pa
    import pyarrow.csv as pv
else:
    logger.info("PyArrow not found, defaulting to pandas for CSV parsing")


@dataclass
class BraidzData:
    """
    Container for data extracted from braidz files.

    Attributes:
        kalman_estimates: DataFrame containing kalman estimates data
        csv_data: Dictionary of additional CSV data files
        source_file: Path to the source braidz file
    """

    kalman_estimates: Optional[pd.DataFrame]
    csv_data: Dict[str, pd.DataFrame]
    source_file: str


class BraidzError(Exception):
    """Base exception class for braidz-related errors."""

    pass


class BraidzFileError(BraidzError):
    """Raised when there are issues with file operations."""

    pass


class BraidzValidationError(BraidzError):
    """Raised when file validation fails."""

    pass


class BraidzParsingError(BraidzError):
    """Raised when data parsing fails."""

    pass


def read_csv(
    file_obj, parser: Literal["pandas", "pyarrow"] = "pyarrow"
) -> Optional[pd.DataFrame]:
    """
    Read a CSV file using the specified parser with automatic fallback.

    Args:
        file_obj: File-like object or path containing CSV data.
        parser: Preferred parser to use ("pandas" or "pyarrow"). Defaults to "pyarrow"
               if available, otherwise falls back to "pandas".

    Returns:
        DataFrame containing the CSV data, or None if invalid/empty.

    Raises:
        BraidzParsingError: If parsing fails for any reason other than empty file.
    """
    # If pyarrow is requested but not available, fall back to pandas
    if parser == "pyarrow" and not PYARROW_AVAILABLE:
        logger.debug("PyArrow requested but not available, falling back to pandas")
        parser = "pandas"

    try:
        if parser == "pyarrow":
            try:
                table = pv.read_csv(
                    file_obj, read_options=pv.ReadOptions(skip_rows_after_names=1)
                )
                return table.to_pandas()
            except pa.ArrowInvalid as e:
                logger.warning(
                    f"PyArrow parsing failed: {str(e)}:{file_obj}, falling back to pandas"
                )
                # If pyarrow fails, try pandas as fallback
                return pd.read_csv(file_obj, comment="#")
        else:  # pandas
            return pd.read_csv(file_obj, comment="#")

    except pd.errors.EmptyDataError:
        logger.warning(f"Empty CSV file {file_obj} encountered")
        return None
    except Exception as e:
        raise BraidzParsingError(f"Failed to parse CSV {file_obj} with {parser}: {str(e)}")


def read_braidz_file(
    filename: str, parser: Literal["pandas", "pyarrow"] = "pyarrow"
) -> BraidzData:
    """
    Read data from a .braidz file using the specified parser.

    Args:
        filename: Path to the .braidz file.
        parser: Parser to use ("pandas" or "pyarrow"). Defaults to "pyarrow"
               if available, otherwise falls back to "pandas".

    Returns:
        BraidzData object containing the extracted data.

    Raises:
        BraidzFileError: If file operations fail.
        BraidzParsingError: If data parsing fails.
        BraidzValidationError: If file validation fails.
    """
    if not os.path.exists(filename):
        raise BraidzFileError(f"File not found: {filename}")

    if not filename.endswith(".braidz"):
        raise BraidzValidationError(f"Invalid file type: {filename}")

    csv_data: Dict[str, pd.DataFrame] = {}

    try:
        with zipfile.ZipFile(filename, "r") as archive:
            logger.info(f"Reading {filename} using {parser}")

            # Read kalman_estimates.csv.gz
            try:
                with archive.open("kalman_estimates.csv.gz") as file:
                    if parser == "pandas":
                        df = read_csv(gzip.open(file, "rt"), parser="pandas")
                    else:
                        with gzip.open(file, "rb") as unzipped:
                            df = read_csv(unzipped, parser=parser)

                if df is None or df.empty:
                    logger.warning(f"Empty kalman_estimates in {filename}")
                    return BraidzData(None, {}, filename)

            except KeyError:
                logger.error(f"kalman_estimates.csv.gz not found in {filename}")
                return BraidzData(None, {}, filename)

            # Read other CSV files
            csv_files = [f for f in archive.namelist() if f.endswith(".csv")]
            for csv_file in csv_files:
                key = os.path.splitext(csv_file)[0]
                try:
                    csv_data[key] = read_csv(archive.open(csv_file), parser=parser)
                except Exception as e:
                    logger.warning(f"Failed to read {csv_file}: {str(e)}")
                    continue

        return BraidzData(df, csv_data, filename)

    except zipfile.BadZipFile:
        raise BraidzFileError(f"Invalid zip file: {filename}")
    except Exception as e:
        raise BraidzFileError(f"Failed to process {filename}: {str(e)}")


def read_multiple_braidz(
    files: Union[List[str], str],
    root_folder: Optional[str] = None,
    parser: Literal["pandas", "pyarrow"] = "pyarrow",
) -> Dict[str, pd.DataFrame]:
    """
    Read and combine data from multiple .braidz files.

    Args:
        files: List of file paths or single file path.
        root_folder: Optional root folder path to prepend to file paths.
        parser: Parser to use ("pandas" or "pyarrow"). Defaults to "pyarrow"
               if available, otherwise falls back to "pandas".

    Returns:
        Dictionary containing combined DataFrames:
            - 'df': Combined kalman estimates
            - 'stim': Combined stimulus data (if present)
            - 'opto': Combined opto data (if present)

    Raises:
        BraidzError: If any file processing fails.
    """
    if isinstance(files, str):
        files = [files]

    dfs, stims, optos = [], [], []
    total_files = len(files)
    logger.info(f"Processing {total_files} files")

    for i, filename in enumerate(files):
        try:
            if root_folder:
                filename = os.path.join(root_folder, filename)

            print(f"Processing file {i+1}/{total_files}: {filename}")
            data = read_braidz_file(filename, parser=parser)

            if data.kalman_estimates is not None:
                df = data.kalman_estimates.copy()
                df["exp_num"] = i
                dfs.append(df)

                if "stim" in data.csv_data:
                    stim = data.csv_data["stim"].copy()
                    stim["exp_num"] = i
                    stims.append(stim)

                if "opto" in data.csv_data:
                    opto = data.csv_data["opto"].copy()
                    opto["exp_num"] = i
                    optos.append(opto)

        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue

    # Combine all dataframes if they exist
    result = {}
    logger.info("Combining processed data")
    if dfs:
        result["df"] = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {len(dfs)} kalman estimate dataframes")
    if stims:
        result["stim"] = pd.concat(stims, ignore_index=True)
        logger.info(f"Combined {len(stims)} stimulus dataframes")
    if optos:
        result["opto"] = pd.concat(optos, ignore_index=True)
        logger.info(f"Combined {len(optos)} optogenetics dataframes")

    return result

# Basic usage - will use pyarrow if available, otherwise pandas
# data = read_braidz_file("path/to/file.braidz")

# # Explicitly specify parser
# data = read_braidz_file("path/to/file.braidz", parser="pandas")

# # Multiple files with specific parser
# combined_data = read_multiple_braidz(
#     ["file1.braidz", "file2.braidz"],
#     root_folder="data/",
#     parser="pyarrow"
# )