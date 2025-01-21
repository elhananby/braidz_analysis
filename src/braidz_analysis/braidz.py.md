# Braidz Data Reader Documentation

## Table of Contents

* [1. Introduction](#1-introduction)
* [2. Module-Level Constants and Configuration](#2-module-level-constants-and-configuration)
* [3. Custom Exceptions](#3-custom-exceptions)
* [4. Data Structures](#4-data-structures)
    * [4.1. `BraidzData` TypedDict](#41-braidzdata-typeddict)
    * [4.2. `CSVFiles` Dataclass](#42-csvfiles-dataclass)
* [5. Functions](#5-functions)
    * [5.1. `_open_filename_or_url` Function](#51-_open_filename_or_url-function)
    * [5.2. `_read_csv_safe` Function](#52-_read_csv_safe-function)
    * [5.3. `_process_other_csvs` Function](#53-_process_other_csvs-function)
    * [5.4. `read_braidz` Function](#54-read_braidz-function)


## 1. Introduction

This module provides functionality to read and process data from Braidz archive files (`.braidz`).  It handles both local files and URLs, supports parallel processing, and offers robust error handling. The module utilizes either `pandas` or `pyarrow` for CSV parsing depending on availability and user preference.


## 2. Module-Level Constants and Configuration

The module defines logging levels and checks for the availability of the `pyarrow` library.  If `pyarrow` is available, it's imported for potentially faster CSV parsing; otherwise, `pandas` is used as a fallback.

| Constant Name       | Description                                                                   | Value/Type       |
|-----------------------|-------------------------------------------------------------------------------|-------------------|
| `LOG_LEVELS`         | Dictionary mapping log level names to their corresponding logging levels.    | `Dict[str, int]` |
| `logger`             | Logger instance for logging messages.                                        | `logging.Logger` |
| `PYARROW_AVAILABLE` | Boolean indicating whether the `pyarrow` library is installed and importable. | `bool`           |


## 3. Custom Exceptions

| Exception Name       | Description                                                                        |
|-----------------------|------------------------------------------------------------------------------------|
| `EmptyKalmanError`    | Raised when the `kalman_estimates.csv.gz` file within a Braidz archive is empty. |


## 4. Data Structures

### 4.1. `BraidzData` TypedDict

This `TypedDict` defines the structure of the dictionary returned by the `read_braidz` function.

| Key          | Type                     | Description                                      |
|---------------|--------------------------|--------------------------------------------------|
| `df`          | `pd.DataFrame`           | Kalman estimates DataFrame.                      |
| `opto`        | `Optional[pd.DataFrame]` | Optogenetics data DataFrame (optional).           |
| `stim`        | `Optional[pd.DataFrame]` | Stimulus data DataFrame (optional).               |
| `other_csvs` | `Dict[str, List[pd.DataFrame]]` | Dictionary of other CSV files found in the archive. |


### 4.2. `CSVFiles` Dataclass

This dataclass stores the filenames of expected CSV files within Braidz archives.

| Field Name | Value                     | Description                                   |
|------------|--------------------------|-----------------------------------------------|
| `KALMAN`   | `"kalman_estimates.csv.gz"` | Filename for Kalman estimates.                 |
| `OPTO`     | `"opto.csv"`             | Filename for optogenetics data (optional).      |
| `STIM`     | `"stim.csv"`             | Filename for stimulus data (optional).         |


## 5. Functions

### 5.1. `_open_filename_or_url` Function

This function opens a file, handling both local file paths and URLs. It uses `urllib` for URLs and standard `open()` for local files, ensuring consistent file object return.  Windows drive letters are also correctly handled.

### 5.2. `_read_csv_safe` Function

This function reads a CSV file from a zip archive using either `pandas` or `pyarrow`, providing robust error handling for missing or empty files.  It handles gzip compression and includes a comment skipping feature for the Kalman estimates file.  If `pyarrow` is used for reading `kalman_estimates.csv.gz`, the function first decompresses the file to handle potential issues with `pyarrow`'s gzip handling.  Error handling includes checking for empty data and raising `EmptyKalmanError` specifically for empty `kalman_estimates.csv.gz` files.  The function adds an `exp_num` column to the DataFrame.

*   **Algorithm:** Checks file existence, determines compression, selects parsing engine, reads CSV using chosen engine, adds experiment number column, handles exceptions, returns DataFrame or None.

### 5.3. `_process_other_csvs` Function

This function processes all CSV files in a zip archive *except* those specified in `excluded_files`.  It uses `pandas` to read each CSV and stores them in a dictionary keyed by filename.  Empty CSV files are logged and skipped.

*   **Algorithm:** Iterates through archive contents, filters for CSV files not in `excluded_files`, reads each using `pd.read_csv`, handles `pd.errors.EmptyDataError`, appends DataFrames to a dictionary.


### 5.4. `read_braidz` Function

This is the main function of the module. It reads data from one or more Braidz files, processes the data, and returns a combined result.  It handles different file input types (single string or list of strings), prepending a `base_folder` if provided. It uses the specified or auto-detected engine (`pyarrow` if available, otherwise `pandas`) for CSV parsing.  The function validates the input log level, and handles errors during file processing gracefully.  It concatenates DataFrames for Kalman estimates, optogenetics, and stimulus data, while also collecting other CSV data.  It raises a `ValueError` if no valid Kalman estimates file is found.

*   **Algorithm:**
    1.  Validates and normalizes input file paths.
    2.  Sets logging level.
    3.  Determines CSV parsing engine.
    4.  Iterates through input Braidz files:
        *   Opens each archive using `_open_filename_or_url`.
        *   Reads Kalman estimates using `_read_csv_safe`.  Skips file if empty Kalman estimates are found.
        *   Reads optional optogenetics and stimulus data using `_read_csv_safe`.
        *   Processes other CSV files using `_process_other_csvs`.
        *   Handles exceptions during file processing.
    5.  Combines data from multiple files using `pd.concat`.
    6.  Returns a `BraidzData` dictionary containing the combined data.  Raises `ValueError` if no valid Kalman data is found.

