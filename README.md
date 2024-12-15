# braidz-analysis

A Python library for reading and analyzing .braidz files. The package provides utilities for efficiently reading single or multiple .braidz files, which are zip archives containing CSV and gzipped CSV data.

## Features

- Single and multi-file reading capabilities
- Support for both pandas and pyarrow parsing engines with automatic fallback
- Robust error handling and validation
- Clear logging of processing operations
- Type hints and comprehensive documentation

## Installation

Currently only via Github:
```bash
git clone https://github.com/elhananby/braidz_analysis.git
cd braidz_analysis
pip install .
```

or for development:
```
pip install -e .
```

## Usage

### Reading a Single File

```python
from braidz_analysis import braidz

# Read a single braidz file
data = braidz.read_braidz_file("path/to/file.braidz")

# Access the data
kalman_data = data.kalman_estimates  # Main tracking data
additional_data = data.csv_data      # Dictionary of other CSV files
```

### Reading Multiple Files

```python
from braidz_analysis import braidz

# Read multiple files
combined_data = braidz.read_multiple_braidz(
    ["file1.braidz", "file2.braidz"],
    root_folder="data/"
)

# Access combined data
kalman_data = combined_data["df"]      # Combined tracking data
stimulus_data = combined_data["stim"]  # Combined stimulus data (if present)
opto_data = combined_data["opto"]      # Combined optogenetics data (if present)
```

### Customizing Parser

The library supports both PyArrow and Pandas for CSV parsing:

```python
# Explicitly use pandas parser
data = braidz.read_braidz_file("file.braidz", parser="pandas")

# Use pyarrow (default if available)
data = braidz.read_braidz_file("file.braidz", parser="pyarrow")
```

## Data Structure

The library returns data in the following structure:

- For single files (`read_braidz_file`):
  - Returns a `BraidzData` object containing:
    - `kalman_estimates`: DataFrame with tracking data
    - `csv_data`: Dictionary of additional CSV files
    - `source_file`: Path to the source file

- For multiple files (`read_multiple_braidz`):
  - Returns a dictionary containing:
    - `df`: Combined kalman estimates
    - `stim`: Combined stimulus data (if present)
    - `opto`: Combined optogenetics data (if present)

## Requirements

- Python ≥ 3.10
- pandas ≥ 2.2.3
- pyarrow ≥ 18.1.0 (optional, for faster CSV parsing)