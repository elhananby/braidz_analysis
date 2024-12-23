# braidz-analysis
A package to process .braidz files from Andrew Straw's [Braid system](https://github.com/strawlab/strand-braid/).

## Installation
Two options - either create a conda/mamba environment using the `env.yaml` file:
```sh
mamba env create -f env.yaml
mamba activate braidz-analysis
```
to update the environment:
```
conda env update -f environment.yml --prune
```

or create a virtual environment using either [uv](https://github.com/astral-sh/uv) or [poetry](https://github.com/python-poetry/poetry) or whatever you like.

## Contents
Currently includes the following modules:
* `braidz.py` - contains the main function to read `.braid` file. I would recommend always using `read_multiple_braidz`, even if reading only a single file.
* `errors.py` - unused currently.
* `filtering.py` - functions to filter `braid` pd.DataFrame objects based on different trajectory paramters.
* `helpers.py` - some basic helper functions
* `plotting.py` - wrappers to plot different trajectory metrics.
* `processing.py` - main processing module. Able to process either `stim` or `opto` dataframes, or extract saccades from entire`df` data.
* `trajectory.py` - trajectory metrics calculations (angular velocity, linear velocity, heading change, etc).
* `types.py` - custom types for better managment.

## Basic usage
See `notebooks\read_braidz_example.py` for some basic usage examples. Alternativley, there's a `analyse_and_plot.py` script you can use to quickly perform an analysis of your choice on braid file(s).