# utils.py
from typing import Dict, List
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from .utils import TrajectoryAnalysisManager


def smooth_trajectory(
    df: pd.DataFrame,
    columns: List[str] = ["x", "y", "z", "xvel", "yvel", "zvel"],
    window: int = 21,
    polyorder: int = 3,
) -> pd.DataFrame:
    """
    Smooth trajectory data using Savitzky-Golay filter.

    Args:
        df: DataFrame containing trajectory data
        columns: Columns to smooth
        window: Window size for filtering
        polyorder: Polynomial order for filtering

    Returns:
        pd.DataFrame: Smoothed DataFrame
    """
    df = df.copy()
    for column in columns:
        df.loc[:, f"original_{column}"] = df[column].copy()
        df.loc[:, column] = savgol_filter(df[column], window, polyorder)
    return df


def dict_list_to_numpy(data_dict: Dict) -> Dict:
    """Convert dictionary of lists to dictionary of numpy arrays."""
    return {k: np.array(v) for k, v in data_dict.items()}


# Example usage
def main():
    """Example usage of the trajectory analysis pipeline."""
    # Load data
    trajectory_df = pd.read_csv("trajectory_data.csv")
    stim_df = pd.read_csv("stim_events.csv")

    # Initialize analysis manager with default parameters
    manager = TrajectoryAnalysisManager()

    # Process dataset
    results = manager.process_dataset(trajectory_df, stim_df, stim_type="opto")

    # Convert to legacy format if needed
    legacy_results = manager.convert_to_legacy_format(results)

    # Access individual results
    for result in results:
        print(
            f"Trajectory {result.metadata['obj_id']}: "
            f"{'Responsive' if result.responsive else 'Non-responsive'}"
        )


if __name__ == "__main__":
    main()
