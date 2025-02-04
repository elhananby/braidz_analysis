# braidz_analysis/src/braidz_analysis/processing/__init__.py
import pandas as pd

from .analysis import AnalysisResult, TrajectoryAnalyzer
from .core import StimulationEvent, TrajectoryData
from .filtering import TrajectoryFilter
from .interface import TrajectoryAnalysisManager

# For backwards compatibility
from .interface import TrajectoryAnalysisManager as Processing
from .params import (
    AnalysisParams,
    BaseAnalysisParams,
    SaccadeParams,
    SpatialParams,
)
from .utils import dict_list_to_numpy, smooth_trajectory

# Version info
__version__ = "2.0.0"

__all__ = [
    # Core data structures
    "TrajectoryData",
    "StimulationEvent",
    # Parameters
    "AnalysisParams",
    "BaseAnalysisParams",
    "SpatialParams",
    "SaccadeParams",
    # Analysis components
    "TrajectoryFilter",
    "AnalysisResult",
    "TrajectoryAnalyzer",
    "TrajectoryAnalysisManager",
    # Utilities
    "smooth_trajectory",
    "dict_list_to_numpy",
    # Backwards compatibility
    "Processing",
]


# Legacy function aliases for backwards compatibility
def get_stim_or_opto_response_data(
    df, stim_df, params=None, type="opto", saccade_params=None
):
    """Legacy interface for stimulus response analysis."""
    manager = TrajectoryAnalysisManager(params)
    results = manager.process_dataset(df, stim_df, stim_type=type)
    return manager.convert_to_legacy_format(results)


def get_stim_or_opto_data(df, stim_df, params=None, type="opto", saccade_params=None):
    """Legacy interface for stimulus data analysis."""
    manager = TrajectoryAnalysisManager(params)
    results = manager.process_dataset(df, stim_df, stim_type=type)
    return manager.convert_to_legacy_format(results)


def get_all_saccades(df, params=None, progressbar=True):
    """Legacy interface for saccade analysis."""
    manager = TrajectoryAnalysisManager(params, show_progress=progressbar)
    # Create a dummy stim_df with all frames
    stim_df = pd.DataFrame(
        {
            "frame": df["frame"].unique(),
            "obj_id": df["obj_id"].iloc[0],
            "exp_num": df["exp_num"].iloc[0] if "exp_num" in df.columns else None,
        }
    )
    results = manager.process_dataset(df, stim_df, stim_type="saccade")
    return manager.convert_to_legacy_format(results)


# Add legacy functions to __all__
__all__.extend(
    [
        "get_stim_or_opto_response_data",
        "get_stim_or_opto_data",
        "get_all_saccades",
    ]
)
