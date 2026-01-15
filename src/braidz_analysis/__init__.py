"""
braidz_analysis: Analysis tools for Strand-Braid fly tracking data.

This package provides tools for analyzing fruit fly (Drosophila) flight
trajectories from the Strand-Braid tracking system, with support for:

    - Loading .braidz trajectory files
    - Kinematic analysis (velocity, heading, saccade detection)
    - Event-triggered analysis (optogenetic and visual stimuli)
    - Publication-quality visualization

Quick Start
-----------
>>> import braidz_analysis as ba

>>> # Load data
>>> data = ba.read_braidz("experiment.braidz")

>>> # Analyze saccades
>>> saccades = ba.analyze_saccades(data.trajectories)

>>> # Analyze optogenetic responses
>>> opto = ba.analyze_event_responses(data.trajectories, data.opto)
>>> print(f"Response rate: {opto.response_rate:.1%}")

>>> # Plot results
>>> ba.plot_angular_velocity(opto.responsive)

Modules
-------
io : Reading and writing braidz files
kinematics : Velocity calculations, saccade detection, flight state
analysis : High-level analysis functions
plotting : Visualization tools
config : Configuration dataclass

"""

__version__ = "0.3.0"

# =============================================================================
# Primary API - Most commonly used functions
# =============================================================================

# Data loading
from .io import read_braidz, BraidzData

# Analysis functions
from .analysis import (
    analyze_saccades,
    analyze_event_responses,
    filter_trajectories,
    compute_response_statistics,
    SaccadeResults,
    EventResults,
)

# Configuration
from .config import Config, DEFAULT_CONFIG, OPTO_CONFIG, STIM_CONFIG

# Plotting
from .plotting import (
    plot_traces,
    plot_angular_velocity,
    plot_linear_velocity,
    plot_heading_distribution,
    plot_heading_comparison,
    plot_trajectory,
    plot_response_rate_by_group,
    add_stimulus_region,
    convert_frames_to_ms,
    create_summary_figure,
)

# =============================================================================
# Secondary API - Kinematics functions for advanced users
# =============================================================================

from .kinematics import (
    compute_angular_velocity,
    compute_linear_velocity,
    compute_heading_change,
    detect_saccades,
    classify_flight_state,
    extract_flight_bouts,
    smooth_trajectory,
    add_kinematics_to_trajectory,
    SaccadeEvent,
)

# =============================================================================
# Legacy module access for backwards compatibility
# =============================================================================

from . import io
from . import kinematics
from . import analysis
from . import plotting
from . import config

# Legacy aliases (deprecated, will be removed in v1.0)
braidz = io  # Old: ba.braidz.read_braidz -> New: ba.read_braidz
processing = analysis  # Old: ba.processing.get_stim_or_opto_data -> New: ba.analyze_event_responses
trajectory = kinematics  # Old: ba.trajectory.detect_saccades -> New: ba.detect_saccades

# =============================================================================
# Public API declaration
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Data loading
    "read_braidz",
    "BraidzData",
    # Analysis
    "analyze_saccades",
    "analyze_event_responses",
    "filter_trajectories",
    "compute_response_statistics",
    "SaccadeResults",
    "EventResults",
    # Configuration
    "Config",
    "DEFAULT_CONFIG",
    "OPTO_CONFIG",
    "STIM_CONFIG",
    # Plotting
    "plot_traces",
    "plot_angular_velocity",
    "plot_linear_velocity",
    "plot_heading_distribution",
    "plot_heading_comparison",
    "plot_trajectory",
    "plot_response_rate_by_group",
    "add_stimulus_region",
    "convert_frames_to_ms",
    "create_summary_figure",
    # Kinematics
    "compute_angular_velocity",
    "compute_linear_velocity",
    "compute_heading_change",
    "detect_saccades",
    "classify_flight_state",
    "extract_flight_bouts",
    "smooth_trajectory",
    "add_kinematics_to_trajectory",
    "SaccadeEvent",
    # Modules
    "io",
    "kinematics",
    "analysis",
    "plotting",
    "config",
]
