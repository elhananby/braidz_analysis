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

Logging
-------
>>> # Set logging level for the package
>>> ba.set_log_level("DEBUG")  # Show all messages
>>> ba.set_log_level("WARNING")  # Only warnings and errors (quieter)

Modules
-------
io : Reading and writing braidz files
kinematics : Velocity calculations, saccade detection, flight state
analysis : High-level analysis functions
plotting : Visualization tools
config : Configuration dataclass

"""

import logging

__version__ = "0.3.0"

# =============================================================================
# Logging Configuration
# =============================================================================

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def set_log_level(level: str = "WARNING") -> None:
    """
    Set the logging level for the braidz_analysis package.

    Args:
        level: Logging level string. One of: 'DEBUG', 'INFO', 'WARNING',
               'ERROR', 'CRITICAL' (case-insensitive).

    Example:
        >>> import braidz_analysis as ba
        >>> ba.set_log_level("DEBUG")  # Show all messages including debug
        >>> ba.set_log_level("WARNING")  # Only warnings and errors (default)
        >>> ba.set_log_level("ERROR")  # Only errors
    """
    level_upper = level.upper()
    level_lower = level.lower()

    if level_lower not in _LOG_LEVELS:
        valid = ", ".join(_LOG_LEVELS.keys())
        raise ValueError(f"Invalid log level: {level}. Must be one of: {valid}")

    log_level = _LOG_LEVELS[level_lower]

    # Set level on the package logger (affects all submodules)
    package_logger = logging.getLogger("braidz_analysis")
    package_logger.setLevel(log_level)

    # Ensure there's at least a handler if none configured
    if not package_logger.handlers and not logging.root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        package_logger.addHandler(handler)

    package_logger.debug(f"Log level set to {level_upper}")


# =============================================================================
# Primary API - Most commonly used functions
# =============================================================================

# Data loading
# =============================================================================
# Module access
# =============================================================================
from . import analysis, config, io, kinematics, plotting

# Analysis functions
from .analysis import (
    EventResults,
    SaccadeResults,
    analyze_event_responses,
    analyze_saccades,
    compute_response_statistics,
    filter_trajectories,
)

# Configuration
from .config import DEFAULT_CONFIG, OPTO_CONFIG, STIM_CONFIG, Config
from .io import BraidzData, read_braidz

# =============================================================================
# Secondary API - Kinematics functions for advanced users
# =============================================================================
from .kinematics import (
    SaccadeEvent,
    add_kinematics_to_trajectory,
    classify_flight_state,
    compute_angular_velocity,
    compute_heading_change,
    compute_linear_velocity,
    compute_mgsd_scores,
    detect_saccades,
    detect_saccades_mgsd,
    extract_flight_bouts,
    smooth_trajectory,
)

# Plotting
from .plotting import (
    add_stimulus_region,
    convert_frames_to_ms,
    create_summary_figure,
    plot_angular_velocity,
    plot_heading_comparison,
    plot_heading_distribution,
    plot_linear_velocity,
    plot_response_rate_by_group,
    plot_traces,
    plot_trajectory,
)

# =============================================================================
# Public API declaration
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Logging
    "set_log_level",
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
    "detect_saccades_mgsd",
    "compute_mgsd_scores",
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
