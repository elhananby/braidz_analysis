"""
Configuration module for braidz_analysis.

Provides a single unified Config class that consolidates all analysis parameters,
replacing the previous multiple parameter classes (SaccadeAnalysisParams,
OptoAnalysisParams, StimAnalysisParams, etc.).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """
    Unified configuration for all braidz analysis operations.

    This class consolidates all analysis parameters into a single, flat structure.
    Users can override any parameter by passing it to the constructor.

    Attributes:
        fps: Frame rate of the tracking system in Hz.
        pre_frames: Number of frames to extract before an event/saccade.
        post_frames: Number of frames to extract after an event/saccade.
        response_delay: Frames after event to wait before looking for response.
            Use this to ignore early saccades (e.g., during stimulus buildup).
        response_window: Frames after event onset to stop searching for response.
            The actual search window is [response_delay, response_window].

        saccade_threshold: Angular velocity threshold for saccade detection (deg/s).
        min_saccade_spacing: Minimum frames between detected saccades.
        heading_window: Frames before/after peak for heading change calculation.

        min_trajectory_frames: Minimum trajectory length to include in analysis.
        z_bounds: Valid vertical position range (min, max) in meters.
        max_radius: Maximum radial distance from arena center in meters.
        min_position_range: Minimum spatial extent to exclude stationary objects.

    Example:
        >>> # Use defaults
        >>> config = Config()

        >>> # Override specific parameters
        >>> config = Config(response_window=50, saccade_threshold=400)

        >>> # For visual stimuli with longer response window
        >>> stim_config = Config(response_window=50, post_frames=150)
    """

    # === Timing Parameters ===
    fps: float = 100.0
    pre_frames: int = 50
    post_frames: int = 100
    response_delay: int = 0  # frames after event before looking for response
    response_window: int = 30  # frames after event to stop looking for response

    # === Saccade Detection ===
    saccade_threshold: float = 300.0  # deg/s
    min_saccade_spacing: int = 50  # frames
    heading_window: int = 10  # frames for heading change calculation

    # === Trajectory Quality Filters ===
    min_trajectory_frames: int = 150
    z_bounds: tuple[float, float] = (0.05, 0.3)
    max_radius: float = 0.23
    min_position_range: float = 0.05  # minimum spatial extent in x, y, z

    # === Smoothing Parameters ===
    smoothing_window: int = 21  # Savitzky-Golay window size
    smoothing_polyorder: int = 3  # Savitzky-Golay polynomial order

    # === Flight State Detection ===
    flight_high_threshold: float = 0.05  # velocity to enter flight state
    flight_low_threshold: float = 0.01  # velocity to exit flight state
    flight_min_frames: int = 20  # minimum sustained frames

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.pre_frames + self.post_frames > self.min_trajectory_frames:
            raise ValueError(
                f"Extraction window ({self.pre_frames} + {self.post_frames} = "
                f"{self.pre_frames + self.post_frames} frames) cannot exceed "
                f"min_trajectory_frames ({self.min_trajectory_frames})"
            )

        if self.response_delay >= self.response_window:
            raise ValueError(
                f"response_delay ({self.response_delay}) must be less than "
                f"response_window ({self.response_window})"
            )

        if self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be odd")

        if self.smoothing_polyorder >= self.smoothing_window:
            raise ValueError("smoothing_polyorder must be less than smoothing_window")

    @property
    def dt(self) -> float:
        """Time step between frames in seconds."""
        return 1.0 / self.fps

    @property
    def saccade_threshold_rad(self) -> float:
        """Saccade threshold in radians/second."""
        import numpy as np

        return np.deg2rad(self.saccade_threshold)

    def with_overrides(self, **kwargs) -> "Config":
        """
        Create a new Config with some parameters overridden.

        Example:
            >>> base_config = Config()
            >>> stim_config = base_config.with_overrides(response_window=50)
        """
        from dataclasses import asdict

        current = asdict(self)
        current.update(kwargs)
        return Config(**current)


# Pre-configured defaults for common analysis types
DEFAULT_CONFIG = Config()

OPTO_CONFIG = Config(
    response_window=30,
    post_frames=100,
)

STIM_CONFIG = Config(
    response_window=50,
    post_frames=150,
    min_trajectory_frames=250,  # Accommodate larger extraction window
)
