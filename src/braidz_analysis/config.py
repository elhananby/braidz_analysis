"""
Configuration module for braidz_analysis.

Provides a unified Config class that consolidates all analysis parameters.
Parameters are organized into logical sections for clarity.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    """
    Unified configuration for all braidz analysis operations.

    Parameters are organized into sections:
        - Recording: Basic recording parameters (frame rate)
        - Saccade Detection: Algorithm selection and tuning
        - Event Analysis: Response window and timing
        - Trajectory Filtering: Quality control thresholds
        - Smoothing: Signal processing parameters
        - Flight State: Walking vs flying classification

    Example:
        >>> # Use defaults (velocity-based saccade detection)
        >>> config = Config()

        >>> # Use mGSD algorithm with custom threshold
        >>> config = Config(saccade_method="mgsd", mgsd_threshold=0.0005)

        >>> # Override from existing config
        >>> new_config = config.with_overrides(mgsd_threshold=0.001)

    See Also:
        DEFAULT_CONFIG: Standard defaults for most analyses
        OPTO_CONFIG: Optimized for optogenetic experiments
        STIM_CONFIG: Optimized for visual stimulus experiments
        MGSD_CONFIG: Defaults using mGSD saccade detection
    """

    # =========================================================================
    # Recording Parameters
    # =========================================================================

    fps: float = 100.0
    """Frame rate of the tracking system in Hz."""

    # =========================================================================
    # Saccade Detection - Algorithm Selection
    # =========================================================================

    saccade_method: Literal["velocity", "mgsd"] = "velocity"
    """
    Saccade detection algorithm to use.

    Options:
        - "velocity": Threshold-based detection on angular velocity peaks.
            Fast and works well for clear, sharp saccades.
        - "mgsd": Modified Geometric Saccade Detection algorithm.
            Uses trajectory geometry; better for slow or noisy data.
    """

    # =========================================================================
    # Saccade Detection - Velocity Method Parameters
    # =========================================================================

    saccade_threshold: float = 300.0
    """Angular velocity threshold for saccade detection (deg/s)."""

    min_saccade_spacing: int = 50
    """Minimum frames between detected saccades."""

    heading_window: int = 10
    """Frames before/after saccade peak for heading change calculation."""

    # =========================================================================
    # Saccade Detection - mGSD Method Parameters
    # =========================================================================

    mgsd_delta_frames: int = 5
    """
    Window size for mGSD algorithm (frames before/after current frame).
    At 100 Hz, delta_frames=5 gives a 100ms window (50ms before + 50ms after).
    """

    mgsd_threshold: float = 0.001
    """
    Minimum mGSD score for peak detection.
    The score is amplitude² × dispersion, so units depend on dispersion_method.
    Typical values: 0.0001-0.01 for 'mean', 0.001-0.1 for 'sum'.
    """

    mgsd_min_spacing: int = 10
    """Minimum frames between mGSD-detected saccades."""

    mgsd_dispersion: Literal["sum", "mean", "std"] = "sum"
    """
    How to compute dispersion in mGSD algorithm.

    Options:
        - "sum": Sum of distances (original implementation).
            Score scales with window size.
        - "mean": Mean distance.
            More consistent across different delta_frames values.
        - "std": Standard deviation of distances (as in paper).
            Most robust to window size changes.
    """

    # =========================================================================
    # Event/Response Analysis Parameters
    # =========================================================================

    pre_frames: int = 50
    """Frames to extract before an event/saccade for trace alignment."""

    post_frames: int = 100
    """Frames to extract after an event/saccade for trace alignment."""

    response_delay: int = 0
    """
    Frames after event onset before looking for response saccades.
    Use to ignore early activity (e.g., during stimulus buildup).
    """

    response_window: int = 30
    """
    Frames after event onset to search for response saccades.
    The search window is [response_delay, response_window].
    """

    detect_in_window_only: bool = False
    """
    If True, run saccade detection only within the response window.
    More sensitive to responses that might be suppressed by min_saccade_spacing
    due to earlier saccades outside the window.
    """

    # =========================================================================
    # Trajectory Quality Filters
    # =========================================================================

    min_trajectory_frames: int = 150
    """Minimum trajectory length (frames) to include in analysis."""

    z_bounds: tuple[float, float] = (0.05, 0.3)
    """Valid vertical position range (min, max) in meters."""

    max_radius: float = 0.23
    """Maximum radial distance from arena center in meters."""

    min_position_range: float = 0.05
    """Minimum spatial extent in x, y, z to exclude stationary objects (meters)."""

    # =========================================================================
    # Smoothing Parameters
    # =========================================================================

    smoothing_window: int = 21
    """Savitzky-Golay filter window size (must be odd)."""

    smoothing_polyorder: int = 3
    """Savitzky-Golay polynomial order (must be < smoothing_window)."""

    # =========================================================================
    # Flight State Detection
    # =========================================================================

    flight_high_threshold: float = 0.05
    """Linear velocity (m/s) to enter flight state."""

    flight_low_threshold: float = 0.01
    """Linear velocity (m/s) to exit flight state."""

    flight_min_frames: int = 20
    """Minimum sustained frames to trigger flight state change."""

    # =========================================================================
    # Validation
    # =========================================================================

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate extraction window
        if self.pre_frames + self.post_frames > self.min_trajectory_frames:
            raise ValueError(
                f"Extraction window ({self.pre_frames} + {self.post_frames} = "
                f"{self.pre_frames + self.post_frames} frames) cannot exceed "
                f"min_trajectory_frames ({self.min_trajectory_frames})"
            )

        # Validate response window
        if self.response_delay >= self.response_window:
            raise ValueError(
                f"response_delay ({self.response_delay}) must be less than "
                f"response_window ({self.response_window})"
            )

        # Validate smoothing
        if self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be odd")

        if self.smoothing_polyorder >= self.smoothing_window:
            raise ValueError("smoothing_polyorder must be less than smoothing_window")

        # Validate saccade method
        if self.saccade_method not in ("velocity", "mgsd"):
            raise ValueError(
                f"saccade_method must be 'velocity' or 'mgsd', got '{self.saccade_method}'"
            )

        # Validate mGSD dispersion method
        if self.mgsd_dispersion not in ("sum", "mean", "std"):
            raise ValueError(
                f"mgsd_dispersion must be 'sum', 'mean', or 'std', got '{self.mgsd_dispersion}'"
            )

        # Validate positive values
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.saccade_threshold <= 0:
            raise ValueError(f"saccade_threshold must be positive, got {self.saccade_threshold}")

        if self.mgsd_threshold <= 0:
            raise ValueError(f"mgsd_threshold must be positive, got {self.mgsd_threshold}")

    # =========================================================================
    # Computed Properties
    # =========================================================================

    @property
    def dt(self) -> float:
        """Time step between frames in seconds."""
        return 1.0 / self.fps

    @property
    def saccade_threshold_rad(self) -> float:
        """Saccade threshold in radians/second."""
        import numpy as np

        return np.deg2rad(self.saccade_threshold)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    def with_overrides(self, **kwargs) -> "Config":
        """
        Create a new Config with some parameters overridden.

        Example:
            >>> base_config = Config()
            >>> mgsd_config = base_config.with_overrides(saccade_method="mgsd")
        """
        from dataclasses import asdict

        current = asdict(self)
        current.update(kwargs)
        return Config(**current)


# =============================================================================
# Pre-configured Defaults
# =============================================================================

DEFAULT_CONFIG = Config()
"""Standard defaults using velocity-based saccade detection."""

OPTO_CONFIG = Config(
    response_window=30,
    post_frames=100,
)
"""Optimized for optogenetic stimulation experiments."""

STIM_CONFIG = Config(
    response_window=50,
    post_frames=150,
    min_trajectory_frames=250,
)
"""Optimized for visual stimulus experiments (longer response windows)."""

MGSD_CONFIG = Config(
    saccade_method="mgsd",
    mgsd_dispersion="mean",
    mgsd_threshold=0.0005,
)
"""Defaults using mGSD saccade detection with mean dispersion."""
