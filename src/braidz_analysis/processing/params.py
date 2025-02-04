# params.py

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass
class BaseAnalysisParams:
    """Base parameters shared across different analysis types."""

    min_frames: int = 300
    pre_frames: int = 50
    post_frames: int = 50
    duration: int = 30

    def validate(self) -> None:
        """Validate parameter values."""
        if self.min_frames < 0:
            raise ValueError("min_frames must be positive")
        if self.pre_frames < 0 or self.post_frames < 0:
            raise ValueError("frame windows must be positive")
        if self.duration < 0:
            raise ValueError("duration must be positive")


@dataclass
class SpatialParams:
    """Parameters for spatial filtering and analysis."""

    z_bounds: Tuple[float, float] = (0.1, 0.3)
    max_radius: float = 0.23
    opto_radius: float = 0.05
    min_movement: float = 0.05  # Minimum movement in any dimension

    def validate(self) -> None:
        """Validate spatial parameters."""
        if self.z_bounds[0] >= self.z_bounds[1]:
            raise ValueError("Invalid z_bounds: min must be less than max")
        if self.max_radius <= 0:
            raise ValueError("max_radius must be positive")
        if self.min_movement <= 0:
            raise ValueError("min_movement must be positive")
        if self.opto_radius <= 0:
            raise ValueError("opto_radius must be positive")


@dataclass
class SaccadeParams:
    """Parameters for saccade detection and analysis."""

    threshold: float = np.deg2rad(300)  # rad/s
    distance: int = 10  # frames
    heading_diff_window: int = 5

    def validate(self) -> None:
        """Validate saccade parameters."""
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if self.distance <= 0:
            raise ValueError("distance must be positive")
        if self.heading_diff_window <= 0:
            raise ValueError("heading_diff_window must be positive")


@dataclass
class AnalysisParams:
    """Complete set of analysis parameters."""

    base: BaseAnalysisParams = field(default_factory=BaseAnalysisParams)
    spatial: SpatialParams = field(default_factory=SpatialParams)
    saccade: SaccadeParams = field(default_factory=SaccadeParams)

    def validate(self) -> None:
        """Validate all parameters."""
        self.base.validate()
        self.spatial.validate()
        self.saccade.validate()
