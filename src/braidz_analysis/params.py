"""
DEPRECATED: This module is deprecated. Use braidz_analysis.config instead.

Migration:
    Old: ba.params.OptoAnalysisParams(duration=30, radius=0.025)
    New: ba.Config(response_window=30)

The new Config class unifies all parameter classes into one.
This module will be removed in version 1.0.
"""
import warnings

warnings.warn(
    "braidz_analysis.params is deprecated. Use braidz_analysis.config instead. "
    "See README.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SaccadeAnalysisParams:
    threshold: float = np.deg2rad(300)
    distance: int = 50
    heading_diff_window: int = 10
    pre_frames: int = 50
    post_frames: int = 50
    zlim: list[float] = field(default_factory=lambda: [0.05, 0.3])
    max_radius: float = 0.23


@dataclass
class BaseAnalysisParams:
    min_frames: int = 150
    pre_frames: int = 50
    post_frames: int = 100

    def __post_init__(self):
        if self.pre_frames + self.post_frames > self.min_frames:
            raise ValueError(
                f"Total frame window ({self.pre_frames + self.post_frames}) "
                f"cannot be larger than minimum frames ({self.min_frames})"
            )


@dataclass
class OptoAnalysisParams(BaseAnalysisParams):
    duration: float = 30.0
    radius: float = 0.025


@dataclass
class StimAnalysisParams(BaseAnalysisParams):
    duration: float = 50.0
