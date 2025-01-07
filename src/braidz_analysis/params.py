from dataclasses import dataclass, field
import numpy as np


@dataclass
class SaccadeAnalysisParams:
    threshold: float = np.deg2rad(300)
    distance: int = 10
    heading_diff_window: int = 10
    pre_frames: int = 50
    post_frames: int = 50
    zlim: list[float] = field(default_factory=lambda: [0.05, 0.3])
    max_radius: float = 0.23


@dataclass
class OptoAnalysisParams:
    duration: float = 30.0
    min_frames: int = 150
    pre_frames: int = 50
    post_frames: int = 100
    radius: float = 0.025


@dataclass
class StimAnalysisParams:
    duration: float = 50.0
    min_frames: int = 150
    pre_frames: int = 50
    post_frames: int = 100
