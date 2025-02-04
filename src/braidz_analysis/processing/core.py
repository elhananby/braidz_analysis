# core.py

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from pynumdiff.smooth_finite_difference import butterdiff


@dataclass
class TrajectoryData:
    """
    Represents a single trajectory with all its computed properties.

    Attributes:
        position: Array of shape (n_frames, 3) containing x,y,z coordinates
        velocity: Array of shape (n_frames, 3) containing velocity components
        obj_id: Object identifier
        exp_num: Optional experiment number
        frames: Array of frame numbers
        _cache: Dictionary for storing computed properties
    """

    position: np.ndarray
    velocity: np.ndarray
    obj_id: int
    exp_num: Optional[int]
    frames: np.ndarray
    _cache: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "TrajectoryData":
        """Create trajectory data from a DataFrame segment."""
        position = df[["x", "y", "z"]].values
        velocity = df[["xvel", "yvel", "zvel"]].values
        obj_id = df["obj_id"].iloc[0]
        exp_num = df["exp_num"].iloc[0] if "exp_num" in df.columns else None
        frames = df["frame"].values

        return cls(position, velocity, obj_id, exp_num, frames)

    def get_segment(self, start_idx: int, end_idx: int) -> "TrajectoryData":
        """Extract a segment of the trajectory."""
        return TrajectoryData(
            position=self.position[start_idx:end_idx],
            velocity=self.velocity[start_idx:end_idx],
            obj_id=self.obj_id,
            exp_num=self.exp_num,
            frames=self.frames[start_idx:end_idx],
        )

    @property
    def angular_velocity(self) -> np.ndarray:
        """Calculate and cache angular velocity."""
        if "angular_velocity" not in self._cache:
            heading, ang_vel = self._calculate_angular_velocity()
            self._cache["heading"] = heading
            self._cache["angular_velocity"] = ang_vel
        return self._cache["angular_velocity"]

    @property
    def heading(self) -> np.ndarray:
        """Get cached heading angles."""
        if "heading" not in self._cache:
            heading, ang_vel = self._calculate_angular_velocity()
            self._cache["heading"] = heading
            self._cache["angular_velocity"] = ang_vel
        return self._cache["heading"]

    @property
    def linear_velocity(self) -> np.ndarray:
        """Calculate and cache linear velocity."""
        if "linear_velocity" not in self._cache:
            self._cache["linear_velocity"] = self._calculate_linear_velocity()
        return self._cache["linear_velocity"]

    def _calculate_angular_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate heading and angular velocity."""
        xvel, yvel = self.velocity[:, 0], self.velocity[:, 1]
        theta = np.arctan2(yvel, xvel)
        theta_unwrap = np.unwrap(theta)
        _, angular_velocity = butterdiff(theta_unwrap, dt=0.01, params=[1, 0.1])
        return theta, angular_velocity

    def _calculate_linear_velocity(self) -> np.ndarray:
        """Calculate linear velocity magnitude."""
        xvel, yvel = self.velocity[:, 0], self.velocity[:, 1]
        return np.sqrt(xvel**2 + yvel**2)


@dataclass
class StimulationEvent:
    """
    Represents a single stimulation/opto event.

    Attributes:
        frame: Frame number when the event occurred
        obj_id: Object identifier
        exp_num: Optional experiment number
        event_type: Type of stimulation ('opto' or 'stim')
        intensity: Optional stimulation intensity
        duration: Optional stimulation duration
        frequency: Optional stimulation frequency
        sham: Whether this was a sham stimulation
    """

    frame: int
    obj_id: int
    exp_num: Optional[int]
    event_type: str  # 'opto' or 'stim'
    intensity: Optional[float] = None
    duration: Optional[float] = None
    frequency: Optional[float] = None
    sham: bool = False

    @classmethod
    def from_series(cls, row: pd.Series, event_type: str) -> "StimulationEvent":
        """Create stimulation event from a pandas Series."""
        return cls(
            frame=int(row["frame"]),
            obj_id=int(row["obj_id"]),
            exp_num=int(row["exp_num"]) if "exp_num" in row else None,
            event_type=event_type,
            intensity=row.get("intensity"),
            duration=row.get("duration"),
            frequency=row.get("frequency"),
            sham=row.get("sham", False),
        )
