# core.py

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from tqdm import tqdm


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
        heading = np.arctan2(yvel, xvel)
        angular_velocity = np.gradient(heading) * 100  # Convert to deg/s
        return heading, angular_velocity

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


# params.py


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
    min_movement: float = 0.05  # Minimum movement in any dimension

    def validate(self) -> None:
        """Validate spatial parameters."""
        if self.z_bounds[0] >= self.z_bounds[1]:
            raise ValueError("Invalid z_bounds: min must be less than max")
        if self.max_radius <= 0:
            raise ValueError("max_radius must be positive")
        if self.min_movement <= 0:
            raise ValueError("min_movement must be positive")


@dataclass
class SaccadeParams:
    """Parameters for saccade detection and analysis."""

    threshold: float = 800  # deg/s
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


# filtering.py


logger = logging.getLogger(__name__)


class TrajectoryFilter:
    """Filters trajectories based on quality criteria."""

    def __init__(self, params: AnalysisParams):
        self.params = params
        self.params.validate()

    def filter_trajectory(self, traj: TrajectoryData) -> bool:
        """
        Check if a trajectory meets quality criteria.

        Returns:
            bool: True if trajectory passes all filters
        """
        # Length check
        if len(traj.position) < self.params.base.min_frames:
            logger.debug(
                f"Trajectory {traj.obj_id} too short: {len(traj.position)} frames"
            )
            return False

        # Spatial bounds check
        z_median = np.median(traj.position[:, 2])
        if not (
            self.params.spatial.z_bounds[0]
            <= z_median
            <= self.params.spatial.z_bounds[1]
        ):
            logger.debug(f"Trajectory {traj.obj_id} outside z bounds: {z_median:.3f}")
            return False

        # Radius check
        radius = np.sqrt(np.sum(traj.position[:, :2] ** 2, axis=1))
        radius_median = np.median(radius)
        if radius_median > self.params.spatial.max_radius:
            logger.debug(
                f"Trajectory {traj.obj_id} radius too large: {radius_median:.3f}"
            )
            return False

        # Movement check
        movement = np.ptp(traj.position, axis=0)
        if any(m < self.params.spatial.min_movement for m in movement):
            logger.debug(f"Trajectory {traj.obj_id} insufficient movement: {movement}")
            return False

        return True

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter trajectories from a DataFrame.

        Args:
            df: DataFrame with trajectory data

        Returns:
            pd.DataFrame: Filtered DataFrame containing only valid trajectories
        """
        required_cols = ["x", "y", "z", "obj_id"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")

        good_trajectories = []
        groupby_cols = ["obj_id", "exp_num"] if "exp_num" in df.columns else ["obj_id"]

        for _, group_df in df.groupby(groupby_cols):
            traj = TrajectoryData.from_dataframe(group_df)
            if self.filter_trajectory(traj):
                good_trajectories.append(group_df)

        if not good_trajectories:
            logger.warning("No trajectories met the quality criteria")
            return pd.DataFrame(columns=df.columns)

        return pd.concat(good_trajectories, ignore_index=True)

    def iter_trajectories(self, df: pd.DataFrame) -> Iterator[TrajectoryData]:
        """
        Iterate over filtered trajectories.

        Args:
            df: DataFrame with trajectory data

        Yields:
            TrajectoryData: Filtered trajectory objects
        """
        groupby_cols = ["obj_id", "exp_num"] if "exp_num" in df.columns else ["obj_id"]

        for _, group_df in df.groupby(groupby_cols):
            traj = TrajectoryData.from_dataframe(group_df)
            if self.filter_trajectory(traj):
                yield traj


# analysis.py


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    angular_velocity: np.ndarray
    linear_velocity: np.ndarray
    position: np.ndarray
    heading_difference: float
    frames_in_radius: Optional[int] = None
    reaction_delay: Optional[int] = None
    responsive: bool = False
    pre_saccade_amplitude: Optional[float] = None
    pre_saccade_distance: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrajectoryAnalyzer:
    """Handles trajectory analysis operations."""

    def __init__(self, params: AnalysisParams):
        self.params = params
        self.params.validate()

    def detect_saccades(self, angular_velocity: np.ndarray) -> np.ndarray:
        """Detect saccades in angular velocity data."""
        peaks, _ = find_peaks(
            np.abs(angular_velocity),
            height=self.params.saccade.threshold,
            distance=self.params.saccade.distance,
        )
        return peaks

    def find_relevant_peak(
        self, peaks: np.ndarray, stim_idx: int, return_none: bool = False
    ) -> tuple[Optional[int], bool]:
        """Find first peak within the stimulus window."""
        window_end = stim_idx + self.params.base.duration

        for peak in peaks:
            if stim_idx < peak < window_end:
                return int(peak), True

        return (
            None if return_none else stim_idx + self.params.base.duration // 2
        ), False

    def get_pre_saccade(
        self, peaks: np.ndarray, stim_idx: int, window: int = 100
    ) -> Optional[int]:
        """Get the closest peak before the stimulation event."""
        valid_peaks = peaks[(peaks < stim_idx) & (peaks > stim_idx - window)]
        return valid_peaks[-1] if len(valid_peaks) > 0 else None

    def calculate_heading_diff(
        self, heading: np.ndarray, start_idx: int, center_idx: int, end_idx: int
    ) -> float:
        """Calculate heading difference around a point."""
        if start_idx < 0 or end_idx >= len(heading):
            return np.nan

        pre_heading = np.mean(heading[start_idx:center_idx])
        post_heading = np.mean(heading[center_idx:end_idx])
        diff = post_heading - pre_heading

        # Normalize to [-pi, pi]
        return np.arctan2(np.sin(diff), np.cos(diff))

    def analyze_segment(
        self, traj: TrajectoryData, stim_event: StimulationEvent
    ) -> Optional[AnalysisResult]:
        """Analyze a trajectory segment around a stimulation event."""
        try:
            # Find stimulation index
            stim_idx = np.where(traj.frames == stim_event.frame)[0][0]

            # Check if we have enough frames
            if (
                stim_idx < self.params.base.pre_frames
                or stim_idx >= len(traj.frames) - self.params.base.post_frames
            ):
                logger.debug("Insufficient frames around stimulation event")
                return None

            # Detect saccades
            peaks = self.detect_saccades(traj.angular_velocity)
            peak_idx, is_responsive = self.find_relevant_peak(peaks, stim_idx)

            if peak_idx is None:
                logger.debug("No relevant peak found")
                return None

            # Extract segment around peak
            segment_start = peak_idx - self.params.base.pre_frames
            segment_end = peak_idx + self.params.base.post_frames
            segment = traj.get_segment(segment_start, segment_end)

            # Calculate heading difference
            heading_diff = self.calculate_heading_diff(
                traj.heading,
                peak_idx - self.params.saccade.heading_diff_window,
                peak_idx,
                peak_idx + self.params.saccade.heading_diff_window,
            )

            # Find pre-saccade if any
            pre_saccade_idx = self.get_pre_saccade(peaks, stim_idx)

            # Calculate frames in radius if applicable
            frames_in_radius = None
            if hasattr(self.params.spatial, "radius"):
                radius = np.sqrt(
                    np.sum(
                        traj.position[
                            stim_idx : stim_idx + self.params.base.duration, :2
                        ]
                        ** 2,
                        axis=1,
                    )
                )
                frames_in_radius = np.sum(radius < self.params.spatial.max_radius)

            return AnalysisResult(
                angular_velocity=segment.angular_velocity,
                linear_velocity=segment.linear_velocity,
                position=segment.position,
                heading_difference=heading_diff,
                frames_in_radius=frames_in_radius,
                reaction_delay=peak_idx - stim_idx if is_responsive else None,
                responsive=is_responsive,
                pre_saccade_amplitude=traj.angular_velocity[pre_saccade_idx]
                if pre_saccade_idx is not None
                else None,
                pre_saccade_distance=pre_saccade_idx - stim_idx
                if pre_saccade_idx is not None
                else None,
                metadata={
                    "obj_id": traj.obj_id,
                    "exp_num": traj.exp_num,
                    "stim_frame": stim_event.frame,
                    "stim_type": stim_event.event_type,
                    "intensity": stim_event.intensity,
                    "duration": stim_event.duration,
                    "frequency": stim_event.frequency,
                    "sham": stim_event.sham,
                },
            )

        except (IndexError, ValueError) as e:
            logger.debug(f"Error analyzing segment: {str(e)}")
            return None


# interface.py

from dataclasses import asdict
from typing import Dict, Optional
import pandas as pd
import numpy as np


class TrajectoryAnalysisManager:
    """Main interface for trajectory analysis."""

    def __init__(
        self,
        analysis_params: Optional[AnalysisParams] = None,
        show_progress: bool = True,
    ):
        """Initialize with optional custom parameters."""
        self.params = analysis_params or AnalysisParams()
        self.params.validate()
        self.show_progress = show_progress

        self.filter = TrajectoryFilter(self.params)
        self.analyzer = TrajectoryAnalyzer(self.params)

    def process_dataset(
        self,
        trajectory_df: pd.DataFrame,
        stim_df: pd.DataFrame,
        stim_type: str = "opto",
    ) -> List[AnalysisResult]:
        """
        Process a complete dataset of trajectories and stimulation events.

        Args:
            trajectory_df: DataFrame containing trajectory data
            stim_df: DataFrame containing stimulation events
            stim_type: Type of stimulation ("opto" or "stim")

        Returns:
            List[AnalysisResult]: List of analysis results
        """
        # Filter trajectories
        filtered_df = self.filter.filter_dataframe(trajectory_df)
        if filtered_df.empty:
            logger.warning("No valid trajectories after filtering")
            return []

        results = []
        stim_iter = (
            tqdm(stim_df.iterrows(), total=len(stim_df))
            if self.show_progress
            else stim_df.iterrows()
        )

        for _, stim_row in stim_iter:
            # Create stimulation event
            stim_event = StimulationEvent.from_series(stim_row, stim_type)

            # Get corresponding trajectory data
            traj_mask = (
                (filtered_df["obj_id"] == stim_event.obj_id)
                & (filtered_df["exp_num"] == stim_event.exp_num)
                if stim_event.exp_num is not None
                else filtered_df["obj_id"] == stim_event.obj_id
            )
            traj_data = TrajectoryData.from_dataframe(filtered_df[traj_mask])

            # Analyze trajectory
            result = self.analyzer.analyze_segment(traj_data, stim_event)
            if result is not None:
                results.append(result)

        return results

    def convert_to_legacy_format(self, results: List[AnalysisResult]) -> Dict:
        """
        Convert analysis results to the old dictionary format for backwards compatibility.

        Args:
            results: List of AnalysisResult objects

        Returns:
            Dict containing results in the old format
        """
        legacy_dict = {
            "angular_velocity": [],
            "linear_velocity": [],
            "position": [],
            "heading_difference": [],
            "frames_in_radius": [],
            "sham": [],
            "intensity": [],
            "duration": [],
            "frequency": [],
            "responsive": [],
            "timestamp": [],
            "reaction_delay": [],
            "pre_saccade_amplitude": [],
            "pre_saccade_distance": [],
        }

        for result in results:
            result_dict = asdict(result)
            metadata = result_dict.pop("metadata")

            for key in legacy_dict:
                if key in result_dict:
                    legacy_dict[key].append(result_dict[key])
                elif key in metadata:
                    legacy_dict[key].append(metadata[key])

        return legacy_dict


# utils.py


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
