import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from tqdm import tqdm
import inspect

from .trajectory import (
    calculate_angular_velocity,
    calculate_linear_velocity,
    detect_saccades,
    heading_diff,
)

# Constants for trajectory analysis
MAX_RADIUS = 0.23  # Maximum allowed radius for valid trajectories
Z_RANGE = [0.05, 0.3]  # Valid range for z-coordinate [min, max]


class SaccadeType(str, Enum):
    SPONTANEOUS = "spontaneous"
    STIM_OR_OPTO = "stim_or_opto"
    NO_RESPONSE = "stim_or_opto_no_response"


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class TrajectoryAnalysisResults:
    """Results from trajectory analysis.

    Contains all possible metrics that could be returned from trajectory analysis functions.
    Not all fields will be populated by every analysis function.

    Attributes:
        angular_velocity: Array of angular velocities (n_samples, n_timepoints)
        linear_velocity: Array of linear velocities (n_samples, n_timepoints)
        xyz: Array of positions (n_samples, n_timepoints, 3)
        heading_diff: Array of heading differences (n_samples, n_timepoints)
        saccade_type: Array of saccade type labels (n_samples,)
        sham: Array of boolean sham indicators (n_samples,)
        frames_in_opto_radius: Array of frame counts within opto radius (n_samples,)
    """

    angular_velocity: np.ndarray
    linear_velocity: np.ndarray
    xyz: np.ndarray
    heading_diff: np.ndarray
    saccade_type: np.ndarray | None = None  # Optional fields that aren't always present
    sham: np.ndarray | None = None
    frames_in_opto_radius: np.ndarray | None = None

    def __post_init__(self):
        """Validate array shapes after initialization."""
        n_samples = len(self.angular_velocity)
        assert len(self.linear_velocity) == n_samples, "Mismatched sample counts"
        assert len(self.xyz) == n_samples, "Mismatched sample counts"
        assert len(self.heading_diff) == n_samples, "Mismatched sample counts"

        if self.saccade_type is not None:
            assert len(self.saccade_type) == n_samples, "Mismatched sample counts"
        if self.sham is not None:
            assert len(self.sham) == n_samples, "Mismatched sample counts"
        if self.frames_in_opto_radius is not None:
            assert (
                len(self.frames_in_opto_radius) == n_samples
            ), "Mismatched sample counts"


@dataclass
class AnalysisParams:
    """Parameters for trajectory analysis.

    Attributes:
        pre_frames (int): Number of frames to analyze before event
        post_frames (int): Number of frames to analyze after event
        min_frames (int): Minimum number of frames required for analysis
        opto_radius (float): Radius threshold for optogenetic stimulation
        opto_duration (int): Duration of optogenetic stimulation in frames
    """

    pre_frames: int = 50
    post_frames: int = 100
    min_frames: int = 300
    opto_radius: float = 0.025
    opto_duration: int = 30

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
        )


@dataclass
class SaccadeParams:
    """Parameters for saccade detection.

    Attributes:
        threshold (float): Angular velocity threshold for saccade detection
        distance (int): Minimum distance between saccades in frames
    """

    threshold: float = np.deg2rad(300)
    distance: int = 10

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
        )


def validate_frame_range(
    frame_idx: int, total_frames: int, pre_frames: int, post_frames: int
) -> bool:
    """Validate if there are enough frames before and after the index for analysis.

    Args:
        frame_idx: Index of the frame to analyze
        total_frames: Total number of frames in dataset
        pre_frames: Number of frames needed before the index
        post_frames: Number of frames needed after the index

    Returns:
        bool: True if frame range is valid, False otherwise
    """
    if frame_idx - pre_frames < 0 or frame_idx + post_frames >= total_frames:
        return False  # Invalid frame range
    else:
        return True  # Valid frame range


def get_frame_range(center_idx: int, pre_frames: int, post_frames: int) -> range:
    """Get range of frames to analyze around a center point.

    Args:
        center_idx: Index of the central frame
        pre_frames: Number of frames to include before center
        post_frames: Number of frames to include after center

    Returns:
        range: Range object spanning the analysis window
    """
    return range(center_idx - pre_frames, center_idx + post_frames)


def check_flying_bounds(x: float, y: float, z: float) -> bool:
    """Check if position is within acceptable flying bounds.

    Args:
        x: X-coordinate position
        y: Y-coordinate position
        z: Z-coordinate position

    Returns:
        bool: True if position is within bounds, False otherwise
    """
    radius = np.sqrt(x**2 + y**2)
    return radius <= MAX_RADIUS and Z_RANGE[0] <= z <= Z_RANGE[1]


def calculate_trajectory_metrics(grp: pd.DataFrame) -> tuple:
    """Calculate basic trajectory metrics from velocity data.

    Args:
        grp: DataFrame containing 'xvel' and 'yvel' columns

    Returns:
        tuple: (heading, angular_velocity, linear_velocity)
    """
    # Calculate heading and angular velocity from x/y velocities
    heading, angular_velocity = calculate_angular_velocity(
        grp["xvel"].values, grp["yvel"].values
    )

    # Calculate linear velocity magnitude
    linear_velocity = calculate_linear_velocity(grp["xvel"].values, grp["yvel"].values)

    return heading, angular_velocity, linear_velocity


def calculate_radius_metrics(
    grp: pd.DataFrame, stim_idx: int, params: AnalysisParams
) -> int:
    """Calculate metrics related to radius around stimulus point.

    Args:
        grp: DataFrame containing position data
        stim_idx: Index of stimulation frame
        params: Analysis parameters

    Returns:
        int: Number of frames where trajectory is within opto radius
    """
    # Calculate radial distance from origin
    radius = np.sqrt(grp["x"].values ** 2 + grp["y"].values ** 2)

    # Count frames within opto radius during stimulation period
    return np.sum(
        radius[stim_idx : stim_idx + params.opto_duration] < params.opto_radius
    )


def determine_saccade_type(
    sac_idx: int, stim_indices: np.ndarray, opto_duration: int
) -> str:
    """Determine if saccade occurred during stimulation period using vectorized operations.

    Args:
        sac_idx: Index of the saccade frame
        stim_indices: Array of stimulation start frame indices
        opto_duration: Duration of stimulation period in frames

    Returns:
        str: 'stim_or_opto' if saccade occurred during stimulation, 'spontaneous' otherwise
    """
    if len(stim_indices) == 0:
        return "spontaneous"

    # Create arrays of start and end indices for each stim period
    stim_starts = stim_indices
    stim_ends = stim_indices + opto_duration

    # Vectorized check if saccade falls within any stim period
    is_during_stim = np.any((sac_idx >= stim_starts) & (sac_idx < stim_ends))

    return "stim_or_opto" if is_during_stim else "spontaneous"


def process_stim_data(grp: pd.DataFrame, stim_idx: int, params: AnalysisParams) -> dict:
    """Process trajectory data around a stimulus point.

    Args:
        grp: DataFrame containing trajectory data
        stim_idx: Index of stimulation frame
        params: Analysis parameters

    Returns:
        dict: Processed metrics or None if invalid
    """
    if not validate_frame_range(
        stim_idx, len(grp), params.pre_frames, params.post_frames
    ):
        return None

    heading, angular_velocity, linear_velocity = calculate_trajectory_metrics(grp)
    stim_range = get_frame_range(stim_idx, params.pre_frames, params.post_frames)

    return {
        "angular_velocity": angular_velocity[stim_range],
        "linear_velocity": linear_velocity[stim_range],
        "heading_diff": heading_diff(heading, stim_idx, window=25),
        "xyz": grp[["x", "y", "z"]].values[stim_range],
        "frames_in_radius": calculate_radius_metrics(grp, stim_idx, params),
    }


def check_median_position(
    grp: pd.DataFrame,
    x_range: tuple[float, float] = (-0.2, 0.2),
    y_range: tuple[float, float] = (-0.2, 0.2),
    z_range: tuple[float, float] = (0.1, 0.25),  # More restricted than Z_RANGE
) -> bool:
    """Check if median position is within central region of arena.

    Verifies if the median x,y,z coordinates fall within a specified central region,
    useful for ensuring trajectory segments are centered in the arena.

    Args:
        grp: DataFrame containing 'x', 'y', 'z' position columns
        xy_radius: Maximum allowed radius from center in x-y plane
        z_range: Tuple of (min_z, max_z) defining valid z-coordinate range

    Returns:
        bool: True if median position is within specified bounds

    Example:
        >>> df = pd.DataFrame({
        ...     'x': [0.05, 0.06, 0.07],
        ...     'y': [-0.02, -0.03, -0.02],
        ...     'z': [0.15, 0.16, 0.15]
        ... })
        >>> is_centered = check_median_position(df, xy_radius=0.1, z_range=(0.1, 0.25))
    """
    # Calculate median position
    median_x = np.median(grp["x"].values)
    median_y = np.median(grp["y"].values)
    median_z = np.median(grp["z"].values)

    # Check if median position is within bounds
    return (
        x_range[0] <= median_x <= x_range[1]
        and y_range[0] <= median_y <= y_range[1]
        and z_range[0] <= median_z <= z_range[1]
    )


def process_saccade_data(
    grp: pd.DataFrame, sac_idx: int, stim_indices: list, params: AnalysisParams
) -> dict:
    """Process trajectory data around a saccade point.

    Args:
        grp: DataFrame containing trajectory data
        sac_idx: Index of saccade frame
        stim_indices: List of stimulation frame indices
        params: Analysis parameters

    Returns:
        dict: Processed metrics or None if invalid
    """
    if not validate_frame_range(
        sac_idx, len(grp), params.pre_frames, params.post_frames
    ):
        return None

    # Validate position is within flying bounds
    x, y, z = grp[["x", "y", "z"]].values[sac_idx]
    if not check_flying_bounds(x, y, z):
        return None

    heading, angular_velocity, linear_velocity = calculate_trajectory_metrics(grp)
    sac_range = get_frame_range(sac_idx, params.pre_frames, params.post_frames)

    # Use vectorized saccade type determination
    saccade_type = determine_saccade_type(
        sac_idx, np.array(stim_indices), params.opto_duration
    )

    return {
        "angular_velocity": angular_velocity[sac_range],
        "linear_velocity": linear_velocity[sac_range],
        "xyz": grp[["x", "y", "z"]].values[sac_range],
        "heading_diff": heading_diff(heading, sac_idx, window=25),
        "saccade_type": saccade_type,
    }


def get_stim_or_opto_data(
    df: pd.DataFrame, stim_or_opto: pd.DataFrame, **kwargs
) -> dict:
    """Analyze trajectory data around stimulus or optogenetic manipulation points.

    Args:
        df: DataFrame containing trajectory data
        stim_or_opto: DataFrame containing stimulation timing information
        **kwargs: Additional parameters for analysis

    Returns:
        dict: Dictionary of numpy arrays containing processed metrics
    """
    params = AnalysisParams.from_dict(kwargs)
    results = {
        "angular_velocity": [],
        "linear_velocity": [],
        "heading_diff": [],
        "xyz": [],
        "sham": [],
        "frames_in_opto_radius": [],
    }

    # Process each stimulation event
    for _, row in tqdm(stim_or_opto.iterrows(), total=len(stim_or_opto)):
        obj_id = row["obj_id"]
        exp_num = row.get("exp_num")

        # Filter dataframe for current object/experiment
        grp = df[(df["obj_id"] == obj_id)]
        if exp_num is not None:
            grp = grp[grp["exp_num"] == exp_num]

        if len(grp) < params.min_frames:
            logger.debug(f"Skipping {obj_id} with {len(grp)} frames")
            continue

        if not check_median_position(grp):
            logger.debug(f"Skipping {obj_id} with invalid median position")
            continue

        try:
            stim_idx = np.where(grp["frame"] == row["frame"])[0][0]
        except IndexError:
            logger.debug(f"Skipping {obj_id} with no frame {row['frame']}")
            continue

        processed_data = process_stim_data(grp, stim_idx, params)
        if processed_data is None:
            continue

        # Collect results
        for key in results:
            if key == "sham":
                results[key].append(row.get("is_sham", False))
            else:
                results[key].append(processed_data.get(key, None))

    return {k: np.array(v) for k, v in results.items()}


def get_all_saccades(
    df: pd.DataFrame,
    stim_or_opto: pd.DataFrame = None,
    use_opto_or_stim_only: bool = False,
    latency: int = 15,  # Default latency for no-response windows
    **kwargs,
) -> TrajectoryAnalysisResults:
    """Analyze all saccades in trajectory data with optional stim/opto event processing.

    Args:
        df: DataFrame containing trajectory data
        stim_or_opto: Optional DataFrame containing stimulation timing information
        use_opto_or_stim_only: If True, only process groups present in stim_or_opto
        latency: Number of frames to delay window center for no-response cases
        **kwargs: Additional parameters for analysis

    Returns:
        TrajectoryAnalysisResults: Processed trajectory metrics and saccade information
    """
    saccade_params = SaccadeParams.from_dict(kwargs)
    analysis_params = AnalysisParams.from_dict(kwargs)

    results = {
        "angular_velocity": [],
        "linear_velocity": [],
        "xyz": [],
        "heading_diff": [],
        "saccade_type": [],
        "frames_in_opto_radius": [],
    }

    # Determine which groups to process
    if use_opto_or_stim_only and stim_or_opto is not None:
        # Only process groups from stim_or_opto
        unique_groups = stim_or_opto[["obj_id", "exp_num"]].drop_duplicates()
    else:
        # Process all groups from df
        unique_groups = df[["obj_id", "exp_num"]].drop_duplicates()

    for _, group_info in tqdm(unique_groups.iterrows(), total=len(unique_groups)):
        obj_id = group_info["obj_id"]
        exp_num = group_info["exp_num"]

        # Get trajectory group
        grp = df[(df["obj_id"] == obj_id) & (df["exp_num"] == exp_num)]

        # Basic validation
        if len(grp) < analysis_params.min_frames:
            logger.debug(f"Skipping {obj_id} with {len(grp)} frames")
            continue

        # Check median position only when processing all groups
        if not use_opto_or_stim_only and not check_median_position(grp):
            logger.debug(f"Skipping {obj_id} with invalid median position")
            continue

        # Calculate trajectory metrics and detect saccades
        heading, angular_velocity, linear_velocity = calculate_trajectory_metrics(grp)
        saccades = detect_saccades(
            angular_velocity,
            height=saccade_params.threshold,
            distance=saccade_params.distance,
        )

        # Get stim/opto events for this group if available
        stim_events = []
        frames_in_radius = np.nan
        if stim_or_opto is not None:
            stim_grp = stim_or_opto[
                (stim_or_opto["obj_id"] == obj_id)
                & (stim_or_opto["exp_num"] == exp_num)
            ]
            for _, row in stim_grp.iterrows():
                try:
                    stim_idx = np.where(grp["frame"] == row["frame"])[0][0]
                    frames_in_radius = calculate_radius_metrics(
                        grp, stim_idx, analysis_params
                    )
                    stim_events.append((stim_idx, frames_in_radius))
                except IndexError:
                    logger.debug(f"No stim frame {row['frame']} found for {obj_id}")

        # Process saccades
        for sac_idx in saccades:
            # Validate frame range (not overflowing start/end of trajectory)
            if not validate_frame_range(
                sac_idx,
                len(grp),
                analysis_params.pre_frames,
                analysis_params.post_frames,
            ):
                continue

            # Check flying bounds (within central region of arena)
            x, y, z = grp[["x", "y", "z"]].values[sac_idx]
            if not check_flying_bounds(x, y, z):
                continue

            # Determine saccade type and which stim event it belongs to
            saccade_type = SaccadeType.SPONTANEOUS
            current_frames_in_radius = np.nan

            for stim_idx, radius_frames in stim_events:
                if stim_idx < sac_idx < stim_idx + analysis_params.opto_duration:
                    saccade_type = SaccadeType.STIM_OR_OPTO
                    current_frames_in_radius = radius_frames
                    # Remove this stim event as we found its first saccade
                    stim_events = [(s, r) for s, r in stim_events if s != stim_idx]
                    break

            # Get frame range for analysis
            sac_range = get_frame_range(
                sac_idx, analysis_params.pre_frames, analysis_params.post_frames
            )

            # Collect saccade results
            results["angular_velocity"].append(angular_velocity[sac_range])
            results["linear_velocity"].append(linear_velocity[sac_range])
            results["xyz"].append(grp[["x", "y", "z"]].values[sac_range])
            results["heading_diff"].append(heading_diff(heading, sac_idx, window=25))
            results["saccade_type"].append(saccade_type)
            results["frames_in_opto_radius"].append(current_frames_in_radius)

        # Process remaining stim events with no response
        for stim_idx, radius_frames in stim_events:
            # Center window at stim_idx + latency
            window_center = stim_idx + latency

            # Validate frame range and flying bounds
            if not validate_frame_range(
                window_center,
                len(grp),
                analysis_params.pre_frames,
                analysis_params.post_frames,
            ):
                continue

            x, y, z = grp[["x", "y", "z"]].values[window_center]
            if not check_flying_bounds(x, y, z):
                continue

            # Get frame range for analysis
            window_range = get_frame_range(
                window_center, analysis_params.pre_frames, analysis_params.post_frames
            )

            # Collect no-response results
            results["angular_velocity"].append(angular_velocity[window_range])
            results["linear_velocity"].append(linear_velocity[window_range])
            results["xyz"].append(grp[["x", "y", "z"]].values[window_range])
            results["heading_diff"].append(
                heading_diff(heading, window_center, window=25)
            )
            results["saccade_type"].append(SaccadeType.NO_RESPONSE)
            results["frames_in_opto_radius"].append(radius_frames)

    # Convert results to numpy arrays and return
    return TrajectoryAnalysisResults(
        angular_velocity=np.array(results["angular_velocity"]),
        linear_velocity=np.array(results["linear_velocity"]),
        xyz=np.array(results["xyz"]),
        heading_diff=np.array(results["heading_diff"]),
        saccade_type=np.array(results["saccade_type"]),
        frames_in_opto_radius=np.array(results["frames_in_opto_radius"]),
    )
