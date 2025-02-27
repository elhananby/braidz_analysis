import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def filter_by_time(
    df: pd.DataFrame, min_duration: float = 1.0, max_duration: float | None = None
) -> List[Tuple[int, int]]:
    """Filter trajectories based on their duration.

    Args:
        df: DataFrame containing trajectory data
        min_duration: Minimum duration in seconds (default: 1.0)
        max_duration: Maximum duration in seconds (optional)

    Returns:
        List of (obj_id, exp_num) tuples for trajectories meeting the duration criteria
    """
    # Group by both identifiers and calculate duration
    grouped = df.groupby(["obj_id", "exp_num"]).agg(
        {"timestamp": lambda x: x.max() - x.min()}
    )

    # Apply minimum duration filter
    mask = grouped["timestamp"] >= min_duration

    # Apply maximum duration filter if specified
    if max_duration is not None:
        mask &= grouped["timestamp"] <= max_duration

    valid_groups = grouped[mask].index.tolist()

    logger.info(
        f"Time filter: {len(valid_groups)} trajectories passed "
        f"(min_duration={min_duration}s"
        f"{f', max_duration={max_duration}s' if max_duration else ''})"
    )

    return valid_groups


def filter_by_distance(
    df: pd.DataFrame, min_distance: float = 1.0, max_distance: float | None = None
) -> List[Tuple[int, int]]:
    """Filter trajectories based on total distance traveled.

    Args:
        df: DataFrame containing trajectory data
        min_distance: Minimum distance in spatial units (default: 1.0)
        max_distance: Maximum distance in spatial units (optional)

    Returns:
        List of (obj_id, exp_num) tuples for trajectories meeting the distance criteria
    """

    def calculate_total_distance(group):
        # Calculate displacement between consecutive points
        dx = np.diff(group["x"].values)
        dy = np.diff(group["y"].values)
        dz = np.diff(group["z"].values)

        # Calculate total distance
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        return np.sum(distances)

    # Calculate total distance for each trajectory
    distances = df.groupby(["obj_id", "exp_num"]).apply(calculate_total_distance)

    # Apply minimum distance filter
    mask = distances >= min_distance

    # Apply maximum distance filter if specified
    if max_distance is not None:
        mask &= distances <= max_distance

    valid_groups = distances[mask].index.tolist()

    logger.info(
        f"Distance filter: {len(valid_groups)} trajectories passed "
        f"(min_distance={min_distance}"
        f"{f', max_distance={max_distance}' if max_distance else ''})"
    )

    return valid_groups


def filter_by_median(
    df: pd.DataFrame,
    xy_range: Tuple[float, float] = (-10, 10),
    z_range: Tuple[float, float] = (-10, 10),
) -> List[Tuple[int, int]]:
    """Filter trajectories based on their median position.

    Args:
        df: DataFrame containing trajectory data
        xy_range: Tuple of (min, max) values for x and y coordinates
        z_range: Tuple of (min, max) values for z coordinate

    Returns:
        List of (obj_id, exp_num) tuples for trajectories within spatial bounds
    """
    # Calculate median positions for each trajectory
    medians = df.groupby(["obj_id", "exp_num"]).agg(
        {"x": "median", "y": "median", "z": "median"}
    )

    # Create masks for each spatial criterion
    x_mask = (medians["x"] > xy_range[0]) & (medians["x"] < xy_range[1])
    y_mask = (medians["y"] > xy_range[0]) & (medians["y"] < xy_range[1])
    z_mask = (medians["z"] > z_range[0]) & (medians["z"] < z_range[1])

    # Combine all masks
    valid_groups = medians[x_mask & y_mask & z_mask].index.tolist()

    logger.info(
        f"Median position filter: {len(valid_groups)} trajectories passed "
        f"(xy_range={xy_range}, z_range={z_range})"
    )

    return valid_groups


def find_common_trajectories(*args) -> List[Tuple[int, int]]:
    """Find trajectories that appear in all provided lists.

    Args:
        *args: Variable number of lists, each containing (obj_id, exp_num) tuples

    Returns:
        List of (obj_id, exp_num) tuples present in all input lists

    Example:
        >>> time_filtered = filter_by_time(df, min_duration=2.0)
        >>> distance_filtered = filter_by_distance(df, min_distance=5.0)
        >>> position_filtered = filter_by_median(df, xy_range=(-5, 5))
        >>> valid = find_common_trajectories(time_filtered, distance_filtered, position_filtered)
    """
    if not args:
        return []

    # Convert all lists to sets
    sets = [set(traj_list) for traj_list in args]

    # Find intersection of all sets
    common_trajectories = list(sets[0].intersection(*sets[1:]))

    logger.info(
        f"Found {len(common_trajectories)} trajectories common to all {len(args)} filters"
    )

    return sorted(common_trajectories)  # Sort for consistent ordering


def filter_by_sham(
    data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Creates masks for splitting data based on sham boolean array.

    Args:
        data: Dictionary containing trajectory data with 'sham' boolean array of shape (k,)

    Returns:
        Dictionary with two boolean arrays of shape (k,):
            - 'real': mask for non-sham data
            - 'sham': mask for sham data
    """
    return {"real": ~data["sham"], "sham": data["sham"]}


def filter_by_frames_in_radius(
    data: Dict[str, np.ndarray], thresholds: Union[float, List[float]] = [15]
) -> Dict[str, np.ndarray]:
    """
    Creates masks for splitting data based on frames_in_radius values.

    Args:
        data: Dictionary containing trajectory data with 'frames_in_radius' array of shape (k,)
        thresholds: Either a single float value or a list of float values
            - If single float: creates two masks (below and above threshold)
            - If list: creates masks for intervals [0, t1], [t1, t2], ..., [tn-1, tn]

    Returns:
        Dictionary mapping threshold descriptions to boolean masks of shape (k,):
            - For single threshold t: {'<=t': mask, '>t': mask}
            - For thresholds [t1,t2]: {'<=t1': mask, 't1-t2': mask, '>t2': mask}
    """
    if isinstance(thresholds, (int, float)):
        thresholds = [float(thresholds)]
    thresholds = sorted(thresholds)

    frames = data["frames_in_radius"]
    masks = {}

    # First bin
    masks[f"<={thresholds[0]}"] = frames <= thresholds[0]

    # Middle bins
    for i in range(len(thresholds) - 1):
        key = f"{thresholds[i]}-{thresholds[i + 1]}"
        masks[key] = (frames > thresholds[i]) & (frames <= thresholds[i + 1])

    # Last bin
    masks[f">{thresholds[-1]}"] = frames > thresholds[-1]

    return masks


def apply_hysteresis_filter(
    linear_velocity, high_threshold=0.1, low_threshold=0.001, max_gap=100
):
    """
    Apply hysteresis filtering to identify flight bouts, combining bouts separated by small gaps.

    This function uses a two-stage approach:
    1. First, it identifies initial flight bouts using hysteresis thresholding
    2. Then, it looks for short gaps between bouts and combines bouts that are close together

    Args:
        linear_velocity: Array of linear velocities over time
        high_threshold: Velocity threshold that must be exceeded to start a flight bout
        low_threshold: Velocity threshold that must be crossed to end a flight bout
        max_gap: Maximum number of frames between bouts to consider them as one continuous bout

    Returns:
        list[np.ndarray]: List of arrays containing indices for each flight bout
    """
    import scipy

    # Smooth the velocity data to reduce noise
    smoothed_velocity = scipy.signal.savgol_filter(
        linear_velocity, window_length=21, polyorder=3
    )

    # First pass: identify initial bout segments
    initial_segments = []
    currently_flying = False
    current_start = None

    # Identify initial bout segments using hysteresis
    for i in range(len(smoothed_velocity)):
        if currently_flying:
            if smoothed_velocity[i] < low_threshold:
                currently_flying = False
                initial_segments.append(np.arange(current_start, i))
        else:
            if smoothed_velocity[i] > high_threshold:
                currently_flying = True
                current_start = i

    # Handle case where trajectory ends during flight
    if currently_flying:
        initial_segments.append(np.arange(current_start, len(smoothed_velocity)))

    # If we found no segments, return empty list
    if not initial_segments:
        return []

    # Second pass: combine segments that have small gaps between them
    combined_segments = []
    current_segment = initial_segments[0]

    # Look at each pair of consecutive segments
    for next_segment in initial_segments[1:]:
        # Calculate the gap between current segment end and next segment start
        gap = next_segment[0] - current_segment[-1] - 1

        if gap <= max_gap:
            # If the gap is small enough, create a new combined segment that includes
            # both segments and the gap between them
            current_segment = np.arange(current_segment[0], next_segment[-1] + 1)
        else:
            # If the gap is too large, store the current segment and start a new one
            combined_segments.append(current_segment)
            current_segment = next_segment

    # Don't forget to add the last segment
    combined_segments.append(current_segment)

    return combined_segments


def _combine_masks(*masks, method="and"):
    """
    Combines multiple boolean masks using a specified logical operation.

    Args:
        *masks: Variable number of boolean numpy arrays (masks) of the same shape
        method: String specifying the combination method ('and' or 'or')
               'and': True only where all masks are True
               'or': True where any mask is True

    Returns:
        numpy.ndarray: Combined boolean mask of the same shape as input masks

    Raises:
        ValueError: If no masks are provided or if masks have different shapes
        ValueError: If invalid method is specified
    """
    import numpy as np

    # Validate that we have at least one mask
    if not masks:
        raise ValueError("At least one mask must be provided")

    # Convert all masks to numpy arrays for consistent handling
    masks = [np.array(mask, dtype=bool) for mask in masks]

    # Validate that all masks have the same shape
    first_shape = masks[0].shape
    if not all(mask.shape == first_shape for mask in masks):
        raise ValueError("All masks must have the same shape")

    # Perform the combination based on the specified method
    if method.lower() == "and":
        # np.logical_and.reduce performs AND operation across all masks
        return np.logical_and.reduce(masks)
    elif method.lower() == "or":
        # np.logical_or.reduce performs OR operation across all masks
        return np.logical_or.reduce(masks)
    else:
        raise ValueError("Method must be either 'and' or 'or'")


def apply_masks(data: Dict[str, np.ndarray], *masks) -> Dict[str, np.ndarray]:
    """
    Applies combined boolean masks to all arrays in a dictionary.

    Args:
        data: Dictionary where values are numpy arrays that the mask will be applied to
        *masks: Variable number of boolean masks to be combined and applied

    Returns:
        Dictionary with the same keys but with masked arrays as values

    Example:
        >>> data = {'values': np.array([1, 2, 3, 4]), 'scores': np.array([10, 20, 30, 40])}
        >>> mask1 = np.array([True, False, True, False])
        >>> mask2 = np.array([True, True, False, False])
        >>> result = apply_masks(data, mask1, mask2)
        >>> print(result)
        {'values': array([1]), 'scores': array([10])}
    """
    # Combine all provided masks into a single mask
    combined_mask = _combine_masks(*masks)

    # Apply the combined mask to each array in the dictionary
    return {
        key: data[key][combined_mask]
        for key in data
        if len(data[key]) == len(combined_mask)
    }


def filter_data_dict(data_dict, mask):
    """
    Filter a dictionary of arrays based on a boolean mask along the first dimension.

    Args:
        data_dict (dict): Dictionary containing arrays of different shapes
        mask (ndarray): Boolean array matching the first dimension of data arrays

    Returns:
        dict: New dictionary with filtered arrays
    """
    filtered_dict = {}

    for key, array in data_dict.items():
        # Handle different dimensional data differently
        if len(array.shape) == 1:
            # For 1D arrays (like heading_difference, frames_in_radius, etc.)
            filtered_dict[key] = array[mask]
        elif len(array.shape) == 2:
            # For 2D arrays (like angular_velocity, linear_velocity)
            filtered_dict[key] = array[mask, :]
        elif len(array.shape) == 3:
            # For 3D arrays (like position)
            filtered_dict[key] = array[mask, :, :]

    return filtered_dict
