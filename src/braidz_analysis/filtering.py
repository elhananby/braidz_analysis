import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

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
