# filtering.py

from typing import Iterator
import logging
import numpy as np
import pandas as pd
from .core import TrajectoryData
from .params import AnalysisParams

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
