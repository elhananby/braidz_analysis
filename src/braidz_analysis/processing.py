import numpy as np
import pandas as pd
from tqdm import tqdm
from .trajectory import (
    calculate_angular_velocity,
    calculate_linear_velocity,
    heading_diff,
)

import logging
from dataclasses import dataclass


@dataclass
class AnalysisParams:
    pre_frames: int = 50
    post_frames: int = 100
    min_frames: int = 300
    opto_radius: float = 0.025
    opto_duration: int = 30


class TrajectoryData:
    angular_velocity: np.ndarray
    linear_velocity: np.ndarray
    heading_diff: np.ndarray
    xyz: np.ndarray
    sham: np.ndarray
    frames_in_opto_radius: np.ndarray

    # a function to generate a mask for the data based on frames in opto radius
    def get_radius_mask(self, threshold: int = 15):
        return self.frames_in_opto_radius > threshold

    def get_mean(self, key):
        return np.mean(getattr(self, key), axis=0)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_stim_or_opto_data(df: pd.DataFrame, stim_or_opto: pd.DataFrame, **kwargs):
    params = AnalysisParams(**kwargs)

    # Initialize lists to store the angular velocity, linear velocity, heading difference, xyz coordinates, and sham values
    angvels = []
    linvels = []
    heading_diffs = []
    xyzs = []
    sham = []
    frames_in_opto_radius = []

    for idx, row in tqdm(stim_or_opto.iterrows(), total=len(stim_or_opto)):
        obj_id = row["obj_id"]
        exp_num = row["exp_num"] if "exp_num" in row else None
        stim_or_opto_frame = row["frame"]

        # get grp df using obj_id and exp_num if exp_num is not none, otherwise use obj_id only
        grp = (
            df[(df["obj_id"] == obj_id) & (df["exp_num"] == exp_num)]
            if exp_num
            else df[df["obj_id"] == obj_id].copy()
        )

        if len(grp) < params.min_frames:
            logger.debug(f"Skipping {obj_id} with {len(grp)} frames")
            continue

        try:
            stim_or_opto_idx = np.where(grp["frame"] == stim_or_opto_frame)[0][0]
        except IndexError:
            logger.debug(f"Skipping {obj_id} with no frame {stim_or_opto_frame}")
            continue

        if (
            stim_or_opto_idx - params.pre_frames < 0
            or stim_or_opto_idx + params.post_frames >= len(grp)
        ):
            logger.debug(f"Skipping {obj_id} with out of range frames")
            continue

        stim_or_opto_range = range(
            stim_or_opto_idx - params.pre_frames,
            stim_or_opto_idx + params.post_frames,
        )

        heading, angular_velocity = calculate_angular_velocity(
            grp["xvel"].values, grp["yvel"].values
        )
        linear_velocity = calculate_linear_velocity(
            grp["xvel"].values, grp["yvel"].values
        )

        # get radius
        radius = np.sqrt(grp["x"].values ** 2 + grp["y"].values ** 2)

        # append values to lists
        frames_in_opto_radius.append(
            np.sum(
                radius[stim_or_opto_idx : stim_or_opto_idx + params.opto_duration]
                < params.opto_radius
            )
        )
        angvels.append(angular_velocity[stim_or_opto_range])
        linvels.append(linear_velocity[stim_or_opto_range])
        heading_diffs.append(heading_diff(heading, stim_or_opto_idx, window=25))
        xyzs.append(grp[["x", "y", "z"]].values[stim_or_opto_range])
        sham.append(row.get("is_sham", False))

    trajectory_data = TrajectoryData()
    trajectory_data.angular_velocity = np.array(angvels)
    trajectory_data.linear_velocity = np.array(linvels)
    trajectory_data.heading_diff = np.array(heading_diffs)
    trajectory_data.xyz = np.array(xyzs)
    trajectory_data.sham = np.array(sham)
    trajectory_data.frames_in_opto_radius = np.array(frames_in_opto_radius)
    return trajectory_data
