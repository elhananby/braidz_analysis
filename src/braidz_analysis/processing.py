import numpy as np
import pandas as pd
from tqdm import tqdm
from .trajectory import (
    calculate_angular_velocity,
    calculate_linear_velocity,
    heading_diff,
    heading_diff_pos
)

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_stim_or_opto_data(df: pd.DataFrame, stim_or_opto: pd.DataFrame, **kwargs):
    # Get the pre and post frames from the kwargs, if not present, use default values
    pre_frames = kwargs.get("pre_frames", 50)
    post_frames = kwargs.get("post_frames", 100)
    
    # Get the minimum number of frames from the kwargs, if not present, use default value
    min_frames = kwargs.get("min_frames", 300)

    # Initialize lists to store the angular velocity, linear velocity, heading difference, xyz coordinates, and sham values
    angvels = []
    linvels = []
    heading_diffs = []
    xyzs = []
    sham = []

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

        if len(grp) < min_frames:
            logger.debug(f"Skipping {obj_id} with {len(grp)} frames")
            continue

        try:
            stim_or_opto_idx = np.where(grp["frame"] == stim_or_opto_frame)[0][0]
        except IndexError:
            logger.debug(f"Skipping {obj_id} with no frame {stim_or_opto_frame}")
            continue

        if stim_or_opto_idx - pre_frames < 0 or stim_or_opto_idx + post_frames >= len(
            grp
        ):
            logger.debug(f"Skipping {obj_id} with out of range frames")
            continue

        stim_or_opto_range = range(
            stim_or_opto_idx - pre_frames,
            stim_or_opto_idx + post_frames,
        )

        heading, angular_velocity = calculate_angular_velocity(
            grp["xvel"].values, grp["yvel"].values
        )
        linear_velocity = calculate_linear_velocity(
            grp["xvel"].values, grp["yvel"].values
        )

        angvels.append(angular_velocity[stim_or_opto_range])
        linvels.append(linear_velocity[stim_or_opto_range])
        heading_diffs.append(heading_diff_pos(grp[["x", "y", "z"]].values, stim_or_opto_idx))
        xyzs.append(grp[["x", "y", "z"]].values[stim_or_opto_range])
        sham.append(row.get("is_sham", False))

    return {
        "angular_velocity": np.array(angvels),
        "linear_velocity": np.array(linvels),
        "heading_diff": np.array(heading_diffs),
        "xyz": np.array(xyzs),
        "sham": np.array(sham),
    }
