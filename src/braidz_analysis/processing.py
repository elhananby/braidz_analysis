import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm
import inspect

from .trajectory import (
    calculate_angular_velocity,
    calculate_linear_velocity,
    detect_saccades,
)

from scipy.stats import circmean

# Constants for trajectory analysis
MAX_RADIUS = 0.23  # Maximum allowed radius for valid trajectories
XY_RANGE = [-0.23, 0.23]
Z_RANGE = [0.05, 0.3]  # Valid range for z-coordinate [min, max]
HEADING_DIFF_WINDOW = 10  # Window size for heading difference calculation

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# @dataclass
# class TrajectoryAnalysisResults:
#     """Results from trajectory analysis.

#     Contains all possible metrics that could be returned from trajectory analysis functions.
#     Not all fields will be populated by every analysis function.

#     Attributes:
#         angular_velocity: Array of angular velocities (n_samples, n_timepoints)
#         linear_velocity: Array of linear velocities (n_samples, n_timepoints)
#         xyz: Array of positions (n_samples, n_timepoints, 3)
#         heading_diff: Array of heading differences (n_samples, n_timepoints)
#         saccade_type: Array of saccade type labels (n_samples,)
#         sham: Array of boolean sham indicators (n_samples,)
#         frames_in_opto_radius: Array of frame counts within opto radius (n_samples,)
#     """

#     angular_velocity: np.ndarray
#     linear_velocity: np.ndarray
#     xyz: np.ndarray
#     heading_diff: np.ndarray
#     saccade_type: np.ndarray | None = None  # Optional fields that aren't always present
#     sham: np.ndarray | None = None
#     frames_in_opto_radius: np.ndarray | None = None

#     def __post_init__(self):
#         """Validate array shapes after initialization."""
#         n_samples = len(self.angular_velocity)
#         assert len(self.linear_velocity) == n_samples, "Mismatched sample counts"
#         assert len(self.xyz) == n_samples, "Mismatched sample counts"
#         assert len(self.heading_diff) == n_samples, "Mismatched sample counts"

#         if self.saccade_type is not None:
#             assert len(self.saccade_type) == n_samples, "Mismatched sample counts"
#         if self.sham is not None:
#             assert len(self.sham) == n_samples, "Mismatched sample counts"
#         if self.frames_in_opto_radius is not None:
#             assert (
#                 len(self.frames_in_opto_radius) == n_samples
#             ), "Mismatched sample counts"


@dataclass
class OptoAnalysis:
    """Parameters for Optogenetic trajectory analysis.

    Attributes:
        pre_frames (int): Number of frames to analyze before event
        post_frames (int): Number of frames to analyze after event
        min_frames (int): Minimum number of frames required for analysis
        opto_radius (float): Radius threshold for optogenetic stimulation
        opto_duration (int): Duration of optogenetic stimulation in frames
    """

    pre_frames: int = 50
    post_frames: int = 100
    min_frames: int = 150
    opto_radius: float = 0.025
    opto_duration: int = 30

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
        )


@dataclass
class LoomingAnalysis:
    """Parameters for Optogenetic trajectory analysis.

    Attributes:
        pre_frames (int): Number of frames to analyze before event
        post_frames (int): Number of frames to analyze after event
        min_frames (int): Minimum number of frames required for analysis
        opto_radius (float): Radius threshold for optogenetic stimulation
        opto_duration (int): Duration of optogenetic stimulation in frames
    """

    pre_frames: int = 50
    post_frames: int = 100
    min_frames: int = 150
    looming_duration: int = 30
    response_delay: int = 20

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

    pre_frames: int = 50
    post_frames: int = 50
    min_trajectory_length: int = 150
    threshold: float = np.deg2rad(300)
    distance: int = 10

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
        )


def get_opto_data(
    df: pd.DataFrame, opto: pd.DataFrame, params: dict | OptoAnalysis | None = None
) -> dict:
    if isinstance(params, dict):
        params = OptoAnalysis.from_dict(params)
    elif params is None:
        params = OptoAnalysis()

    opto_data = {
        "angular_velocity": [],
        "angular_velocity_peak_centered": [],
        "linear_velocity": [],
        "linear_velocity_peak_centered": [],
        "xyz": [],
        "heading_difference": [],
        "heading_difference_peak_centered": [],
        "frames_in_opto_radius": [],
        "sham": [],
        "reaction_delay": [],
    }

    for _, row in opto.iterrows():
        if "exp_num" in row:
            grp = df[df["obj_id"] == row["obj_id"] and df["exp_num"] == row["exp_num"]]
        else:
            grp = df[df["obj_id"] == row["obj_id"]]

        if len(grp) < params.min_frames:
            logger.debug(f"Skipping trajectory with {len(grp)} frames")
            continue

        try:
            opto_idx = np.where(grp["frame"] == row["frame"])[0][0]
        except IndexError:
            logger.debug("Skipping trajectory with no opto frame")
            continue

        if opto_idx - params.pre_frames < 0 or opto_idx + params.post_frames >= len(
            grp
        ):
            logger.debug("Skipping trajectory with insufficient frames")
            continue

        # get angular and linear velocity
        heading, angular_velocity = calculate_angular_velocity(
            grp.xvel.values, grp.yvel.values
        )
        linear_velocity = calculate_linear_velocity(grp.xvel.values, grp.yvel.values)

        # calculate how many frames in opto radius
        radius = np.sqrt(grp.x.values**2 + grp.y.values**2)
        frames_in_opto_radius = np.sum(
            radius[opto_idx : opto_idx + params.opto_duration] < params.opto_radius
        )

        # get opto range
        opto_range = range(opto_idx - params.pre_frames, opto_idx + params.post_frames)

        opto_data["angular_velocity"].append(angular_velocity[opto_range])
        opto_data["linear_velocity"].append(linear_velocity[opto_range])
        opto_data["xyz"].append(grp[["x", "y", "z"]].values[opto_range])
        opto_data["frames_in_opto_radius"].append(frames_in_opto_radius)

        # calculate heading difference
        heading_before = circmean(
            heading[opto_idx - HEADING_DIFF_WINDOW : opto_idx], low=-np.pi, high=np.pi
        )
        heading_after = circmean(
            heading[
                opto_idx + params.opto_duration : opto_idx
                + params.opto_duration
                + HEADING_DIFF_WINDOW
            ],
            low=-np.pi,
            high=np.pi,
        )

        heading_difference = np.arctan2(
            np.sin(heading_after - heading_before),
            np.cos(heading_after - heading_before),
        )
        opto_data["heading_difference"].append(heading_difference)
        opto_data["sham"].append(row["sham"] if "sham" in row else False)

        # find all peaks
        peaks = detect_saccades(angular_velocity, params.threshold, params.distance)
        opto_peak = [
            peak for peak in peaks if opto_idx < peak < opto_idx + params.opto_duration
        ]
        if len(opto_peak) == 0:
            opto_data["angular_velocity_peak_centered"].append(np.nan)
            opto_data["linear_velocity_peak_centered"].append(np.nan)
            opto_data["heading_difference_peak_centered"].append(np.nan)
        else:
            opto_peak = opto_peak[0]  # only take the first peak
            peak_centered_angular_velocity = angular_velocity[
                opto_peak - params.pre_frames : opto_peak + params.post_frames
            ]
            peak_centered_linear_velocity = linear_velocity[
                opto_peak - params.pre_frames : opto_peak + params.post_frames
            ]

            peak_centered_heading_before = circmean(
                heading[opto_peak - HEADING_DIFF_WINDOW : opto_peak],
                low=-np.pi,
                high=np.pi,
            )
            peak_centered_heading_after = circmean(
                heading[opto_peak : opto_peak + HEADING_DIFF_WINDOW],
                low=-np.pi,
                high=np.pi,
            )

            peak_centered_heading_difference = np.arctan2(
                np.sin(peak_centered_heading_after - peak_centered_heading_before),
                np.cos(peak_centered_heading_after - peak_centered_heading_before),
            )
            opto_data["angular_velocity_peak_centered"].append(
                peak_centered_angular_velocity
            )
            opto_data["linear_velocity_peak_centered"].append(
                peak_centered_linear_velocity
            )
            opto_data["heading_difference_peak_centered"].append(
                peak_centered_heading_difference
            )

            # calculate reaction delay
            reaction_delay = opto_peak - opto_idx
            opto_data["reaction_delay"].append(reaction_delay)

    # convert to numpy arrays
    return {k: np.array(v) for k, v in opto_data.items()}


def get_stim_data(
    df: pd.DataFrame,
    stim: pd.DataFrame,
    looming_params: dict | LoomingAnalysis | None = None,
    saccade_params: dict | SaccadeParams | None = None,
) -> dict:
    if isinstance(looming_params, dict):
        looming_params = LoomingAnalysis.from_dict(looming_params)
    elif looming_params is None:
        looming_params = LoomingAnalysis()

    if isinstance(saccade_params, dict):
        saccade_params = SaccadeParams.from_dict(saccade_params)
    elif saccade_params is None:
        saccade_params = SaccadeParams()

    stim_data = {
        "angular_velocity": [],
        "angular_velocity_peak_centered": [],
        "linear_velocity": [],
        "linear_velocity_peak_centered": [],
        "xyz": [],
        "heading_difference": [],
        "heading_difference_peak_centered": [],
        "reaction_delay": [],
    }

    for _, row in stim.iterrows():
        if "exp_num" in row:
            grp = df[
                (df["obj_id"] == row["obj_id"]) & (df["exp_num"] == row["exp_num"])
            ]
        else:
            grp = df[df["obj_id"] == row["obj_id"]]

        if len(grp) < looming_params.min_frames:
            logger.debug(f"Skipping trajectory with {len(grp)} frames")
            continue

        try:
            stim_idx = np.where(grp["frame"] == row["frame"])[0][0]
        except IndexError:
            logger.debug("Skipping trajectory with no opto frame")
            continue

        if (
            stim_idx - looming_params.pre_frames < 0
            or stim_idx + looming_params.post_frames >= len(grp)
        ):
            logger.debug("Skipping trajectory with insufficient frames")
            continue

        # get angular and linear velocity
        heading, angular_velocity = calculate_angular_velocity(
            grp.xvel.values, grp.yvel.values
        )
        linear_velocity = calculate_linear_velocity(grp.xvel.values, grp.yvel.values)

        # get opto range
        stim_range = range(
            stim_idx - looming_params.pre_frames, stim_idx + looming_params.post_frames
        )

        stim_data["angular_velocity"].append(angular_velocity[stim_range])
        stim_data["linear_velocity"].append(linear_velocity[stim_range])
        stim_data["xyz"].append(grp[["x", "y", "z"]].values[stim_range])

        # calculate heading difference
        heading_before = circmean(
            heading[stim_idx - HEADING_DIFF_WINDOW : stim_idx], low=-np.pi, high=np.pi
        )
        heading_after = circmean(
            heading[
                stim_idx + looming_params.looming_duration : stim_idx
                + looming_params.looming_duration
                + HEADING_DIFF_WINDOW
            ],
            low=-np.pi,
            high=np.pi,
        )

        heading_difference = np.arctan2(
            np.sin(heading_after - heading_before),
            np.cos(heading_after - heading_before),
        )
        stim_data["heading_difference"].append(heading_difference)

        # find all peaks
        peaks = detect_saccades(
            angular_velocity, saccade_params.threshold, saccade_params.distance
        )
        stim_peak = [
            peak
            for peak in peaks
            if stim_idx
            < peak
            < stim_idx + looming_params.looming_duration + looming_params.response_delay
        ]

        # if no peaks found, append nan
        if len(stim_peak) == 0:
            stim_data["angular_velocity_peak_centered"].append(np.nan)
            stim_data["linear_velocity_peak_centered"].append(np.nan)
            stim_data["heading_difference_peak_centered"].append(np.nan)

        # if multiple peaks found, append the first peak
        else:
            stim_peak = stim_peak[0]  # only take the first peak

            if (
                stim_peak - looming_params.pre_frames < 0
                or stim_peak + looming_params.post_frames >= len(grp)
            ):
                logger.debug("Skipping peak with insufficient frames")
                continue

            # get peak centered data
            peak_centered_angular_velocity = angular_velocity[
                stim_peak - looming_params.pre_frames : stim_peak
                + looming_params.post_frames
            ]
            peak_centered_linear_velocity = linear_velocity[
                stim_peak - looming_params.pre_frames : stim_peak
                + looming_params.post_frames
            ]

            # get peak centered heading difference
            peak_centered_heading_before = circmean(
                heading[stim_peak - HEADING_DIFF_WINDOW : stim_peak],
                low=-np.pi,
                high=np.pi,
            )
            peak_centered_heading_after = circmean(
                heading[stim_peak : stim_peak + HEADING_DIFF_WINDOW],
                low=-np.pi,
                high=np.pi,
            )

            peak_centered_heading_difference = np.arctan2(
                np.sin(peak_centered_heading_after - peak_centered_heading_before),
                np.cos(peak_centered_heading_after - peak_centered_heading_before),
            )
            # append peak centered data
            stim_data["angular_velocity_peak_centered"].append(
                peak_centered_angular_velocity
            )
            stim_data["linear_velocity_peak_centered"].append(
                peak_centered_linear_velocity
            )
            stim_data["heading_difference_peak_centered"].append(
                peak_centered_heading_difference
            )

            # calculate reaction delay
            reaction_delay = stim_peak - stim_idx
            stim_data["reaction_delay"].append(reaction_delay)

    # convert to numpy arrays
    return {k: np.array(v) for k, v in stim_data.items()}


def get_all_saccades(
    df: pd.DataFrame, params: dict | SaccadeParams | None = None
) -> dict:
    if isinstance(params, dict):
        params = SaccadeParams.from_dict(params)
    elif params is None:
        params = SaccadeParams()

    saccade_data = {
        "angular_velocity": [],
        "linear_velocity": [],
        "xyz": [],
        "heading_difference": [],
    }

    grouped_data = (
        df.groupby(["obj_id", "exp_num"])
        if "exp_num" in df.columns
        else df.groupby("obj_id")
    )

    for _, grp in tqdm(
        grouped_data, desc="Processing trajectories", total=len(grouped_data)
    ):
        # check length
        if len(grp) < params.min_trajectory_length:
            logger.debug(f"Skipping trajectory with {len(grp)} frames")
            continue

        # check median
        if (
            not (XY_RANGE[0] < grp.x.median() < XY_RANGE[1])
            and (XY_RANGE[0] < grp.y.median() < XY_RANGE[1])
            and (Z_RANGE[0] < grp.z.median() < Z_RANGE[1])
        ):
            logger.debug("Skipping trajectory outside of valid median range")
            continue

        # get angular and linear velocity
        heading, angular_velocity = calculate_angular_velocity(
            grp.xvel.values, grp.yvel.values
        )
        linear_velocity = calculate_linear_velocity(grp.xvel.values, grp.yvel.values)

        # detect saccades
        saccades = detect_saccades(angular_velocity, params.threshold, params.distance)
        logger.debug(
            f"Detected {len(saccades)} saccades for obj_id {grp.obj_id.iloc[0]}, exp_num {grp.exp_num.iloc[0]}"
        )

        # loop and extract saccade traces
        for sac in saccades:
            # check if saccade has enough frames
            if sac - params.pre_frames < 0 or sac + params.post_frames >= len(grp):
                logger.debug("Skipping saccade with insufficient frames")
                continue

            # check if the fly is flying
            if not (
                (XY_RANGE[0] < grp.x.iloc[sac] < XY_RANGE[1])
                and (XY_RANGE[0] < grp.y.iloc[sac] < XY_RANGE[1])
                and (Z_RANGE[0] < grp.z.iloc[sac] < Z_RANGE[1])
            ):
                continue

            # get saccade traces
            saccade_angular_velocity = angular_velocity[
                sac - params.pre_frames : sac + params.post_frames
            ]
            saccade_linear_velocity = linear_velocity[
                sac - params.pre_frames : sac + params.post_frames
            ]
            saccade_xyz = grp[["x", "y", "z"]].values[
                sac - params.pre_frames : sac + params.post_frames
            ]

            # calculate heading difference
            heading_before = circmean(
                heading[sac - HEADING_DIFF_WINDOW : sac], low=-np.pi, high=np.pi
            )
            heading_after = circmean(
                heading[sac : sac + HEADING_DIFF_WINDOW], low=-np.pi, high=np.pi
            )
            heading_difference = np.arctan2(
                np.sin(heading_after - heading_before),
                np.cos(heading_after - heading_before),
            )

            saccade_data["angular_velocity"].append(saccade_angular_velocity)
            saccade_data["linear_velocity"].append(saccade_linear_velocity)
            saccade_data["xyz"].append(saccade_xyz)
            saccade_data["heading_difference"].append(heading_difference)

    return {k: np.array(v) for k, v in saccade_data.items()}
