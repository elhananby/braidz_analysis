import logging
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm

from .filtering import apply_hysteresis_filter
from .helpers import dict_list_to_numpy
from .params import OptoAnalysisParams, SaccadeAnalysisParams, StimAnalysisParams
from .trajectory import (
    calculate_angular_velocity,
    calculate_heading_diff,
    calculate_linear_velocity,
    calculate_smoothed_linear_velocity,
    detect_saccades,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _find_relevant_peak(peaks, stim_idx, duration=30, return_none=False):
    """Find first peak within the stimulus window."""
    window_end = stim_idx + duration

    for peak in peaks:
        if stim_idx < peak < window_end:
            return int(peak)

    return None if return_none else (stim_idx + duration // 2)


def _smooth_df(
    df: pd.DataFrame, columns: List[str] = ["x", "y", "z", "xvel", "yvel", "zvel"]
) -> pd.DataFrame:
    """Smooth a DataFrame using a Savitzky-Golay filter.

    Args:
        df: DataFrame to smooth
        columns: List of columns to smooth

    Returns:
        pd.DataFrame: Smoothed DataFrame
    """
    df = df.copy()
    for column in columns:
        df.loc[:, f"original_{column}"] = df[column].copy()
        df.loc[:, column] = savgol_filter(df[column], 21, 3)
    return df


def get_stim_or_opto_response_data(
    df: pd.DataFrame,
    opto_or_stim: pd.DataFrame,
    params: Optional[Union[OptoAnalysisParams, StimAnalysisParams]] = None,
    type: Literal["opto", "stim"] = "opto",
    saccade_params: Optional[SaccadeAnalysisParams] = None,
) -> dict:
    """Analyze response data around stimulation events."""
    if params is None:
        params = OptoAnalysisParams() if type == "opto" else StimAnalysisParams()

    if saccade_params is None:
        saccade_params = SaccadeAnalysisParams()

    output_data = {
        "angular_velocity": [],
        "linear_velocity": [],
        "position": [],
        "heading_difference": [],
        "frames_in_radius": [],
        "sham": [],
        "reaction_delay": [],
        "responsive": [],
    }

    for _, row in opto_or_stim.iterrows():
        grp = df[
            (df["obj_id"] == row["obj_id"]) & (df["exp_num"] == row["exp_num"])
            if "exp_num" in row
            else df["obj_id"] == row["obj_id"]
        ]

        if len(grp) < params.min_frames:
            logger.debug("Skipping trajectory: insufficient length")
            continue

        try:
            stim_idx = np.where(grp["frame"] == row["frame"])[0][0]
            grp = _smooth_df(grp, ["x", "y", "z"])

            heading, angular_velocity = calculate_angular_velocity(
                grp.xvel.values, grp.yvel.values
            )
            linear_velocity = calculate_linear_velocity(
                grp.xvel.values, grp.yvel.values
            )

            peaks = detect_saccades(
                angular_velocity,
                height=saccade_params.threshold,
                distance=saccade_params.distance,
            )
            peak = _find_relevant_peak(
                peaks, stim_idx, duration=params.duration, return_none=True
            )

            # Initialize empty arrays
            peak_data = {
                "angular_velocity": np.full(
                    params.pre_frames + params.post_frames, np.nan
                ),
                "linear_velocity": np.full(
                    params.pre_frames + params.post_frames, np.nan
                ),
                "position": np.full(
                    (params.pre_frames + params.post_frames, 3), np.nan
                ),
                "heading_difference": np.nan,
                "reaction_delay": np.nan,
                "responsive": False,
            }

            if (
                peak is not None
                and params.pre_frames <= peak < len(grp) - params.post_frames
            ):
                range_to_extract = range(
                    peak - params.pre_frames, peak + params.post_frames
                )

                peak_data.update(
                    {
                        "angular_velocity": angular_velocity[range_to_extract],
                        "linear_velocity": linear_velocity[range_to_extract],
                        "position": grp[["x", "y", "z"]].to_numpy()[range_to_extract],
                        "heading_difference": calculate_heading_diff(
                            heading,
                            peak - saccade_params.heading_diff_window,
                            peak,
                            peak + saccade_params.heading_diff_window,
                        ),
                        "reaction_delay": peak - stim_idx,
                        "responsive": True,
                    }
                )

            # Calculate radius-based metrics
            if hasattr(params, "radius"):
                radius = np.sqrt(grp.x.values**2 + grp.y.values**2)
                frames_in_radius = np.sum(
                    radius[stim_idx : stim_idx + int(params.duration)] < params.radius
                )
            else:
                frames_in_radius = None

            # Update output data
            for key in [
                "angular_velocity",
                "linear_velocity",
                "position",
                "heading_difference",
                "reaction_delay",
            ]:
                output_data[key].append(peak_data[key])
            output_data["frames_in_radius"].append(frames_in_radius)
            output_data["sham"].append(row.get("sham", False))
            output_data["responsive"].append(peak_data["responsive"])

        except (IndexError, ValueError) as e:
            logger.debug(f"Skipping trajectory: {str(e)}")
            continue

    return dict_list_to_numpy(output_data)


def get_stim_or_opto_data(
    df: pd.DataFrame,
    stim_or_opto: pd.DataFrame,
    params: Optional[Union[OptoAnalysisParams, StimAnalysisParams]] = None,
    type: Literal["opto", "stim"] = "opto",
    saccade_params: Optional[SaccadeAnalysisParams] = None,
) -> dict:
    """Analyze stimulation data with parameter validation.

    Args:
        df: DataFrame containing trajectory data
        stim_or_opto: DataFrame containing stimulation events
        params: Analysis parameters. If None, uses defaults based on type
        type: Either "opto" or "stim", used when params is None
        saccade_params: Parameters for saccade detection
    """
    if params is None:
        params = OptoAnalysisParams() if type == "opto" else StimAnalysisParams()

    if saccade_params is None:
        saccade_params = SaccadeAnalysisParams()

    opto_data = {
        "angular_velocity": [],
        "linear_velocity": [],
        "position": [],
        "heading_difference": [],
        "frames_in_radius": [],
        "sham": [],
    }

    for _, row in stim_or_opto.iterrows():
        grp = df[
            (df["obj_id"] == row["obj_id"]) & (df["exp_num"] == row["exp_num"])
            if "exp_num" in row
            else df["obj_id"] == row["obj_id"]
        ]

        if len(grp) < params.min_frames:
            logger.debug(f"Skipping trajectory with {len(grp)} frames")
            continue

        try:
            stim_idx = np.where(grp["frame"] == row["frame"])[0][0]

            if stim_idx < params.pre_frames:
                logger.debug("Skipping: insufficient preceding frames")
                continue

            range_to_extract = range(
                stim_idx - params.pre_frames,
                stim_idx + params.post_frames,
            )

            if range_to_extract[-1] >= len(grp) or range_to_extract[0] < 0:
                logger.debug("Skipping: range out of bounds")
                continue

            heading, angular_velocity = calculate_angular_velocity(
                grp.xvel.values, grp.yvel.values
            )
            linear_velocity = calculate_smoothed_linear_velocity(grp)

            peaks = detect_saccades(
                angular_velocity,
                height=saccade_params.threshold,
                distance=saccade_params.distance,
            )

            peak = _find_relevant_peak(
                peaks,
                stim_idx,
                duration=params.duration,
                return_none=False,
            )

            heading_difference = calculate_heading_diff(
                heading,
                peak - saccade_params.heading_diff_window,
                peak,
                peak + saccade_params.heading_diff_window,
            )

            if hasattr(params, "radius"):
                radius = np.sqrt(grp.x.values**2 + grp.y.values**2)
                frames_in_radius = np.sum(
                    radius[stim_idx : stim_idx + int(params.duration)] < params.radius
                )
            else:
                frames_in_radius = 0

            opto_data["angular_velocity"].append(angular_velocity[range_to_extract])
            opto_data["linear_velocity"].append(linear_velocity[range_to_extract])
            opto_data["position"].append(grp[["x", "y", "z"]].values[range_to_extract])
            opto_data["frames_in_radius"].append(frames_in_radius)
            opto_data["heading_difference"].append(heading_difference)
            opto_data["sham"].append(row.get("sham", False))

        except (IndexError, ValueError) as e:
            logger.debug(f"Skipping trajectory: {str(e)}")
            continue

    return dict_list_to_numpy(opto_data)


def get_all_saccades(
    df: pd.DataFrame,
    params: Optional[SaccadeAnalysisParams] = None,
) -> dict:
    """Extract and analyze all saccades from trajectory data."""
    if params is None:
        params = SaccadeAnalysisParams()

    filtered_df = filter_trajectories(df)
    grouped_data = filtered_df.groupby(
        ["obj_id", "exp_num"] if "exp_num" in filtered_df.columns else "obj_id"
    )

    saccade_data = {
        "angular_velocity": [],
        "linear_velocity": [],
        "position": [],
        "heading_difference": [],
        "ISI": [],
        "saccade_per_second": [],
        "trajectory_duration": [],
        "num_of_saccades": [],
    }

    for _, grp in tqdm(grouped_data, desc="Processing trajectories"):
        heading, angular_velocity = calculate_angular_velocity(
            grp.xvel.values, grp.yvel.values
        )
        linear_velocity = calculate_linear_velocity(grp.xvel.values, grp.yvel.values)

        normalized_linear_velocity = linear_velocity / np.max(linear_velocity)
        flight_bouts = apply_hysteresis_filter(normalized_linear_velocity)

        for bout_indices in flight_bouts:
            bout_angular_velocity = angular_velocity[bout_indices]
            bout_saccades = detect_saccades(
                bout_angular_velocity,
                height=params.threshold,
                distance=params.distance,
            )
            trajectory_saccades = bout_indices[bout_saccades]
            good_saccades = []

            for sac in trajectory_saccades:
                if sac < params.pre_frames or sac + params.post_frames >= len(grp):
                    continue

                radius = np.sqrt(grp.x.iloc[sac] ** 2 + grp.y.iloc[sac] ** 2)
                if not (
                    radius < params.max_radius
                    and params.zlim[0] < grp.z.iloc[sac] < params.zlim[1]
                ):
                    continue

                range_to_extract = range(
                    sac - params.pre_frames, sac + params.post_frames
                )
                heading_difference = calculate_heading_diff(
                    heading,
                    sac - params.heading_diff_window,
                    sac,
                    sac + params.heading_diff_window,
                )

                saccade_data["angular_velocity"].append(
                    angular_velocity[range_to_extract]
                )
                saccade_data["linear_velocity"].append(
                    linear_velocity[range_to_extract]
                )
                saccade_data["position"].append(
                    grp[["x", "y", "z"]].values[range_to_extract]
                )
                saccade_data["heading_difference"].append(heading_difference)
                good_saccades.append(sac)

            bout_duration = (
                grp.timestamp.iloc[bout_indices[-1]]
                - grp.timestamp.iloc[bout_indices[0]]
            )

            if len(good_saccades) < 2:
                saccade_data["ISI"].append(np.nan)
                saccade_data["saccade_per_second"].append(np.nan)
            else:
                isis = np.diff(good_saccades)
                valid_isis = isis[(isis > 1) & (isis < 500)]
                saccade_data["ISI"].append(np.nanmean(valid_isis))
                saccade_data["saccade_per_second"].append(
                    len(good_saccades) / bout_duration
                )

            saccade_data["num_of_saccades"].append(len(good_saccades))
            saccade_data["trajectory_duration"].append(bout_duration)

    return dict_list_to_numpy(saccade_data)


def filter_trajectories(
    df: pd.DataFrame,
    **kwargs: Optional[dict],
) -> pd.DataFrame:
    """Filter trajectory data based on length and spatial criteria.

    Args:
        df: DataFrame containing trajectory data
        saccade_params: Saccade detection parameters
        trajectory_params: Trajectory analysis parameters including spatial bounds

    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid trajectories
    """
    # Define grouping columns
    group_cols = ["obj_id", "exp_num"] if "exp_num" in df.columns else ["obj_id"]

    # Calculate group statistics in a single pass
    group_stats = (
        df.groupby(group_cols)
        .agg(
            {
                "x": "median",
                "y": "median",
                "z": "median",
                "obj_id": "size",  # This gives us the length of each group
            }
        )
        .rename(columns={"obj_id": "trajectory_length"})
    )

    # Create masks for each filtering criterion
    length_mask = group_stats["trajectory_length"] >= kwargs.get("min_frames", 300)

    xlim = kwargs.get("xlim", [-0.23, 0.23])
    x_median_mask = (group_stats["x"] > xlim[0]) & (group_stats["x"] < xlim[1])
    ylim = kwargs.get("ylim", [-0.23, 0.23])
    y_median_mask = (group_stats["y"] > ylim[0]) & (group_stats["y"] < ylim[1])
    zlim = kwargs.get("zlim", [-0.1, 0.1])
    z_median_mask = (group_stats["z"] > zlim[0]) & (group_stats["z"] < zlim[1])

    # Combine all masks
    valid_groups = group_stats[
        length_mask & x_median_mask & y_median_mask & z_median_mask
    ].index

    # Log filtering results
    total_groups = len(group_stats)
    valid_group_count = len(valid_groups)
    logger.info(f"Filtered {total_groups - valid_group_count} trajectories:")
    logger.info(f"- {total_groups - length_mask.sum()} failed length criterion")
    logger.info(
        f"- {total_groups - (x_median_mask & y_median_mask & z_median_mask).sum()} failed spatial criteria"
    )
    logger.info(f"Retained {valid_group_count} valid trajectories")

    # Filter the original DataFrame
    if isinstance(valid_groups, pd.MultiIndex):
        return df.set_index(group_cols).loc[valid_groups].reset_index()
    else:
        return df[df[group_cols[0]].isin(valid_groups)]
