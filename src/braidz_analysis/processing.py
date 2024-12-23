import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from .helpers import dict_list_to_numpy

from .trajectory import (
    calculate_angular_velocity,
    calculate_heading_diff,
    calculate_linear_velocity,
    detect_saccades,
)
from .types import (
    AnalysisParamType,
    LoomingAnalysis,
    OptoAnalysis,
    ParameterManager,
    SaccadeParams,
    TrajectoryParams,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_opto_data(
    df: pd.DataFrame,
    opto: pd.DataFrame,
    opto_params: dict | OptoAnalysis | None = None,
    saccade_params: dict | SaccadeParams | None = None,
    trajectory_params: dict | TrajectoryParams | None = None,
) -> dict:
    """Analyze optogenetic stimulation data with parameter validation.

    Args:
        df: DataFrame containing trajectory data
        opto: DataFrame containing optogenetic stimulation events
        opto_params: Optogenetic analysis parameters
        saccade_params: Saccade detection parameters
        trajectory_params: Trajectory analysis parameters

    Returns:
        dict: Dictionary containing analyzed data arrays
    """
    # Initialize parameters using ParameterManager
    params = {
        "opto": ParameterManager.initialize_params(
            AnalysisParamType.OPTOGENETICS, opto_params
        ),
        "saccade": ParameterManager.initialize_params(
            AnalysisParamType.SACCADE, saccade_params
        ),
        "trajectory": ParameterManager.initialize_params(
            AnalysisParamType.TRAJECTORY, trajectory_params
        ),
    }

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
        "responsive": [],
    }

    for _, row in opto.iterrows():
        if "exp_num" in row:
            grp = df[
                (df["obj_id"] == row["obj_id"]) & (df["exp_num"] == row["exp_num"])
            ]
        else:
            grp = df[df["obj_id"] == row["obj_id"]]

        if len(grp) < params["opto"].min_frames:
            logger.debug(f"Skipping trajectory with {len(grp)} frames")
            continue

        try:
            opto_idx = np.where(grp["frame"] == row["frame"])[0][0]
        except IndexError:
            logger.debug("Skipping trajectory with no opto frame")
            continue

        if opto_idx - params["opto"].pre_frames < 0 or opto_idx + params[
            "opto"
        ].post_frames >= len(grp):
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
            radius[opto_idx : opto_idx + params["opto"].opto_duration]
            < params["opto"].opto_radius
        )

        # get opto range
        opto_range = range(
            opto_idx - params["opto"].pre_frames, opto_idx + params["opto"].post_frames
        )

        opto_data["angular_velocity"].append(angular_velocity[opto_range])
        opto_data["linear_velocity"].append(linear_velocity[opto_range])
        opto_data["xyz"].append(grp[["x", "y", "z"]].values[opto_range])
        opto_data["frames_in_opto_radius"].append(frames_in_opto_radius)

        # calculate heading difference using trajectory params
        heading_difference = calculate_heading_diff(
            heading,
            range(opto_idx - params["trajectory"].heading_diff_window, opto_idx),
            range(
                opto_idx,
                opto_idx
                + params["opto"].opto_duration
                + params["trajectory"].heading_diff_window,
            ),
        )

        opto_data["heading_difference"].append(heading_difference)
        opto_data["sham"].append(row["sham"] if "sham" in row else False)

        # find all peaks
        peaks = detect_saccades(
            angular_velocity, params["saccade"].threshold, params["saccade"].distance
        )
        opto_peak = [
            peak
            for peak in peaks
            if opto_idx < peak < opto_idx + params["opto"].opto_duration
        ]

        # initialize peak centered data
        peak_centered_angular_velocity = np.full(
            params["opto"].pre_frames + params["opto"].post_frames, np.nan
        )
        peak_centered_linear_velocity = np.full(
            params["opto"].pre_frames + params["opto"].post_frames, np.nan
        )
        peak_centered_heading_difference = np.nan
        reaction_delay = np.nan
        responsive = False

        if len(opto_peak) == 0:
            logger.debug("No peak found")
            continue
        # if multiple peaks found, append the first peak
        elif opto_peak[0] - params["saccade"].pre_frames < 0 or opto_peak[0] + params[
            "saccade"
        ].post_frames >= len(grp):
            logger.debug("Skipping peak with insufficient frames")
            continue
        else:
            opto_peak = opto_peak[0]  # only take the first peak

            # get peak centered data
            peak_centered_angular_velocity = angular_velocity[
                opto_peak - params["saccade"].pre_frames : opto_peak
                + params["saccade"].post_frames
            ]
            peak_centered_linear_velocity = linear_velocity[
                opto_peak - params["saccade"].pre_frames : opto_peak
                + params["saccade"].post_frames
            ]

            # get peak centered heading difference
            peak_centered_heading_difference = calculate_heading_diff(
                heading,
                range(opto_peak - params["trajectory"].heading_diff_window, opto_peak),
                range(opto_peak, opto_peak + params["trajectory"].heading_diff_window),
            )

            # calculate reaction delay
            reaction_delay = opto_peak - opto_idx
            responsive = True

        # append peak centered data
        opto_data["angular_velocity_peak_centered"].append(
            peak_centered_angular_velocity
        )
        opto_data["linear_velocity_peak_centered"].append(peak_centered_linear_velocity)
        opto_data["heading_difference_peak_centered"].append(
            peak_centered_heading_difference
        )
        opto_data["reaction_delay"].append(reaction_delay)
        opto_data["responsive"].append(responsive)

    return dict_list_to_numpy(opto_data)


def get_stim_data(
    df: pd.DataFrame,
    stim: pd.DataFrame,
    looming_params: dict | LoomingAnalysis | None = None,
    saccade_params: dict | SaccadeParams | None = None,
    trajectory_params: dict
    | TrajectoryParams
    | None = None,  # Added for heading_diff_window
) -> dict:
    """Analyze looming stimulus data with parameter validation.

    Args:
        df: DataFrame containing trajectory data
        stim: DataFrame containing stimulus events
        looming_params: Looming stimulus analysis parameters
        saccade_params: Saccade detection parameters
        trajectory_params: Trajectory analysis parameters

    Returns:
        dict: Dictionary containing analyzed data arrays
    """
    # Initialize parameters using ParameterManager
    params = {
        "looming": ParameterManager.initialize_params(
            AnalysisParamType.LOOMING, looming_params
        ),
        "saccade": ParameterManager.initialize_params(
            AnalysisParamType.SACCADE, saccade_params
        ),
        "trajectory": ParameterManager.initialize_params(
            AnalysisParamType.TRAJECTORY, trajectory_params
        ),
    }

    stim_data = {
        "angular_velocity": [],
        "angular_velocity_peak_centered": [],
        "linear_velocity": [],
        "linear_velocity_peak_centered": [],
        "xyz": [],
        "heading_difference": [],
        "heading_difference_peak_centered": [],
        "reaction_delay": [],
        "responsive": [],
    }

    for _, row in stim.iterrows():
        if "exp_num" in row:
            grp = df[
                (df["obj_id"] == row["obj_id"]) & (df["exp_num"] == row["exp_num"])
            ]
        else:
            grp = df[df["obj_id"] == row["obj_id"]]

        if len(grp) < params["looming"].min_frames:
            logger.debug(f"Skipping trajectory with {len(grp)} frames")
            continue

        try:
            stim_idx = np.where(grp["frame"] == row["frame"])[0][0]
        except IndexError:
            logger.debug("Skipping trajectory with no stimulus frame")
            continue

        if stim_idx - params["looming"].pre_frames < 0 or stim_idx + params[
            "looming"
        ].post_frames >= len(grp):
            logger.debug("Skipping trajectory with insufficient frames")
            continue

        # get angular and linear velocity
        heading, angular_velocity = calculate_angular_velocity(
            grp.xvel.values, grp.yvel.values
        )
        linear_velocity = calculate_linear_velocity(grp.xvel.values, grp.yvel.values)

        # get stimulus range
        stim_range = range(
            stim_idx - params["looming"].pre_frames,
            stim_idx + params["looming"].post_frames,
        )
        stim_data["angular_velocity"].append(angular_velocity[stim_range])
        stim_data["linear_velocity"].append(linear_velocity[stim_range])
        stim_data["xyz"].append(grp[["x", "y", "z"]].values[stim_range])

        # calculate heading difference
        heading_difference = calculate_heading_diff(
            heading,
            range(stim_idx, stim_idx + params["looming"].looming_duration),
            range(
                stim_idx + params["looming"].looming_duration,
                stim_idx
                + params["looming"].looming_duration
                + params["trajectory"].heading_diff_window,
            ),
        )

        stim_data["heading_difference"].append(heading_difference)

        # find all peaks
        peaks = detect_saccades(
            angular_velocity, params["saccade"].threshold, params["saccade"].distance
        )
        stim_peak = [
            peak
            for peak in peaks
            if stim_idx
            < peak
            < stim_idx
            + params["looming"].looming_duration
            + params["looming"].response_delay
        ]

        # initialize peak centered data
        peak_centered_angular_velocity = np.full(
            params["looming"].pre_frames + params["looming"].post_frames, np.nan
        )
        peak_centered_linear_velocity = np.full(
            params["looming"].pre_frames + params["looming"].post_frames, np.nan
        )
        peak_centered_heading_difference = np.nan
        reaction_delay = np.nan
        responsive = False

        if len(stim_peak) == 0:
            logger.debug("No peak found")
            continue
        # if multiple peaks found, append the first peak
        elif stim_peak[0] - params["looming"].pre_frames < 0 or stim_peak[0] + params[
            "looming"
        ].post_frames >= len(grp):
            logger.debug("Skipping peak with insufficient frames")
            continue
        else:
            stim_peak = stim_peak[0]  # only take the first peak

            # get peak centered data
            peak_centered_angular_velocity = angular_velocity[
                stim_peak - params["looming"].pre_frames : stim_peak
                + params["looming"].post_frames
            ]
            peak_centered_linear_velocity = linear_velocity[
                stim_peak - params["looming"].pre_frames : stim_peak
                + params["looming"].post_frames
            ]

            # get peak centered heading difference
            peak_centered_heading_difference = calculate_heading_diff(
                heading,
                range(stim_peak - params["trajectory"].heading_diff_window, stim_peak),
                range(stim_peak, stim_peak + params["trajectory"].heading_diff_window),
            )

            # calculate reaction delay
            reaction_delay = stim_peak - stim_idx
            responsive = True

        # append peak centered data
        stim_data["angular_velocity_peak_centered"].append(
            peak_centered_angular_velocity
        )
        stim_data["linear_velocity_peak_centered"].append(peak_centered_linear_velocity)
        stim_data["heading_difference_peak_centered"].append(
            peak_centered_heading_difference
        )
        stim_data["reaction_delay"].append(reaction_delay)
        stim_data["responsive"].append(responsive)

    return dict_list_to_numpy(stim_data)


def filter_trajectories(
    df: pd.DataFrame,
    saccade_params: dict | SaccadeParams | None = None,
    trajectory_params: dict | TrajectoryParams | None = None,
) -> pd.DataFrame:
    """Filter trajectory data based on length and spatial criteria.

    Args:
        df: DataFrame containing trajectory data
        saccade_params: Saccade detection parameters
        trajectory_params: Trajectory analysis parameters including spatial bounds

    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid trajectories
    """
    # Initialize parameters using ParameterManager
    params = {
        "saccade": ParameterManager.initialize_params(
            AnalysisParamType.SACCADE, saccade_params
        ),
        "trajectory": ParameterManager.initialize_params(
            AnalysisParamType.TRAJECTORY, trajectory_params
        ),
    }

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
    length_mask = (
        group_stats["trajectory_length"] >= params["saccade"].min_trajectory_length
    )

    x_median_mask = (group_stats["x"] > params["trajectory"].xy_range[0]) & (
        group_stats["x"] < params["trajectory"].xy_range[1]
    )
    y_median_mask = (group_stats["y"] > params["trajectory"].xy_range[0]) & (
        group_stats["y"] < params["trajectory"].xy_range[1]
    )
    z_median_mask = (group_stats["z"] > params["trajectory"].z_range[0]) & (
        group_stats["z"] < params["trajectory"].z_range[1]
    )

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


def get_all_saccades(
    df: pd.DataFrame,
    saccade_params: dict | SaccadeParams | None = None,
    trajectory_params: dict | TrajectoryParams | None = None,
) -> dict:
    """Analyze all saccades in trajectory data with parameter validation.

    Args:
        df: DataFrame containing trajectory data
        saccade_params: Saccade detection parameters
        trajectory_params: Trajectory analysis parameters including spatial bounds

    Returns:
        dict: Dictionary containing analyzed saccade data arrays
    """
    # Filter trajectories first
    filtered_df = filter_trajectories(df, saccade_params, trajectory_params)

    # Initialize parameters using ParameterManager
    params = {
        "saccade": ParameterManager.initialize_params(
            AnalysisParamType.SACCADE, saccade_params
        ),
        "trajectory": ParameterManager.initialize_params(
            AnalysisParamType.TRAJECTORY, trajectory_params
        ),
    }

    saccade_data = {
        "angular_velocity": [],
        "linear_velocity": [],
        "xyz": [],
        "heading_difference": [],
        "ISI": [],
        "saccade_per_second": [],
        "trajectory_duration": [],
        "num_of_saccades": [],
    }

    # Group the filtered data
    grouped_data = (
        filtered_df.groupby(["obj_id", "exp_num"])
        if "exp_num" in filtered_df.columns
        else filtered_df.groupby("obj_id")
    )

    for _, grp in tqdm(
        grouped_data, desc="Processing trajectories", total=len(grouped_data)
    ):
        # Calculate velocities for the entire trajectory
        heading, angular_velocity = calculate_angular_velocity(
            grp.xvel.values, grp.yvel.values
        )
        linear_velocity = calculate_linear_velocity(grp.xvel.values, grp.yvel.values)

        # Normalize linear velocity and get flight bouts
        normalized_linear_velocity = linear_velocity / np.max(linear_velocity)
        flight_bouts = apply_hysteresis_filter(normalized_linear_velocity)

        # Process each flight bout separately
        for bout_indices in flight_bouts:
            # Detect saccades within this bout
            bout_angular_velocity = angular_velocity[bout_indices]
            bout_saccades = detect_saccades(
                bout_angular_velocity,
                params["saccade"].threshold,
                params["saccade"].distance,
            )

            # Convert bout-relative saccade indices to trajectory-relative indices (so we can index into the full angular_velocity and linear_velocity)
            trajectory_saccades = bout_indices[bout_saccades]

            good_saccades = []
            # Process each saccade in this bout
            for sac in trajectory_saccades:
                # Check if we have enough frames before and after the saccade
                if sac - params["saccade"].pre_frames < 0 or sac + params[
                    "saccade"
                ].post_frames >= len(grp):
                    logger.debug("Skipping saccade with insufficient frames")
                    continue

                # Check spatial bounds at saccade time
                if not (
                    (
                        np.sqrt(grp.x.iloc[sac] ** 2 + grp.y.iloc[sac] ** 2)
                        < params["trajectory"].max_radius
                    )
                    and (
                        params["trajectory"].z_range[0]
                        < grp.z.iloc[sac]
                        < params["trajectory"].z_range[1]
                    )
                ):
                    continue

                # Extract saccade traces
                saccade_angular_velocity = angular_velocity[
                    sac - params["saccade"].pre_frames : sac
                    + params["saccade"].post_frames
                ]
                saccade_linear_velocity = linear_velocity[
                    sac - params["saccade"].pre_frames : sac
                    + params["saccade"].post_frames
                ]
                saccade_xyz = grp[["x", "y", "z"]].values[
                    sac - params["saccade"].pre_frames : sac
                    + params["saccade"].post_frames
                ]

                heading_difference = calculate_heading_diff(
                    heading,
                    range(sac - params["trajectory"].heading_diff_window, sac),
                    range(sac, sac + params["trajectory"].heading_diff_window),
                )

                # Store saccade data
                saccade_data["angular_velocity"].append(saccade_angular_velocity)
                saccade_data["linear_velocity"].append(saccade_linear_velocity)
                saccade_data["xyz"].append(saccade_xyz)
                saccade_data["heading_difference"].append(heading_difference)
                good_saccades.append(sac)

            # Calculate bout-specific metrics
            bout_duration = (
                grp.timestamp.iloc[bout_indices[-1]]
                - grp.timestamp.iloc[bout_indices[0]]
            )

            if len(good_saccades) < 2:
                saccade_data["ISI"].append(np.nan)
                saccade_data["saccade_per_second"].append(np.nan)
            else:
                isis = np.diff(good_saccades)

                # remove any isis less than 1 and bigger than 500
                isis = isis[(isis > 1) & (isis < 500)]
                saccade_data["ISI"].append(np.nanmean(isis))
                saccade_data["saccade_per_second"].append(
                    len(good_saccades) / bout_duration
                )

            saccade_data["num_of_saccades"].append(len(good_saccades))
            saccade_data["trajectory_duration"].append(bout_duration)

    return dict_list_to_numpy(saccade_data)
