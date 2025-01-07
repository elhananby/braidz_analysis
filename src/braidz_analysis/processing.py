import logging
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .helpers import dict_list_to_numpy
from .trajectory import (
    calculate_angular_velocity,
    calculate_heading_diff,
    calculate_linear_velocity,
    detect_saccades,
    calculate_smoothed_linear_velocity,
)
from .filtering import apply_hysteresis_filter


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_stim_or_opto_response_data(
    df: pd.DataFrame, opto_or_stim: pd.DataFrame, **kwargs: Dict[str, Any]
):
    output_data = {
        "angular_velocity": [],
        "linear_velocity": [],
        "position": [],
        "heading_difference": [],
        "frames_in_radius": [],
        "sham": [],
        "reaction_delay": [],
        "responsive": [],
        "intensity": [],
        "duration": [],
        "frequency": [],
    }

    for _, row in opto_or_stim.iterrows():
        if "exp_num" in row:
            grp = df[
                (df["obj_id"] == row["obj_id"]) & (df["exp_num"] == row["exp_num"])
            ]
        else:
            grp = df[df["obj_id"] == row["obj_id"]]

        if len(grp) < kwargs.get("min_frames", 300):
            logger.debug(f"Skipping trajectory with {len(grp)} frames")
            continue

        try:
            opto_or_stim_idx = np.where(grp["frame"] == row["frame"])[0][0]
        except IndexError:
            logger.debug("Skipping trajectory with no opto frame")
            continue

        if opto_or_stim_idx - kwargs.get(
            "pre_frames", 50
        ) < 0 or opto_or_stim_idx + kwargs.get("post_frames", 50) >= len(grp):
            logger.debug("Skipping trajectory with insufficient frames")
            continue

        # get angular and linear velocity
        heading, angular_velocity = calculate_angular_velocity(
            grp.xvel.values, grp.yvel.values
        )
        linear_velocity = calculate_smoothed_linear_velocity(grp)

        # calculate how many frames in opto radius
        radius = np.sqrt(grp.x.values**2 + grp.y.values**2)
        frames_in_radius = np.sum(
            radius[opto_or_stim_idx : opto_or_stim_idx + kwargs.get("duration", 30)]
            < kwargs.get("radius", 0.025)
        )

        peaks = detect_saccades(
            angular_velocity,
            kwargs.get("threshold", np.deg2rad(300)),
            kwargs.get("distance", 10),
        )

        opto_or_stim_peaks = [
            peak
            for peak in peaks
            if opto_or_stim_idx < peak < opto_or_stim_idx + kwargs.get("duration", 30)
        ]

        # initialize empty peak centered data
        peak_centered_angular_velocity = np.full(
            kwargs.get("pre_frames", 50) + kwargs.get("post_frames", 50), np.nan
        )
        peak_centered_linear_velocity = np.full(
            kwargs.get("pre_frames", 50) + kwargs.get("post_frames", 50), np.nan
        )
        peak_centered_position = np.full(
            (kwargs.get("pre_frames", 50) + kwargs.get("post_frames", 50), 3), np.nan
        )
        peak_centered_heading_difference = np.nan
        reaction_delay = np.nan
        responsive = False

        # check if no peaks found
        if len(opto_or_stim_peaks) == 0:
            logger.debug("No peak found")
            
        # check if the first peak is within bounds
        elif opto_or_stim_peaks[0] - kwargs.get(
            "pre_frames", 50
        ) < 0 or opto_or_stim_peaks[0] + kwargs.get("post_frames", 50) >= len(grp):
            logger.debug("Skipping peak with insufficient frames")
            continue

        else:
            opto_or_stim_peaks = opto_or_stim_peaks[0]  # only take the first peak

            range_to_extract = range(
                opto_or_stim_peaks - kwargs.get("pre_frames", 50),
                opto_or_stim_peaks + kwargs.get("post_frames", 50),
            )

            # get peak centered data
            peak_centered_angular_velocity = angular_velocity[range_to_extract]
            peak_centered_linear_velocity = linear_velocity[range_to_extract]

            # get peak centered heading difference
            peak_centered_heading_difference = calculate_heading_diff(
                heading,
                range(
                    opto_or_stim_peaks - kwargs.get("heading_diff_window", 10),
                    opto_or_stim_peaks,
                ),
                range(
                    opto_or_stim_peaks,
                    opto_or_stim_peaks + kwargs.get("heading_diff_window", 10),
                ),
            )

            # calculate reaction delay
            reaction_delay = opto_or_stim_peaks - opto_or_stim_idx
            responsive = True
            peak_centered_position = grp[["x", "y", "z"]].to_numpy()[range_to_extract]

        # append peak centered data
        output_data["angular_velocity"].append(peak_centered_angular_velocity)
        output_data["linear_velocity"].append(peak_centered_linear_velocity)
        output_data["position"].append(peak_centered_position)

        output_data["heading_difference"].append(peak_centered_heading_difference)
        output_data["frames_in_radius"].append(frames_in_radius)
        output_data["sham"].append(row.get("sham", False))
        output_data["reaction_delay"].append(reaction_delay)
        output_data["responsive"].append(responsive)

    return dict_list_to_numpy(output_data)


def get_stim_or_opto_data(
    df: pd.DataFrame,
    stim_or_opto: pd.DataFrame,
    **kwargs: Optional[dict],
) -> dict:
    """Analyze optogenetic stimulation data with parameter validation.

    Args:
        df: DataFrame containing trajectory data.
        stim_or_opto: DataFrame containing optogenetic or stimulation events.
        min_frames (optional): Minimum number of frames required for a trajectory. Defaults to 300.
        pre_frames (optional): Number of frames before the event to extract. Defaults to 50.
        post_frames (optional): Number of frames after the event to extract. Defaults to 100.
        duration (optional): Duration of the event in frames. Defaults to 300.
        radius (optional): Radius of the event in meters. Defaults to 0.025.
        heading_diff_window (optional): Number of frames to consider for heading difference. Defaults to 10.

    Returns:
        dict: Dictionary containing analyzed data arrays
    """

    opto_data = {
        "angular_velocity": [],
        "linear_velocity": [],
        "position": [],
        "heading_difference": [],
        "frames_in_radius": [],
        "sham": [],
        "intensity": [],
        "duration": [],
        "frequency": [],
    }

    for _, row in stim_or_opto.iterrows():
        if "exp_num" in row:
            grp = df[
                (df["obj_id"] == row["obj_id"]) & (df["exp_num"] == row["exp_num"])
            ]
        else:
            grp = df[df["obj_id"] == row["obj_id"]]

        if len(grp) < kwargs.get("min_frames", 300):
            logger.debug(f"Skipping trajectory with {len(grp)} frames")
            continue

        try:
            stim_or_opto_idx = np.where(grp["frame"] == row["frame"])[0][0]
        except IndexError:
            logger.debug("Skipping trajectory with no opto frame")
            continue

        range_to_extract = range(
            stim_or_opto_idx - kwargs.get("pre_frames", 50),
            stim_or_opto_idx + kwargs.get("post_frames", 100),
        )

        # check if any values in `range_to_extract` are out of bounds
        if range_to_extract[0] < 0 or range_to_extract[-1] >= len(grp):
            logger.debug("Skipping trajectory with insufficient frames")
            continue

        # get angular and linear velocity
        heading, angular_velocity = calculate_angular_velocity(
            grp.xvel.values, grp.yvel.values
        )
        linear_velocity = calculate_smoothed_linear_velocity(grp)

        # calculate how many frames in opto radius
        radius = np.sqrt(grp.x.values**2 + grp.y.values**2)
        frames_in_radius = np.sum(
            radius[stim_or_opto_idx : stim_or_opto_idx + kwargs.get("duration", 30)]
            < kwargs.get("radius", 0.025)
        )

        # calculate heading difference using trajectory params
        heading_difference = calculate_heading_diff(
            heading,
            range(
                stim_or_opto_idx - kwargs.get("heading_diff_window", 10),
                stim_or_opto_idx,
            ),
            range(stim_or_opto_idx, stim_or_opto_idx + kwargs.get("duration", 30)),
        )

        # append all data
        opto_data["angular_velocity"].append(angular_velocity[range_to_extract])
        opto_data["linear_velocity"].append(linear_velocity[range_to_extract])
        opto_data["position"].append(grp[["x", "y", "z"]].values[range_to_extract])
        opto_data["frames_in_radius"].append(frames_in_radius)
        opto_data["heading_difference"].append(heading_difference)
        
        # append stim or opto specific data
        opto_data["sham"].append(row["sham"] if "sham" in row else False)
        opto_data["intensity"].append(row["intensity"] if "intensity" in row else np.nan)
        opto_data["duration"].append(row["duration"] if "duration" in row else np.nan)
        opto_data["frequency"].append(row["frequency"] if "frequency" in row else np.nan)

    return dict_list_to_numpy(opto_data)


def get_all_saccades(
    df: pd.DataFrame,
    **kwargs: Optional[dict],
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
    filtered_df = filter_trajectories(df, **kwargs)

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
                kwargs.get("threshold", np.deg2rad(300)),
                kwargs.get("distance", 10),
            )

            # Convert bout-relative saccade indices to trajectory-relative indices (so we can index into the full angular_velocity and linear_velocity)
            trajectory_saccades = bout_indices[bout_saccades]

            good_saccades = []

            # Process each saccade in this bout
            for sac in trajectory_saccades:
                # Check if we have enough frames before and after the saccade
                if sac - kwargs.get("pre_frames", 50) < 0 or sac + kwargs.get(
                    "post_frames", 50
                ) >= len(grp):
                    logger.debug("Skipping saccade with insufficient frames")
                    continue

                # Check spatial bounds at saccade time
                if not (
                    (
                        np.sqrt(grp.x.iloc[sac] ** 2 + grp.y.iloc[sac] ** 2)
                        < kwargs.get("radius", 0.025)
                    )
                    and (
                        kwargs.get("zlim", [0.05, 0.3])[0]
                        < grp.z.iloc[sac]
                        < kwargs.get("zlim", [0.05, 0.3])[1]
                    )
                ):
                    continue

                # Extract saccade traces
                range_to_extract = range(
                    sac - kwargs.get("pre_frames", 50),
                    sac + kwargs.get("post_frames", 50),
                )
                saccade_angular_velocity = angular_velocity[range_to_extract]
                saccade_linear_velocity = linear_velocity[range_to_extract]
                saccade_xyz = grp[["x", "y", "z"]].values[range_to_extract]

                heading_difference = calculate_heading_diff(
                    heading,
                    range(sac - kwargs.get("heading_diff_windows, 10"), sac),
                    range(sac, sac + kwargs.get("heading_diff_windows, 10")),
                )

                # Store saccade data
                saccade_data["angular_velocity"].append(saccade_angular_velocity)
                saccade_data["linear_velocity"].append(saccade_linear_velocity)
                saccade_data["position"].append(saccade_xyz)
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
