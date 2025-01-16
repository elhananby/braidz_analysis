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
    detect_saccades,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _find_relevant_peak(
    peaks: Union[list, np.ndarray],
    stim_idx: int,
    duration: int = 30,
    return_none: bool = False,
):
    """Find first peak within the stimulus window."""
    window_end = stim_idx + duration

    for peak in peaks:
        if stim_idx < peak < window_end:
            return int(peak), True

    return None if return_none else (stim_idx + duration // 2), False


def _get_pre_saccade(
    peaks: Union[list, np.ndarray], stim_idx: int, window: int = 100
) -> int:
    """
    Get the closest peak before the stimulation event.

    Args:
        peaks: List of peak indices
        stim_idx: Index of the stimulation event
        window: Number of frames to search before the stimulation event

    Returns:
        peak: Index of the closest peak before the stimulation event
    """
    # Filter peaks that are within window before stim
    valid_peaks = [p for p in peaks if stim_idx - window < p < stim_idx]

    if not valid_peaks:
        return None

    # Find peak closest to stim_idx
    return max(valid_peaks)


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
        "intensity": [],
        "duration": [],
        "frequency": [],
    }

    for _, row in tqdm(opto_or_stim.iterrows(), total=len(opto_or_stim)):
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

            peak, _ = _find_relevant_peak(
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
            output_data["intensity"].append(row.get("intensity", np.nan))
            output_data["duration"].append(row.get("duration", np.nan))
            output_data["frequency"].append(row.get("frequency", np.nan))

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
        "timestamp": [],
        "angular_velocity": [],
        "linear_velocity": [],
        "position": [],
        "heading_difference": [],
        "frames_in_radius": [],
        "pre_saccade_amplitude": [],
        "pre_saccade_distance_from_stim_idx": [],
        "sham": [],
        "intensity": [],
        "duration": [],
        "frequency": [],
        "responsive": [],
    }

    first_frame = df["frame"].iloc[0]

    for _, row in tqdm(stim_or_opto.iterrows(), total=len(stim_or_opto)):
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
            linear_velocity = calculate_linear_velocity(
                grp.xvel.values, grp.yvel.values
            )

            # detect all saccades in the trajectory
            peaks = detect_saccades(
                angular_velocity,
                height=saccade_params.threshold,
                distance=saccade_params.distance,
            )

            # find saccades that happened before the stimulation event
            pre_saccade_idx = _get_pre_saccade(
                peaks,
                stim_idx,
            )

            if pre_saccade_idx is not None:
                pre_saccade_amplitude = angular_velocity[pre_saccade_idx]
                pre_saccade_distance_from_stim_idx = pre_saccade_idx - stim_idx
            else:
                pre_saccade_amplitude = np.nan
                pre_saccade_distance_from_stim_idx = np.nan

            # find the first peak within the stimulus window
            peak, responsive = _find_relevant_peak(
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

            opto_data["timestamp"].append((row["frame"] - first_frame)*0.01)
            opto_data["angular_velocity"].append(angular_velocity[range_to_extract])
            opto_data["linear_velocity"].append(linear_velocity[range_to_extract])
            opto_data["position"].append(grp[["x", "y", "z"]].values[range_to_extract])
            opto_data["frames_in_radius"].append(frames_in_radius)
            opto_data["heading_difference"].append(heading_difference)
            opto_data["sham"].append(row.get("sham", False))
            opto_data["responsive"].append(responsive)
            opto_data["intensity"].append(row.get("intensity", np.nan))
            opto_data["duration"].append(row.get("duration", np.nan))
            opto_data["frequency"].append(row.get("frequency", np.nan))
            opto_data["pre_saccade_amplitude"].append(pre_saccade_amplitude)
            opto_data["pre_saccade_distance_from_stim_idx"].append(
                pre_saccade_distance_from_stim_idx
            )

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
        "timestamp": [],
    }

    for _, grp in tqdm(grouped_data, desc="Processing trajectories"):
        heading, angular_velocity = calculate_angular_velocity(
            grp.xvel.values, grp.yvel.values
        )
        linear_velocity = calculate_linear_velocity(grp.xvel.values, grp.yvel.values)

        peaks = detect_saccades(
            angular_velocity,
            height=params.threshold,
            distance=params.distance,
        )

        x = savgol_filter(grp.x.values, 21, 3)
        y = savgol_filter(grp.y.values, 21, 3)
        z = savgol_filter(grp.z.values, 21, 3)
        pos = np.column_stack((x, y, z))
        timestamps = []
        for sac in peaks:
            if sac - params.pre_frames < 0 or sac + params.post_frames >= len(grp):
                continue

            if (
                np.sqrt(grp.x.iloc[sac] ** 2 + grp.y.iloc[sac] ** 2) < params.max_radius
                and params.zlim[0] < grp.z.iloc[sac] < params.zlim[1]
            ):
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
                    pos[range_to_extract, :]
                )
                saccade_data["heading_difference"].append(heading_difference)
                timestamps.append(grp["timestamp"].iloc[sac])
        
        saccade_data["timestamp"].append(timestamps)
        saccade_data["isi"].append(np.diff(timestamps))

    return dict_list_to_numpy(saccade_data)

def filter_trajectories(
    df: pd.DataFrame,
    min_frames: int = 200,
    z_bounds: tuple = (0.1, 0.3),
    max_radius: float = 0.23,
    required_cols: tuple = ('x', 'y', 'z', 'obj_id', 'exp_num')
) -> pd.DataFrame:
    """
    Filter animal locomotor trajectories based on spatial and temporal criteria.
    
    Parameters:
    -----------
    df : pd.DataFrame
        High-resolution tracking data containing positional coordinates and metadata
    min_frames : int
        Minimum trajectory duration in frames for reliable behavioral analysis
    z_bounds : tuple
        Acceptable vertical position bounds (min, max) in normalized units
    max_radius : float
        Maximum allowed radial distance from origin in normalized units
    required_cols : tuple
        Required column names for trajectory analysis
        
    Returns:
    --------
    pd.DataFrame
        Filtered trajectories meeting all quality criteria for subsequent analysis
    """
    # Validate input data structure
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    zmin, zmax = z_bounds
    good_grps = []
    
    # Create trajectory groups with progress bar
    trajectory_groups = df.groupby(by=["obj_id", "exp_num"])
    
    # Iterate through trajectories with tqdm progress tracking
    for (obj_id, exp_num), grp in tqdm(
        trajectory_groups,
        desc="Filtering trajectories",
        total=len(trajectory_groups)
    ):
        if len(grp) < min_frames:
            continue
            
        # Compute spatial metrics for quality assessment
        radius = np.sqrt(grp.x.values**2 + grp.y.values**2)
        radius_median = np.nanmedian(radius)
        z_median = np.nanmedian(grp.z)
        
        # Apply spatial filtering criteria
        if (radius_median > max_radius) or (z_median < zmin) or (z_median > zmax):
            continue
            
        good_grps.append(grp)
    
    if not good_grps:
        logging.debug("No trajectories met the specified quality criteria")
        return pd.DataFrame(columns=df.columns)
        
    return pd.concat(good_grps, ignore_index=True)

def get_pre_saccade(
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

    results = {
        "saccade_amplitude": [],
        "saccade_direction": [],
        "pre_saccade_amplitude": [],
        "pre_saccade_direction": [],
        "distance": [],
        "responsive": [],
    }

    for _, row in tqdm(stim_or_opto.iterrows(), total=len(stim_or_opto)):
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

            heading, angular_velocity = calculate_angular_velocity(
                grp.xvel.values, grp.yvel.values
            )

            # detect all saccades in the trajectory
            peaks = detect_saccades(
                angular_velocity,
                height=saccade_params.threshold,
                distance=saccade_params.distance,
            )

            saccade_peak_idx = np.nan

            # find the first peak within the stimulus window
            for peak in peaks:
                if stim_idx < peak < stim_idx + params.duration:
                    saccade_peak_idx = peak
                    break

            # find the closest peak before the stimulation event
            pre_saccade_peak_idx = [
                peak for peak in peaks if stim_idx - params.pre_frames < peak < stim_idx
            ]
            pre_saccade_peak_idx = (
                np.max(pre_saccade_peak_idx) if pre_saccade_peak_idx else np.nan
            )

            # get amplitude for each
            saccade_amplitude = angular_velocity[saccade_peak_idx]
            pre_saccade_amplitude = angular_velocity[pre_saccade_peak_idx]

            # get direction
            saccade_direction = np.sign(angular_velocity[saccade_peak_idx])
            pre_saccade_direction = np.sign(angular_velocity[pre_saccade_peak_idx])

            # get distance from stim_idx
            pre_saccade_distance = pre_saccade_peak_idx - stim_idx

            # append to dict
            results["saccade_amplitude"].append(saccade_amplitude)
            results["saccade_direction"].append(saccade_direction)
            results["pre_saccade_amplitude"].append(pre_saccade_amplitude)
            results["pre_saccade_direction"].append(pre_saccade_direction)
            results["distance"].append(pre_saccade_distance)
            results["responsive"].append(not np.isnan(saccade_peak_idx))

        except (IndexError, ValueError) as e:
            logger.debug(f"Skipping trajectory: {str(e)}")
            continue

    return dict_list_to_numpy(results)
