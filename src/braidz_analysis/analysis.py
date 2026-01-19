"""
High-level analysis module for braidz trajectory data.

This module provides the main entry points for analyzing fly behavior:
    - analyze_saccades: Detect and characterize all saccades in trajectories
    - analyze_event_responses: Analyze responses to opto/stim events

Both functions return structured result objects with traces, metrics, and
convenient filtering methods.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm

from .config import Config, DEFAULT_CONFIG
from .kinematics import (
    classify_flight_state,
    compute_angular_velocity,
    compute_heading_change,
    compute_linear_velocity,
    detect_saccades,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Result Classes
# =============================================================================


@dataclass
class SaccadeResults:
    """
    Results from saccade analysis.

    Attributes:
        traces: Dictionary of time-aligned arrays (n_saccades, n_frames):
            - angular_velocity: Angular velocity around each saccade
            - linear_velocity: Linear velocity around each saccade
            - position: Position (n_saccades, n_frames, 3) for x, y, z

        metrics: DataFrame with one row per saccade:
            - heading_change: Net heading change (radians)
            - peak_velocity: Peak angular velocity (rad/s)
            - amplitude: Total angular displacement (radians)
            - direction: Turn direction (+1=left/CCW, -1=right/CW)
            - obj_id, exp_num: Source trajectory identifiers
            - frame: Frame index of saccade peak

        config: Configuration used for analysis.
    """

    traces: Dict[str, np.ndarray]
    metrics: pd.DataFrame
    config: Config = field(default_factory=lambda: DEFAULT_CONFIG)

    def __len__(self) -> int:
        return len(self.metrics)

    def __repr__(self) -> str:
        return f"SaccadeResults({len(self)} saccades)"

    def filter(self, mask: np.ndarray = None, **kwargs) -> "SaccadeResults":
        """
        Filter results by boolean mask or column conditions.

        Args:
            mask: Boolean array of length n_saccades.
            **kwargs: Column conditions, e.g., direction=1 for left turns.

        Returns:
            New SaccadeResults with filtered data.

        Example:
            >>> left_turns = results.filter(direction=1)
            >>> large_turns = results.filter(mask=np.abs(results.metrics['heading_change']) > 0.5)
        """
        if mask is None:
            mask = np.ones(len(self), dtype=bool)

        # Apply column conditions
        for col, val in kwargs.items():
            if col in self.metrics.columns:
                mask = mask & (self.metrics[col].values == val)

        # Filter traces
        filtered_traces = {}
        for key, arr in self.traces.items():
            if arr.ndim == 1:
                filtered_traces[key] = arr[mask]
            elif arr.ndim == 2:
                filtered_traces[key] = arr[mask, :]
            elif arr.ndim == 3:
                filtered_traces[key] = arr[mask, :, :]

        return SaccadeResults(
            traces=filtered_traces,
            metrics=self.metrics[mask].reset_index(drop=True),
            config=self.config,
        )

    @property
    def left_turns(self) -> "SaccadeResults":
        """Filter to left/counterclockwise turns only."""
        return self.filter(direction=1)

    @property
    def right_turns(self) -> "SaccadeResults":
        """Filter to right/clockwise turns only."""
        return self.filter(direction=-1)


@dataclass
class EventResults:
    """
    Results from event-triggered analysis (opto or stim).

    Attributes:
        traces: Dictionary of time-aligned arrays (n_events, n_frames):
            - angular_velocity: Angular velocity around each event
            - linear_velocity: Linear velocity around each event
            - position: Position (n_events, n_frames, 3)

        metrics: DataFrame with one row per event:
            - responded: Whether a saccade occurred in response window
            - heading_change: Heading change at response peak (or reference point if no response)
            - reaction_time: Frames from event to response peak (NaN if no response)
            - peak_velocity: Angular velocity at response peak (or reference point)
            - max_velocity_in_window: Maximum |angular velocity| in response window

        metadata: DataFrame with event metadata (from opto/stim file):
            - May include: intensity, duration, frequency, sham, etc.
            - Columns depend on what's in the source event file

        config: Configuration used for analysis.

    Note:
        For non-responsive trials, metrics are computed at a reference point:
        - If 'duration' is in metadata (opto): center of stimulus duration
        - Otherwise: center of response window
        This allows comparison between responsive and non-responsive trials.
    """

    traces: Dict[str, np.ndarray]
    metrics: pd.DataFrame
    metadata: pd.DataFrame
    config: Config = field(default_factory=lambda: DEFAULT_CONFIG)

    def __len__(self) -> int:
        return len(self.metrics)

    def __repr__(self) -> str:
        n_resp = self.metrics["responded"].sum() if "responded" in self.metrics else "?"
        return f"EventResults({len(self)} events, {n_resp} responses)"

    def filter(self, mask: np.ndarray = None, **kwargs) -> "EventResults":
        """
        Filter results by boolean mask or column conditions.

        Supports filtering by both metrics and metadata columns.

        Args:
            mask: Boolean array of length n_events.
            **kwargs: Column conditions from metrics or metadata.

        Returns:
            New EventResults with filtered data.

        Example:
            >>> responsive = results.filter(responded=True)
            >>> high_intensity = results.filter(intensity=100)
            >>> non_sham = results.filter(sham=False)
        """
        if mask is None:
            mask = np.ones(len(self), dtype=bool)

        # Apply conditions from metrics
        for col, val in kwargs.items():
            if col in self.metrics.columns:
                mask = mask & (self.metrics[col].values == val)
            elif col in self.metadata.columns:
                mask = mask & (self.metadata[col].values == val)

        # Filter traces
        filtered_traces = {}
        for key, arr in self.traces.items():
            if arr.ndim == 1:
                filtered_traces[key] = arr[mask]
            elif arr.ndim == 2:
                filtered_traces[key] = arr[mask, :]
            elif arr.ndim == 3:
                filtered_traces[key] = arr[mask, :, :]

        return EventResults(
            traces=filtered_traces,
            metrics=self.metrics[mask].reset_index(drop=True),
            metadata=self.metadata[mask].reset_index(drop=True),
            config=self.config,
        )

    @property
    def responsive(self) -> "EventResults":
        """Filter to trials where fly responded."""
        return self.filter(responded=True)

    @property
    def non_responsive(self) -> "EventResults":
        """Filter to trials where fly did not respond."""
        return self.filter(responded=False)

    @property
    def real(self) -> "EventResults":
        """Filter to non-sham trials (if sham column exists)."""
        if "sham" in self.metadata.columns:
            return self.filter(sham=False)
        return self

    @property
    def sham(self) -> "EventResults":
        """Filter to sham trials only (if sham column exists)."""
        if "sham" in self.metadata.columns:
            return self.filter(sham=True)
        return EventResults(
            traces={k: v[:0] for k, v in self.traces.items()},
            metrics=self.metrics.iloc[:0],
            metadata=self.metadata.iloc[:0],
            config=self.config,
        )

    @property
    def response_rate(self) -> float:
        """Fraction of trials with a response."""
        if len(self) == 0:
            return 0.0
        return self.metrics["responded"].mean()


# =============================================================================
# Trajectory Filtering
# =============================================================================


def filter_trajectories(
    df: pd.DataFrame,
    config: Config = None,
    progressbar: bool = False,
) -> pd.DataFrame:
    """
    Filter trajectories based on quality criteria.

    Removes trajectories that are:
        - Too short (< min_trajectory_frames)
        - Outside z bounds (fly not in tracking volume)
        - Too far from center (beyond max_radius)
        - Stationary (not enough movement in x, y, z)

    Args:
        df: DataFrame with trajectory data.
        config: Analysis configuration.
        progressbar: Show progress bar.

    Returns:
        Filtered DataFrame containing only valid trajectories.
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Determine grouping columns
    group_cols = ["obj_id", "exp_num"] if "exp_num" in df.columns else ["obj_id"]

    valid_groups = []
    grouped = df.groupby(group_cols)

    iterator = tqdm(grouped, desc="Filtering trajectories", disable=not progressbar)
    for group_key, grp in iterator:
        # Check minimum length
        if len(grp) < config.min_trajectory_frames:
            continue

        # Check z bounds
        z_median = grp["z"].median()
        if not (config.z_bounds[0] <= z_median <= config.z_bounds[1]):
            continue

        # Check radial position
        radius = np.sqrt(grp["x"].values ** 2 + grp["y"].values ** 2)
        if np.median(radius) > config.max_radius:
            continue

        # Check for sufficient movement (not stationary)
        if (
            np.ptp(grp["x"]) < config.min_position_range
            or np.ptp(grp["y"]) < config.min_position_range
            or np.ptp(grp["z"]) < config.min_position_range
        ):
            continue

        valid_groups.append(grp)

    if not valid_groups:
        logger.warning("No trajectories passed quality filters")
        return pd.DataFrame(columns=df.columns)

    return pd.concat(valid_groups, ignore_index=True)


# =============================================================================
# Main Analysis Functions
# =============================================================================


def analyze_saccades(
    df: pd.DataFrame,
    config: Config = None,
    filter_trajectories_first: bool = True,
    flight_only: bool = True,
    progressbar: bool = True,
) -> SaccadeResults:
    """
    Detect and analyze all saccades in trajectory data.

    This function:
        1. Filters trajectories for quality (optional)
        2. Computes kinematics for each trajectory
        3. Detects saccades based on angular velocity threshold
        4. Extracts time-aligned traces around each saccade
        5. Computes saccade metrics (heading change, amplitude, etc.)

    Args:
        df: DataFrame with trajectory data (from read_braidz).
        config: Analysis configuration. Uses defaults if None.
        filter_trajectories_first: Apply quality filters before analysis.
        flight_only: Only analyze saccades during flight (not walking).
        progressbar: Show progress bar.

    Returns:
        SaccadeResults with traces and metrics for all detected saccades.

    Example:
        >>> data = read_braidz("experiment.braidz")
        >>> saccades = analyze_saccades(data.trajectories)
        >>> print(f"Found {len(saccades)} saccades")
        >>> print(saccades.metrics[['heading_change', 'peak_velocity']].describe())
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Apply trajectory filters
    if filter_trajectories_first:
        df = filter_trajectories(df, config, progressbar=False)

    if len(df) == 0:
        return SaccadeResults(
            traces={"angular_velocity": np.array([]), "linear_velocity": np.array([]), "position": np.array([])},
            metrics=pd.DataFrame(),
            config=config,
        )

    # Determine grouping
    group_cols = ["obj_id", "exp_num"] if "exp_num" in df.columns else ["obj_id"]
    grouped = df.groupby(group_cols)

    # Storage for results
    all_angular_vel = []
    all_linear_vel = []
    all_positions = []
    all_metrics = []

    window_size = config.pre_frames + config.post_frames

    iterator = tqdm(grouped, desc="Analyzing saccades", disable=not progressbar)
    for group_key, grp in iterator:
        # Compute kinematics
        heading, angular_velocity = compute_angular_velocity(
            grp["xvel"].values, grp["yvel"].values, dt=config.dt
        )
        linear_velocity = compute_linear_velocity(
            grp["xvel"].values, grp["yvel"].values, grp["zvel"].values
        )

        # Get positions
        x = savgol_filter(grp["x"].values, config.smoothing_window, config.smoothing_polyorder)
        y = savgol_filter(grp["y"].values, config.smoothing_window, config.smoothing_polyorder)
        z = savgol_filter(grp["z"].values, config.smoothing_window, config.smoothing_polyorder)
        position = np.column_stack([x, y, z])

        # Filter to flight only if requested
        if flight_only:
            flight_mask = classify_flight_state(
                linear_velocity,
                high_threshold=config.flight_high_threshold,
                low_threshold=config.flight_low_threshold,
                min_frames=config.flight_min_frames,
            )
            if flight_mask.sum() < 100:
                continue

            # Apply mask to all arrays
            angular_velocity = angular_velocity[flight_mask]
            linear_velocity = linear_velocity[flight_mask]
            position = position[flight_mask]
            heading = heading[flight_mask]

        # Detect saccades
        peaks = detect_saccades(
            angular_velocity,
            threshold=config.saccade_threshold,
            min_spacing=config.min_saccade_spacing,
        )

        # Extract data around each saccade
        for peak in peaks:
            # Check if we have enough frames
            if peak < config.pre_frames or peak >= len(angular_velocity) - config.post_frames:
                continue

            # Extract window
            start = peak - config.pre_frames
            end = peak + config.post_frames

            all_angular_vel.append(angular_velocity[start:end])
            all_linear_vel.append(linear_velocity[start:end])
            all_positions.append(position[start:end])

            # Compute metrics
            heading_change = compute_heading_change(
                heading, peak, window=config.heading_window
            )

            metrics = {
                "heading_change": heading_change,
                "peak_velocity": angular_velocity[peak],
                "amplitude": np.abs(heading_change),  # Simplified
                "direction": int(np.sign(angular_velocity[peak])),
                "frame": peak,
            }

            # Add trajectory identifiers
            if isinstance(group_key, tuple):
                metrics["obj_id"] = group_key[0]
                metrics["exp_num"] = group_key[1]
            else:
                metrics["obj_id"] = group_key

            all_metrics.append(metrics)

    # Convert to arrays
    if all_angular_vel:
        traces = {
            "angular_velocity": np.array(all_angular_vel),
            "linear_velocity": np.array(all_linear_vel),
            "position": np.array(all_positions),
        }
        metrics_df = pd.DataFrame(all_metrics)
    else:
        traces = {
            "angular_velocity": np.empty((0, window_size)),
            "linear_velocity": np.empty((0, window_size)),
            "position": np.empty((0, window_size, 3)),
        }
        metrics_df = pd.DataFrame()

    return SaccadeResults(traces=traces, metrics=metrics_df, config=config)


def analyze_event_responses(
    df: pd.DataFrame,
    events: pd.DataFrame,
    config: Config = None,
    progressbar: bool = True,
) -> EventResults:
    """
    Analyze fly responses to events (optogenetic or visual stimuli).

    This function:
        1. For each event, finds the corresponding trajectory
        2. Computes kinematics around the event
        3. Detects if a response saccade occurred within the response window
        4. Extracts time-aligned traces centered on event onset
        5. Computes response metrics

    Args:
        df: DataFrame with trajectory data (from read_braidz).
        events: DataFrame with event data (opto or stim from read_braidz).
        config: Analysis configuration. Uses defaults if None.
        progressbar: Show progress bar.

    Returns:
        EventResults with traces, metrics, and metadata for all events.

    Example:
        >>> data = read_braidz("experiment.braidz")
        >>> opto_results = analyze_event_responses(data.trajectories, data.opto)
        >>> print(f"Response rate: {opto_results.response_rate:.1%}")

        >>> # Filter to responsive trials
        >>> responsive = opto_results.responsive
        >>> plot_traces(responsive)

        >>> # Compare by intensity
        >>> for intensity in opto_results.metadata['intensity'].unique():
        ...     subset = opto_results.filter(intensity=intensity)
        ...     print(f"Intensity {intensity}: {subset.response_rate:.1%}")
    """
    if config is None:
        config = DEFAULT_CONFIG

    if events is None or len(events) == 0:
        logger.warning("No events provided")
        return EventResults(
            traces={},
            metrics=pd.DataFrame(),
            metadata=pd.DataFrame(),
            config=config,
        )

    # Storage for results
    all_angular_vel = []
    all_linear_vel = []
    all_positions = []
    all_metrics = []
    all_metadata = []

    window_size = config.pre_frames + config.post_frames

    # Process each event
    iterator = tqdm(events.iterrows(), total=len(events), disable=not progressbar)
    for _, event_row in iterator:
        try:
            obj_id = int(event_row["obj_id"])
            frame = int(event_row["frame"])
            exp_num = int(event_row["exp_num"]) if "exp_num" in event_row else None
        except (ValueError, KeyError):
            continue

        # Find corresponding trajectory
        if exp_num is not None:
            grp = df[(df["obj_id"] == obj_id) & (df["exp_num"] == exp_num)]
        else:
            grp = df[df["obj_id"] == obj_id]

        # Check minimum length
        if len(grp) < config.min_trajectory_frames:
            logger.debug(f"Trajectory too short: {len(grp)} frames")
            continue

        # Find event index in trajectory
        try:
            frame_indices = np.where(grp["frame"].values == frame)[0]
            if len(frame_indices) == 0:
                continue
            event_idx = frame_indices[0]
        except (IndexError, ValueError):
            continue

        # Check if we have enough frames around the event
        if event_idx < config.pre_frames:
            logger.debug("Insufficient frames before event")
            continue
        if event_idx >= len(grp) - config.post_frames:
            logger.debug("Insufficient frames after event")
            continue

        # Compute kinematics
        heading, angular_velocity = compute_angular_velocity(
            grp["xvel"].values, grp["yvel"].values, dt=config.dt
        )
        linear_velocity = compute_linear_velocity(
            grp["xvel"].values, grp["yvel"].values, grp["zvel"].values
        )

        # Get positions
        x = savgol_filter(grp["x"].values, config.smoothing_window, config.smoothing_polyorder)
        y = savgol_filter(grp["y"].values, config.smoothing_window, config.smoothing_polyorder)
        z = savgol_filter(grp["z"].values, config.smoothing_window, config.smoothing_polyorder)
        position = np.column_stack([x, y, z])

        # Detect saccades
        peaks = detect_saccades(
            angular_velocity,
            threshold=config.saccade_threshold,
            min_spacing=config.min_saccade_spacing,
        )

        # Find response saccade (first saccade in response window)
        # Window is [event + response_delay, event + response_window]
        response_window_start = event_idx + config.response_delay
        response_window_end = event_idx + config.response_window
        response_peaks = peaks[(peaks >= response_window_start) & (peaks < response_window_end)]

        if len(response_peaks) > 0:
            # Responsive trial: use the detected saccade peak
            response_peak = response_peaks[0]
            responded = True
            reaction_time = response_peak - event_idx
            reference_idx = response_peak
        else:
            # Non-responsive trial: use a reference point for metric calculation
            responded = False
            reaction_time = np.nan

            # Determine reference index based on available metadata
            # For opto: use center of stimulus duration
            # For stim: use end of response window (or could be customized)
            if "duration" in event_row.index and pd.notna(event_row["duration"]):
                # Duration is typically in ms, convert to frames
                duration_frames = int(event_row["duration"] / 1000 * config.fps)
                reference_idx = event_idx + duration_frames // 2
            else:
                # Fallback: use center of response window
                reference_idx = event_idx + config.response_window // 2

            # Ensure reference_idx is within bounds
            reference_idx = min(reference_idx, len(angular_velocity) - config.heading_window - 1)

        # Calculate metrics at the reference point
        heading_change = compute_heading_change(
            heading, reference_idx, window=config.heading_window
        )
        peak_velocity = angular_velocity[reference_idx]

        # Also compute max angular velocity in response window (useful for non-responsive)
        window_slice = angular_velocity[response_window_start:response_window_end]
        max_velocity_in_window = np.max(np.abs(window_slice)) if len(window_slice) > 0 else np.nan

        # Extract traces around event
        start = event_idx - config.pre_frames
        end = event_idx + config.post_frames

        all_angular_vel.append(angular_velocity[start:end])
        all_linear_vel.append(linear_velocity[start:end])
        all_positions.append(position[start:end])

        # Store metrics
        all_metrics.append(
            {
                "responded": responded,
                "heading_change": heading_change,
                "reaction_time": reaction_time,
                "peak_velocity": peak_velocity,
                "max_velocity_in_window": max_velocity_in_window,
                "obj_id": obj_id,
                "exp_num": exp_num,
                "frame": frame,
            }
        )

        # Store metadata from event row (all columns that aren't identifiers)
        metadata_cols = [
            c
            for c in event_row.index
            if c not in ["obj_id", "exp_num", "frame"]
        ]
        all_metadata.append({c: event_row[c] for c in metadata_cols})

    # Convert to arrays
    if all_angular_vel:
        traces = {
            "angular_velocity": np.array(all_angular_vel),
            "linear_velocity": np.array(all_linear_vel),
            "position": np.array(all_positions),
        }
        metrics_df = pd.DataFrame(all_metrics)
        metadata_df = pd.DataFrame(all_metadata)
    else:
        traces = {
            "angular_velocity": np.empty((0, window_size)),
            "linear_velocity": np.empty((0, window_size)),
            "position": np.empty((0, window_size, 3)),
        }
        metrics_df = pd.DataFrame()
        metadata_df = pd.DataFrame()

    return EventResults(
        traces=traces,
        metrics=metrics_df,
        metadata=metadata_df,
        config=config,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def compute_response_statistics(results: EventResults) -> pd.DataFrame:
    """
    Compute summary statistics for event responses.

    Groups by available metadata columns (e.g., intensity, duration, sham)
    and computes response rates and kinematic statistics.

    Args:
        results: EventResults from analyze_event_responses.

    Returns:
        DataFrame with statistics per condition.
    """
    if len(results) == 0:
        return pd.DataFrame()

    # Combine metrics and metadata
    combined = pd.concat([results.metrics, results.metadata], axis=1)

    # Find categorical columns to group by
    group_cols = []
    for col in results.metadata.columns:
        if results.metadata[col].nunique() < 20:  # Likely categorical
            group_cols.append(col)

    if not group_cols:
        # No grouping - compute overall stats
        return pd.DataFrame(
            {
                "n_events": [len(results)],
                "n_responses": [results.metrics["responded"].sum()],
                "response_rate": [results.response_rate],
                "mean_heading_change": [results.metrics["heading_change"].abs().mean()],
                "mean_reaction_time": [results.metrics["reaction_time"].mean()],
            }
        )

    # Group and compute statistics
    stats = []
    for group_vals, group_df in combined.groupby(group_cols):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)

        stat_row = dict(zip(group_cols, group_vals))
        stat_row["n_events"] = len(group_df)
        stat_row["n_responses"] = group_df["responded"].sum()
        stat_row["response_rate"] = group_df["responded"].mean()
        stat_row["mean_heading_change"] = group_df["heading_change"].abs().mean()
        stat_row["mean_reaction_time"] = group_df["reaction_time"].mean()

        stats.append(stat_row)

    return pd.DataFrame(stats)
