"""
Kinematics module for computing trajectory metrics.

This module provides functions for calculating kinematic properties of fly
trajectories, including:
    - Angular and linear velocity
    - Heading direction
    - Saccade detection
    - Flight state classification

All functions are designed to work with numpy arrays and polars DataFrames
from the Strand-Braid tracking system.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
from pynumdiff.smooth_finite_difference import butterdiff
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import circmean

from .config import DEFAULT_CONFIG, Config

# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SaccadeEvent:
    """
    Properties of a detected saccade.

    Attributes:
        frame: Frame index where the saccade peak occurred.
        amplitude_rad: Total angular change during the saccade (radians).
        duration_frames: Duration of the saccade in frames.
        peak_velocity_rad_s: Maximum angular velocity during saccade (rad/s).
        heading_change_rad: Net heading change from before to after saccade.
        direction: Sign of the turn (+1 for left/CCW, -1 for right/CW).
    """

    frame: int
    amplitude_rad: float
    duration_frames: int
    peak_velocity_rad_s: float
    heading_change_rad: float = 0.0
    direction: int = 0

    @property
    def amplitude_deg(self) -> float:
        """Amplitude in degrees."""
        return np.degrees(self.amplitude_rad)

    @property
    def peak_velocity_deg_s(self) -> float:
        """Peak angular velocity in degrees/second."""
        return np.degrees(self.peak_velocity_rad_s)

    @property
    def heading_change_deg(self) -> float:
        """Heading change in degrees."""
        return np.degrees(self.heading_change_rad)

    def duration_ms(self, fps: float = 100.0) -> float:
        """Duration in milliseconds."""
        return (self.duration_frames / fps) * 1000


# =============================================================================
# Core Velocity Functions
# =============================================================================


def compute_angular_velocity(
    xvel: np.ndarray,
    yvel: np.ndarray,
    dt: float = 0.01,
    filter_params: Tuple[int, float] = (1, 0.1),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute heading direction and angular velocity from velocity components.

    Uses a Butterworth filter for smooth differentiation of the heading angle.

    Args:
        xvel: Array of x-velocity values (m/s).
        yvel: Array of y-velocity values (m/s).
        dt: Time step between frames (seconds). Default is 0.01 (100 fps).
        filter_params: Butterworth filter parameters (order, cutoff).

    Returns:
        Tuple of:
            - heading: Heading angle at each frame (radians, -pi to pi)
            - angular_velocity: Angular velocity at each frame (rad/s)

    Example:
        >>> heading, omega = compute_angular_velocity(df['xvel'], df['yvel'])
        >>> print(f"Peak angular velocity: {np.max(np.abs(omega)):.1f} rad/s")
    """
    # Compute heading from velocity direction
    heading = np.arctan2(yvel, xvel)

    # Unwrap to handle discontinuities at +/- pi
    heading_unwrapped = np.unwrap(heading)

    # Differentiate with Butterworth filter for smooth angular velocity
    _, angular_velocity = butterdiff(heading_unwrapped, dt=dt, params=list(filter_params))

    return heading, angular_velocity


def compute_linear_velocity(
    xvel: np.ndarray,
    yvel: np.ndarray,
    zvel: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute linear speed from velocity components.

    Args:
        xvel: Array of x-velocity values (m/s).
        yvel: Array of y-velocity values (m/s).
        zvel: Array of z-velocity values (m/s). Optional.

    Returns:
        Array of linear speeds (m/s).

    Example:
        >>> speed = compute_linear_velocity(df['xvel'], df['yvel'], df['zvel'])
        >>> print(f"Mean speed: {np.mean(speed):.3f} m/s")
    """
    xvel = np.asarray(xvel)
    yvel = np.asarray(yvel)

    if zvel is None:
        return np.sqrt(xvel**2 + yvel**2)
    else:
        zvel = np.asarray(zvel)
        return np.sqrt(xvel**2 + yvel**2 + zvel**2)


def compute_heading_change(
    heading: np.ndarray,
    idx: int,
    window: int = 10,
) -> float:
    """
    Compute heading change around a specific frame.

    Calculates the circular mean heading before and after the specified index,
    then returns the angular difference.

    Args:
        heading: Array of heading angles (radians).
        idx: Index of the point (e.g., saccade peak).
        window: Number of frames before/after idx for averaging.

    Returns:
        Signed heading change in radians (-pi to pi).

    Example:
        >>> heading_change = compute_heading_change(heading, saccade_idx, window=10)
        >>> print(f"Turned {np.degrees(heading_change):.1f} degrees")
    """
    # Define ranges for before and after
    start = max(0, idx - window)
    end = min(len(heading), idx + window)

    if start >= idx or idx >= end:
        return 0.0

    # Compute circular means
    heading_before = circmean(heading[start:idx], low=-np.pi, high=np.pi)
    heading_after = circmean(heading[idx:end], low=-np.pi, high=np.pi)

    # Compute signed angular difference
    diff = heading_after - heading_before
    heading_change = np.arctan2(np.sin(diff), np.cos(diff))

    return heading_change


# =============================================================================
# Smoothing Functions
# =============================================================================


def smooth_trajectory(
    df: pl.DataFrame,
    columns: List[str] = None,
    window: int = 21,
    polyorder: int = 3,
) -> pl.DataFrame:
    """
    Apply Savitzky-Golay smoothing to trajectory columns.

    Args:
        df: DataFrame with trajectory data.
        columns: Columns to smooth. Default: ['x', 'y', 'z', 'xvel', 'yvel', 'zvel']
        window: Smoothing window size (must be odd).
        polyorder: Polynomial order for smoothing.

    Returns:
        DataFrame with smoothed values (original data preserved with '_raw' suffix).
    """
    if columns is None:
        columns = ["x", "y", "z", "xvel", "yvel", "zvel"]

    df = df.clone()
    new_columns = []

    for col in columns:
        if col in df.columns:
            # Store raw values
            raw_values = df[col].to_numpy()
            smoothed_values = savgol_filter(raw_values, window, polyorder)
            new_columns.append(pl.Series(f"{col}_raw", raw_values))
            new_columns.append(pl.Series(col, smoothed_values))

    # Build expressions to update the dataframe
    if new_columns:
        # First add raw columns, then update smoothed ones
        raw_cols = [c for c in new_columns if c.name.endswith("_raw")]
        smoothed_cols = [c for c in new_columns if not c.name.endswith("_raw")]

        # Add raw columns
        for raw_col in raw_cols:
            df = df.with_columns(raw_col)

        # Update smoothed columns
        for smooth_col in smoothed_cols:
            df = df.with_columns(smooth_col)

    return df


# =============================================================================
# Saccade Detection
# =============================================================================


def detect_saccades(
    angular_velocity: np.ndarray,
    threshold: float = 300.0,
    min_spacing: int = 50,
    mode: str = "absolute",
    return_properties: bool = False,
) -> np.ndarray:
    """
    Detect saccades as peaks in angular velocity.

    Uses scipy's find_peaks to detect peaks in angular velocity. The mode
    parameter controls which direction of turns to detect.

    Args:
        angular_velocity: Angular velocity in rad/s.
        threshold: Minimum angular velocity for detection (deg/s).
        min_spacing: Minimum frames between peaks.
        mode: Which peaks to detect:
            - "both": Detect positive and negative peaks separately, then combine.
                      This is direction-aware and preserves turn direction info.
            - "absolute": Run peak detection on |angular_velocity|.
                          Simpler but loses direction information in peak detection.
            - "positive": Only detect positive peaks (left/counterclockwise turns).
            - "negative": Only detect negative peaks (right/clockwise turns).
        return_properties: If True, also return peak properties dict.

    Returns:
        Array of frame indices where saccades occur.
        If return_properties=True, also returns dict with peak properties.

    Example:
        >>> # Detect all saccades (default)
        >>> peaks = detect_saccades(omega, threshold=300)
        >>>
        >>> # Detect only left turns
        >>> left_peaks = detect_saccades(omega, threshold=300, mode="positive")
        >>>
        >>> # Detect on absolute value trace
        >>> peaks = detect_saccades(omega, threshold=300, mode="absolute")
    """
    valid_modes = {"both", "absolute", "positive", "negative"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

    threshold_rad = np.deg2rad(threshold)

    if mode == "absolute":
        # Run peak detection on absolute value of angular velocity
        peaks, props = find_peaks(
            np.abs(angular_velocity), height=threshold_rad, distance=min_spacing
        )
        if return_properties:
            return peaks, {"peaks": peaks, "heights": props.get("peak_heights", [])}
        return peaks

    elif mode == "positive":
        # Only detect positive peaks (counterclockwise/left turns)
        peaks, props = find_peaks(angular_velocity, height=threshold_rad, distance=min_spacing)
        if return_properties:
            return peaks, {"pos_peaks": peaks, "heights": props.get("peak_heights", [])}
        return peaks

    elif mode == "negative":
        # Only detect negative peaks (clockwise/right turns)
        peaks, props = find_peaks(-angular_velocity, height=threshold_rad, distance=min_spacing)
        if return_properties:
            return peaks, {"neg_peaks": peaks, "heights": props.get("peak_heights", [])}
        return peaks

    else:  # mode == "both"
        # Find positive peaks (counterclockwise turns)
        pos_peaks, pos_props = find_peaks(
            angular_velocity, height=threshold_rad, distance=min_spacing
        )

        # Find negative peaks (clockwise turns)
        neg_peaks, neg_props = find_peaks(
            -angular_velocity, height=threshold_rad, distance=min_spacing
        )

        # Combine and sort
        all_peaks = np.sort(np.concatenate([pos_peaks, neg_peaks]))

        if return_properties:
            return all_peaks, {"pos_peaks": pos_peaks, "neg_peaks": neg_peaks}

        return all_peaks


def extract_saccade_events(
    angular_velocity: np.ndarray,
    heading: np.ndarray,
    peaks: np.ndarray,
    config: Config = None,
) -> List[SaccadeEvent]:
    """
    Extract detailed properties for each detected saccade.

    Args:
        angular_velocity: Angular velocity array (rad/s).
        heading: Heading angle array (radians).
        peaks: Array of peak frame indices.
        config: Analysis configuration.

    Returns:
        List of SaccadeEvent objects with computed properties.
    """
    if config is None:
        config = DEFAULT_CONFIG

    events = []
    for peak in peaks:
        # Compute heading change
        heading_change = compute_heading_change(heading, peak, window=config.heading_window)

        # Estimate saccade duration (frames where |omega| > threshold/2)
        half_thresh = config.saccade_threshold_rad / 2
        in_saccade = np.abs(angular_velocity) > half_thresh

        # Find saccade boundaries around peak
        start = peak
        while start > 0 and in_saccade[start - 1]:
            start -= 1
        end = peak
        while end < len(in_saccade) - 1 and in_saccade[end + 1]:
            end += 1

        duration = end - start + 1

        # Compute amplitude (integral of angular velocity)
        amplitude = np.abs(np.trapz(angular_velocity[start : end + 1])) * config.dt

        events.append(
            SaccadeEvent(
                frame=int(peak),
                amplitude_rad=amplitude,
                duration_frames=duration,
                peak_velocity_rad_s=float(angular_velocity[peak]),
                heading_change_rad=heading_change,
                direction=int(np.sign(angular_velocity[peak])),
            )
        )

    return events


# =============================================================================
# Modified Geometric Saccade Detection (mGSD)
# =============================================================================


def _circular_median(angles: np.ndarray) -> float:
    """
    Compute the circular median of an array of angles.

    Uses the arctan2(median(sin), median(cos)) approach which is robust
    to angle wrapping at ±π.

    Args:
        angles: Array of angles in radians.

    Returns:
        Circular median angle in radians.
    """
    return np.arctan2(np.median(np.sin(angles)), np.median(np.cos(angles)))


def _circular_distance(angle1: float, angle2: float) -> float:
    """
    Compute the absolute angular distance between two angles.

    Uses the dot product formula: arccos(cos(a)cos(b) + sin(a)sin(b))
    which gives the absolute difference in [0, π].

    Args:
        angle1: First angle in radians.
        angle2: Second angle in radians.

    Returns:
        Absolute angular distance in radians [0, π].
    """
    dot_product = np.cos(angle1) * np.cos(angle2) + np.sin(angle1) * np.sin(angle2)
    # Clamp to [-1, 1] to handle numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return np.abs(np.arccos(dot_product))


def compute_mgsd_scores(
    x: np.ndarray,
    y: np.ndarray,
    angular_velocity: np.ndarray,
    delta_frames: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Modified Geometric Saccade Detection (mGSD) scores.

    The mGSD algorithm detects saccades based on geometric properties of the
    trajectory rather than angular velocity peaks. For each frame, it computes:

    - Amplitude (A): The angular change between the median direction to "before"
      points and the median direction to "after" points.
    - Dispersion (D): Sum of distances from the current position to all points
      in the surrounding window.
    - Score (S): S = A² × D × sign(angular_velocity)

    Reference:
        Cellini, B., et al. (2024). Current Biology.
        Modified from Censi, A., et al. original GSD algorithm.

    Args:
        x: Array of x positions.
        y: Array of y positions.
        angular_velocity: Angular velocity array (rad/s), used for determining
            turn direction (sign of the score).
        delta_frames: Number of frames before/after current frame to include
            in the window. Default is 5 (50ms at 100Hz).

    Returns:
        Tuple of three arrays:
            - scores: The mGSD score at each frame (signed by turn direction)
            - amplitudes: The amplitude component A at each frame
            - dispersions: The dispersion component D at each frame

    Example:
        >>> scores, amps, disps = compute_mgsd_scores(x, y, angular_velocity)
        >>> # High |scores| indicate likely saccades
    """
    n = len(x)
    scores = np.zeros(n)
    amplitudes = np.zeros(n)
    dispersions = np.zeros(n)

    # Process each frame (excluding edges where we don't have full windows)
    for k in range(delta_frames, n - delta_frames):
        # Center coordinates on current position
        ref_x, ref_y = x[k], y[k]
        norm_x = x - ref_x
        norm_y = y - ref_y

        # Define before and after intervals
        before_slice = slice(k - delta_frames, k)
        after_slice = slice(k + 1, k + delta_frames + 1)

        # Compute angles from current position to other points
        # For "before" points: angle TO the before position (where fly came from)
        alpha_before = np.arctan2(norm_y[before_slice], norm_x[before_slice])
        # For "after" points: angle AWAY from the after position (opposite direction)
        # This makes both angles represent "came from" direction
        alpha_after = np.arctan2(-norm_y[after_slice], -norm_x[after_slice])

        # Compute median angles
        theta_before = _circular_median(alpha_before)
        theta_after = _circular_median(alpha_after)

        # Amplitude: angular difference between before and after directions
        amp = _circular_distance(theta_before, theta_after)

        # Dispersion: sum of distances in the full window
        full_slice = slice(k - delta_frames, k + delta_frames + 1)
        distances = np.sqrt(norm_x[full_slice] ** 2 + norm_y[full_slice] ** 2)
        disp = np.sum(distances)

        # Score: A² × D × sign(angular_velocity)
        # Using amp² emphasizes larger turns
        sign = np.sign(angular_velocity[k]) if angular_velocity[k] != 0 else 1
        score = (amp**2) * disp * sign

        scores[k] = score
        amplitudes[k] = amp
        dispersions[k] = disp

    return scores, amplitudes, dispersions


def detect_saccades_mgsd(
    x: np.ndarray,
    y: np.ndarray,
    angular_velocity: np.ndarray,
    delta_frames: int = 5,
    threshold: float = 0.001,
    min_spacing: int = 10,
    return_scores: bool = False,
) -> np.ndarray:
    """
    Detect saccades using the Modified Geometric Saccade Detection algorithm.

    This algorithm detects saccades based on geometric properties of the
    trajectory, combining amplitude (direction change) with dispersion
    (movement magnitude). It is more robust than velocity-based detection
    for slow or noisy trajectories.

    Args:
        x: Array of x positions.
        y: Array of y positions.
        angular_velocity: Angular velocity array (rad/s).
        delta_frames: Window size in frames (default 5 = 50ms at 100Hz).
        threshold: Minimum |score| for peak detection (default 0.001).
        min_spacing: Minimum frames between detected peaks (default 10).
        return_scores: If True, also return the score arrays.

    Returns:
        Array of frame indices where saccades were detected.
        If return_scores=True, returns tuple of (peaks, scores, amplitudes, dispersions).

    Example:
        >>> # Basic usage
        >>> peaks = detect_saccades_mgsd(x, y, angular_velocity)
        >>>
        >>> # Get scores for visualization
        >>> peaks, scores, amps, disps = detect_saccades_mgsd(
        ...     x, y, angular_velocity, return_scores=True
        ... )
    """
    # Compute mGSD scores
    scores, amplitudes, dispersions = compute_mgsd_scores(
        x, y, angular_velocity, delta_frames=delta_frames
    )

    # Find peaks in absolute score
    # Use prominence = 2*threshold as in reference implementation
    peaks, _ = find_peaks(
        np.abs(scores),
        height=threshold,
        distance=min_spacing,
        prominence=threshold * 2,
    )

    if return_scores:
        return peaks, scores, amplitudes, dispersions

    return peaks


# =============================================================================
# Flight State Classification
# =============================================================================


def classify_flight_state(
    linear_velocity: np.ndarray,
    high_threshold: float = 0.05,
    low_threshold: float = 0.01,
    min_frames: int = 20,
    fps: float = 100.0,
) -> np.ndarray:
    """
    Classify flight vs. non-flight states using hysteresis thresholding.

    Uses a state machine with hysteresis to identify sustained flight bouts,
    avoiding rapid switching between states.

    Args:
        linear_velocity: Array of linear speeds (m/s).
        high_threshold: Velocity to transition into flight state.
        low_threshold: Velocity to transition out of flight state.
        min_frames: Minimum consecutive frames to trigger state change.
        fps: Frame rate for rolling statistics.

    Returns:
        Boolean array where True indicates flight state.

    Example:
        >>> is_flying = classify_flight_state(speed)
        >>> flight_frames = np.sum(is_flying)
        >>> print(f"Flying {flight_frames / len(is_flying) * 100:.1f}% of time")
    """
    # Compute rolling mean for noise reduction using polars
    window = int(fps)  # 1-second window
    rolling_mean = pl.Series(linear_velocity).rolling_mean(window_size=window, center=True)
    rolling_mean_arr = rolling_mean.to_numpy()

    # State machine with hysteresis
    states = np.zeros(len(linear_velocity), dtype=bool)
    current_state = False  # Start in non-flight
    frame_count = 0

    for i in range(len(rolling_mean_arr)):
        val = rolling_mean_arr[i]
        # Handle NaN values at boundaries
        if np.isnan(val):
            states[i] = current_state
            continue

        # Check for potential state transition
        if current_state:
            # Currently flying - check for exit
            if val < low_threshold:
                frame_count += 1
            else:
                frame_count = 0
        else:
            # Currently not flying - check for entry
            if val > high_threshold:
                frame_count += 1
            else:
                frame_count = 0

        # Trigger state change after sustained threshold crossing
        if frame_count >= min_frames:
            current_state = not current_state
            frame_count = 0

        states[i] = current_state

    return states


def extract_flight_bouts(
    linear_velocity: np.ndarray,
    high_threshold: float = 0.1,
    low_threshold: float = 0.01,
    max_gap: int = 100,
) -> List[np.ndarray]:
    """
    Extract contiguous flight bout indices, merging nearby bouts.

    Args:
        linear_velocity: Array of linear speeds.
        high_threshold: Velocity to start a bout.
        low_threshold: Velocity to end a bout.
        max_gap: Maximum gap (frames) to merge adjacent bouts.

    Returns:
        List of arrays, each containing frame indices for one bout.
    """
    # Smooth velocity
    smoothed = savgol_filter(linear_velocity, window_length=21, polyorder=3)

    # Find initial segments using hysteresis
    segments = []
    flying = False
    start = None

    for i in range(len(smoothed)):
        if flying:
            if smoothed[i] < low_threshold:
                segments.append(np.arange(start, i))
                flying = False
        else:
            if smoothed[i] > high_threshold:
                start = i
                flying = True

    # Handle case where recording ends during flight
    if flying:
        segments.append(np.arange(start, len(smoothed)))

    if not segments:
        return []

    # Merge segments with small gaps
    merged = [segments[0]]
    for seg in segments[1:]:
        gap = seg[0] - merged[-1][-1] - 1
        if gap <= max_gap:
            # Merge: create continuous range spanning both segments
            merged[-1] = np.arange(merged[-1][0], seg[-1] + 1)
        else:
            merged.append(seg)

    return merged


# =============================================================================
# Convenience Functions
# =============================================================================


def add_kinematics_to_trajectory(
    df: pl.DataFrame,
    config: Config = None,
) -> pl.DataFrame:
    """
    Add computed kinematic columns to a trajectory DataFrame.

    Adds the following columns:
        - heading: Direction of movement (radians)
        - angular_velocity: Rate of heading change (rad/s)
        - linear_velocity: Speed (m/s)
        - is_flying: Boolean flight state

    Args:
        df: DataFrame with xvel, yvel, zvel columns.
        config: Analysis configuration.

    Returns:
        DataFrame with added kinematic columns.
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Compute velocities
    heading, angular_velocity = compute_angular_velocity(
        df["xvel"].to_numpy(), df["yvel"].to_numpy(), dt=config.dt
    )
    linear_velocity = compute_linear_velocity(
        df["xvel"].to_numpy(), df["yvel"].to_numpy(), df["zvel"].to_numpy()
    )

    # Classify flight state
    is_flying = classify_flight_state(
        linear_velocity,
        high_threshold=config.flight_high_threshold,
        low_threshold=config.flight_low_threshold,
        min_frames=config.flight_min_frames,
        fps=config.fps,
    )

    # Add columns to DataFrame
    df = df.with_columns(
        [
            pl.Series("heading", heading),
            pl.Series("angular_velocity", angular_velocity),
            pl.Series("linear_velocity", linear_velocity),
            pl.Series("is_flying", is_flying),
        ]
    )

    return df
