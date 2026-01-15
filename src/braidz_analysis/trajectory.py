"""
DEPRECATED: This module is deprecated. Use braidz_analysis.kinematics instead.

Migration:
    Old: ba.trajectory.calculate_angular_velocity(xvel, yvel)
    New: ba.compute_angular_velocity(xvel, yvel)

    Old: ba.trajectory.detect_saccades(omega)
    New: ba.detect_saccades(omega)

This module will be removed in version 1.0.
"""
import warnings

warnings.warn(
    "braidz_analysis.trajectory is deprecated. Use braidz_analysis.kinematics instead. "
    "See README.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np
from pynumdiff.smooth_finite_difference import butterdiff
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import circmean
import pandas as pd
from typing import List, Tuple
from pycircstat2.descriptive import circ_mean, circ_var
from .helpers import calculate_angle_between_three_points


def calculate_angular_velocity(xvel, yvel):
    """
    Calculate the angular velocity given the x and y velocities.

    Args:
        xvel (ndarray): Array of x velocities.
        yvel (ndarray): Array of y velocities.

    Returns:
        tuple: A tuple containing the theta and angular velocity arrays.
    """
    theta = np.arctan2(yvel, xvel)
    theta_unwrap = np.unwrap(theta)
    _, angular_velocity = butterdiff(theta_unwrap, dt=0.01, params=[1, 0.1])
    return theta, angular_velocity


def calculate_linear_velocity(xvel, yvel, zvel=None):
    """
    Calculate the linear velocity given the x, y, and z velocities.

    Args:
        xvel (ndarray): Array of x velocities.
        yvel (ndarray): Array of y velocities.
        zvel (ndarray, optional): Array of z velocities. Defaults to None.

    Returns:
        ndarray: Array of linear velocities.
    """

    # convert all vels to numpy array if pandas series
    xvel = np.array(xvel)
    yvel = np.array(yvel)

    if zvel is None:
        zvel = np.zeros_like(xvel)
    else:
        zvel = np.array(zvel)

    linear_velocity = np.sqrt(xvel**2 + yvel**2 + zvel**2)
    return linear_velocity


def detect_saccades(angular_velocity, height=np.deg2rad(300), distance=10):
    """
    Detect saccades in the angular velocity array.

    Args:
        angular_velocity (ndarray): Array of angular velocities.
        height (float, optional): Minimum height of peaks to be considered as saccades. Defaults to np.deg2rad(300).
        distance (int, optional): Minimum distance between peaks to be considered as separate saccades. Defaults to 10.

    Returns:
        ndarray: Array of indices where saccades occur.
    """
    positive_peaks, _ = find_peaks(angular_velocity, height=height, distance=distance)
    negative_peaks, _ = find_peaks(-angular_velocity, height=height, distance=distance)
    peaks = np.sort(np.concatenate([positive_peaks, negative_peaks]))
    return peaks


def sg_smooth(arr, **kwargs):
    """
    Apply Savitzky-Golay smoothing to the input array.

    Args:
        arr (ndarray): Input array to be smoothed.
        **kwargs: Additional keyword arguments for the savgol_filter function.

    Returns:
        ndarray: Smoothed array.
    """
    return savgol_filter(
        arr, kwargs.get("window_length", 21), kwargs.get("polyorder", 3)
    )


def calculate_heading_diff(heading, p1, p2, p3):
    """
    Calculate the difference in heading at a specific index.

    Args:
        heading (ndarray): Array of headings.
        incidices_before (ndarray): Array of indices for the before heading calculation.
        indices_after (ndarray): Array of indices for the after heading calculation.

    Returns:
        float: Difference in heading.
    """

    # convert indices_before and indices_after to np.array if not already
    p1, p2, p3 = int(p1), int(p2), int(p3)

    indices_before = np.array(range(p1, p2))
    indices_after = np.array(range(p2, p3))

    # check if indices are within range, raise error if not
    if np.any(indices_before < 0) or np.any(indices_before >= len(heading)):
        raise ValueError("indices_before out of range")

    # calculate the circular mean of the headings
    heading_before = circmean(heading[indices_before], low=-np.pi, high=np.pi)
    heading_after = circmean(heading[indices_after], low=-np.pi, high=np.pi)

    # calculate the difference in heading
    heading_difference = np.arctan2(
        np.sin(heading_after - heading_before), np.cos(heading_after - heading_before)
    )

    return heading_difference


def heading_diff_pos(xyz, idx, window=25):
    """
    Calculate the heading difference at a specific index in a trajectory.

    Parameters:
    xyz (numpy.ndarray): The trajectory data.
    idx (int): The index at which to calculate the heading difference.
    window (int, optional): The window size for calculating the heading difference. Default is 25.

    Returns:
    float: The heading difference in degrees.
    """
    if np.shape(xyz)[1] == 3:
        xyz = xyz[:, :2]

    v1 = xyz[max(0, idx - window), :]
    v2 = xyz[idx, :]
    v3 = xyz[min(idx + window, xyz.shape[1]), :]

    angle = calculate_angle_between_three_points(v1, v2, v3)
    return angle


import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SaccadeEvent:
    """Data class to store properties of a detected saccade."""

    frame: int
    amplitude_rad: float
    duration_ms: float
    peak_angular_velocity_rad_s: float

    @property
    def amplitude_deg(self) -> float:
        return np.degrees(self.amplitude_rad)

    @property
    def peak_angular_velocity_deg_s(self) -> float:
        return np.degrees(self.peak_angular_velocity_rad_s)


class ModifiedGeometricSaccadeDetector:
    """
    Implementation of the Modified Geometric Saccade Detection Algorithm (mGSD)
    as described in Stupski & van Breugel, 2024.

    The algorithm detects saccades in fruit fly trajectories by analyzing
    geometric properties of the flight path.
    """

    def __init__(
        self,
        delta: int = 10,  # Window size (in frames)
        min_amplitude_rad: float = 0.2,  # Minimum amplitude (~11 degrees)
        min_velocity_rad_s: float = 5.0,  # Minimum angular velocity
        min_peak_distance_ms: float = 50,  # Minimum time between saccades
        dt: float = 0.01,  # Time step (seconds)
    ):
        """
        Initialize the detector.

        Args:
            delta: Time step for analysis window (in frames)
            min_amplitude_rad: Minimum angular amplitude for saccade detection (radians)
            min_velocity_rad_s: Minimum angular velocity for saccade detection (rad/s)
            min_peak_distance_ms: Minimum time between detected saccades (milliseconds)
            dt: Time step between frames (seconds)
        """
        self.delta = delta
        self.min_amplitude_rad = min_amplitude_rad
        self.min_velocity_rad_s = min_velocity_rad_s
        self.min_peak_distance_frames = int(min_peak_distance_ms / (dt * 1000))
        self.dt = dt

    def _circular_mean(self, angles: np.ndarray) -> float:
        """Calculate the mean of circular data (angles)."""
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))
        return np.arctan2(sin_sum, cos_sum)

    def _circular_diff(self, angle1: float, angle2: float) -> float:
        """Calculate the smallest angular difference between two angles."""
        return np.pi - np.abs(np.abs(angle1 - angle2) - np.pi)

    def _calculate_angles(
        self, x: np.ndarray, y: np.ndarray, k: int
    ) -> Tuple[float, float]:
        """
        Calculate mean angles for before and after intervals around point k.
        Uses circular statistics to properly handle angular data.
        """
        # Redefine origin to current point
        x_centered = x - x[k]
        y_centered = y - y[k]

        # Calculate angles for before interval
        before_x = x_centered[k - self.delta : k]
        before_y = y_centered[k - self.delta : k]
        before_angles = np.arctan2(before_y, before_x)

        # Calculate angles for after interval
        after_x = x_centered[k + 1 : k + self.delta + 1]
        after_y = y_centered[k + 1 : k + self.delta + 1]
        after_angles = np.arctan2(after_y, after_x)

        return self._circular_mean(before_angles), self._circular_mean(after_angles)

    def _calculate_angular_velocity(self, angles: np.ndarray) -> float:
        """
        Calculate RMS angular velocity over a window of angles.

        Args:
            angles: Array of angles in radians

        Returns:
            RMS angular velocity in radians/second
        """
        angle_diffs = np.array(
            [
                self._circular_diff(angles[i + 1], angles[i])
                for i in range(len(angles) - 1)
            ]
        )
        return np.sqrt(np.mean(angle_diffs**2)) / self.dt

    def _compute_scores(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute saccade detection scores for each point in the trajectory.
        """
        n_frames = len(x)
        scores = np.zeros(n_frames)

        # Calculate instantaneous angles
        angles = np.arctan2(np.diff(y), np.diff(x))

        # Calculate scores for each point
        for k in range(self.delta, n_frames - self.delta - 1):
            # Calculate before and after angles
            before_angle, after_angle = self._calculate_angles(x, y, k)

            # Calculate amplitude
            amplitude = self._circular_diff(after_angle, before_angle)

            # Calculate angular velocity in window
            window_angles = angles[k - self.delta : k + self.delta]
            angular_velocity = self._calculate_angular_velocity(window_angles)

            # Compute score - only if amplitude exceeds minimum
            if amplitude > self.min_amplitude_rad:
                scores[k] = angular_velocity

        return scores

    def _find_peaks(self, scores: np.ndarray) -> List[int]:
        """
        Find peaks in score array with minimum distance between peaks.
        """
        peaks = []
        for i in range(1, len(scores) - 1):
            if scores[i] > self.min_velocity_rad_s:  # Threshold check
                if (
                    scores[i] > scores[i - 1] and scores[i] > scores[i + 1]
                ):  # Peak check
                    if not peaks or (i - peaks[-1]) >= self.min_peak_distance_frames:
                        peaks.append(i)
        return peaks

    def detect_saccades(self, df: pd.DataFrame) -> List[SaccadeEvent]:
        """
        Detect saccades in trajectory data.

        Args:
            df: DataFrame with 'x' and 'y' columns

        Returns:
            List of SaccadeEvent objects
        """
        x = df["x"].values
        y = df["y"].values

        # Compute scores
        scores = self._compute_scores(x, y)

        # Find peaks in scores
        peak_frames = self._find_peaks(scores)

        # Calculate properties for each detected saccade
        saccades = []
        window_frames = self.delta  # Use same window as detection

        for frame in peak_frames:
            # Extract window around saccade
            start_frame = max(0, frame - window_frames)
            end_frame = min(len(df), frame + window_frames)

            # Get x, y data for window
            window_x = x[start_frame:end_frame]
            window_y = y[start_frame:end_frame]

            # Calculate angles
            angles = np.arctan2(np.diff(window_y), np.diff(window_x))

            # Calculate amplitude (total angular change)
            angle_diffs = np.array(
                [
                    self._circular_diff(angles[i + 1], angles[i])
                    for i in range(len(angles) - 1)
                ]
            )
            amplitude = np.sum(angle_diffs)

            # Calculate duration (assume full window for now)
            duration_ms = (end_frame - start_frame) * self.dt * 1000

            # Calculate peak angular velocity
            angular_velocities = angle_diffs / self.dt
            peak_velocity = np.max(np.abs(angular_velocities))

            saccades.append(
                SaccadeEvent(
                    frame=frame,
                    amplitude_rad=amplitude,
                    duration_ms=duration_ms,
                    peak_angular_velocity_rad_s=peak_velocity,
                )
            )

        return saccades

    def get_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get the raw detection scores for visualization/debugging.

        Args:
            df: DataFrame with 'x' and 'y' columns

        Returns:
            Array of scores
        """
        return self._compute_scores(df["x"].values, df["y"].values)
