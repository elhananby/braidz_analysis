import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import circmean
from pynumdiff.smooth_finite_difference import butterdiff
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
    _, angular_velocity = butterdiff(theta_unwrap, dt=0.01, params=[1, 0.01])
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
    if zvel is None:
        zvel = np.zeros_like(xvel)
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


def heading_diff(heading, idx, window=25):
    """
    Calculate the difference in heading at a specific index.

    Args:
        heading (ndarray): Array of headings.
        idx (int): Index at which to calculate the heading difference.
        window (int, optional): Number of data points to consider before and after the index. Defaults to 25.

    Returns:
        float: Difference in heading.
    """
    heading_before = circmean(
        heading[max(0, idx - window) : idx], low=-np.pi, high=np.pi
    )
    heading_after = circmean(
        heading[idx : min(len(heading), idx + window)], low=-np.pi, high=np.pi
    )

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
