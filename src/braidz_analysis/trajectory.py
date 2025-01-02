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
    _, angular_velocity = butterdiff(theta_unwrap, dt=0.01, params=[1, 0.1])
    return theta, angular_velocity


def calculate_smoothed_linear_velocity(df):
    """
    Calculate smoothed linear velocity using Kalman uncertainties for adaptive smoothing.

    Args:
        df: DataFrame with columns xvel, yvel, zvel and P matrix elements

    Returns:
        ndarray: Smoothed linear velocities
    """
    # Extract velocities
    xvel = df["xvel"].values
    yvel = df["yvel"].values
    zvel = df["zvel"].values

    # Initialize smoothed arrays
    smoothed_x = np.zeros_like(xvel)
    smoothed_y = np.zeros_like(yvel)
    smoothed_z = np.zeros_like(zvel)

    # Initialize with first values
    smoothed_x[0] = xvel[0]
    smoothed_y[0] = yvel[0]
    smoothed_z[0] = zvel[0]

    # Set base_alpha based on typical uncertainty scale (mean + 1 std)
    base_alpha = 1e-3  # Matched to order of magnitude of uncertainties

    # Process each timestep
    for i in range(1, len(df)):
        # Extract velocity uncertainties from P matrix
        vx_var = np.clip(df["P33"].iloc[i], 1e-6, 1e-1)  # Prevent extreme values
        vy_var = np.clip(df["P44"].iloc[i], 1e-6, 1e-1)
        vz_var = np.clip(df["P55"].iloc[i], 1e-6, 1e-1)

        # Calculate adaptive smoothing factors
        k_x = base_alpha / (base_alpha + vx_var)
        k_y = base_alpha / (base_alpha + vy_var)
        k_z = base_alpha / (base_alpha + vz_var)

        # Update smoothed estimates with bounded coefficients
        smoothed_x[i] = k_x * xvel[i] + (1 - k_x) * smoothed_x[i - 1]
        smoothed_y[i] = k_y * yvel[i] + (1 - k_y) * smoothed_y[i - 1]
        smoothed_z[i] = k_z * zvel[i] + (1 - k_z) * smoothed_z[i - 1]

    # Calculate final linear velocity
    linear_velocity = np.sqrt(smoothed_x**2 + smoothed_y**2 + smoothed_z**2)

    return linear_velocity


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


def calculate_heading_diff(heading, indices_before, indices_after):
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
    indices_before = np.array(indices_before)
    indices_after = np.array(indices_after)

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
