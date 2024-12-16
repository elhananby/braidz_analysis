from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .helpers import get_mean_and_std


def plot_mean_and_std(
    arr: np.ndarray, ax: Optional[plt.Axes] = None, **kwargs
) -> plt.Axes:
    """Plot mean with standard deviation envelope.

    Creates a line plot of the mean values with a shaded region representing
    ±1 standard deviation around the mean.

    Args:
        arr: Input array of shape (n_samples, n_timepoints) to calculate statistics from
        ax: Matplotlib axes to plot on. If None, creates new figure and axes
        **kwargs: Additional keyword arguments passed to both plot() and fill_between()
                 (e.g., color, label, linewidth)

    Returns:
        plt.Axes: The axes object containing the plot

    Example:
        >>> data = np.random.randn(100, 10)  # 100 samples, 10 timepoints
        >>> ax = plot_mean_and_std(data, color='blue', label='Group A')
    """
    # Calculate statistics across samples (first dimension)
    mean, std = get_mean_and_std(arr)

    # Create new figure if no axes provided
    if ax is None:
        _, ax = plt.subplots()

    # Plot mean line
    ax.plot(mean, **kwargs)

    # Add standard deviation envelope
    ax.fill_between(
        range(len(mean)),  # x coordinates
        mean - std,  # lower bound
        mean + std,  # upper bound
        alpha=0.3,  # transparency of fill
        **kwargs,
    )

    return ax


def plot_histogram(
    arr: np.ndarray, ax: Optional[plt.Axes] = None, **kwargs
) -> plt.Axes:
    """Create a histogram of the input array.

    Args:
        arr: Input array of values to create histogram from
        ax: Matplotlib axes to plot on. If None, creates new figure and axes
        **kwargs: Additional keyword arguments passed to hist()
                 (e.g., bins, density, color, alpha)

    Returns:
        plt.Axes: The axes object containing the histogram

    Example:
        >>> data = np.random.randn(1000)
        >>> ax = plot_histogram(data, bins=30, color='red', alpha=0.7)
    """
    # Create new figure if no axes provided
    if ax is None:
        _, ax = plt.subplots()

    # Create histogram
    ax.hist(arr, **kwargs)

    return ax


def add_shaded_region(
    ax: plt.Axes, start: int, end: int, color: str = "gray", alpha: float = 0.3
) -> plt.Axes:
    """Add a vertical shaded region to an existing plot.

    Useful for highlighting specific regions of interest, such as stimulus periods
    or events.

    Args:
        ax: Matplotlib axes to add shading to
        start: Starting x-coordinate of shaded region
        end: Ending x-coordinate of shaded region
        color: Color of shaded region (default: "gray")
        alpha: Transparency of shaded region, 0 to 1 (default: 0.3)

    Returns:
        plt.Axes: The axes object with added shaded region

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot(data)
        >>> add_shaded_region(ax, start=50, end=100, color='red', alpha=0.2)
    """
    # Add vertical span to highlight region
    ax.axvspan(
        start,  # x start
        end,  # x end
        color=color,
        alpha=alpha,
    )

    return ax


def create_statistical_plot(
    data: np.ndarray,
    highlight_regions: Optional[list[Tuple[int, int]]] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a complete statistical plot with mean, std, and optional highlighted regions.

    A convenience function that combines plot_mean_and_std with add_shaded_region.

    Args:
        data: Input array of shape (n_samples, n_timepoints)
        highlight_regions: List of (start, end) tuples defining regions to shade
        **kwargs: Additional keyword arguments passed to plot_mean_and_std

    Returns:
        Tuple containing:
            - plt.Figure: The figure object
            - plt.Axes: The axes object containing the plot

    Example:
        >>> data = np.random.randn(100, 200)  # 100 samples, 200 timepoints
        >>> regions = [(50, 75), (150, 175)]  # Two regions to highlight
        >>> fig, ax = create_statistical_plot(
        ...     data,
        ...     highlight_regions=regions,
        ...     color='blue',
        ...     label='Group A'
        ... )
    """
    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot mean and standard deviation
    ax = plot_mean_and_std(data, ax=ax, **kwargs)

    # Add highlighted regions if specified
    if highlight_regions is not None:
        for start, end in highlight_regions:
            ax = add_shaded_region(ax, start, end)

    return fig, ax
