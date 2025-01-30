from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .helpers import get_mean_and_std, subtract_baseline


def plot_angular_velocity(
    data: dict,
    ax: Optional[plt.Axes] = None,
    use_abs: bool = False,
    baseline_range: Union[list, Tuple, None] = None,
    shaded_region: list = [50, 80],
    shaded_color: str = "tab:red",
    convert_to_degrees: bool = False,
    **kwargs,
) -> plt.Axes:
    """
    Plots the angular velocity data.

    Args:
        data (dict): A dictionary containing the angular velocity data.
        ax (Optional[plt.Axes], optional): The matplotlib Axes object to plot on. If not provided, a new figure and axes will be created. Defaults to None.
        use_abs (bool, optional): Whether to use the absolute values of the angular velocity. Defaults to False.
        baseline_range (list, optional): The range of indices to use for baseline subtraction. Only applicable if use_abs is True. Defaults to [0, 50].
        shaded_region (list, optional): The range of indices to shade in the plot. Defaults to [50, 80].
        convert_to_degrees (bool, optional): Whether to convert the angular velocity to degrees. Defaults to False.

    Returns:
        plt.Axes: The matplotlib Axes object containing the plot.
    """
    velocity = data["angular_velocity"]

    if baseline_range and not use_abs:
        raise ValueError("Baseline subtraction requires absolute values (use_abs=True)")

    elif use_abs:
        velocity = np.abs(velocity)
        if baseline_range:
            velocity = subtract_baseline(velocity, *baseline_range)

    if convert_to_degrees:
        velocity = np.rad2deg(velocity)

    ax = plot_mean_and_std(velocity, ax=ax, **kwargs)

    if shaded_region is not None:
        ax = add_shaded_region(
            ax, shaded_region[0], shaded_region[1], color=shaded_color, alpha=0.3
        )
    ax.set_xlabel("Time (frames)")
    value = "deg" if convert_to_degrees else "radians"
    ax.set_ylabel(f"Angular Velocity ({value}/s)")
    ax.set_xlim(0, velocity.shape[1])
    return ax


def plot_linear_velocity(
    data: dict,
    ax: Optional[plt.Axes] = None,
    shaded_region: list = [50, 80],
    shaded_color: str = "tab:red",
    **kwargs,
) -> plt.Axes:
    """
    Plots the linear velocity data.

    Parameters:
        data (dict): A dictionary containing the data.
        ax (Optional[plt.Axes]): The matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        shaded_region (list): A list specifying the range of the shaded region. Default is [50, 80].

    Returns:
        plt.Axes: The matplotlib Axes object with the plotted data.
    """

    velocity = data["linear_velocity"]
    ax = plot_mean_and_std(velocity, ax=ax, **kwargs)

    if shaded_region is not None:
        ax = add_shaded_region(
            ax, shaded_region[0], shaded_region[1], color=shaded_color, alpha=0.3
        )
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Linear Velocity (m/s)")
    ax.set_xlim(0, velocity.shape[1])
    return ax


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

    # remove `label` from kwargs to avoid passing it to fill_between
    kwargs.pop("label", None)

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


def plot_heading_difference_alternative(
    data: dict,
    ax: Optional[plt.Axes] = None,
    convert_to_degrees: bool = False,
    value_range: list = [-np.pi, np.pi],
    bins=36,
    **kwargs,
):
    ax.grid(zorder=0)  # Grid in background
    bin_edges = np.linspace(
        value_range[0], value_range[1], bins + 1
    )  # 37 to get 36 bins (n+1 edges for n bins)

    # Manual normalization
    hist1, _ = np.histogram(
        data["heading_difference"], bins=bin_edges, **kwargs.get("density", True)
    )

    # Normalize to make peaks similar heights
    hist1_norm = hist1 / hist1.max()

    # Plot the normalized histograms
    ax.bar(
        bin_edges[:-1], hist1_norm, width=np.diff(bin_edges), alpha=0.5, color="black"
    )
    # Check if the axes is using polar projection
    try:
        ax.set_theta_zero_location("N")
    except AttributeError:
        pass

    if convert_to_degrees:
        xticks = np.deg2rad([0, 90, 180, 270])
        xticklabels = ["0", "+90", "±180", "-90"]
    else:
        xticks = np.linspace(-np.pi, np.pi, 4)
        xticklabels = ["-π", "-π/2", "π", "π/2"]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    return ax


def plot_heading_difference(
    data: dict,
    ax: Optional[plt.Axes] = None,
    convert_to_degrees: bool = False,
    value_range: list = [-np.pi, np.pi],
    bins=36,
    **kwargs,
) -> plt.Axes:
    """
    Plots the histogram of heading differences.

    Args:
        data (dict): The data dictionary containing the heading differences.
        ax (Optional[plt.Axes]): The matplotlib Axes object to plot on. If not provided, a new figure and axes will be created.
        convert_to_degrees (bool): Whether to convert the heading differences to degrees. Default is False.
        value_range (list): The range of values to display on the x-axis. Default is [-np.pi, np.pi].

    Returns:
        plt.Axes: The matplotlib Axes object containing the histogram plot.
    """
    differences = data["heading_difference"]

    if convert_to_degrees:
        differences = np.rad2deg(differences)
        value_range = [-180, 180]

    ax = plot_histogram(differences, ax=ax, bins=bins, range=value_range, **kwargs)

    # Check if the axes is using polar projection
    if hasattr(ax, "projection") and getattr(ax.projection, "name", None) == "polar":
        ax.set_theta_zero_location("N")
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(["0", "+90", "±180", "-90"])
        ax.set_yticklabels([])
        ax.set_ylabel("")
    else:
        value = "degrees" if convert_to_degrees else "radians"
        ax.set_xlabel(f"Heading Difference ({value})")
        ax.set_ylabel("Count")

    return ax


def add_shaded_region(
    ax: plt.Axes,
    start: int,
    end: int,
    color: str = "gray",
    alpha: float = 0.3,
    **kwargs,
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
        linewidth=0,
        **kwargs,
    )

    return ax


def create_statistical_plot(
    data: np.ndarray,
    highlight_regions: Optional["list[Tuple[int, int]]"] = None,
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


def convert_frames_to_ms(ax: plt.Axes, fps: int = 100, step: int = 300) -> plt.Axes:
    """Convert x-axis ticks from frames to milliseconds.

    Args:
        ax: Matplotlib axes to convert x-axis ticks
        fps: Frames per second of the video. Default is 100.
        step: Time step in milliseconds between ticks. Default is 300

    Returns:
        plt.Axes: The axes object with updated x-axis ticks

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot(data)  # data in frames (0-150)
        >>> ax = convert_frames_to_ms(ax, fps=100, step=500)  # Will show ticks at 0, 500, 1000, 1500 ms
    """
    # Get current x-axis limits in frames
    x_min, x_max = ax.get_xlim()

    # Convert frame limits to milliseconds
    ms_min = (x_min / fps) * 1000
    ms_max = (x_max / fps) * 1000

    # Create tick positions in milliseconds with the specified step
    tick_positions_ms = np.arange(int(ms_min), int(ms_max) + step, step)

    # Convert millisecond positions back to frames for the axis
    tick_positions_frames = (tick_positions_ms / 1000) * fps

    # Set new ticks and labels
    ax.set_xticks(tick_positions_frames)
    ax.set_xticklabels([f"{int(ms)}" for ms in tick_positions_ms])

    # Set x-axis label
    ax.set_xlabel("Time (ms)")

    return ax


def plot_trajectory_with_arrows(
    data: dict,
    index: int,
    opto_range: list = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot the trajectory with arrows.

    Args:
        data (dict): The data containing the trajectory information.
        index (int): The index of the trajectory to plot.
        opto_range (list, optional): The range of indices to highlight with a different color. Defaults to None.
        ax (Optional[plt.Axes], optional): The axes to plot on. If None, a new figure will be created. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        plt.Axes: The axes object containing the plot.
    """

    # extract x and y coordinates
    x = data["position"][index][:, 0]
    y = data["position"][index][:, 1]

    # create new figure if no axes provided
    line = ax.plot(x, y, color="k")[0]
    if opto_range:
        ax.plot(
            x[opto_range[0] : opto_range[1]],
            y[opto_range[0] : opto_range[1]],
            color="tab:red",
        )
    ax.axis("off")

    for xi in range(0, len(x), kwargs.get("step", 5)):
        if opto_range:
            color = "tab:red" if xi in range(opto_range[0], opto_range[1]) else "k"
        _add_arrow(
            line,
            position=x[xi],
            direction="right",
            size=kwargs.get("size", 5),
            color=color,
        )

    return ax


def _add_arrow(line, position=None, direction="right", size=15, color=None):
    """
    Add an arrow to a line plot.

    Parameters:
    line (matplotlib.lines.Line2D): The line to add the arrow to.
    position (float, optional): The x-coordinate of the arrow's starting position. If not provided, it will be set to the mean of the line's x-data.
    direction (str, optional): The direction of the arrow. Can be 'right' or 'left'. Defaults to 'right'.
    size (int, optional): The size of the arrow. Defaults to 15.
    color (str, optional): The color of the arrow. If not provided, it will be set to the color of the line.

    Returns:
    None
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()

    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == "right":
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate(
        "",
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="-|>", color=color),
        size=size,
    )
