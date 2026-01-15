"""
Plotting module for braidz analysis visualization.

Provides functions for creating publication-quality plots of:
    - Response traces (angular velocity, linear velocity)
    - Heading change distributions
    - Trajectory visualizations
    - Summary statistics

All plotting functions accept result objects from the analysis module
and return matplotlib axes for further customization.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .analysis import EventResults, SaccadeResults


# =============================================================================
# Utility Functions
# =============================================================================


def _get_mean_and_std(arr: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard deviation, ignoring NaN values."""
    return np.nanmean(arr, axis=axis), np.nanstd(arr, axis=axis)


def _subtract_baseline(
    arr: np.ndarray, start: int, end: int
) -> np.ndarray:
    """Subtract baseline (mean of range) from each row."""
    baseline = np.nanmean(arr[:, start:end], axis=1)
    return arr - baseline[:, np.newaxis]


# =============================================================================
# Trace Plotting
# =============================================================================


def plot_traces(
    results: Union[SaccadeResults, EventResults],
    trace_type: str = "angular_velocity",
    ax: Optional[plt.Axes] = None,
    use_abs: bool = False,
    baseline_range: Optional[Tuple[int, int]] = None,
    convert_to_degrees: bool = True,
    color: Optional[str] = None,
    label: Optional[str] = None,
    alpha: float = 0.3,
    **kwargs,
) -> plt.Axes:
    """
    Plot mean trace with standard deviation envelope.

    Args:
        results: SaccadeResults or EventResults object.
        trace_type: Which trace to plot ('angular_velocity', 'linear_velocity').
        ax: Matplotlib axes. Creates new if None.
        use_abs: Use absolute values (for angular velocity).
        baseline_range: (start, end) frames for baseline subtraction.
        convert_to_degrees: Convert angular velocity to degrees.
        color: Line color.
        label: Legend label.
        alpha: Transparency for std envelope.
        **kwargs: Additional arguments for ax.plot().

    Returns:
        Matplotlib axes with the plot.

    Example:
        >>> plot_traces(opto_results, trace_type='angular_velocity')
        >>> plot_traces(opto_results.responsive, color='red', label='Responsive')
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    # Get trace data
    if trace_type not in results.traces:
        raise ValueError(f"Unknown trace type: {trace_type}. Available: {list(results.traces.keys())}")

    data = results.traces[trace_type].copy()

    if len(data) == 0:
        return ax

    # Process data
    if use_abs:
        data = np.abs(data)
        if baseline_range is not None:
            data = _subtract_baseline(data, baseline_range[0], baseline_range[1])
    elif baseline_range is not None:
        raise ValueError("Baseline subtraction requires use_abs=True")

    if trace_type == "angular_velocity" and convert_to_degrees:
        data = np.degrees(data)

    # Compute statistics
    mean, std = _get_mean_and_std(data)

    # Plot
    x = np.arange(len(mean))
    line = ax.plot(x, mean, color=color, label=label, **kwargs)[0]
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=line.get_color(),
        alpha=alpha,
        linewidth=0,
    )

    # Labels
    if trace_type == "angular_velocity":
        unit = "deg/s" if convert_to_degrees else "rad/s"
        ylabel = f"Angular Velocity ({unit})"
        if use_abs:
            ylabel = f"|Angular Velocity| ({unit})"
    else:
        ylabel = "Linear Velocity (m/s)"

    ax.set_xlabel("Frames")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, len(mean))

    return ax


def plot_angular_velocity(
    results: Union[SaccadeResults, EventResults],
    ax: Optional[plt.Axes] = None,
    use_abs: bool = True,
    baseline_range: Optional[Tuple[int, int]] = (0, 50),
    stimulus_range: Optional[Tuple[int, int]] = (50, 80),
    stimulus_color: str = "tab:red",
    convert_to_degrees: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Plot angular velocity traces with optional stimulus highlight.

    A convenience wrapper around plot_traces() with common defaults
    for event-triggered analysis.

    Args:
        results: Analysis results object.
        ax: Matplotlib axes.
        use_abs: Use absolute values (default True).
        baseline_range: Frames for baseline subtraction.
        stimulus_range: Frames to highlight as stimulus period.
        stimulus_color: Color for stimulus highlight.
        convert_to_degrees: Convert to degrees.
        **kwargs: Additional arguments for plot_traces().

    Returns:
        Matplotlib axes.
    """
    ax = plot_traces(
        results,
        trace_type="angular_velocity",
        ax=ax,
        use_abs=use_abs,
        baseline_range=baseline_range,
        convert_to_degrees=convert_to_degrees,
        **kwargs,
    )

    if stimulus_range is not None:
        add_stimulus_region(ax, stimulus_range[0], stimulus_range[1], color=stimulus_color)

    return ax


def plot_linear_velocity(
    results: Union[SaccadeResults, EventResults],
    ax: Optional[plt.Axes] = None,
    stimulus_range: Optional[Tuple[int, int]] = (50, 80),
    stimulus_color: str = "tab:red",
    **kwargs,
) -> plt.Axes:
    """
    Plot linear velocity traces with optional stimulus highlight.

    Args:
        results: Analysis results object.
        ax: Matplotlib axes.
        stimulus_range: Frames to highlight as stimulus period.
        stimulus_color: Color for stimulus highlight.
        **kwargs: Additional arguments for plot_traces().

    Returns:
        Matplotlib axes.
    """
    ax = plot_traces(
        results,
        trace_type="linear_velocity",
        ax=ax,
        **kwargs,
    )

    if stimulus_range is not None:
        add_stimulus_region(ax, stimulus_range[0], stimulus_range[1], color=stimulus_color)

    return ax


# =============================================================================
# Heading Change Plots
# =============================================================================


def plot_heading_distribution(
    results: Union[SaccadeResults, EventResults],
    ax: Optional[plt.Axes] = None,
    convert_to_degrees: bool = False,
    bins: int = 36,
    polar: bool = False,
    density: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Plot histogram of heading changes.

    Args:
        results: Analysis results object.
        ax: Matplotlib axes. Creates polar axes if polar=True and ax=None.
        convert_to_degrees: Convert to degrees for display.
        bins: Number of histogram bins.
        polar: Use polar projection.
        density: Normalize as density.
        **kwargs: Additional arguments for histogram.

    Returns:
        Matplotlib axes.

    Example:
        >>> # Standard histogram
        >>> plot_heading_distribution(results)

        >>> # Polar plot
        >>> plot_heading_distribution(results, polar=True)
    """
    # Get heading changes
    heading_changes = results.metrics["heading_change"].dropna().values

    if len(heading_changes) == 0:
        if ax is None:
            _, ax = plt.subplots()
        return ax

    # Set up value range
    if convert_to_degrees:
        heading_changes = np.degrees(heading_changes)
        value_range = (-180, 180)
    else:
        value_range = (-np.pi, np.pi)

    # Create axes
    if ax is None:
        if polar:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})
        else:
            _, ax = plt.subplots()

    # Plot histogram
    if polar:
        # For polar plots, use bar chart
        hist, bin_edges = np.histogram(heading_changes, bins=bins, range=value_range, density=density)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = np.diff(bin_edges)[0]

        if not convert_to_degrees:
            ax.bar(bin_centers, hist, width=width, alpha=0.7, **kwargs)
        else:
            ax.bar(np.radians(bin_centers), hist, width=np.radians(width), alpha=0.7, **kwargs)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
    else:
        ax.hist(heading_changes, bins=bins, range=value_range, density=density, **kwargs)
        unit = "degrees" if convert_to_degrees else "radians"
        ax.set_xlabel(f"Heading Change ({unit})")
        ax.set_ylabel("Density" if density else "Count")

    return ax


def plot_heading_comparison(
    groups: List[str],
    results_list: List[Union[SaccadeResults, EventResults]],
    ax: Optional[plt.Axes] = None,
    colors: Union[str, List[str]] = "tab10",
) -> plt.Axes:
    """
    Compare heading change distributions across groups using violin plots.

    Args:
        groups: List of group names.
        results_list: List of result objects (one per group).
        ax: Matplotlib axes.
        colors: Color palette name or list of colors.

    Returns:
        Matplotlib axes.

    Example:
        >>> plot_heading_comparison(
        ...     ["Control", "Opto"],
        ...     [control_results, opto_results]
        ... )
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    # Build DataFrame for seaborn
    data = []
    for group, results in zip(groups, results_list):
        hc = results.metrics["heading_change"].dropna().values
        for val in hc:
            data.append({"Group": group, "Heading Change": val})

    import pandas as pd

    df = pd.DataFrame(data)

    # Plot
    sns.violinplot(
        data=df,
        x="Heading Change",
        y="Group",
        ax=ax,
        hue="Group",
        palette=colors,
        legend=False,
    )

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(["-180°", "0°", "+180°"])

    return ax


# =============================================================================
# Trajectory Visualization
# =============================================================================


def plot_trajectory(
    results: Union[SaccadeResults, EventResults],
    index: int,
    dims: Tuple[str, str] = ("x", "y"),
    ax: Optional[plt.Axes] = None,
    highlight_range: Optional[Tuple[int, int]] = None,
    highlight_color: str = "tab:red",
    show_arrows: bool = True,
    arrow_spacing: int = 5,
    normalize: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Plot a single trajectory in 2D.

    Args:
        results: Analysis results object.
        index: Index of trajectory to plot.
        dims: Tuple of dimensions ('x', 'y', or 'z').
        ax: Matplotlib axes.
        highlight_range: Frame range to highlight (e.g., stimulus period).
        highlight_color: Color for highlighted portion.
        show_arrows: Add direction arrows.
        arrow_spacing: Frames between arrows.
        normalize: Normalize coordinates to [-1, 1].
        **kwargs: Additional arguments for plot().

    Returns:
        Matplotlib axes.

    Example:
        >>> # Plot first trial with stimulus highlighted
        >>> plot_trajectory(results, 0, highlight_range=(50, 80))
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Dimension mapping
    dim_map = {"x": 0, "y": 1, "z": 2}
    d1, d2 = dim_map[dims[0]], dim_map[dims[1]]

    # Extract position
    pos = results.traces["position"][index]
    coord1 = pos[:, d1]
    coord2 = pos[:, d2]

    # Normalize if requested
    if normalize:
        coord1 = 2 * (coord1 - coord1.min()) / (coord1.max() - coord1.min() + 1e-10) - 1
        coord2 = 2 * (coord2 - coord2.min()) / (coord2.max() - coord2.min() + 1e-10) - 1

    # Plot main trajectory
    line = ax.plot(coord1, coord2, "k-", linewidth=kwargs.get("linewidth", 1))[0]

    # Highlight range if specified
    if highlight_range is not None:
        start, end = highlight_range
        ax.plot(
            coord1[start:end],
            coord2[start:end],
            color=highlight_color,
            linewidth=kwargs.get("linewidth", 1) * 1.5,
        )

    # Add arrows
    if show_arrows:
        for i in range(0, len(coord1) - 1, arrow_spacing):
            color = highlight_color if highlight_range and highlight_range[0] <= i < highlight_range[1] else "k"
            if i + 1 < len(coord1):
                ax.annotate(
                    "",
                    xy=(coord1[i + 1], coord2[i + 1]),
                    xytext=(coord1[i], coord2[i]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.5),
                )

    ax.set_xlabel(dims[0])
    ax.set_ylabel(dims[1])
    ax.set_aspect("equal")
    ax.axis("off")

    return ax


# =============================================================================
# Summary and Statistics Plots
# =============================================================================


def plot_response_rate_by_group(
    results: EventResults,
    group_by: str,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot response rate as a function of a metadata variable.

    Args:
        results: EventResults object.
        group_by: Metadata column to group by (e.g., 'intensity').
        ax: Matplotlib axes.
        **kwargs: Additional arguments for bar plot.

    Returns:
        Matplotlib axes.

    Example:
        >>> plot_response_rate_by_group(results, group_by='intensity')
    """
    if ax is None:
        _, ax = plt.subplots()

    if group_by not in results.metadata.columns:
        raise ValueError(f"Column '{group_by}' not in metadata. Available: {list(results.metadata.columns)}")

    # Compute response rates
    import pandas as pd

    combined = pd.concat([results.metrics, results.metadata], axis=1)
    stats = combined.groupby(group_by)["responded"].agg(["mean", "sum", "count"])
    stats.columns = ["rate", "n_responses", "n_total"]

    # Plot
    ax.bar(range(len(stats)), stats["rate"], **kwargs)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels([str(x) for x in stats.index])
    ax.set_xlabel(group_by.replace("_", " ").title())
    ax.set_ylabel("Response Rate")
    ax.set_ylim(0, 1)

    # Add count annotations
    for i, (rate, n) in enumerate(zip(stats["rate"], stats["n_total"])):
        ax.annotate(
            f"n={int(n)}",
            xy=(i, rate + 0.02),
            ha="center",
            fontsize=8,
        )

    return ax


# =============================================================================
# Helper Functions
# =============================================================================


def add_stimulus_region(
    ax: plt.Axes,
    start: int,
    end: int,
    color: str = "gray",
    alpha: float = 0.3,
    **kwargs,
) -> plt.Axes:
    """
    Add a vertical shaded region to highlight stimulus period.

    Args:
        ax: Matplotlib axes.
        start: Start frame.
        end: End frame.
        color: Fill color.
        alpha: Transparency.

    Returns:
        Matplotlib axes.
    """
    ax.axvspan(start, end, color=color, alpha=alpha, linewidth=0, **kwargs)
    return ax


def convert_frames_to_ms(
    ax: plt.Axes,
    fps: float = 100.0,
    tick_step_ms: int = 100,
) -> plt.Axes:
    """
    Convert x-axis from frames to milliseconds.

    Args:
        ax: Matplotlib axes.
        fps: Frame rate.
        tick_step_ms: Interval between tick marks in milliseconds.

    Returns:
        Matplotlib axes with updated ticks.
    """
    x_min, x_max = ax.get_xlim()
    ms_min = (x_min / fps) * 1000
    ms_max = (x_max / fps) * 1000

    tick_ms = np.arange(int(ms_min), int(ms_max) + tick_step_ms, tick_step_ms)
    tick_frames = (tick_ms / 1000) * fps

    ax.set_xticks(tick_frames)
    ax.set_xticklabels([str(int(ms)) for ms in tick_ms])
    ax.set_xlabel("Time (ms)")

    return ax


def create_summary_figure(
    results: EventResults,
    title: str = None,
    figsize: Tuple[float, float] = (12, 4),
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create a three-panel summary figure for event responses.

    Panels:
        1. Angular velocity traces
        2. Linear velocity traces
        3. Heading change distribution

    Args:
        results: EventResults object.
        title: Optional figure title.
        figsize: Figure size.

    Returns:
        Tuple of (figure, list of axes).

    Example:
        >>> fig, axes = create_summary_figure(opto_results, title="Optogenetic Response")
        >>> plt.savefig("summary.png")
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Angular velocity
    plot_angular_velocity(results, ax=axes[0], use_abs=True, baseline_range=(0, 50))
    axes[0].set_title("Angular Velocity")

    # Linear velocity
    plot_linear_velocity(results, ax=axes[1])
    axes[1].set_title("Linear Velocity")

    # Heading distribution
    plot_heading_distribution(results, ax=axes[2], polar=False)
    axes[2].set_title("Heading Change")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, axes
