import matplotlib.pyplot as plt
from .helpers import get_mean_and_std
import numpy as np


def plot_mean_and_std(arr: np.array, ax: plt.Axes = None, **kwargs):
    mean, std = get_mean_and_std(arr)
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(mean, **kwargs)
    ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3, **kwargs)
    return ax


def plot_histogram(arr: np.array, ax: plt.Axes = None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(arr, **kwargs)
    return ax


def add_shaded_region(
    ax: plt.Axes, start: int, end: int, color: str = "gray", alpha: float = 0.3
):
    ax.axvspan(start, end, color=color, alpha=alpha)
    return ax
