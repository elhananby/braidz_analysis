import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Type, TypeVar, Union

import numpy as np


@dataclass
class TrajectoryParams:
    max_radius = 0.23
    xy_range = [-0.23, 0.23]
    z_range = [0.05, 0.3]
    heading_diff_window = 10

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
        )


@dataclass
class OptoAnalysis:
    """Parameters for Optogenetic trajectory analysis.

    Attributes:
        pre_frames (int): Number of frames to analyze before event
        post_frames (int): Number of frames to analyze after event
        min_frames (int): Minimum number of frames required for analysis
        opto_radius (float): Radius threshold for optogenetic stimulation
        opto_duration (int): Duration of optogenetic stimulation in frames
    """

    pre_frames: int = 50
    post_frames: int = 100
    min_frames: int = 150
    opto_radius: float = 0.025
    opto_duration: int = 30

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
        )


@dataclass
class LoomingAnalysis:
    """Parameters for Optogenetic trajectory analysis.

    Attributes:
        pre_frames (int): Number of frames to analyze before event
        post_frames (int): Number of frames to analyze after event
        min_frames (int): Minimum number of frames required for analysis
        opto_radius (float): Radius threshold for optogenetic stimulation
        opto_duration (int): Duration of optogenetic stimulation in frames
    """

    pre_frames: int = 50
    post_frames: int = 100
    min_frames: int = 150
    looming_duration: int = 30
    response_delay: int = 20

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
        )


@dataclass
class SaccadeParams:
    """Parameters for saccade detection.

    Attributes:
        threshold (float): Angular velocity threshold for saccade detection
        distance (int): Minimum distance between saccades in frames
    """

    pre_frames: int = 50
    post_frames: int = 50
    min_trajectory_length: int = 150
    threshold: float = np.deg2rad(300)
    distance: int = 10

    @classmethod
    def from_dict(cls, env):
        return cls(
            **{k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
        )


class AnalysisParamType(Enum):
    """Enum class for different types of behavioral analysis parameters.

    Each enum value corresponds to a specific parameter dataclass and includes
    metadata about the analysis type.
    """

    TRAJECTORY = (
        "TrajectoryParams",
        TrajectoryParams,
        "Basic trajectory analysis parameters",
    )
    OPTOGENETICS = (
        "OptoAnalysis",
        OptoAnalysis,
        "Optogenetic stimulation analysis parameters",
    )
    LOOMING = (
        "LoomingAnalysis",
        LoomingAnalysis,
        "Looming stimulus analysis parameters",
    )
    SACCADE = ("SaccadeParams", SaccadeParams, "Saccade detection parameters")

    def __init__(self, class_name: str, param_class: Type, description: str):
        self.class_name = class_name
        self.param_class = param_class
        self.description = description

    def create_params(
        self, **kwargs
    ) -> Union[TrajectoryParams, OptoAnalysis, LoomingAnalysis, SaccadeParams]:
        """Create a parameter instance of the appropriate type."""
        return self.param_class(**kwargs)

    @classmethod
    def from_class_name(cls, name: str) -> "AnalysisParamType":
        """Get enum member by class name."""
        for member in cls:
            if member.class_name == name:
                return member
        raise ValueError(f"No parameter type found for class name: {name}")


T = TypeVar("T", TrajectoryParams, OptoAnalysis, LoomingAnalysis, SaccadeParams)


class ParameterManager:
    """Manages parameter initialization and validation for behavioral analysis."""

    @staticmethod
    def initialize_params(
        param_type: AnalysisParamType, params: Optional[Union[dict, T]] = None
    ) -> T:
        """Initialize parameters of a specific type from dict or existing instance."""
        if isinstance(params, dict):
            return param_type.param_class.from_dict(params)
        elif params is None:
            return param_type.param_class()
        elif isinstance(params, param_type.param_class):
            return params
        else:
            raise TypeError(
                f"Parameters must be dict, {param_type.class_name}, or None, "
                f"not {type(params)}"
            )
