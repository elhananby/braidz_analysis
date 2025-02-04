# analysis.py

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np
from scipy.signal import find_peaks
from .params import AnalysisParams
import logging
from .core import TrajectoryData, StimulationEvent
from scipy.stats import circmean

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    angular_velocity: np.ndarray
    linear_velocity: np.ndarray
    position: np.ndarray
    heading_difference: float
    frames_in_radius: Optional[int] = None
    reaction_delay: Optional[int] = None
    responsive: bool = False
    pre_saccade_amplitude: Optional[float] = None
    pre_saccade_distance: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrajectoryAnalyzer:
    """Handles trajectory analysis operations."""

    def __init__(self, params: AnalysisParams):
        self.params = params
        self.params.validate()

    def detect_saccades(self, angular_velocity: np.ndarray) -> np.ndarray:
        """Detect saccades for both positive and negative direction in angular velocity data."""

        positive_peaks, _ = find_peaks(
            angular_velocity,
            height=self.params.saccade.threshold,
            distance=self.params.saccade.distance,
        )
        negative_peaks, _ = find_peaks(
            -angular_velocity,
            height=self.params.saccade.threshold,
            distance=self.params.saccade.distance,
        )

        peaks = np.sort(np.concatenate([positive_peaks, negative_peaks]))
        return peaks

    def find_relevant_peak(
        self, peaks: np.ndarray, stim_idx: int, return_none: bool = False
    ) -> tuple[Optional[int], bool]:
        """Find first peak within the stimulus window."""
        window_end = stim_idx + self.params.base.duration

        for peak in peaks:
            if stim_idx < peak < window_end:
                return int(peak), True

        return (
            None if return_none else stim_idx + self.params.base.duration // 2
        ), False

    def get_pre_saccade(
        self, peaks: np.ndarray, stim_idx: int, window: int = 100
    ) -> Optional[int]:
        """Get the closest peak before the stimulation event."""
        valid_peaks = peaks[(peaks < stim_idx) & (peaks > stim_idx - window)]
        return valid_peaks[-1] if len(valid_peaks) > 0 else None

    def calculate_heading_diff(
        self, heading: np.ndarray, start_idx: int, center_idx: int, end_idx: int
    ) -> float:
        """Calculate heading difference around a point."""
        # convert indices_before and indices_after to np.array if not already
        p1, p2, p3 = int(start_idx), int(center_idx), int(end_idx)

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
            np.sin(heading_after - heading_before),
            np.cos(heading_after - heading_before),
        )

        return heading_difference

    def analyze_segment(
        self, traj: TrajectoryData, stim_event: StimulationEvent
    ) -> Optional[AnalysisResult]:
        """Analyze a trajectory segment around a stimulation event."""
        try:
            # Find stimulation index
            stim_idx = np.where(traj.frames == stim_event.frame)[0][0]

            # Check if we have enough frames
            if (
                stim_idx < self.params.base.pre_frames
                or stim_idx >= len(traj.frames) - self.params.base.post_frames
            ):
                logger.debug("Insufficient frames around stimulation event")
                return None

            # Detect saccades
            peaks = self.detect_saccades(traj.angular_velocity)
            peak_idx, is_responsive = self.find_relevant_peak(peaks, stim_idx)

            # Extract segment around peak
            segment_start = peak_idx - self.params.base.pre_frames
            segment_end = peak_idx + self.params.base.post_frames
            segment = traj.get_segment(segment_start, segment_end)

            # Calculate heading difference
            heading_diff = self.calculate_heading_diff(
                traj.heading,
                peak_idx - self.params.saccade.heading_diff_window,
                peak_idx,
                peak_idx + self.params.saccade.heading_diff_window,
            )

            # Find pre-saccade if any
            pre_saccade_idx = self.get_pre_saccade(peaks, stim_idx)

            # Calculate frames in radius if applicable
            frames_in_radius = None
            radius = np.sqrt(
                np.sum(
                    traj.position[stim_idx : stim_idx + self.params.base.duration, :2]
                    ** 2,
                    axis=1,
                )
            )
            frames_in_radius = np.sum(radius < self.params.spatial.opto_radius)

            return AnalysisResult(
                angular_velocity=segment.angular_velocity,
                linear_velocity=segment.linear_velocity,
                position=segment.position,
                heading_difference=heading_diff,
                frames_in_radius=frames_in_radius,
                reaction_delay=peak_idx - stim_idx if is_responsive else None,
                responsive=is_responsive,
                pre_saccade_amplitude=traj.angular_velocity[pre_saccade_idx]
                if pre_saccade_idx is not None
                else None,
                pre_saccade_distance=pre_saccade_idx - stim_idx
                if pre_saccade_idx is not None
                else None,
                metadata={
                    "obj_id": traj.obj_id,
                    "exp_num": traj.exp_num,
                    "stim_frame": stim_event.frame,
                    "stim_type": stim_event.event_type,
                    "intensity": stim_event.intensity,
                    "duration": stim_event.duration,
                    "frequency": stim_event.frequency,
                    "sham": stim_event.sham,
                },
            )

        except (IndexError, ValueError) as e:
            logger.debug(f"Error analyzing segment: {str(e)}")
            return None
