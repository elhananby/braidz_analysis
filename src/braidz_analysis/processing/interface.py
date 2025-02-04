# interface.py

import logging
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .analysis import (
    AnalysisResult,
    TrajectoryAnalyzer,
)
from .core import StimulationEvent, TrajectoryData
from .filtering import TrajectoryFilter
from .params import AnalysisParams

logger = logging.getLogger(__name__)


class TrajectoryAnalysisManager:
    """Main interface for trajectory analysis."""

    def __init__(
        self,
        analysis_params: Optional[AnalysisParams] = None,
        show_progress: bool = True,
    ):
        """Initialize with optional custom parameters."""
        self.params = analysis_params or AnalysisParams()
        self.params.validate()
        self.show_progress = show_progress

        self.filter = TrajectoryFilter(self.params)
        self.analyzer = TrajectoryAnalyzer(self.params)

    def process_dataset(
        self,
        trajectory_df: pd.DataFrame,
        stim_df: pd.DataFrame,
        stim_type: str = "opto",
    ) -> List[AnalysisResult]:
        """
        Process a complete dataset of trajectories and stimulation events.

        Args:
            trajectory_df: DataFrame containing trajectory data
            stim_df: DataFrame containing stimulation events
            stim_type: Type of stimulation ("opto" or "stim")

        Returns:
            List[AnalysisResult]: List of analysis results
        """
        # Filter trajectories
        filtered_df = self.filter.filter_dataframe(trajectory_df)
        if filtered_df.empty:
            logger.warning("No valid trajectories after filtering")
            return []

        results = []
        stim_iter = (
            tqdm(stim_df.iterrows(), total=len(stim_df))
            if self.show_progress
            else stim_df.iterrows()
        )

        for _, stim_row in stim_iter:
            # Create stimulation event
            stim_event = StimulationEvent.from_series(stim_row, stim_type)

            # Get corresponding trajectory data
            traj_mask = (
                (filtered_df["obj_id"] == stim_event.obj_id)
                & (filtered_df["exp_num"] == stim_event.exp_num)
                if stim_event.exp_num is not None
                else filtered_df["obj_id"] == stim_event.obj_id
            )
            traj_data = TrajectoryData.from_dataframe(filtered_df[traj_mask])

            # Analyze trajectory
            result = self.analyzer.analyze_segment(traj_data, stim_event)
            if result is not None:
                results.append(result)

        return results

    def convert_to_legacy_format(self, results: List[AnalysisResult]) -> Dict:
        """
        Convert analysis results to the old dictionary format for backwards compatibility.

        Args:
            results: List of AnalysisResult objects

        Returns:
            Dict containing results in the old format
        """
        legacy_dict = {
            "angular_velocity": [],
            "linear_velocity": [],
            "position": [],
            "heading_difference": [],
            "frames_in_radius": [],
            "sham": [],
            "intensity": [],
            "duration": [],
            "frequency": [],
            "responsive": [],
            "timestamp": [],
            "reaction_delay": [],
            "pre_saccade_amplitude": [],
            "pre_saccade_distance": [],
        }

        for result in results:
            result_dict = asdict(result)
            metadata = result_dict.pop("metadata")

            for key in legacy_dict:
                if key in result_dict:
                    legacy_dict[key].append(result_dict[key])
                elif key in metadata:
                    legacy_dict[key].append(metadata[key])

        return legacy_dict
