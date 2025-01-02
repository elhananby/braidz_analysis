import numpy as np

DEFAULT_TRAJECTORY_PARAMS = {
    "min_frames": 300,
    "radius": 0.025,
    "heading_diff_window": 10,
}

DEFAULT_SACCADE_PARAMS = {
    "threshold": np.deg2rad(300),
    "distance": 10,
}

DEFAULT_OPTO_PARAMS = {
    "pre_frames": 50,
    "post_frames": 100,
    "duration": 30,  # in frames
}

DEFAULT_STIM_PARAMS = {
    "pre_frames": 50,
    "post_frames": 100,
    "duration": 50,
}
