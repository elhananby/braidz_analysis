# braidz-analysis

Analysis tools for fruit fly (Drosophila) flight trajectories from the [Strand-Braid](https://github.com/strawlab/strand-braid/) tracking system.

## Features

- **Load trajectory data** from `.braidz` files (single or multiple)
- **Saccade detection** using angular velocity thresholds
- **Event-triggered analysis** for optogenetic and visual stimuli experiments
- **Flight state classification** to separate flying from walking/stationary
- **Publication-quality plotting** with customizable visualizations

## Installation

### Using uv (recommended)

```bash
uv sync
```

### Using conda/mamba

```bash
mamba env create -f environment.yaml
mamba activate braidz-analysis
```

### Using pip

```bash
pip install -e .
```

## Quick Start

```python
import braidz_analysis as ba

# Load data from a braidz file
data = ba.read_braidz("experiment.braidz")
print(data)  # BraidzData(42 trajectories, 125000 frames, 500 opto events)

# Analyze all saccades in the dataset
saccades = ba.analyze_saccades(data.trajectories)
print(f"Found {len(saccades)} saccades")

# Analyze optogenetic responses
opto = ba.analyze_event_responses(data.trajectories, data.opto)
print(f"Response rate: {opto.response_rate:.1%}")

# Filter to responsive trials and plot
ba.plot_angular_velocity(opto.responsive)
```

## Core Concepts

### Data Loading

The `read_braidz()` function loads one or more `.braidz` files and returns a `BraidzData` object:

```python
# Single file
data = ba.read_braidz("experiment.braidz")

# Multiple files
data = ba.read_braidz(["exp1.braidz", "exp2.braidz"])

# With base folder
data = ba.read_braidz("20230101.braidz", base_folder="/data/experiments")
```

The `BraidzData` object contains:
- `trajectories`: DataFrame with Kalman filter estimates (position, velocity)
- `opto`: DataFrame of optogenetic events (if present)
- `stim`: DataFrame of visual stimulus events (if present)

### Analysis Functions

#### Saccade Analysis

Detect and characterize all saccades (rapid turns) in the trajectory data:

```python
saccades = ba.analyze_saccades(data.trajectories)

# Access results
print(saccades.metrics[['heading_change', 'peak_velocity', 'amplitude']].describe())

# Filter by direction
left_turns = saccades.left_turns
right_turns = saccades.right_turns
```

#### Event-Triggered Analysis

Analyze responses to optogenetic or visual stimuli:

```python
# Works for both opto and stim - just pass the appropriate event DataFrame
opto = ba.analyze_event_responses(data.trajectories, data.opto)
stim = ba.analyze_event_responses(data.trajectories, data.stim)

# Access results
print(f"Response rate: {opto.response_rate:.1%}")
print(opto.metrics[['responded', 'heading_change', 'reaction_time']].head())

# Filter by response
responsive = opto.responsive
non_responsive = opto.non_responsive

# Filter by metadata (available columns depend on your event file)
high_intensity = opto.filter(intensity=100)
non_sham = opto.real  # Shortcut for filter(sham=False)
```

### Result Objects

Both analysis functions return structured result objects with:

- **traces**: Dictionary of time-aligned arrays
  - `angular_velocity`: (n_events, n_frames)
  - `linear_velocity`: (n_events, n_frames)
  - `position`: (n_events, n_frames, 3)

- **metrics**: DataFrame with computed metrics per event
  - For saccades: `heading_change`, `peak_velocity`, `amplitude`, `direction`
  - For events: `responded`, `heading_change`, `reaction_time`, `peak_velocity`, `max_velocity_in_window`
  - Note: Metrics are computed for all trials (not just responsive ones)

- **filter()**: Method to subset results by any column

### Configuration

Customize analysis parameters using the `Config` class:

```python
# Use defaults
results = ba.analyze_saccades(df)

# Custom configuration
config = ba.Config(
    saccade_threshold=400,       # deg/s (stricter threshold)
    response_window=50,          # frames to detect response
    pre_frames=100,              # frames before event
    post_frames=200,             # frames after event
)
results = ba.analyze_event_responses(df, events, config=config)

# Modify existing config
stim_config = ba.DEFAULT_CONFIG.with_overrides(response_window=50)
```

### Plotting

All plotting functions work with result objects:

```python
import matplotlib.pyplot as plt

# Angular velocity trace with stimulus highlight
ba.plot_angular_velocity(results, stimulus_range=(50, 80))

# Linear velocity
ba.plot_linear_velocity(results)

# Heading change distribution
ba.plot_heading_distribution(results, polar=True)

# Compare groups
ba.plot_heading_comparison(
    ["Control", "Opto"],
    [control_results, opto_results]
)

# Response rate by metadata
ba.plot_response_rate_by_group(results, group_by='intensity')

# Single trajectory visualization
ba.plot_trajectory(results, index=0, highlight_range=(50, 80))

# Create summary figure (3 panels)
fig, axes = ba.create_summary_figure(results, title="Experiment Results")
plt.savefig("summary.png")
```

## Modules

| Module | Description |
|--------|-------------|
| `io` | Read/write `.braidz` files |
| `kinematics` | Velocity, heading, saccade detection, flight state |
| `analysis` | High-level analysis functions |
| `plotting` | Visualization tools |
| `config` | Configuration dataclass |

### Advanced Kinematics

For lower-level access to kinematic calculations:

```python
# Compute velocities
heading, omega = ba.compute_angular_velocity(df['xvel'], df['yvel'])
speed = ba.compute_linear_velocity(df['xvel'], df['yvel'], df['zvel'])

# Detect saccades (mode: 'both', 'absolute', 'positive', 'negative')
peaks = ba.detect_saccades(omega, threshold=300, min_spacing=50, mode='absolute')

# Classify flight state
is_flying = ba.classify_flight_state(speed)

# Extract flight bouts
bouts = ba.extract_flight_bouts(speed)

# Add kinematics to DataFrame
df = ba.add_kinematics_to_trajectory(df)
```

## Example Notebooks

See the `notebooks/` directory for complete examples:

- `01_basic_usage.ipynb` - Loading data and basic analysis
- `02_opto_analysis.ipynb` - Optogenetic response analysis
- `03_saccade_analysis.ipynb` - Detailed saccade characterization
- `04_visual_stim_analysis.ipynb` - Visual stimuli (looming) analysis

## API Reference

### Main Functions

```python
# Data loading
ba.read_braidz(files, base_folder=None, engine='auto', pre_filter=True) -> BraidzData

# Analysis
ba.analyze_saccades(df, config=None, flight_only=True) -> SaccadeResults
ba.analyze_event_responses(df, events, config=None) -> EventResults
ba.filter_trajectories(df, config=None) -> pd.DataFrame

# Plotting
ba.plot_angular_velocity(results, ax=None, use_abs=True, baseline_range=(0, 50), ...)
ba.plot_linear_velocity(results, ax=None, stimulus_range=(50, 80), ...)
ba.plot_heading_distribution(results, ax=None, polar=False, ...)
ba.plot_trajectory(results, index, dims=('x', 'y'), highlight_range=None, ...)
ba.create_summary_figure(results, title=None) -> (fig, axes)
```

### Configuration Options

```python
ba.Config(
    # Timing
    fps=100.0,              # Frame rate (Hz)
    pre_frames=50,          # Frames before event
    post_frames=100,        # Frames after event
    response_delay=0,       # Frames to wait before looking for response
    response_window=30,     # Frames after event to stop looking

    # Saccade detection
    saccade_threshold=300,  # deg/s
    min_saccade_spacing=50, # frames
    heading_window=10,      # frames for heading calculation

    # Quality filters
    min_trajectory_frames=150,
    z_bounds=(0.05, 0.3),
    max_radius=0.23,
)

# Example: For looming stimuli with 300ms expansion time
looming_config = ba.Config(
    response_delay=30,      # Ignore first 300ms (stimulus expanding)
    response_window=60,     # Look up to 600ms after onset
    post_frames=150,
    min_trajectory_frames=250,
)
```

## Migration from v0.2

The v0.3 API has been completely rewritten with a simplified architecture.
The old modules (`braidz`, `processing`, `trajectory`, `params`) have been removed.

| Old (v0.2) | New (v0.3) |
|------------|------------|
| `ba.braidz.read_braidz(file)` | `ba.read_braidz(file)` |
| `ba.processing.get_stim_or_opto_data(df, opto, type='opto')` | `ba.analyze_event_responses(df, opto)` |
| `ba.processing.get_all_saccades(df)` | `ba.analyze_saccades(df)` |
| `ba.params.OptoAnalysisParams(...)` | `ba.Config(...)` |
| `data['df']`, `data['opto']` | `data.trajectories`, `data.opto` |
| Dict results | `SaccadeResults` / `EventResults` with `.filter()` methods |

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
