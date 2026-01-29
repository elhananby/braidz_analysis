# braidz-analysis

Analysis tools for fruit fly (Drosophila) flight trajectories from the [Strand-Braid](https://github.com/strawlab/strand-braid/) tracking system.

## Features

- **Load trajectory data** from `.braidz` files (single or multiple)
- **Saccade detection** with two algorithms:
  - Velocity-based: Fast threshold detection on angular velocity
  - mGSD: Modified Geometric Saccade Detection for noisy/slow data
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

## Configuration

The `Config` class controls all analysis parameters. Create one config and use it throughout your analysis for consistency:

```python
import braidz_analysis as ba

# Configure once at the start
config = ba.Config(
    # Saccade detection
    saccade_method="mgsd",        # "velocity" or "mgsd"
    mgsd_dispersion="mean",       # "sum", "mean", or "std"
    mgsd_threshold=0.0005,

    # Response analysis
    response_window=50,
    pre_frames=50,
    post_frames=100,

    # Quality filters
    min_trajectory_frames=200,
    z_bounds=(0.05, 0.25),
)

# Use the same config everywhere
data = ba.read_braidz("experiment.braidz")
filtered = ba.filter_trajectories(data.trajectories, config=config)
saccades = ba.analyze_saccades(filtered, config=config)
opto = ba.analyze_event_responses(filtered, data.opto, config=config)
```

### Config Presets

Pre-configured defaults for common use cases:

```python
ba.DEFAULT_CONFIG   # Velocity-based saccade detection (default)
ba.MGSD_CONFIG      # mGSD with mean dispersion
ba.OPTO_CONFIG      # Optimized for optogenetic experiments
ba.STIM_CONFIG      # Optimized for visual stimulus experiments
```

### Config Sections

| Section | Parameters |
|---------|------------|
| **Recording** | `fps` |
| **Saccade Detection** | `saccade_method`, `saccade_threshold`, `min_saccade_spacing`, `heading_window` |
| **mGSD Parameters** | `mgsd_delta_frames`, `mgsd_threshold`, `mgsd_min_spacing`, `mgsd_dispersion` |
| **Event Analysis** | `pre_frames`, `post_frames`, `response_delay`, `response_window`, `detect_in_window_only` |
| **Quality Filters** | `min_trajectory_frames`, `z_bounds`, `max_radius`, `min_position_range` |
| **Smoothing** | `smoothing_window`, `smoothing_polyorder` |
| **Flight State** | `flight_high_threshold`, `flight_low_threshold`, `flight_min_frames` |

## Saccade Detection

Two algorithms are available:

### Velocity-Based (default)

Fast detection based on angular velocity peaks:

```python
config = ba.Config(
    saccade_method="velocity",
    saccade_threshold=300,      # deg/s
    min_saccade_spacing=50,     # frames
)
saccades = ba.analyze_saccades(df, config=config)
```

### Modified Geometric Saccade Detection (mGSD)

Geometric algorithm that combines amplitude (direction change) with dispersion (movement magnitude). Better for slow or noisy trajectories:

```python
config = ba.Config(
    saccade_method="mgsd",
    mgsd_delta_frames=5,        # window size (frames)
    mgsd_threshold=0.001,       # score threshold
    mgsd_dispersion="mean",     # "sum", "mean", or "std"
)
saccades = ba.analyze_saccades(df, config=config)

# Or use the preset
saccades = ba.analyze_saccades(df, config=ba.MGSD_CONFIG)
```

**Dispersion methods:**
- `"sum"`: Original implementation (score scales with window size)
- `"mean"`: Normalized (consistent across different `mgsd_delta_frames`)
- `"std"`: Standard deviation (as described in the paper)

**Reference:** Cellini, van Breugel,"; SMELLING IN REVERSE: PRESENTATION OF A FICTIVE ODOR GRADIENT AFTER PLUME DEPARTURE ELICITS A REVERSE SACCADE IN FLYING DROSOPHILA", Current Biology, 2024.

## Data Loading

```python
# Single file
data = ba.read_braidz("experiment.braidz")

# Multiple files
data = ba.read_braidz(["exp1.braidz", "exp2.braidz"])

# With base folder
data = ba.read_braidz("20230101.braidz", base_folder="/data/experiments")
```

The `BraidzData` object contains:
- `trajectories`: Polars DataFrame with Kalman filter estimates (position, velocity)
- `opto`: DataFrame of optogenetic events (if present)
- `stim`: DataFrame of visual stimulus events (if present)

## Analysis Functions

### Saccade Analysis

```python
saccades = ba.analyze_saccades(data.trajectories, config=config)

# Access results
print(saccades.metrics[['heading_change', 'peak_velocity', 'amplitude']].describe())

# Filter by direction
left_turns = saccades.left_turns
right_turns = saccades.right_turns
```

### Event-Triggered Analysis

```python
opto = ba.analyze_event_responses(data.trajectories, data.opto, config=config)

# Access results
print(f"Response rate: {opto.response_rate:.1%}")
print(opto.metrics[['responded', 'heading_change', 'reaction_time']].head())

# Filter by response
responsive = opto.responsive
non_responsive = opto.non_responsive

# Filter by metadata
high_intensity = opto.filter(intensity=100)
non_sham = opto.real  # Shortcut for filter(sham=False)
```

## Result Objects

Both analysis functions return structured result objects with:

- **traces**: Dictionary of time-aligned arrays
  - `angular_velocity`: (n_events, n_frames)
  - `linear_velocity`: (n_events, n_frames)
  - `position`: (n_events, n_frames, 3)

- **metrics**: Polars DataFrame with computed metrics per event
  - For saccades: `heading_change`, `peak_velocity`, `amplitude`, `direction`
  - For events: `responded`, `heading_change`, `reaction_time`, `peak_velocity`

- **filter()**: Method to subset results by any column

## Plotting

```python
import matplotlib.pyplot as plt

# Angular velocity trace with stimulus highlight
ba.plot_angular_velocity(results, stimulus_range=(50, 80))

# Linear velocity
ba.plot_linear_velocity(results)

# Heading change distribution
ba.plot_heading_distribution(results, polar=True)

# Compare groups
ba.plot_heading_comparison(["Control", "Opto"], [control_results, opto_results])

# Response rate by metadata
ba.plot_response_rate_by_group(results, group_by='intensity')

# Single trajectory visualization
ba.plot_trajectory(results, index=0, highlight_range=(50, 80))

# Create summary figure (3 panels)
fig, axes = ba.create_summary_figure(results, title="Experiment Results")
plt.savefig("summary.png")
```

## Advanced Kinematics

For lower-level access to kinematic calculations:

```python
# Compute velocities
heading, omega = ba.compute_angular_velocity(df['xvel'], df['yvel'])
speed = ba.compute_linear_velocity(df['xvel'], df['yvel'], df['zvel'])

# Detect saccades directly
peaks = ba.detect_saccades(omega, threshold=300, min_spacing=50)
peaks_mgsd = ba.detect_saccades_mgsd(x, y, omega, threshold=0.001)

# Get mGSD scores for visualization
peaks, scores, amps, disps = ba.detect_saccades_mgsd(
    x, y, omega, return_scores=True
)

# Compute mGSD scores without peak detection
scores, amplitudes, dispersions = ba.compute_mgsd_scores(x, y, omega)

# Classify flight state
is_flying = ba.classify_flight_state(speed)

# Extract flight bouts
bouts = ba.extract_flight_bouts(speed)

# Add kinematics to DataFrame
df = ba.add_kinematics_to_trajectory(df, config=config)
```

## Modules

| Module | Description |
|--------|-------------|
| `io` | Read/write `.braidz` files |
| `kinematics` | Velocity, heading, saccade detection, flight state |
| `analysis` | High-level analysis functions |
| `plotting` | Visualization tools |
| `config` | Configuration dataclass |

## Example Notebooks

See the `notebooks/` directory for complete examples:

- `01_basic_usage.ipynb` - Loading data and basic analysis
- `02_opto_analysis.ipynb` - Optogenetic response analysis
- `03_saccade_analysis.ipynb` - Detailed saccade characterization
- `04_visual_stim_analysis.ipynb` - Visual stimuli (looming) analysis

## Full Config Reference

```python
ba.Config(
    # === Recording ===
    fps=100.0,                    # Frame rate (Hz)

    # === Saccade Detection - Algorithm ===
    saccade_method="velocity",    # "velocity" or "mgsd"

    # === Saccade Detection - Velocity Method ===
    saccade_threshold=300.0,      # deg/s
    min_saccade_spacing=50,       # frames
    heading_window=10,            # frames for heading calculation

    # === Saccade Detection - mGSD Method ===
    mgsd_delta_frames=5,          # window size (frames before/after)
    mgsd_threshold=0.001,         # score threshold
    mgsd_min_spacing=10,          # frames between peaks
    mgsd_dispersion="sum",        # "sum", "mean", or "std"

    # === Event Analysis ===
    pre_frames=50,                # frames before event
    post_frames=100,              # frames after event
    response_delay=0,             # frames to wait before looking
    response_window=30,           # frames to search for response
    detect_in_window_only=False,  # more sensitive response detection

    # === Quality Filters ===
    min_trajectory_frames=150,
    z_bounds=(0.05, 0.3),         # meters
    max_radius=0.23,              # meters
    min_position_range=0.05,      # meters

    # === Smoothing ===
    smoothing_window=21,          # Savitzky-Golay window (odd)
    smoothing_polyorder=3,        # polynomial order

    # === Flight State ===
    flight_high_threshold=0.05,   # m/s to enter flight
    flight_low_threshold=0.01,    # m/s to exit flight
    flight_min_frames=20,         # sustained frames for state change
)
```

## Migration from v0.3

v0.4 changes:
- **Polars instead of Pandas**: All DataFrames are now Polars DataFrames
- **Config-based saccade method**: `saccade_method` and `mgsd_dispersion` moved from function arguments to Config
- **New mGSD algorithm**: `detect_saccades_mgsd()` and `compute_mgsd_scores()`
- **New preset**: `MGSD_CONFIG` for mGSD defaults

```python
# Old (v0.3)
saccades = ba.analyze_saccades(df, method="mgsd", dispersion_method="mean")

# New (v0.4)
config = ba.Config(saccade_method="mgsd", mgsd_dispersion="mean")
saccades = ba.analyze_saccades(df, config=config)

# Or use preset
saccades = ba.analyze_saccades(df, config=ba.MGSD_CONFIG)
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
