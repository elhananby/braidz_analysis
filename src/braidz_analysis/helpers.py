from typing import Tuple, Union
import numpy as np
import numpy.typing as npt

# Type aliases for clarity
Point2D = Union[Tuple[float, float], npt.NDArray[np.float64]]
Vector = npt.NDArray[np.float64]


def calculate_angle_between_three_points(
    p1: Point2D, p2: Point2D, p3: Point2D
) -> float:
    """Calculate the signed angle between three 2D points with p2 as the vertex.

    Computes the angle between vectors (p1 - p2) and (p3 - p2), where p2 is the vertex.
    The angle is signed: positive for counterclockwise angles and negative for clockwise angles.

    Mathematical Steps:
    1. Create vectors v1 = p1 - p2 and v2 = p3 - p2
    2. Calculate angle using dot product: cos(θ) = (v1 · v2) / (|v1| * |v2|)
    3. Determine sign using cross product: sign(v1 × v2)

    Args:
        p1: First point coordinates (x, y)
        p2: Vertex point coordinates (x, y)
        p3: Third point coordinates (x, y)

    Returns:
        float: Signed angle in radians between -π and π

    Example:
        >>> p1 = (0, 0)
        >>> p2 = (1, 1)
        >>> p3 = (2, 0)
        >>> angle = calculate_angle_between_three_points(p1, p2, p3)

    Notes:
        - Input points can be either tuples or numpy arrays
        - Uses arccos for angle calculation, with handling for floating-point errors
        - Sign is determined by the direction of rotation from v1 to v2
    """
    # Convert points to numpy arrays if they aren't already
    v1 = np.asarray(p1) - np.asarray(p2)
    v2 = np.asarray(p3) - np.asarray(p2)

    # Calculate dot product and vector magnitudes
    dot_product = np.dot(v1, v2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)

    # Avoid division by zero
    if magnitude_product == 0:
        return 0.0

    # Calculate cosine of angle with floating point error handling
    cos_angle = np.clip(dot_product / magnitude_product, -1.0, 1.0)

    # Calculate the unsigned angle
    angle = np.arccos(cos_angle)

    # Determine rotation direction using cross product
    # For 2D vectors, cross product is scalar (z-component)
    cross_product = np.cross(v1, v2)

    # Apply sign based on rotation direction
    return -angle if cross_product < 0 else angle


def get_mean_and_std(
    arr: npt.NDArray[np.float64], axis: int = 0, ignore_nan: bool = True
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate mean and standard deviation along specified axis.

    Computes the mean and standard deviation of an array, optionally handling NaN values.
    Uses numpy's optimized functions for calculation.

    Args:
        arr: Input array of shape (n_samples, n_features) or arbitrary dimensions
        axis: Axis along which to compute statistics (default: 0)
        ignore_nan: Whether to ignore NaN values in calculation (default: True)

    Returns:
        Tuple containing:
            - Mean array of shape (n_features,) if axis=0
            - Standard deviation array of shape (n_features,) if axis=0

    Example:
        >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> mean, std = get_mean_and_std(data, axis=0)
        >>> print(f"Mean: {mean}")  # [4. 5. 6.]
        >>> print(f"Std: {std}")    # [3. 3. 3.]

    Notes:
        - If ignore_nan is True, uses np.nanmean and np.nanstd
        - If ignore_nan is False, uses np.mean and np.std
        - NaN handling affects both mean and standard deviation calculations
    """
    if ignore_nan:
        mean = np.nanmean(arr, axis=axis)
        std = np.nanstd(arr, axis=axis)
    else:
        mean = np.mean(arr, axis=axis)
        std = np.std(arr, axis=axis)

    return mean, std


def calculate_vector_angle(v1: Vector, v2: Vector) -> float:
    """Calculate the signed angle between two 2D vectors.

    A helper function that directly calculates the angle between two vectors
    without first constructing them from points.

    Args:
        v1: First vector as numpy array [x, y]
        v2: Second vector as numpy array [x, y]

    Returns:
        float: Signed angle in radians between -π and π

    Example:
        >>> v1 = np.array([1, 0])
        >>> v2 = np.array([0, 1])
        >>> angle = calculate_vector_angle(v1, v2)  # Returns π/2
    """
    # Calculate dot product and magnitudes
    dot_product = np.dot(v1, v2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)

    # Handle zero vectors
    if magnitude_product == 0:
        return 0.0

    # Calculate angle with floating point error handling
    cos_angle = np.clip(dot_product / magnitude_product, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    # Determine sign using cross product
    cross_product = np.cross(v1, v2)
    return -angle if cross_product < 0 else angle


def dict_list_to_numpy(data: dict) -> dict:
    # convert to numpy arrays and return
    for k, v in data.items():
        try:
            data[k] = np.array(v)
        except ValueError:
            print(f"Error converting {k} to numpy array")
    return data


def subtract_baseline(arr: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    """Subtract baseline from input array along the second axis.

    Args:
        arr: Input array of shape (n_samples, n_timepoints)
        start_idx: Start index for baseline subtraction
        end_idx: End index for baseline subtraction

    Returns:
        np.ndarray: Array with baseline subtracted along the second axis

    Example:
        >>> data = np.random.randn(100, 10)  # 100 samples, 10 timepoints
        >>> baseline_subtracted = subtract_baseline(data, start_idx=0, end_idx=5)
    """
    baseline = np.nanmean(arr[:, start_idx:end_idx], axis=1)
    normalized_data = arr - baseline[:, np.newaxis]
    return normalized_data
