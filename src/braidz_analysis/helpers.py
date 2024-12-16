import numpy as np

def calculate_angle_between_three_points(u1, u2, u3):
    """
    Calculate the angle between three points with u2 as the vertex.
    
    Args:
        u1 (tuple): First point coordinates (x, y)
        u2 (tuple): Vertex point coordinates (x, y)
        u3 (tuple): Third point coordinates (x, y)
        
    Returns:
        float: Angle in radians between -π and π
    """
    # Convert points to numpy arrays for easier vector operations
    p1 = np.array(u1)
    p2 = np.array(u2)
    p3 = np.array(u3)
    
    # Calculate vectors from vertex to other points
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate dot product
    dot_product = np.dot(v1, v2)
    
    # Calculate magnitudes
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    # Calculate cosine of angle
    cos_angle = dot_product / (mag1 * mag2)
    
    # Handle floating point errors
    cos_angle = min(1.0, max(-1.0, cos_angle))
    
    # Calculate angle using arccos
    angle = np.arccos(cos_angle)
    
    # Determine sign of angle using cross product
    cross_product = np.cross(v1, v2)
    if cross_product < 0:
        angle = -angle
        
    return angle


def get_mean_and_std(arr):
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)