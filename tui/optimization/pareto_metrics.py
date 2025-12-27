import logging
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try to import pymoo for hypervolume calculation
try:
    from pymoo.indicators.hv import HV
    HAS_PYMOO_HV = True
except ImportError:
    HAS_PYMOO_HV = False
    logger.debug("pymoo not available for hypervolume calculation. Using custom implementation.")

def calculate_hypervolume(
    pareto_points: List[Tuple[float, float]],
    reference_point: Optional[Tuple[float, float]] = None,
    maximize: Optional[List[bool]] = None
) -> float:
    """Calculate hypervolume indicator for 2D Pareto front.
    
    Args:
        pareto_points: List of (obj1_value, obj2_value) tuples on the Pareto front
        reference_point: Reference point for hypervolume calculation. If None, uses worst point.
        maximize: List of [maximize_obj1, maximize_obj2] booleans. 
                 If None, assumes both should be maximized.
        
    Returns:
        Hypervolume value (area in 2D)
    """
    if not pareto_points or len(pareto_points) < 2:
        return 0.0
    
    np = _ensure_numpy()
    
    # Normalize points (convert minimize to maximize)
    if maximize is None:
        maximize = [True, True]
    
    normalized_points = []
    for obj1_val, obj2_val in pareto_points:
        norm_obj1 = obj1_val if maximize[0] else -obj1_val
        norm_obj2 = obj2_val if maximize[1] else -obj2_val
        normalized_points.append((norm_obj1, norm_obj2))
    
    points_array = np.array(normalized_points)
    
    # Determine reference point
    if reference_point is None:
        # Use worst point (minimum of each objective) as reference
        ref_obj1 = np.min(points_array[:, 0]) - 0.1 * (np.max(points_array[:, 0]) - np.min(points_array[:, 0]))
        ref_obj2 = np.min(points_array[:, 1]) - 0.1 * (np.max(points_array[:, 1]) - np.min(points_array[:, 1]))
        reference_point = (ref_obj1, ref_obj2)
    else:
        # Normalize reference point
        ref_obj1 = reference_point[0] if maximize[0] else -reference_point[0]
        ref_obj2 = reference_point[1] if maximize[1] else -reference_point[1]
        reference_point = (ref_obj1, ref_obj2)
    
    if HAS_PYMOO_HV:
        try:
            # Use pymoo's hypervolume indicator
            hv = HV(ref_point=np.array(reference_point))
            hv_value = hv(points_array)
            return float(hv_value)
        except Exception as e:
            logger.warning(f"Error using pymoo hypervolume: {e}, falling back to custom calculation")
    
    # Custom 2D hypervolume calculation (area calculation)
    # Sort points by first objective
    sorted_points = sorted(normalized_points, key=lambda p: p[0])
    
    if len(sorted_points) == 0:
        return 0.0
    
    # Calculate hypervolume as sum of rectangles
    hv_value = 0.0
    ref_obj1, ref_obj2 = reference_point
    
    prev_obj1 = ref_obj1
    for obj1, obj2 in sorted_points:
        # Rectangle: (obj1 - prev_obj1) * (obj2 - ref_obj2)
        width = obj1 - prev_obj1
        height = obj2 - ref_obj2
        if width > 0 and height > 0:
            hv_value += width * height
        prev_obj1 = obj1
    
    return float(hv_value)

def find_knee_points(
    pareto_points: List[Tuple[float, float]],
    maximize: Optional[List[bool]] = None
) -> List[int]:
    """Find knee points on the Pareto front using normalized Euclidean distance.
    
    Knee points are points that maximize the distance from the line connecting
    the extreme points of the Pareto front.
    
    Args:
        pareto_points: List of (obj1_value, obj2_value) tuples on the Pareto front
        maximize: List of [maximize_obj1, maximize_obj2] booleans.
                 If None, assumes both should be maximized.
        
    Returns:
        List of indices of knee points in pareto_points
    """
    if not pareto_points or len(pareto_points) < 3:
        return []
    
    np = _ensure_numpy()
    
    # Normalize points
    if maximize is None:
        maximize = [True, True]
    
    normalized_points = []
    for obj1_val, obj2_val in pareto_points:
        norm_obj1 = obj1_val if maximize[0] else -obj1_val
        norm_obj2 = obj2_val if maximize[1] else -obj2_val
        normalized_points.append((norm_obj1, norm_obj2))
    
    points_array = np.array(normalized_points)
    
    # Find extreme points (best in each objective)
    if len(points_array) < 3:
        return []
    
    # Get min/max for normalization
    obj1_min, obj1_max = np.min(points_array[:, 0]), np.max(points_array[:, 0])
    obj2_min, obj2_max = np.min(points_array[:, 1]), np.max(points_array[:, 1])
    
    # Normalize to [0, 1] range for distance calculation
    if obj1_max != obj1_min:
        obj1_norm = (points_array[:, 0] - obj1_min) / (obj1_max - obj1_min)
    else:
        obj1_norm = np.zeros(len(points_array))
    
    if obj2_max != obj2_min:
        obj2_norm = (points_array[:, 1] - obj2_min) / (obj2_max - obj2_min)
    else:
        obj2_norm = np.zeros(len(points_array))
    
    normalized_array = np.column_stack([obj1_norm, obj2_norm])
    
    # Find extreme points
    idx_obj1_best = np.argmax(normalized_array[:, 0])
    idx_obj2_best = np.argmax(normalized_array[:, 1])
    
    # If same point, no knee points
    if idx_obj1_best == idx_obj2_best:
        return []
    
    # Line connecting extreme points: ax + by + c = 0
    p1 = normalized_array[idx_obj1_best]
    p2 = normalized_array[idx_obj2_best]
    
    # Vector from p1 to p2
    line_vec = p2 - p1
    line_length = np.linalg.norm(line_vec)
    
    if line_length < 1e-10:
        return []
    
    # Normalize line vector
    line_vec_norm = line_vec / line_length
    
    # Calculate distance from each point to the line
    distances = []
    for i, point in enumerate(normalized_array):
        if i == idx_obj1_best or i == idx_obj2_best:
            distances.append(0.0)  # Extreme points have zero distance
        else:
            # Vector from p1 to point
            vec_to_point = point - p1
            # Project onto line
            proj_length = np.dot(vec_to_point, line_vec_norm)
            proj_point = p1 + proj_length * line_vec_norm
            # Distance from point to line
            dist = np.linalg.norm(point - proj_point)
            distances.append(dist)
    
    distances = np.array(distances)
    
    # Find points with maximum distance (knee points)
    # Use a threshold: points with distance > 0.1 * max_distance
    max_dist = np.max(distances)
    if max_dist < 1e-10:
        return []
    
    threshold = 0.1 * max_dist
    knee_indices = [i for i, d in enumerate(distances) if d >= threshold and i != idx_obj1_best and i != idx_obj2_best]
    
    # If no points above threshold, return the point with maximum distance
    if not knee_indices:
        knee_idx = np.argmax(distances)
        if knee_idx != idx_obj1_best and knee_idx != idx_obj2_best:
            knee_indices = [knee_idx]
    
    return knee_indices

def _ensure_numpy():
    """Ensure numpy is available."""
    try:
        import numpy as np
        return np
    except ImportError:
        raise ImportError("numpy is required for Pareto metrics calculation")


