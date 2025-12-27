import logging
from typing import List, Tuple, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try to import pymoo for advanced Pareto algorithms
try:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False
    logger.debug("pymoo not available. Using custom Pareto front calculation.")

def calculate_pareto_front(
    objective1: str,
    objective2: str,
    data: List[Tuple[Dict[str, float], float, float]],
    algorithm: str = "dominance",
    maximize: Optional[List[bool]] = None
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Calculate Pareto front for two objectives.
    
    Args:
        objective1: Name of first objective
        objective2: Name of second objective
        data: List of (params_dict, obj1_value, obj2_value) tuples
        algorithm: Algorithm to use ('dominance', 'nsga2', 'nsga3')
        maximize: List of [maximize_obj1, maximize_obj2] booleans. 
                 If None, assumes both should be maximized.
                 If True, maximize; if False, minimize.
        
    Returns:
        Tuple of (pareto_points, dominated_points)
        Each point is (obj1_value, obj2_value)
    """
    if not data:
        return [], []
    
    np = _ensure_numpy()
    
    # Extract objective values
    points = []
    for params, obj1_val, obj2_val in data:
        points.append((obj1_val, obj2_val))
    
    if len(points) < 2:
        return points, []
    
    # Normalize points for dominance checking (convert minimize to maximize)
    if maximize is None:
        maximize = [True, True]  # Default: maximize both
    
    normalized_points = []
    for obj1_val, obj2_val in points:
        # If minimizing, negate the value
        norm_obj1 = obj1_val if maximize[0] else -obj1_val
        norm_obj2 = obj2_val if maximize[1] else -obj2_val
        normalized_points.append((norm_obj1, norm_obj2))
    
    # Convert to numpy array for efficient computation
    points_array = np.array(normalized_points)
    
    if algorithm == "dominance":
        # Custom pairwise dominance checking
        pareto_indices = []
        dominated_indices = []
        
        for i in range(len(points)):
            is_dominated = False
            for j in range(len(points)):
                if i == j:
                    continue
                # Point i is dominated by point j if j is better in all objectives
                if (points_array[j, 0] >= points_array[i, 0] and 
                    points_array[j, 1] >= points_array[i, 1] and
                    (points_array[j, 0] > points_array[i, 0] or 
                     points_array[j, 1] > points_array[i, 1])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_indices.append(i)
            else:
                dominated_indices.append(i)
        
        pareto_points = [points[i] for i in pareto_indices]
        dominated_points = [points[i] for i in dominated_indices]
        
    elif algorithm in ["nsga2", "nsga3"]:
        if not HAS_PYMOO:
            logger.warning(f"pymoo not available, falling back to dominance algorithm")
            return calculate_pareto_front(objective1, objective2, data, algorithm="dominance", maximize=maximize)
        
        try:
            # Use pymoo's non-dominated sorting
            # First front (rank 0) is the Pareto front
            nds = NonDominatedSorting()
            fronts = nds.do(points_array, only_non_dominated_front=True)
            
            if len(fronts) > 0:
                pareto_indices = fronts[0].tolist()
                dominated_indices = [i for i in range(len(points)) if i not in pareto_indices]
            else:
                pareto_indices = []
                dominated_indices = list(range(len(points)))
            
            pareto_points = [points[i] for i in pareto_indices]
            dominated_points = [points[i] for i in dominated_indices]
            
        except Exception as e:
            logger.error(f"Error using pymoo for Pareto front calculation: {e}")
            # Fallback to dominance
            return calculate_pareto_front(objective1, objective2, data, algorithm="dominance", maximize=maximize)
    else:
        logger.warning(f"Unknown algorithm: {algorithm}, using dominance")
        return calculate_pareto_front(objective1, objective2, data, algorithm="dominance", maximize=maximize)
    
    return pareto_points, dominated_points

def _ensure_numpy():
    """Ensure numpy is available."""
    try:
        import numpy as np
        return np
    except ImportError:
        raise ImportError("numpy is required for Pareto front calculation")
