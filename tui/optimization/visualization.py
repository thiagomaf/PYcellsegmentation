"""Visualization utilities for parameter space coverage."""
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
# Lazy import numpy - only import when functions are actually called
# import numpy as np

# Lazy import sklearn and pyDOE3 - only check when needed
HAS_SKLEARN = None  # Will be set lazily
HAS_PYDOE = None  # Will be set lazily

logger = logging.getLogger(__name__)

from tui.optimization.models import ParameterRanges, OptimizationProject, SuggestedParameterPoint
from tui.models import ProjectConfig, ParameterConfiguration

def _lazy_import_numpy():
    """Lazy import numpy - only import when actually needed."""
    import numpy as np
    return np

def _lazy_check_sklearn():
    """Lazy check for sklearn - only check when actually needed."""
    global HAS_SKLEARN
    if HAS_SKLEARN is None:
        try:
            from sklearn.decomposition import PCA
            HAS_SKLEARN = True
        except ImportError:
            HAS_SKLEARN = False
    return HAS_SKLEARN

def _lazy_check_pydoe():
    """Lazy check for pyDOE3 - only check when actually needed."""
    global HAS_PYDOE
    if HAS_PYDOE is None:
        try:
            from pyDOE3 import lhs
            HAS_PYDOE = True
        except ImportError:
            HAS_PYDOE = False
    return HAS_PYDOE


def generate_grid_samples(
    parameter_ranges: ParameterRanges, 
    num_samples: int,
    existing_points: Optional[List[Dict[str, float]]] = None
) -> List[Dict[str, float]]:
    """Generate evenly spaced grid samples for the parameter space.
    
    This creates a deterministic, evenly distributed grid of points across
    the parameter space. The same input will always produce the same output.
    
    If existing_points are provided, the function will select new points that
    maximize coverage by avoiding regions already covered by existing points.
    
    Args:
        parameter_ranges: The parameter ranges defining the search space
        num_samples: Number of samples to generate (will be adjusted to fit grid)
        existing_points: Optional list of existing parameter dictionaries to avoid
        
    Returns:
        List of parameter dictionaries
    """
    # Determine which parameters are active
    active_params = []
    
    if parameter_ranges.optimize_diameter:
        active_params.append(('diameter', parameter_ranges.diameter_min, parameter_ranges.diameter_max))
    if parameter_ranges.optimize_min_size:
        active_params.append(('min_size', parameter_ranges.min_size_min, parameter_ranges.min_size_max))
    if parameter_ranges.optimize_flow_threshold:
        active_params.append(('flow_threshold', parameter_ranges.flow_threshold_min, parameter_ranges.flow_threshold_max))
    if parameter_ranges.optimize_cellprob_threshold:
        active_params.append(('cellprob_threshold', parameter_ranges.cellprob_threshold_min, parameter_ranges.cellprob_threshold_max))
    
    if len(active_params) == 0:
        return []
    
    # Lazy import numpy
    np = _lazy_import_numpy()
    
    num_dims = len(active_params)
    
    # Calculate points per dimension for a roughly uniform grid
    # For N dimensions and M samples, we want roughly M^(1/N) points per dimension
    # But we need to ensure we get at least M unique combinations
    if num_dims == 1:
        points_per_dim = [num_samples]
    else:
        # For multiple dimensions, calculate points per dimension
        # Strategy: ensure we have enough points per dimension for good coverage
        # Minimum 3 points per dimension (unless num_samples is very small)
        # This ensures we get more than just min/max values
        base_points = max(3, int(np.ceil(num_samples ** (1.0 / num_dims))))
        
        # Calculate total combinations
        total_combinations = base_points ** num_dims
        
        # If we have fewer combinations than needed, increase points
        if total_combinations < num_samples:
            points_per_dim = [base_points] * num_dims
            # Increase points until we have enough combinations
            while np.prod(points_per_dim) < num_samples:
                # Round-robin: add a point to each dimension in turn
                for i in range(num_dims):
                    if np.prod(points_per_dim) >= num_samples:
                        break
                    points_per_dim[i] += 1
        else:
            # We have enough or more than enough, use base_points
            points_per_dim = [base_points] * num_dims
    
    # Generate grid using numpy meshgrid
    # Create linspace for each dimension
    # Note: np.linspace always includes endpoints (min and max), which is optimal
    # When n_points=2, it returns [min, max] - the best coverage for 2 values
    grids = []
    for i, (param_name, param_min, param_max) in enumerate(active_params):
        n_points = points_per_dim[i]
        if param_name in ['diameter', 'min_size']:
            # Integer parameters: use integer steps
            if param_max == param_min:
                values = [int(param_min)]
            else:
                # np.linspace with dtype=int ensures we get integer values
                # When n_points=2, this gives [min, max] - optimal for 2 values
                values = np.linspace(param_min, param_max, n_points, dtype=int).tolist()
        else:
            # Float parameters: use float steps
            # When n_points=2, this gives [min, max] - optimal for 2 values
            values = np.linspace(param_min, param_max, n_points).tolist()
        grids.append(values)
    
    # Create meshgrid and flatten
    if num_dims == 1:
        # Single dimension: just use the values directly
        samples = []
        for val in grids[0]:
            params = {}
            param_name, _, _ = active_params[0]
            params[param_name] = val
            samples.append(params)
    else:
        # Multiple dimensions: create meshgrid
        mesh = np.meshgrid(*grids, indexing='ij')
        samples = []
        for indices in np.ndindex(*points_per_dim):
            params = {}
            for i, (param_name, _, _) in enumerate(active_params):
                params[param_name] = mesh[i][indices]
                # Convert numpy types to Python types
                if isinstance(params[param_name], (np.integer, np.floating)):
                    if param_name in ['diameter', 'min_size']:
                        params[param_name] = int(params[param_name])
                    else:
                        params[param_name] = float(params[param_name])
            samples.append(params)
    
    # If we have existing points, select new points that maximize coverage
    if existing_points and len(existing_points) > 0:
        return _select_coverage_maximizing_points(
            candidate_samples=samples,
            existing_points=existing_points,
            parameter_ranges=parameter_ranges,
            num_samples=num_samples
        )
    
    # Limit to requested number of samples (in case grid is larger)
    return samples[:num_samples]


def _select_coverage_maximizing_points(
    candidate_samples: List[Dict[str, float]],
    existing_points: List[Dict[str, float]],
    parameter_ranges: ParameterRanges,
    num_samples: int
) -> List[Dict[str, float]]:
    """Select points from candidates that maximize coverage (minimize overlap with existing).
    
    Uses a spatial binning approach combined with distance maximization to ensure
    even coverage across the entire parameter space. The algorithm:
    1. Divides the parameter space into bins/regions
    2. Counts existing points in each bin
    3. Prefers selecting candidates from under-sampled bins
    4. Within each bin, selects points that maximize distance from existing points
    
    Args:
        candidate_samples: List of candidate parameter dictionaries
        existing_points: List of existing parameter dictionaries to avoid
        parameter_ranges: Parameter ranges for normalization
        num_samples: Number of points to select
        
    Returns:
        List of selected parameter dictionaries
    """
    # Lazy import numpy
    np = _lazy_import_numpy()
    
    if not candidate_samples:
        return []
    
    # Determine active parameters for normalization
    active_params = []
    if parameter_ranges.optimize_diameter:
        active_params.append(('diameter', parameter_ranges.diameter_min, parameter_ranges.diameter_max))
    if parameter_ranges.optimize_min_size:
        active_params.append(('min_size', parameter_ranges.min_size_min, parameter_ranges.min_size_max))
    if parameter_ranges.optimize_cellprob_threshold:
        active_params.append(('cellprob_threshold', parameter_ranges.cellprob_threshold_min, parameter_ranges.cellprob_threshold_max))
    
    if not active_params:
        return candidate_samples[:num_samples]
    
    num_dims = len(active_params)
    
    # Normalize all points to [0, 1] for distance calculation
    def normalize_point(params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dict to normalized numpy array."""
        point = []
        for param_name, param_min, param_max in active_params:
            val = params.get(param_name, param_min)
            # Normalize to [0, 1]
            if param_max == param_min:
                norm_val = 0.0
            else:
                norm_val = (val - param_min) / (param_max - param_min)
            point.append(norm_val)
        return np.array(point)
    
    # Normalize existing points
    existing_normalized = [normalize_point(p) for p in existing_points]
    
    # Normalize candidate points
    candidate_normalized = [normalize_point(p) for p in candidate_samples]
    
    # Remove candidates that are too close to existing points (within a small threshold)
    threshold = 0.01  # 1% of normalized space
    filtered_candidates = []
    filtered_candidate_normalized = []
    
    for cand, cand_norm in zip(candidate_samples, candidate_normalized):
        # Check minimum distance to any existing point
        if existing_normalized:
            min_dist = min(np.linalg.norm(cand_norm - exist_norm) for exist_norm in existing_normalized)
            if min_dist < threshold:
                continue  # Too close to existing point, skip
        filtered_candidates.append(cand)
        filtered_candidate_normalized.append(cand_norm)
    
    if not filtered_candidates:
        # All candidates too close to existing, return original candidates
        return candidate_samples[:num_samples]
    
    # Divide parameter space into bins for spatial coverage
    # Use roughly sqrt(num_samples) bins per dimension for good coverage
    bins_per_dim = max(3, int(np.ceil(np.sqrt(num_samples))))
    
    def get_bin_index(point: np.ndarray) -> tuple:
        """Get the bin index for a normalized point."""
        bin_indices = []
        for coord in point:
            # Clamp to [0, 1] and convert to bin index
            coord = max(0.0, min(1.0, coord))
            bin_idx = int(coord * bins_per_dim)
            # Handle edge case: if coord == 1.0, put it in the last bin
            bin_idx = min(bin_idx, bins_per_dim - 1)
            bin_indices.append(bin_idx)
        return tuple(bin_indices)
    
    # Count existing points in each bin
    bin_counts = {}
    for point in existing_normalized:
        bin_idx = get_bin_index(point)
        bin_counts[bin_idx] = bin_counts.get(bin_idx, 0) + 1
    
    # Group candidates by bin
    candidates_by_bin = {}
    for cand, cand_norm in zip(filtered_candidates, filtered_candidate_normalized):
        bin_idx = get_bin_index(cand_norm)
        if bin_idx not in candidates_by_bin:
            candidates_by_bin[bin_idx] = []
        candidates_by_bin[bin_idx].append((cand, cand_norm))
    
    # Calculate target points per bin for even distribution
    # We want to distribute num_samples across bins, preferring under-sampled bins
    target_per_bin = num_samples / max(1, len(candidates_by_bin))
    
    # Sort bins by how under-sampled they are (fewer existing points = higher priority)
    # Deterministic sorting: sort by bin index first, then by count
    sorted_bins = sorted(candidates_by_bin.keys(), key=lambda b: (bin_counts.get(b, 0), b))
    
    selected = []
    selected_normalized = []
    remaining_samples = num_samples
    
    # First pass: distribute points across bins, prioritizing under-sampled bins
    for bin_idx in sorted_bins:
        if remaining_samples <= 0:
            break
        
        candidates_in_bin = candidates_by_bin[bin_idx]
        existing_in_bin = bin_counts.get(bin_idx, 0)
        
        # Calculate how many points to select from this bin
        # Prefer bins with fewer existing points
        priority = 1.0 / (1.0 + existing_in_bin)  # Higher priority for fewer existing points
        points_from_bin = max(1, int(np.ceil(target_per_bin * priority)))
        points_from_bin = min(points_from_bin, len(candidates_in_bin), remaining_samples)
        
        if points_from_bin <= 0:
            continue
        
        # Within this bin, select points that maximize distance from existing points
        # Sort candidates deterministically first
        def sort_key(item):
            cand, _ = item
            return tuple(cand.get(param_name, 0) for param_name, _, _ in active_params)
        
        candidates_in_bin.sort(key=sort_key)
        
        # Select points from this bin
        bin_selected = []
        bin_selected_normalized = []
        
        for cand, cand_norm in candidates_in_bin:
            if len(bin_selected) >= points_from_bin:
                break
            
            # Calculate minimum distance to existing and already-selected points
            all_known_points = existing_normalized + selected_normalized + bin_selected_normalized
            if all_known_points:
                min_dist = min(np.linalg.norm(cand_norm - known_norm) for known_norm in all_known_points)
            else:
                min_dist = float('inf')
            
            # Add to selection
            bin_selected.append(cand)
            bin_selected_normalized.append(cand_norm)
        
        selected.extend(bin_selected)
        selected_normalized.extend(bin_selected_normalized)
        remaining_samples -= len(bin_selected)
    
    # If we still need more points, fill from remaining candidates
    if remaining_samples > 0:
        # Collect all remaining candidates not yet selected
        remaining_candidates = []
        for bin_idx, candidates in candidates_by_bin.items():
            for cand, cand_norm in candidates:
                if cand not in selected:
                    remaining_candidates.append((cand, cand_norm))
        
        # Sort deterministically
        def sort_key(item):
            cand, _ = item
            return tuple(cand.get(param_name, 0) for param_name, _, _ in active_params)
        
        remaining_candidates.sort(key=sort_key)
        
        # Select remaining points using distance maximization
        for _ in range(min(remaining_samples, len(remaining_candidates))):
            if not remaining_candidates:
                break
            
            best_candidate = None
            best_candidate_norm = None
            best_min_distance = -1
            best_idx = -1
            
            for idx, (cand, cand_norm) in enumerate(remaining_candidates):
                all_known_points = existing_normalized + selected_normalized
                if all_known_points:
                    min_dist = min(np.linalg.norm(cand_norm - known_norm) for known_norm in all_known_points)
                else:
                    min_dist = float('inf')
                
                if min_dist > best_min_distance:
                    best_min_distance = min_dist
                    best_candidate = cand
                    best_candidate_norm = cand_norm
                    best_idx = idx
            
            if best_candidate is not None:
                selected.append(best_candidate)
                selected_normalized.append(best_candidate_norm)
                remaining_candidates.pop(best_idx)
    
    return selected


def generate_lhs_samples(parameter_ranges: ParameterRanges, num_samples: int) -> List[Dict[str, float]]:
    """Generate Latin Hypercube samples for the parameter space.
    
    Note: LHS is stochastic - different points each time. For evenly spaced
    deterministic points, use generate_grid_samples() instead.
    
    Args:
        parameter_ranges: The parameter ranges defining the search space
        num_samples: Number of samples to generate
        
    Returns:
        List of parameter dictionaries
    """
    # Try to import lhs from pyDOE3
    try:
        from pyDOE3 import lhs
    except ImportError:
        # Fallback to random sampling if pyDOE3 is not available
        import random
        samples = []
        for _ in range(num_samples):
            params = {}
            if parameter_ranges.optimize_diameter:
                params['diameter'] = random.randint(parameter_ranges.diameter_min, parameter_ranges.diameter_max)
            if parameter_ranges.optimize_min_size:
                params['min_size'] = random.randint(parameter_ranges.min_size_min, parameter_ranges.min_size_max)
            if parameter_ranges.optimize_flow_threshold:
                params['flow_threshold'] = random.uniform(parameter_ranges.flow_threshold_min, parameter_ranges.flow_threshold_max)
            if parameter_ranges.optimize_cellprob_threshold:
                params['cellprob_threshold'] = random.uniform(parameter_ranges.cellprob_threshold_min, parameter_ranges.cellprob_threshold_max)
            samples.append(params)
        return samples
    
    # Determine which parameters are active
    active_params = []
    param_names = []
    param_mins = []
    param_maxs = []
    
    if parameter_ranges.optimize_diameter:
        active_params.append(('diameter', parameter_ranges.diameter_min, parameter_ranges.diameter_max))
        param_names.append('diameter')
        param_mins.append(parameter_ranges.diameter_min)
        param_maxs.append(parameter_ranges.diameter_max)
    if parameter_ranges.optimize_min_size:
        active_params.append(('min_size', parameter_ranges.min_size_min, parameter_ranges.min_size_max))
        param_names.append('min_size')
        param_mins.append(parameter_ranges.min_size_min)
        param_maxs.append(parameter_ranges.min_size_max)
    if parameter_ranges.optimize_flow_threshold:
        active_params.append(('flow_threshold', parameter_ranges.flow_threshold_min, parameter_ranges.flow_threshold_max))
        param_names.append('flow_threshold')
        param_mins.append(parameter_ranges.flow_threshold_min)
        param_maxs.append(parameter_ranges.flow_threshold_max)
    if parameter_ranges.optimize_cellprob_threshold:
        active_params.append(('cellprob_threshold', parameter_ranges.cellprob_threshold_min, parameter_ranges.cellprob_threshold_max))
        param_names.append('cellprob_threshold')
        param_mins.append(parameter_ranges.cellprob_threshold_min)
        param_maxs.append(parameter_ranges.cellprob_threshold_max)
    
    if len(active_params) == 0:
        return []
    
    # Generate LHS samples in [0, 1] space
    num_dims = len(active_params)
    lhs_samples = lhs(num_dims, samples=num_samples, criterion='maximin')
    
    # Scale to actual parameter ranges
    samples = []
    for sample in lhs_samples:
        params = {}
        for i, (param_name, param_min, param_max) in enumerate(active_params):
            # Scale from [0, 1] to [param_min, param_max]
            if param_name in ['diameter', 'min_size']:
                # Integer parameters
                value = int(param_min + sample[i] * (param_max - param_min))
            else:
                # Float parameters
                value = param_min + sample[i] * (param_max - param_min)
            params[param_name] = value
        samples.append(params)
    
    return samples


def setup_parameter_coverage_plot(
    plt,  # plotext API object from PlotextPlot.plt
    parameter_ranges: ParameterRanges,
    project: Optional[OptimizationProject] = None,
    config_filepaths: Optional[List[str]] = None
) -> None:
    """Set up a plotext plot for parameter space coverage visualization.
    
    Args:
        plt: The plotext API object from PlotextPlot.plt
        parameter_ranges: The parameter ranges defining the search space
        project: Optional optimization project to extract config files from
        config_filepaths: Optional list of config file paths to visualize
    """
    # Lazy import numpy - only import when actually needed
    try:
        np = _lazy_import_numpy()
    except ImportError:
        plt.title("numpy is required for visualization.\nInstall with: pip install numpy")
        return
    
    try:
        # Clear any existing plot data
        # Important: clear both data and figure to reset state
        plt.clear_data()
        plt.clear_figure()
        # Also clear any color/marker settings that might persist
        plt.clear_color()
        # #region agent log
        try:
            import json, time
            with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1","location":"visualization.py:setup_parameter_coverage_plot","message":"Cleared plot in setup function","data":{},"timestamp":time.time()*1000})+"\n")
        except: pass
        # #endregion
    except Exception as e:
        # #region agent log
        try:
            with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                import json, traceback
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"DIV2","location":"visualization.py:clear","message":"error clearing plot","data":{"error":str(e),"traceback":traceback.format_exc()},"timestamp":__import__("time").time()*1000})+"\n")
        except: pass
        # #endregion
        # Don't set plotsize here - let the widget handle it
        plt.title(f"Error initializing plot: {e}")
        return
    
    try:
        # Determine which parameters are being optimized
        active_params = []
        if parameter_ranges.optimize_diameter:
            active_params.append(('diameter', parameter_ranges.diameter_min, parameter_ranges.diameter_max))
        if parameter_ranges.optimize_min_size:
            active_params.append(('min_size', parameter_ranges.min_size_min, parameter_ranges.min_size_max))
        if parameter_ranges.optimize_flow_threshold:
            active_params.append(('flow_threshold', parameter_ranges.flow_threshold_min, parameter_ranges.flow_threshold_max))
        if parameter_ranges.optimize_cellprob_threshold:
            active_params.append(('cellprob_threshold', parameter_ranges.cellprob_threshold_min, parameter_ranges.cellprob_threshold_max))
        
        if len(active_params) < 2:
            # #region agent log
            try:
                import json, time
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H2","location":"visualization.py:setup_parameter_coverage_plot","message":"EARLY RETURN - less than 2 params","data":{"active_params_count":len(active_params)},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            # Don't set plotsize here - let the widget handle it
            plt.clear_data()
            plt.clear_figure()
            plt.title("Need at least 2 active parameters\nto visualize coverage")
            plt.xlabel("Enable at least 2 parameters\nin the checkboxes above")
            plt.ylabel("")
            # Add visible axes by setting limits and adding a grid
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.grid(True)
            # Add a visible point to ensure plot builds something
            plt.scatter([0], [0], marker='+', color='blue', label='')
            return
        
        # Collect config files
        config_files = []
        if project:
            config_files = [cf.filepath for cf in project.config_files if cf.included]
        elif config_filepaths:
            config_files = config_filepaths
        
        if not config_files:
            # #region agent log
            try:
                import json, time
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H2","location":"visualization.py:setup_parameter_coverage_plot","message":"EARLY RETURN - no config files","data":{"has_project":project is not None},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            # Don't set plotsize here - let the widget handle it
            plt.clear_data()
            plt.clear_figure()
            plt.title("No config files to visualize")
            plt.xlabel("Add config files to the project\nin the project dashboard")
            plt.ylabel("")
            # Add visible axes by setting limits and adding a grid
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.grid(True)
            # Add a visible point to ensure plot builds something
            plt.scatter([0], [0], marker='+', color='blue', label='')
            return
        
        # Collect all parameter values from all config files
        all_data_points = []  # List of lists: each inner list is [p1, p2, p3, ...] for all active params
        labels = []
        colors_list = []
        
        # Define colors/markers for different config files
        # Use a larger palette to ensure unique colors for each config
        # Plotext supports: 'red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'white', 'black', 'gray', 'orange'
        # And their bright variants: 'red+', 'blue+', 'green+', 'yellow+', 'magenta+', 'cyan+', 'white+', 'gray+', 'orange+'
        markers = ['+', 'x', '*', 'o', 'dot', 'braille', 'hd', 'sd']
        # Use plotext color names - note: plotext uses '+' suffix for bright colors, not 'bright_'
        color_names = ['blue', 'red', 'green', 'yellow', 'magenta', 'cyan', 'white', 'orange',
                      'blue+', 'red+', 'green+', 'yellow+', 'magenta+', 'cyan+', 'white+', 'orange+']
        
        # Create a mapping from config name to color/marker to ensure consistency
        config_color_map = {}
        config_marker_map = {}
        
        for idx, config_filepath in enumerate(config_files):
            params_list = extract_parameters_from_config(config_filepath)
            if not params_list:
                continue
            
            config_name = Path(config_filepath).stem
            
            # Assign unique color and marker to each config (only once per config name)
            if config_name not in config_color_map:
                config_color_map[config_name] = color_names[idx % len(color_names)]
                config_marker_map[config_name] = markers[idx % len(markers)]
            
            marker = config_marker_map[config_name]
            color = config_color_map[config_name]
            
            for params in params_list:
                # Collect all active parameter values in order
                point = []
                for param_name, _, _ in active_params:
                    val = params.get(param_name, 0)
                    point.append(val)
                
                all_data_points.append(point)
                labels.append(config_name)
                colors_list.append(color)
        
        if not all_data_points:
            # #region agent log
            try:
                import json, time
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H2","location":"visualization.py:setup_parameter_coverage_plot","message":"EARLY RETURN - no data points","data":{"config_files_count":len(config_files)},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            # Don't set plotsize here - let the widget handle it
            plt.clear_data()
            plt.clear_figure()
            plt.title("No parameter data found in config files")
            plt.xlabel("Config files may not contain\nactive parameter sets")
            plt.ylabel("")
            # Add visible axes by setting limits and adding a grid
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.grid(True)
            # Add a visible point to ensure plot builds something
            plt.scatter([0], [0], marker='+', color='blue', label='')
            return
    
        # Convert to numpy array
        data_matrix = np.array(all_data_points)
        
        # Determine visualization approach
        if len(active_params) == 2:
            # Simple 2D: use first two parameters directly
            param1_name, param1_min, param1_max = active_params[0]
            param2_name, param2_min, param2_max = active_params[1]
            
            all_x = data_matrix[:, 0].tolist()
            all_y = data_matrix[:, 1].tolist()
            
            x_min, x_max = param1_min, param1_max
            y_min, y_max = param2_min, param2_max
            x_label = f"{param1_name.replace('_', ' ').title()}"
            y_label = f"{param2_name.replace('_', ' ').title()}"
            title = f"Parameter Space: {param1_name.replace('_', ' ').title()} vs {param2_name.replace('_', ' ').title()}"
            
        else:
            # More than 2 parameters: use PCA for dimension reduction
            if not HAS_SKLEARN:
                # Don't set plotsize here - let the widget handle it
                plt.title("Need sklearn for multi-parameter visualization.\nInstall with: pip install scikit-learn")
                return
            
            # Normalize parameters to [0, 1] for PCA (since they have different scales)
            normalized_data = []
            for i, (param_name, param_min, param_max) in enumerate(active_params):
                col = data_matrix[:, i]
                # Normalize array of values
                # #region agent log
                try:
                    with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                        import json
                        param_range = float(param_max - param_min) if param_max != param_min else 0.0
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"DIV1","location":"visualization.py:normalize","message":"normalizing parameter","data":{"param_name":param_name,"param_min":float(param_min),"param_max":float(param_max),"col_min":float(np.min(col)),"col_max":float(np.max(col)),"range":param_range,"will_divide":param_max != param_min},"timestamp":__import__("time").time()*1000})+"\n")
                except: pass
                # #endregion
                if param_max == param_min or abs(param_max - param_min) < 1e-10:
                    normalized_col = np.full_like(col, 0.5, dtype=float)
                else:
                    param_range = param_max - param_min
                    normalized_col = (col - param_min) / param_range
                normalized_data.append(normalized_col)
            normalized_matrix = np.column_stack(normalized_data)
            
            # Apply PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(normalized_matrix)
            
            all_x = pca_result[:, 0].tolist()
            all_y = pca_result[:, 1].tolist()
            
            # Get axis ranges from PCA results
            x_min, x_max = float(np.min(pca_result[:, 0])), float(np.max(pca_result[:, 0]))
            y_min, y_max = float(np.min(pca_result[:, 1])), float(np.max(pca_result[:, 1]))
            
            # Add padding (handle case where all points are the same)
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            if x_range == 0:
                x_min -= 0.1
                x_max += 0.1
            else:
                x_min -= x_range * 0.1
                x_max += x_range * 0.1
            
            if y_range == 0:
                y_min -= 0.1
                y_max += 0.1
            else:
                y_min -= y_range * 0.1
                y_max += y_range * 0.1
            
            # Create axis labels showing which parameters contribute most
            # Find parameters with highest absolute loadings for each PC
            pc1_loadings = np.abs(pca.components_[0])
            pc2_loadings = np.abs(pca.components_[1])
            
            pc1_idx = np.argmax(pc1_loadings)
            pc2_idx = np.argmax(pc2_loadings)
            
            pc1_param = active_params[pc1_idx][0].replace('_', ' ').title()
            pc2_param = active_params[pc2_idx][0].replace('_', ' ').title()
            
            # Show variance explained
            var1 = pca.explained_variance_ratio_[0] * 100
            var2 = pca.explained_variance_ratio_[1] * 100
            
            x_label = f"PC1 ({pc1_param}, {var1:.1f}% var)"
            y_label = f"PC2 ({pc2_param}, {var2:.1f}% var)"
            
            param_names = [p[0].replace('_', ' ').title() for p in active_params]
            title = f"Parameter Space (PCA): {', '.join(param_names)}"
        
        # Set up the plot (common for both 2D and PCA cases)
        # Note: plot size should be set by the widget based on its dimensions
        # Don't override it here - the widget will handle sizing dynamically
        plt.title(title)
        
        # Set axis labels
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        # Set axis limits (ensure they're not equal to avoid division by zero in plotext)
        if x_max == x_min:
            x_min -= 0.1
            x_max += 0.1
        if y_max == y_min:
            y_min -= 0.1
            y_max += 0.1
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        # Plot points grouped by config file (solid symbols)
        # Use the same color/marker mapping as when collecting data
        plotted_configs = set()
        config_index = 0  # Track index for color assignment
        for idx, config_filepath in enumerate(config_files):
            config_name = Path(config_filepath).stem
            if config_name in plotted_configs:
                continue
            
            # Collect points for this config
            config_x = []
            config_y = []
            for i, label in enumerate(labels):
                if label == config_name:
                    config_x.append(all_x[i])
                    config_y.append(all_y[i])
            
            if config_x:
                # Use the same color/marker mapping as when collecting data
                # If not in map, assign based on current index
                if config_name in config_color_map:
                    color = config_color_map[config_name]
                    marker = config_marker_map[config_name]
                else:
                    # Fallback: assign based on current plotting index
                    color = color_names[config_index % len(color_names)]
                    marker = markers[config_index % len(markers)]
                    config_index += 1
                
                # Plotext scatter takes x and y as separate arguments
                plt.scatter(config_x, config_y, marker=marker, color=color, label=config_name)
                plotted_configs.add(config_name)
        
        # Plot suggested points (open/hollow symbols) if project has them
        if project and project.suggested_points:
            # Extract suggested points for active parameters
            suggested_data_points = []
            for suggested in project.suggested_points:
                point = []
                for param_name, _, _ in active_params:
                    val = suggested.parameters.get(param_name, 0)
                    point.append(val)
                suggested_data_points.append(point)
            
            if suggested_data_points:
                suggested_matrix = np.array(suggested_data_points)
                
                if len(active_params) == 2:
                    # Simple 2D: use first two parameters directly
                    suggested_x = suggested_matrix[:, 0].tolist()
                    suggested_y = suggested_matrix[:, 1].tolist()
                    
                    # Update axis ranges to include suggested points
                    all_x_combined = all_x + suggested_x
                    all_y_combined = all_y + suggested_y
                    x_min_new = min(all_x_combined)
                    x_max_new = max(all_x_combined)
                    y_min_new = min(all_y_combined)
                    y_max_new = max(all_y_combined)
                    
                    # Add padding
                    x_range = x_max_new - x_min_new
                    y_range = y_max_new - y_min_new
                    if x_range == 0:
                        x_min_new -= 0.1
                        x_max_new += 0.1
                    else:
                        x_min_new -= x_range * 0.1
                        x_max_new += x_range * 0.1
                    if y_range == 0:
                        y_min_new -= 0.1
                        y_max_new += 0.1
                    else:
                        y_min_new -= y_range * 0.1
                        y_max_new += y_range * 0.1
                    
                    plt.xlim(x_min_new, x_max_new)
                    plt.ylim(y_min_new, y_max_new)
                else:
                    # More than 2 parameters: normalize and apply PCA
                    normalized_suggested = []
                    for i, (param_name, param_min, param_max) in enumerate(active_params):
                        col = suggested_matrix[:, i]
                        if param_max == param_min or abs(param_max - param_min) < 1e-10:
                            normalized_col = np.full_like(col, 0.5, dtype=float)
                        else:
                            param_range = param_max - param_min
                            normalized_col = (col - param_min) / param_range
                        normalized_suggested.append(normalized_col)
                    normalized_suggested_matrix = np.column_stack(normalized_suggested)
                    
                    # Use the same PCA transform as the config points
                    # We need to refit PCA on combined data or use the same transform
                    # For simplicity, we'll refit on combined normalized data
                    all_normalized = np.vstack([normalized_matrix, normalized_suggested_matrix])
                    pca_combined = PCA(n_components=2)
                    pca_combined_result = pca_combined.fit_transform(all_normalized)
                    suggested_pca = pca_combined_result[len(normalized_matrix):]
                    suggested_x = suggested_pca[:, 0].tolist()
                    suggested_y = suggested_pca[:, 1].tolist()
                    
                    # Update axis ranges to include suggested points
                    all_x_combined = all_x + suggested_x
                    all_y_combined = all_y + suggested_y
                    x_min_new = min(all_x_combined)
                    x_max_new = max(all_x_combined)
                    y_min_new = min(all_y_combined)
                    y_max_new = max(all_y_combined)
                    
                    # Add padding
                    x_range = x_max_new - x_min_new
                    y_range = y_max_new - y_min_new
                    if x_range == 0:
                        x_min_new -= 0.1
                        x_max_new += 0.1
                    else:
                        x_min_new -= x_range * 0.1
                        x_max_new += x_range * 0.1
                    if y_range == 0:
                        y_min_new -= 0.1
                        y_max_new += 0.1
                    else:
                        y_min_new -= y_range * 0.1
                        y_max_new += y_range * 0.1
                    
                    plt.xlim(x_min_new, x_max_new)
                    plt.ylim(y_min_new, y_max_new)
                
                # Plot suggested points with a distinct transient symbol (use 'dot' for small transient appearance)
                # This indicates they are temporary/transient until converted to configs
                # Use plotext color format: 'yellow+' for bright yellow
                plt.scatter(suggested_x, suggested_y, marker='dot', color='yellow+', label='Suggested (transient)')
        
        # Add grid
        plt.grid(True)
        
        # #region agent log
        try:
            import json, time
            with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H2","location":"visualization.py:setup_parameter_coverage_plot","message":"END of setup - plot configured","data":{"all_x_count":len(all_x) if all_x else 0,"all_y_count":len(all_y) if all_y else 0,"has_title":True},"timestamp":time.time()*1000})+"\n")
        except: pass
        # #endregion
        
        # Ensure plot is ready to be built
        # Note: PlotextPlot widget handles rendering automatically
        # But we should ensure at least something is plotted
        if not all_x or not all_y:
            plt.clear_data()
            plt.clear_figure()
            plt.title("No data points to plot")
            plt.xlabel("Check config files and parameters")
            plt.ylabel("")
            # Add visible content so plot builds
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.grid(True)
            plt.scatter([0], [0], marker='+', color='blue', label='')
    except Exception as e:
        # #region agent log
        try:
            with open(r"g:\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                import json, traceback
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"DIV3","location":"visualization.py:setup_plot","message":"error setting up plot","data":{"error":str(e),"error_type":type(e).__name__,"traceback":traceback.format_exc()},"timestamp":__import__("time").time()*1000})+"\n")
        except: pass
        # #endregion
        # Don't set plotsize here - let the widget handle it
        plt.title(f"Error creating visualization: {e}")
        return


def create_parameter_coverage_ascii(
    parameter_ranges: ParameterRanges,
    project: Optional[OptimizationProject] = None,
    config_filepaths: Optional[List[str]] = None,
    width: int = 60,
    height: int = 30
) -> str:
    """Create an ASCII art visualization of parameter space coverage.
    
    Args:
        parameter_ranges: The parameter ranges defining the search space
        project: Optional optimization project to extract config files from
        config_filepaths: Optional list of config file paths to visualize
        width: Width of ASCII canvas in characters
        height: Height of ASCII canvas in characters
        
    Returns:
        String containing ASCII art representation
    """
    # Determine which parameters are being optimized
    active_params = []
    if parameter_ranges.optimize_diameter:
        active_params.append(('diameter', parameter_ranges.diameter_min, parameter_ranges.diameter_max))
    if parameter_ranges.optimize_min_size:
        active_params.append(('min_size', parameter_ranges.min_size_min, parameter_ranges.min_size_max))
    if parameter_ranges.optimize_flow_threshold:
        active_params.append(('flow_threshold', parameter_ranges.flow_threshold_min, parameter_ranges.flow_threshold_max))
    if parameter_ranges.optimize_cellprob_threshold:
        active_params.append(('cellprob_threshold', parameter_ranges.cellprob_threshold_min, parameter_ranges.cellprob_threshold_max))
    
    if len(active_params) < 2:
        return "Need at least 2 active parameters\nto visualize coverage"
    
    # Select first two active parameters for 2D projection
    param1_name, param1_min, param1_max = active_params[0]
    param2_name, param2_min, param2_max = active_params[1]
    
    # Collect config files
    config_files = []
    if project:
        config_files = [cf.filepath for cf in project.config_files if cf.included]
    elif config_filepaths:
        config_files = config_filepaths
    
    if not config_files:
        return "No config files to visualize"
    
    # Create ASCII canvas (height x width)
    # We'll use a 2D array where each cell can hold a character
    canvas = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Define symbols for different config files
    symbols = ['*', 'o', '+', 'x', '#', '@', '%', '&', '=', '~']
    
    # Collect all points from all config files
    all_points = []
    for idx, config_filepath in enumerate(config_files):
        params_list = extract_parameters_from_config(config_filepath)
        if not params_list:
            continue
        
        symbol = symbols[idx % len(symbols)]
        
        for params in params_list:
            # Get parameter values
            p1_val = params.get(param1_name, 0)
            p2_val = params.get(param2_name, 0)
            
            # Normalize to [0, 1]
            x_norm = normalize_parameter(p1_val, param1_min, param1_max)
            y_norm = normalize_parameter(p2_val, param2_min, param2_max)
            
            # Convert to canvas coordinates
            # Note: y is inverted because ASCII goes top to bottom
            # Handle division by zero if width or height is 1
            if width <= 1:
                x_pos = 0
            else:
                x_pos = int(x_norm * (width - 1))
            
            if height <= 1:
                y_pos = 0
            else:
                y_pos = int((1.0 - y_norm) * (height - 1))
            
            # Clamp to canvas bounds
            x_pos = max(0, min(width - 1, x_pos))
            y_pos = max(0, min(height - 1, y_pos))
            
            all_points.append((x_pos, y_pos, symbol, config_filepath))
    
    # Plot points on canvas
    for x, y, symbol, _ in all_points:
        canvas[y][x] = symbol
    
    # Build the output string
    lines = []
    
    # Add title
    title = f"Parameter Space Coverage: {param1_name} vs {param2_name}"
    lines.append(title)
    lines.append("=" * max(len(title), width))
    lines.append("")
    
    # Add y-axis label on the left
    # Create border and labels
    for y in range(height):
        row = []
        # Add y-axis indicator (every 5 rows)
        if y % 5 == 0:
            if height <= 1:
                y_val = 0.5
            else:
                y_val = 1.0 - (y / (height - 1))
            y_label = f"{y_val:.1f}"
            row.append(y_label[:5].ljust(6))
        else:
            row.append(" " * 6)
        
        # Add left border
        row.append("|")
        
        # Add canvas row
        row.extend(canvas[y])
        
        # Add right border
        row.append("|")
        
        lines.append("".join(row))
    
    # Add bottom border and x-axis
    lines.append(" " * 6 + "+" + "-" * width + "+")
    
    # Add x-axis labels
    x_label_line = " " * 6 + " "
    for x in range(0, width, 10):
        if width <= 1:
            x_val = 0.5
        else:
            x_val = x / (width - 1)
        label = f"{x_val:.1f}"
        # Position label approximately
        if x < width - len(label):
            x_label_line += label.ljust(10)
    lines.append(x_label_line)
    
    # Add legend
    lines.append("")
    lines.append("Legend:")
    for idx, config_filepath in enumerate(config_files):
        symbol = symbols[idx % len(symbols)]
        config_name = Path(config_filepath).stem
        lines.append(f"  {symbol} = {config_name}")
    
    # Add axis labels
    lines.append("")
    lines.append(f"X-axis: {param1_name.replace('_', ' ').title()} (normalized)")
    lines.append(f"Y-axis: {param2_name.replace('_', ' ').title()} (normalized)")
    
    return "\n".join(lines)


def extract_parameters_from_config(config_filepath: str) -> List[Dict[str, float]]:
    """Extract parameter values from a config file.
    
    Returns a list of parameter dictionaries, one for each parameter configuration.
    """
    try:
        config = ProjectConfig.from_json_file(config_filepath)
        params_list = []
        
        for param_config in config.cellpose_parameter_configurations:
            if not param_config.is_active:
                continue
                
            cp_params = param_config.cellpose_parameters
            params = {
                'diameter': cp_params.DIAMETER if cp_params.DIAMETER is not None else 0,
                'min_size': cp_params.MIN_SIZE,
                'flow_threshold': getattr(cp_params, 'FLOW_THRESHOLD', 0.0) if hasattr(cp_params, 'FLOW_THRESHOLD') else 0.0,
                'cellprob_threshold': cp_params.CELLPROB_THRESHOLD,
            }
            params_list.append(params)
            
        return params_list
    except Exception as e:
        print(f"Error loading config {config_filepath}: {e}")
        return []


def normalize_parameter(value: float, min_val: float, max_val: float) -> float:
    """Normalize a parameter value to [0, 1] range."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)



