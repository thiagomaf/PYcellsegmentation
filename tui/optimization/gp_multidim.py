import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

def fit_multidim_gp_for_objective(
    objective_name: str,
    data: List[Tuple[Dict[str, float], float]],
    active_params: List[str],
    kernel_type: str = "rbf",
    **kernel_kwargs
) -> Optional[object]:
    """Fit a multi-dimensional Gaussian Process model for an objective.
    
    Args:
        objective_name: Name of the objective function
        data: List of (parameters_dict, objective_value) tuples
        active_params: List of parameter names to use as features
        kernel_type: Type of kernel ('rbf', 'matern', 'rational_quadratic', 'exp_sine_squared')
        **kernel_kwargs: Additional kernel parameters
        
    Returns:
        Fitted GaussianProcessRegressor model, or None if fitting fails
    """
    if not data or not active_params:
        return None
    
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.preprocessing import StandardScaler
        from tui.optimization.gp_regression import create_kernel
    except ImportError:
        logger.warning("scikit-learn not installed. Cannot perform multi-dimensional GP regression.")
        return None
    
    try:
        # Extract X (parameters) and y (objective)
        X_list = []
        y_list = []
        
        for params, value in data:
            # Extract only active parameters
            x_vec = []
            for param_name in active_params:
                if param_name in params:
                    x_vec.append(params[param_name])
                else:
                    logger.warning(f"Parameter {param_name} not found in data point, skipping")
                    break
            else:
                # All parameters found
                if len(x_vec) == len(active_params):
                    X_list.append(x_vec)
                    y_list.append(value)
        
        if len(X_list) < 2:
            logger.warning(f"Not enough data points for multi-dimensional GP (need at least 2, got {len(X_list)})")
            return None
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create kernel
        kernel = create_kernel(kernel_type, **kernel_kwargs)
        if kernel is None:
            return None
        
        # Create and fit GP
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=42
        )
        gp.fit(X_scaled, y)
        
        # Store scaler with the model for later use
        gp.scaler = scaler
        gp.active_params = active_params
        
        return gp
        
    except Exception as e:
        logger.error(f"Error fitting multi-dimensional GP for {objective_name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def project_gp_predictions(
    gp_model: object,
    param_name: str,
    fixed_params: Optional[Dict[str, float]] = None,
    param_range: Optional[Tuple[float, float]] = None,
    num_points: int = 50,
    observed_data: Optional[List[Tuple[Dict[str, float], float]]] = None
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Project multi-dimensional GP predictions onto a single parameter axis.
    
    This function creates a 1D visualization by fixing all other parameters
    and varying only the specified parameter.
    
    Args:
        gp_model: Fitted GaussianProcessRegressor (with scaler and active_params attributes)
        param_name: Name of the parameter to vary (x-axis)
        fixed_params: Dictionary of parameter values to fix. If None, uses mean values.
        param_range: Optional (min, max) range for the parameter. If None, uses observed data or reasonable defaults.
        num_points: Number of points for prediction curve
        observed_data: Optional list of (parameters_dict, objective_value) tuples to determine range from observed data
        
    Returns:
        Tuple of (x_pred, y_mean, y_std) or None if projection fails
        x_pred: 1D array of parameter values
        y_mean: 1D array of predicted mean objective values
        y_std: 1D array of predicted standard deviation
    """
    if not hasattr(gp_model, 'scaler') or not hasattr(gp_model, 'active_params'):
        logger.error("GP model missing scaler or active_params attributes")
        return None
    
    try:
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        active_params = gp_model.active_params
        scaler = gp_model.scaler
        
        if param_name not in active_params:
            logger.error(f"Parameter {param_name} not in active parameters: {active_params}")
            return None
        
        # Determine parameter range
        if param_range is None:
            # Use observed data range if available (like 1D GP does)
            if observed_data:
                x_vals = []
                for params, _ in observed_data:
                    if param_name in params:
                        x_vals.append(params[param_name])
                if x_vals:
                    x_min, x_max = min(x_vals), max(x_vals)
                    padding = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
                    x_min -= padding
                    x_max += padding
                    param_range = (x_min, x_max)
                else:
                    # Fallback to defaults if no data for this parameter
                    if param_name == "diameter":
                        param_range = (15, 100)
                    elif param_name == "min_size":
                        param_range = (5, 50)
                    elif param_name == "flow_threshold":
                        param_range = (0.0, 1.0)
                    elif param_name == "cellprob_threshold":
                        param_range = (-6.0, 6.0)
                    else:
                        param_range = (0.0, 100.0)
            else:
                # Use reasonable defaults based on parameter name
                if param_name == "diameter":
                    param_range = (15, 100)
                elif param_name == "min_size":
                    param_range = (5, 50)
                elif param_name == "flow_threshold":
                    param_range = (0.0, 1.0)
                elif param_name == "cellprob_threshold":
                    param_range = (-6.0, 6.0)
                else:
                    param_range = (0.0, 100.0)
        
        x_min, x_max = param_range
        
        # Create fixed parameter values
        if fixed_params is None:
            fixed_params = {}
        
        # Build prediction points
        X_pred_list = []
        x_pred_values = np.linspace(x_min, x_max, num_points)
        
        for x_val in x_pred_values:
            x_vec = []
            for pname in active_params:
                if pname == param_name:
                    x_vec.append(x_val)
                elif pname in fixed_params:
                    x_vec.append(fixed_params[pname])
                else:
                    # Use mean value from training data (approximate)
                    # This is a simplification - ideally we'd store training data statistics
                    if pname == "diameter":
                        x_vec.append(30.0)  # Default
                    elif pname == "min_size":
                        x_vec.append(15.0)  # Default
                    elif pname == "flow_threshold":
                        x_vec.append(0.4)  # Default
                    elif pname == "cellprob_threshold":
                        x_vec.append(0.0)  # Default
                    else:
                        x_vec.append(0.0)  # Fallback
            
            X_pred_list.append(x_vec)
        
        X_pred = np.array(X_pred_list)
        
        # Scale features
        X_pred_scaled = scaler.transform(X_pred)
        
        # Predict
        y_mean, y_std = gp_model.predict(X_pred_scaled, return_std=True)
        
        return x_pred_values, y_mean, y_std
        
    except Exception as e:
        logger.error(f"Error projecting GP predictions for {param_name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def get_active_parameters(project) -> List[str]:
    """Get list of active (optimizable) parameters from project.
    
    Args:
        project: OptimizationProject instance
        
    Returns:
        List of parameter names that are active for optimization
    """
    active_params = []
    ranges = project.parameter_ranges
    
    if ranges.optimize_diameter:
        active_params.append("diameter")
    if ranges.optimize_min_size:
        active_params.append("min_size")
    if ranges.optimize_flow_threshold:
        active_params.append("flow_threshold")
    if ranges.optimize_cellprob_threshold:
        active_params.append("cellprob_threshold")
    
    # Always include at least diameter for visualization
    if not active_params:
        active_params = ["diameter"]
    
    return active_params


