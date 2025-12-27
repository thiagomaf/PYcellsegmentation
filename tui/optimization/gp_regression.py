import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

def create_kernel(kernel_type: str = "rbf", **kwargs):
    """Create a GP kernel based on the specified type.
    
    Args:
        kernel_type: Type of kernel ('rbf', 'matern', 'rational_quadratic', 'exp_sine_squared')
        **kwargs: Additional kernel parameters
        
    Returns:
        Kernel object wrapped with ConstantKernel and WhiteKernel
    """
    try:
        from sklearn.gaussian_process.kernels import (
            RBF, Matern, RationalQuadratic, ExpSineSquared,
            WhiteKernel, ConstantKernel
        )
    except ImportError:
        logger.warning("scikit-learn not installed. Cannot create kernel.")
        return None
    
    # Base kernel selection
    if kernel_type == "rbf":
        base_kernel = RBF(length_scale=kwargs.get('length_scale', 1.0), 
                         length_scale_bounds=kwargs.get('length_scale_bounds', (1e-2, 1e2)))
    elif kernel_type == "matern":
        nu = kwargs.get('nu', 1.5)
        base_kernel = Matern(length_scale=kwargs.get('length_scale', 1.0),
                            nu=nu,
                            length_scale_bounds=kwargs.get('length_scale_bounds', (1e-2, 1e2)))
    elif kernel_type == "rational_quadratic":
        base_kernel = RationalQuadratic(
            length_scale=kwargs.get('length_scale', 1.0),
            alpha=kwargs.get('alpha', 1.0),
            length_scale_bounds=kwargs.get('length_scale_bounds', (1e-2, 1e2))
        )
    elif kernel_type == "exp_sine_squared":
        base_kernel = ExpSineSquared(
            length_scale=kwargs.get('length_scale', 1.0),
            periodicity=kwargs.get('periodicity', 1.0),
            length_scale_bounds=kwargs.get('length_scale_bounds', (1e-2, 1e2))
        )
    else:
        logger.warning(f"Unknown kernel type: {kernel_type}, using RBF")
        base_kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    
    # Wrap with ConstantKernel and add WhiteKernel for noise
    kernel = ConstantKernel(1.0) * base_kernel + \
             WhiteKernel(noise_level=kwargs.get('noise_level', 1e-5),
                        noise_level_bounds=kwargs.get('noise_level_bounds', (1e-10, 1e-1)))
    
    return kernel

def fit_gp_for_objective(
    objective_name: str, 
    parameter_name: str, 
    data: List[Tuple[Dict[str, float], float]],
    parameter_range: Optional[Tuple[float, float]] = None,
    num_points: int = 50,
    kernel_type: str = "rbf",
    **kernel_kwargs
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fit a Gaussian Process model for a specific parameter and objective.
    
    Args:
        objective_name: Name of the objective function
        parameter_name: Name of the parameter to vary (x-axis)
        data: List of (all_params, objective_value) tuples
        parameter_range: Optional (min, max) range for the parameter. If None, inferred from data.
        num_points: Number of points for prediction curve
        
    Returns:
        Tuple of (x_pred, y_mean, y_std, x_observed, y_observed) or None if fitting fails.
        x_pred: 1D array of parameter values for the curve
        y_mean: 1D array of predicted mean objective values
        y_std: 1D array of predicted standard deviation (uncertainty)
        x_observed: 1D array of observed parameter values
        y_observed: 1D array of observed objective values
    """
    if not data:
        return None
        
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("scikit-learn not installed. Cannot perform GP regression.")
        return None

    try:
        # Extract X (parameter of interest) and y (objective)
        # Note: In a real scenario, we should marginalized over other parameters or fix them.
        # For this visualization, we'll project all points onto this parameter axis.
        # This is a simplification but useful for 1D visualization.
        
        x_vals = []
        y_vals = []
        
        for params, value in data:
            if parameter_name in params:
                x_vals.append(params[parameter_name])
                y_vals.append(value)
                
        if len(x_vals) < 2:
            # Not enough points to fit
            return None
            
        X = np.array(x_vals).reshape(-1, 1)
        y = np.array(y_vals)
        
        # Determine range for prediction
        if parameter_range:
            x_min, x_max = parameter_range
        else:
            x_min, x_max = X.min(), X.max()
            padding = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
            x_min -= padding
            x_max += padding
            
        # Create kernel based on type
        kernel = create_kernel(kernel_type, **kernel_kwargs)
        if kernel is None:
            return None
                 
        # Create and fit GP
        # Normalize y for better fitting, though sklearn can normalize y with normalize_y=True
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=42)
        gp.fit(X, y)
        
        # Generate prediction points
        X_pred = np.linspace(x_min, x_max, num_points).reshape(-1, 1)
        y_mean, y_std = gp.predict(X_pred, return_std=True)
        
        return X_pred.flatten(), y_mean, y_std, X.flatten(), y
        
    except Exception as e:
        logger.error(f"Error fitting GP for {objective_name} vs {parameter_name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

