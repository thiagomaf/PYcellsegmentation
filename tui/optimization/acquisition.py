import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try to import scikit-optimize acquisition functions
try:
    from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    logger.debug("scikit-optimize not available. Using custom acquisition functions.")

def upper_confidence_bound(mean: np.ndarray, std: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Upper Confidence Bound acquisition function.
    
    UCB(x) = μ(x) + β * σ(x)
    
    Args:
        mean: Mean predictions from GP
        std: Standard deviation predictions from GP
        beta: Exploration-exploitation trade-off parameter
        
    Returns:
        UCB values
    """
    return mean + beta * std

def expected_improvement(mean: np.ndarray, std: np.ndarray, best_value: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition function.
    
    EI(x) = σ(x) * [Φ(Z) + Z * φ(Z)]
    where Z = (μ(x) - best - ξ) / σ(x)
    
    Args:
        mean: Mean predictions from GP
        std: Standard deviation predictions from GP
        best_value: Best observed value so far
        xi: Exploration parameter
        
    Returns:
        EI values
    """
    np = _ensure_numpy()
    std = np.maximum(std, 1e-9)
    z = (mean - best_value - xi) / std
    
    # Standard normal CDF and PDF
    from scipy.stats import norm
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    return std * (phi + z * Phi)

def probability_of_improvement(mean: np.ndarray, std: np.ndarray, best_value: float, xi: float = 0.01) -> np.ndarray:
    """Probability of Improvement acquisition function.
    
    PI(x) = Φ((μ(x) - best - ξ) / σ(x))
    
    Args:
        mean: Mean predictions from GP
        std: Standard deviation predictions from GP
        best_value: Best observed value so far
        xi: Exploration parameter
        
    Returns:
        PI values
    """
    np = _ensure_numpy()
    std = np.maximum(std, 1e-9)
    z = (mean - best_value - xi) / std
    
    from scipy.stats import norm
    return norm.cdf(z)

def _ensure_numpy():
    """Ensure numpy is available."""
    try:
        import numpy as np
        return np
    except ImportError:
        raise ImportError("numpy is required for acquisition functions")

def compute_acquisition_function(
    mean: np.ndarray,
    std: np.ndarray,
    method: str = "ucb",
    best_value: Optional[float] = None,
    beta: float = 2.0,
    xi: float = 0.01
) -> np.ndarray:
    """Compute acquisition function values.
    
    Args:
        mean: Mean predictions from GP
        std: Standard deviation predictions from GP
        method: Acquisition function method ('ucb', 'ei', 'pi', 'lcb')
        best_value: Best observed value (required for EI and PI)
        beta: Exploration parameter for UCB/LCB
        xi: Exploration parameter for EI/PI
        
    Returns:
        Acquisition function values
    """
    np = _ensure_numpy()
    std = np.maximum(std, 1e-9)
    
    # Note: scikit-optimize's acquisition functions expect a fitted GP model,
    # not raw mean/std. Since we already have mean and std from our GP,
    # we'll use our custom implementations which are more appropriate here.
    
    if method == "ucb":
        return upper_confidence_bound(mean, std, beta)
    elif method == "lcb":
        return mean - beta * std
    elif method == "ei":
        if best_value is None:
            best_value = float(np.max(mean))
        return expected_improvement(mean, std, best_value, xi)
    elif method == "pi":
        if best_value is None:
            best_value = float(np.max(mean))
        return probability_of_improvement(mean, std, best_value, xi)
    else:
        raise ValueError(f"Unknown acquisition function method: {method}")
