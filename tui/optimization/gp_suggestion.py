import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def compute_suggestion_rationale(
    params: Dict[str, float],
    project,
    method: str,
    gp_models: Optional[Dict[str, object]] = None,
    sets_with_results: Optional[List] = None
) -> str:
    """Compute rationale/explanation for a parameter suggestion.
    
    Args:
        params: Parameter dictionary
        project: OptimizationProject instance
        method: Suggestion method used ('acquisition', 'pareto', 'hybrid')
        gp_models: Optional dict of GP models (obj_name -> GP model)
        sets_with_results: Optional list of parameter sets with results
        
    Returns:
        Rationale string explaining the suggestion
    """
    rationale_parts = [f"Method: {method}"]
    
    try:
        from tui.optimization.gp_multidim import fit_multidim_gp_for_objective, get_active_parameters
        from tui.optimization.acquisition import compute_acquisition_function
        from tui.optimization.models import OptimizationResult
        
        # Get active parameters
        active_params = get_active_parameters(project)
        
        # If GP models not provided, try to fit them
        if gp_models is None:
            # Get enabled objectives
            objective_fields = OptimizationResult.get_objective_fields()
            enabled_objectives = []
            for obj_name in objective_fields:
                direction = project.objective_directions.get(obj_name, None)
                if direction is None:
                    direction = OptimizationResult.get_objective_direction(obj_name)
                if direction is not None:
                    enabled_objectives.append(obj_name)
            
            # Fit GPs for key objectives (limit to 3 for performance)
            gp_models = {}
            for obj_name in enabled_objectives[:3]:
                if sets_with_results is None:
                    data = project.get_objective_data(obj_name)
                else:
                    data = []
                    for ps in sets_with_results:
                        if all(pname in ps.parameters for pname in active_params):
                            obj_value = getattr(ps.result, obj_name, None)
                            if obj_value is not None:
                                data.append((ps.parameters, float(obj_value)))
                
                if len(data) >= 3:
                    gp_model = fit_multidim_gp_for_objective(
                        obj_name,
                        data,
                        active_params,
                        kernel_type=project.gp_kernel_type
                    )
                    if gp_model:
                        gp_models[obj_name] = gp_model
        
        # Get predictions for each objective
        if gp_models:
            # Convert params to array for prediction
            param_array = []
            for pname in active_params:
                if pname in params:
                    param_array.append(params[pname])
                else:
                    # Use default if missing
                    if pname == "diameter":
                        param_array.append(30)
                    elif pname == "min_size":
                        param_array.append(15)
                    elif pname == "flow_threshold":
                        param_array.append(0.4)
                    elif pname == "cellprob_threshold":
                        param_array.append(0.0)
            
            param_array = np.array(param_array).reshape(1, -1)
            
            # Get predictions
            for obj_name, gp_model in list(gp_models.items())[:3]:  # Limit to 3 for display
                try:
                    X_scaled = gp_model.scaler.transform(param_array)
                    y_mean, y_std = gp_model.predict(X_scaled, return_std=True)
                    rationale_parts.append(f"{obj_name}: {y_mean[0]:.3f} ± {y_std[0]:.3f}")
                except Exception as e:
                    logger.debug(f"Error predicting {obj_name}: {e}")
        
        # Distance to existing points (exploration vs exploitation)
        if sets_with_results:
            try:
                distances = []
                for ps in sets_with_results:
                    if all(pname in ps.parameters for pname in active_params):
                        # Compute normalized distance
                        dist = 0.0
                        for pname in active_params:
                            if pname in params and pname in ps.parameters:
                                # Normalize by parameter range
                                ranges = project.parameter_ranges
                                if pname == "diameter":
                                    param_range = ranges.diameter_max - ranges.diameter_min
                                    if param_range > 0:
                                        dist += ((params[pname] - ps.parameters[pname]) / param_range) ** 2
                                elif pname == "min_size":
                                    param_range = ranges.min_size_max - ranges.min_size_min
                                    if param_range > 0:
                                        dist += ((params[pname] - ps.parameters[pname]) / param_range) ** 2
                                elif pname == "flow_threshold":
                                    param_range = ranges.flow_threshold_max - ranges.flow_threshold_min
                                    if param_range > 0:
                                        dist += ((params[pname] - ps.parameters[pname]) / param_range) ** 2
                                elif pname == "cellprob_threshold":
                                    param_range = ranges.cellprob_threshold_max - ranges.cellprob_threshold_min
                                    if param_range > 0:
                                        dist += ((params[pname] - ps.parameters[pname]) / param_range) ** 2
                        if dist > 0:
                            distances.append(np.sqrt(dist))
                
                if distances:
                    min_distance = min(distances)
                    if min_distance > 0.2:  # Threshold for "far"
                        rationale_parts.append("High exploration (far from existing data)")
                    else:
                        rationale_parts.append("Exploitation (near observed points)")
            except Exception as e:
                logger.debug(f"Error computing distance: {e}")
    
    except Exception as e:
        logger.debug(f"Error computing rationale: {e}")
        rationale_parts.append("Rationale computation failed")
    
    return " | ".join(rationale_parts)

def suggest_next_parameters_gp(
    project,
    method: str = "acquisition",
    num_suggestions: int = 3
) -> List[Dict[str, float]]:
    """Suggest next parameter sets using Gaussian Process-based optimization.
    
    Args:
        project: OptimizationProject instance
        method: Suggestion method ('acquisition', 'pareto', 'hybrid')
        num_suggestions: Number of parameter sets to suggest
        
    Returns:
        List of parameter dictionaries
    """
    # Get filtered parameter sets with results
    filtered_sets = project.get_filtered_parameter_sets()
    sets_with_results = [ps for ps in filtered_sets if ps.result is not None]
    
    # Need at least 5 points for GP-based suggestions
    if len(sets_with_results) < 5:
        logger.info(f"Insufficient data ({len(sets_with_results)} points), using random sampling")
        return _random_sampling(project, num_suggestions)
    
    # Get active parameters
    from tui.optimization.gp_multidim import get_active_parameters
    active_params = get_active_parameters(project)
    
    if method == "acquisition":
        return _suggest_acquisition_based(project, sets_with_results, active_params, num_suggestions)
    elif method == "pareto":
        return _suggest_pareto_based(project, sets_with_results, active_params, num_suggestions)
    elif method == "hybrid":
        return _suggest_hybrid(project, sets_with_results, active_params, num_suggestions)
    else:
        logger.warning(f"Unknown method {method}, using acquisition")
        return _suggest_acquisition_based(project, sets_with_results, active_params, num_suggestions)

def _random_sampling(project, num_suggestions: int) -> List[Dict[str, float]]:
    """Generate random parameter samples within bounds."""
    import random
    ranges = project.parameter_ranges
    suggestions = []
    
    for _ in range(num_suggestions):
        params = {}
        if ranges.optimize_diameter:
            params['diameter'] = random.randint(ranges.diameter_min, ranges.diameter_max)
        else:
            params['diameter'] = 30
        
        if ranges.optimize_min_size:
            params['min_size'] = random.randint(ranges.min_size_min, ranges.min_size_max)
        else:
            params['min_size'] = 15
        
        if ranges.optimize_flow_threshold:
            params['flow_threshold'] = random.uniform(ranges.flow_threshold_min, ranges.flow_threshold_max)
        else:
            params['flow_threshold'] = 0.4
        
        if ranges.optimize_cellprob_threshold:
            params['cellprob_threshold'] = random.uniform(ranges.cellprob_threshold_min, ranges.cellprob_threshold_max)
        else:
            params['cellprob_threshold'] = 0.0
        
        suggestions.append(params)
    
    return suggestions

def _suggest_acquisition_based(
    project,
    sets_with_results: List,
    active_params: List[str],
    num_suggestions: int
) -> List[Dict[str, float]]:
    """Suggest parameters by maximizing acquisition function."""
    try:
        from tui.optimization.gp_multidim import fit_multidim_gp_for_objective
        from tui.optimization.acquisition import compute_acquisition_function
        from scipy.optimize import minimize
        
        # Use primary objective (mean_solidity) for acquisition-based suggestion
        # mean_solidity is a key quality metric that should be maximized
        objective_name = "mean_solidity"
        
        # Collect data
        data = []
        for ps in sets_with_results:
            if all(pname in ps.parameters for pname in active_params):
                obj_value = getattr(ps.result, objective_name, 0.0)
                data.append((ps.parameters, obj_value))
        
        if len(data) < 3:
            return _random_sampling(project, num_suggestions)
        
        # Fit multi-dimensional GP
        gp_model = fit_multidim_gp_for_objective(
            objective_name,
            data,
            active_params,
            kernel_type=project.gp_kernel_type
        )
        
        if gp_model is None:
            return _random_sampling(project, num_suggestions)
        
        # Get parameter bounds
        ranges = project.parameter_ranges
        bounds = []
        param_order = []
        for pname in active_params:
            if pname == "diameter":
                bounds.append((ranges.diameter_min, ranges.diameter_max))
                param_order.append(pname)
            elif pname == "min_size":
                bounds.append((ranges.min_size_min, ranges.min_size_max))
                param_order.append(pname)
            elif pname == "flow_threshold":
                bounds.append((ranges.flow_threshold_min, ranges.flow_threshold_max))
                param_order.append(pname)
            elif pname == "cellprob_threshold":
                bounds.append((ranges.cellprob_threshold_min, ranges.cellprob_threshold_max))
                param_order.append(pname)
        
        # Get best observed value
        best_value = max([val for _, val in data])
        
        # Objective function: negative acquisition (we minimize, so negate)
        def neg_acquisition(x):
            try:
                X_scaled = gp_model.scaler.transform(x.reshape(1, -1))
                y_mean, y_std = gp_model.predict(X_scaled, return_std=True)
                acquisition = compute_acquisition_function(
                    y_mean,
                    y_std,
                    method=project.acquisition_method,
                    best_value=best_value
                )
                return -acquisition[0]  # Negate for minimization
            except Exception:
                return 1e10  # Large penalty for invalid points
        
        # Optimize from multiple starting points
        suggestions = []
        for _ in range(num_suggestions * 3):  # Try more to get diverse suggestions
            # Random starting point
            x0 = []
            for pname in param_order:
                if pname == "diameter":
                    x0.append(np.random.uniform(ranges.diameter_min, ranges.diameter_max))
                elif pname == "min_size":
                    x0.append(np.random.uniform(ranges.min_size_min, ranges.min_size_max))
                elif pname == "flow_threshold":
                    x0.append(np.random.uniform(ranges.flow_threshold_min, ranges.flow_threshold_max))
                elif pname == "cellprob_threshold":
                    x0.append(np.random.uniform(ranges.cellprob_threshold_min, ranges.cellprob_threshold_max))
            
            x0 = np.array(x0)
            
            try:
                result = minimize(neg_acquisition, x0, bounds=bounds, method='L-BFGS-B')
                if result.success:
                    # Convert back to parameter dict
                    params = {}
                    for i, pname in enumerate(param_order):
                        if pname == "diameter":
                            params['diameter'] = int(round(result.x[i]))
                        elif pname == "min_size":
                            params['min_size'] = int(round(result.x[i]))
                        else:
                            params[pname] = float(result.x[i])
                    
                    # Add fixed parameters
                    if not ranges.optimize_diameter and 'diameter' not in params:
                        params['diameter'] = 30
                    if not ranges.optimize_min_size and 'min_size' not in params:
                        params['min_size'] = 15
                    if not ranges.optimize_flow_threshold and 'flow_threshold' not in params:
                        params['flow_threshold'] = 0.4
                    if not ranges.optimize_cellprob_threshold and 'cellprob_threshold' not in params:
                        params['cellprob_threshold'] = 0.0
                    
                    # Check if unique (simple check)
                    is_duplicate = False
                    for existing in suggestions:
                        if all(abs(existing.get(p, 0) - params.get(p, 0)) < 0.1 for p in active_params):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        suggestions.append(params)
                        if len(suggestions) >= num_suggestions:
                            break
            except Exception as e:
                logger.debug(f"Optimization failed: {e}")
                continue
        
        # If we didn't get enough, fill with random
        while len(suggestions) < num_suggestions:
            suggestions.extend(_random_sampling(project, 1))
        
        return suggestions[:num_suggestions]
        
    except ImportError:
        logger.warning("scipy not available, using random sampling")
        return _random_sampling(project, num_suggestions)
    except Exception as e:
        logger.error(f"Error in acquisition-based suggestion: {e}")
        return _random_sampling(project, num_suggestions)

def _suggest_pareto_based(
    project,
    sets_with_results: List,
    active_params: List[str],
    num_suggestions: int
) -> List[Dict[str, float]]:
    """Suggest parameters using Pareto front optimization with weighted sum scalarization."""
    try:
        from tui.optimization.gp_multidim import fit_multidim_gp_for_objective
        from tui.optimization.acquisition import compute_acquisition_function
        from tui.optimization.models import OptimizationResult
        from scipy.optimize import minimize
        
        # 1. Get objectives with defined directions (not UNDEFINED)
        objective_fields = OptimizationResult.get_objective_fields()
        enabled_objectives = []
        for obj_name in objective_fields:
            direction = project.objective_directions.get(obj_name, None)
            if direction is None:
                direction = OptimizationResult.get_objective_direction(obj_name)
            if direction is not None:  # Only include if direction is defined
                enabled_objectives.append(obj_name)
        
        if len(enabled_objectives) < 2:
            logger.info(f"Not enough objectives with defined directions ({len(enabled_objectives)}), falling back to acquisition-based")
            return _suggest_acquisition_based(project, sets_with_results, active_params, num_suggestions)
        
        # 2. Fit GP for each objective
        gp_models = {}
        objective_directions = {}
        best_values = {}
        
        for obj_name in enabled_objectives:
            # Get direction
            direction = project.objective_directions.get(obj_name, None)
            if direction is None:
                direction = OptimizationResult.get_objective_direction(obj_name)
            objective_directions[obj_name] = direction
            
            # Collect data
            data = []
            for ps in sets_with_results:
                if all(pname in ps.parameters for pname in active_params):
                    obj_value = getattr(ps.result, obj_name, None)
                    if obj_value is not None and isinstance(obj_value, (int, float)):
                        data.append((ps.parameters, float(obj_value)))
            
            if len(data) < 3:
                logger.debug(f"Objective {obj_name}: insufficient data ({len(data)} points), skipping")
                continue
            
            # Fit multi-dimensional GP
            gp_model = fit_multidim_gp_for_objective(
                obj_name,
                data,
                active_params,
                kernel_type=project.gp_kernel_type
            )
            
            if gp_model is None:
                logger.debug(f"Objective {obj_name}: GP fitting failed, skipping")
                continue
            
            gp_models[obj_name] = gp_model
            # Get best observed value for acquisition function
            values = [val for _, val in data]
            if direction is True:  # Maximize
                best_values[obj_name] = max(values)
            else:  # Minimize
                best_values[obj_name] = min(values)
        
        if len(gp_models) < 2:
            logger.info(f"Only {len(gp_models)} objectives with valid GPs, falling back to acquisition-based")
            return _suggest_acquisition_based(project, sets_with_results, active_params, num_suggestions)
        
        # 3. Generate diverse weight vectors for scalarization
        # Use Dirichlet distribution to sample uniformly from simplex
        num_objectives = len(gp_models)
        weights = _generate_diverse_weights(num_suggestions, num_objectives)
        
        # 4. Get parameter bounds
        ranges = project.parameter_ranges
        bounds = []
        param_order = []
        for pname in active_params:
            if pname == "diameter":
                bounds.append((ranges.diameter_min, ranges.diameter_max))
                param_order.append(pname)
            elif pname == "min_size":
                bounds.append((ranges.min_size_min, ranges.min_size_max))
                param_order.append(pname)
            elif pname == "flow_threshold":
                bounds.append((ranges.flow_threshold_min, ranges.flow_threshold_max))
                param_order.append(pname)
            elif pname == "cellprob_threshold":
                bounds.append((ranges.cellprob_threshold_min, ranges.cellprob_threshold_max))
                param_order.append(pname)
        
        # 5. For each weight vector, optimize scalarized acquisition
        suggestions = []
        obj_names_list = list(gp_models.keys())
        
        for weight_idx, weight_vec in enumerate(weights):
            # Objective function: negative scalarized acquisition (we minimize, so negate)
            def neg_scalarized_acquisition(x):
                try:
                    X_scaled = None
                    total_acquisition = 0.0
                    
                    for i, obj_name in enumerate(obj_names_list):
                        gp_model = gp_models[obj_name]
                        if X_scaled is None:
                            X_scaled = gp_model.scaler.transform(x.reshape(1, -1))
                        else:
                            # Reuse scaling (should be same for all GPs with same active_params)
                            X_scaled = gp_model.scaler.transform(x.reshape(1, -1))
                        
                        y_mean, y_std = gp_model.predict(X_scaled, return_std=True)
                        acquisition = compute_acquisition_function(
                            y_mean,
                            y_std,
                            method=project.acquisition_method,
                            best_value=best_values[obj_name]
                        )
                        
                        # Weight the acquisition by the weight vector
                        total_acquisition += weight_vec[i] * acquisition[0]
                    
                    return -total_acquisition  # Negate for minimization
                except Exception as e:
                    logger.debug(f"Error in scalarized acquisition: {e}")
                    return 1e10  # Large penalty for invalid points
            
            # Optimize from multiple starting points
            best_result = None
            best_acq = float('inf')
            
            for attempt in range(3):  # Try 3 different starting points
                # Random starting point
                x0 = []
                for pname in param_order:
                    if pname == "diameter":
                        x0.append(np.random.uniform(ranges.diameter_min, ranges.diameter_max))
                    elif pname == "min_size":
                        x0.append(np.random.uniform(ranges.min_size_min, ranges.min_size_max))
                    elif pname == "flow_threshold":
                        x0.append(np.random.uniform(ranges.flow_threshold_min, ranges.flow_threshold_max))
                    elif pname == "cellprob_threshold":
                        x0.append(np.random.uniform(ranges.cellprob_threshold_min, ranges.cellprob_threshold_max))
                
                x0 = np.array(x0)
                
                try:
                    result = minimize(neg_scalarized_acquisition, x0, bounds=bounds, method='L-BFGS-B')
                    if result.success:
                        acq_value = -result.fun  # Negate back to get actual acquisition
                        if acq_value > best_acq:
                            best_acq = acq_value
                            best_result = result
                except Exception as e:
                    logger.debug(f"Optimization attempt {attempt} failed: {e}")
                    continue
            
            if best_result is not None:
                # Convert back to parameter dict
                params = {}
                for i, pname in enumerate(param_order):
                    if pname == "diameter":
                        params['diameter'] = int(round(best_result.x[i]))
                    elif pname == "min_size":
                        params['min_size'] = int(round(best_result.x[i]))
                    else:
                        params[pname] = float(best_result.x[i])
                
                # Add fixed parameters
                if not ranges.optimize_diameter and 'diameter' not in params:
                    params['diameter'] = 30
                if not ranges.optimize_min_size and 'min_size' not in params:
                    params['min_size'] = 15
                if not ranges.optimize_flow_threshold and 'flow_threshold' not in params:
                    params['flow_threshold'] = 0.4
                if not ranges.optimize_cellprob_threshold and 'cellprob_threshold' not in params:
                    params['cellprob_threshold'] = 0.0
                
                # Check if unique (simple check)
                is_duplicate = False
                for existing in suggestions:
                    if all(abs(existing.get(p, 0) - params.get(p, 0)) < 0.1 for p in active_params):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    suggestions.append(params)
                    if len(suggestions) >= num_suggestions:
                        break
        
        # If we didn't get enough, fill with random
        while len(suggestions) < num_suggestions:
            suggestions.extend(_random_sampling(project, 1))
        
        return suggestions[:num_suggestions]
        
    except ImportError:
        logger.warning("scipy not available, using acquisition-based")
        return _suggest_acquisition_based(project, sets_with_results, active_params, num_suggestions)
    except Exception as e:
        logger.error(f"Error in Pareto-based suggestion: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return _suggest_acquisition_based(project, sets_with_results, active_params, num_suggestions)

def _generate_diverse_weights(num_weights: int, num_objectives: int) -> List[List[float]]:
    """Generate diverse weight vectors for scalarization.
    
    Uses Dirichlet distribution to sample uniformly from the simplex.
    For 2 objectives, this gives weights on [0,1] that sum to 1.
    For more objectives, ensures weights sum to 1.
    
    Args:
        num_weights: Number of weight vectors to generate
        num_objectives: Number of objectives
        
    Returns:
        List of weight vectors, each summing to 1.0
    """
    # Use Dirichlet distribution with alpha=1 for uniform sampling
    # This ensures weights sum to 1 and are uniformly distributed on the simplex
    weights = np.random.dirichlet([1.0] * num_objectives, size=num_weights)
    return weights.tolist()

def _suggest_hybrid(
    project,
    sets_with_results: List,
    active_params: List[str],
    num_suggestions: int
) -> List[Dict[str, float]]:
    """Hybrid suggestion: 50% acquisition-based, 50% exploration."""
    # Get half from acquisition, half from random exploration
    num_acq = num_suggestions // 2
    num_explore = num_suggestions - num_acq
    
    suggestions = []
    suggestions.extend(_suggest_acquisition_based(project, sets_with_results, active_params, num_acq))
    suggestions.extend(_random_sampling(project, num_explore))
    
    return suggestions[:num_suggestions]

