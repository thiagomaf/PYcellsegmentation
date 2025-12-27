import os
import json
import random
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime

from src.file_paths import PROJECT_ROOT
from tui.optimization.models import OptimizationProject, OptimizationResult, ParameterSetEntry
# Lazy import stats to avoid importing numpy/tifffile at module level
# from tui.optimization.stats import calculate_stats_for_config

logger = logging.getLogger(__name__)


def identify_best_parameter_set(
    project: OptimizationProject,
    method: str = "acquisition"
) -> Optional[Tuple[str, str]]:
    """
    Identify the best parameter set based on GP/acquisition/Pareto logic.
    
    Args:
        project: OptimizationProject instance
        method: Method to use ('acquisition', 'pareto', 'hybrid', or 'score')
        
    Returns:
        Tuple of (config_file_path, param_set_id) for the best set, or None
    """
    from tui.optimization.models import OptimizationResult
    from tui.optimization.gp_multidim import fit_multidim_gp_for_objective, get_active_parameters
    from tui.optimization.acquisition import compute_acquisition_function
    import numpy as np
    
    # Helper to check if result has default metrics
    def _has_default_metrics(result: OptimizationResult) -> bool:
        """Check if result has default/zero quality metrics."""
        return project._result_has_default_metrics(result)
    
    # Get parameter sets with results
    sets_with_results = [
        ps for ps in project.get_filtered_parameter_sets()
        if ps.result is not None and not _has_default_metrics(ps.result)
    ]
    
    if not sets_with_results:
        return None
    
    active_params = get_active_parameters(project)
    
    if method == "score":
        # Simple: best score
        best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
        return (best_ps.config_file_path, best_ps.param_set_id)
    
    elif method == "acquisition":
        # Use acquisition function to find best
        # Fit GP for the primary objective (score or first enabled objective)
        objective_fields = OptimizationResult.get_objective_fields()
        enabled_objectives = []
        for obj_name in objective_fields:
            direction = project.objective_directions.get(obj_name, None)
            if direction is None:
                direction = OptimizationResult.get_objective_direction(obj_name)
            if direction is not None:
                enabled_objectives.append(obj_name)
        
        if not enabled_objectives:
            # Fallback to score
            best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
            return (best_ps.config_file_path, best_ps.param_set_id)
        
        # Use first enabled objective (or score if available)
        primary_obj = "score" if "score" in enabled_objectives else enabled_objectives[0]
        
        # Get data for primary objective
        data = []
        for ps in sets_with_results:
            if all(pname in ps.parameters for pname in active_params):
                if primary_obj == "score":
                    obj_value = ps.result.score if ps.result else 0.0
                else:
                    obj_value = getattr(ps.result, primary_obj, None)
                if obj_value is not None:
                    data.append((ps.parameters, float(obj_value)))
        
        if len(data) < 3:
            # Not enough data for GP, use score
            best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
            return (best_ps.config_file_path, best_ps.param_set_id)
        
        # Fit GP
        gp_model = fit_multidim_gp_for_objective(
            primary_obj,
            data,
            active_params,
            kernel_type=project.gp_kernel_type
        )
        
        if not gp_model:
            # GP fitting failed, use score
            best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
            return (best_ps.config_file_path, best_ps.param_set_id)
        
        # Prepare parameter matrix for batch prediction (more efficient)
        param_arrays = []
        valid_param_sets = []
        
        for ps in sets_with_results:
            if not all(pname in ps.parameters for pname in active_params):
                continue
            param_arrays.append([ps.parameters[pname] for pname in active_params])
            valid_param_sets.append(ps)
        
        if not valid_param_sets:
            # Fallback to score
            best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
            return (best_ps.config_file_path, best_ps.param_set_id)
        
        # Batch predict using sklearn (more efficient than loop)
        X_batch = np.array(param_arrays)
        X_scaled = gp_model.scaler.transform(X_batch)
        mean, std = gp_model.predict(X_scaled, return_std=True)
        
        # Get best value for acquisition
        best_value = max(d[1] for d in data)
        
        # Compute acquisition for all points at once using numpy
        acq_values = compute_acquisition_function(
            mean,
            std,
            method=project.acquisition_method,
            best_value=best_value
        )
        
        # Find best using numpy argmax
        best_idx = np.argmax(acq_values)
        best_ps = valid_param_sets[best_idx]
        
        return (best_ps.config_file_path, best_ps.param_set_id)
    
    elif method in ["pareto", "hybrid"]:
        # Use Pareto front to find best - leveraging pymoo directly
        objective_fields = OptimizationResult.get_objective_fields()
        enabled_objectives = []
        for obj_name in objective_fields:
            direction = project.objective_directions.get(obj_name, None)
            if direction is None:
                direction = OptimizationResult.get_objective_direction(obj_name)
            if direction is not None:
                enabled_objectives.append(obj_name)
        
        if len(enabled_objectives) < 2:
            # Not enough objectives for Pareto, use score
            best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
            return (best_ps.config_file_path, best_ps.param_set_id)
        
        # Use first two objectives for Pareto
        obj1, obj2 = enabled_objectives[0], enabled_objectives[1]
        
        # Get directions for normalization
        dir1 = project.objective_directions.get(obj1, OptimizationResult.get_objective_direction(obj1))
        dir2 = project.objective_directions.get(obj2, OptimizationResult.get_objective_direction(obj2))
        maximize = [dir1 is True, dir2 is True]  # True = maximize, False = minimize
        
        # Prepare objective matrix for pymoo (n_points x n_objectives)
        # Store mapping from index to parameter set
        objective_matrix = []
        index_to_param_set = []
        
        for ps in sets_with_results:
            if not all(pname in ps.parameters for pname in active_params):
                continue
            
            val1 = getattr(ps.result, obj1, None) if obj1 != "score" else (ps.result.score if ps.result else 0.0)
            val2 = getattr(ps.result, obj2, None) if obj2 != "score" else (ps.result.score if ps.result else 0.0)
            if val1 is None or val2 is None:
                continue
            
            # Normalize by direction (pymoo expects maximization, so negate if minimizing)
            norm_val1 = float(val1) if maximize[0] else -float(val1)
            norm_val2 = float(val2) if maximize[1] else -float(val2)
            
            objective_matrix.append([norm_val1, norm_val2])
            index_to_param_set.append(ps)
        
        if len(objective_matrix) < 2:
            # Not enough data, use score
            best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
            return (best_ps.config_file_path, best_ps.param_set_id)
        
        # Use pymoo's NonDominatedSorting directly
        try:
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            
            F = np.array(objective_matrix)
            nds = NonDominatedSorting()
            fronts = nds.do(F, only_non_dominated_front=True)
            
            if len(fronts) == 0 or len(fronts[0]) == 0:
                # No Pareto front found, use score
                best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
                return (best_ps.config_file_path, best_ps.param_set_id)
            
            # Get Pareto-optimal indices
            pareto_indices = fronts[0].tolist()
            
            # Use numpy for weighted sum scalarization (equal weights)
            weights = np.array([0.5, 0.5])
            pareto_objectives = F[pareto_indices]
            
            # Compute weighted sum for each Pareto-optimal solution
            weighted_sums = np.dot(pareto_objectives, weights)
            
            # Find best index
            best_pareto_idx = pareto_indices[np.argmax(weighted_sums)]
            best_ps = index_to_param_set[best_pareto_idx]
            
            return (best_ps.config_file_path, best_ps.param_set_id)
            
        except ImportError:
            # pymoo not available, fallback to score
            logger.warning("pymoo not available for Pareto analysis, using score-based selection")
            best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
            return (best_ps.config_file_path, best_ps.param_set_id)
        except Exception as e:
            logger.error(f"Error using pymoo for Pareto analysis: {e}")
            # Fallback to score
            best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
            return (best_ps.config_file_path, best_ps.param_set_id)
    
    # Fallback: use score
    best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
    return (best_ps.config_file_path, best_ps.param_set_id) if best_ps else None

class FileBasedOptimizer:
    """
    Manages the optimization process using file-based configuration iterations.
    Replaces the MockOptimizer for real-world execution.
    """
    
    def __init__(self, project: OptimizationProject):
        self.project = project
        self.iterations_dir = os.path.join(PROJECT_ROOT, "config", "iterations")
        if not os.path.exists(self.iterations_dir):
            try:
                os.makedirs(self.iterations_dir)
            except OSError as e:
                logger.error(f"Failed to create iterations directory {self.iterations_dir}: {e}")

    def generate_initial_samples(self, count: int = 5) -> List[Dict[str, float]]:
        """
        Generate initial random samples (Latin Hypercube logic adapted from Mock).
        """
        ranges = self.project.parameter_ranges
        samples = []
        
        for _ in range(count):
            params = {}
            if ranges.optimize_diameter:
                params['diameter'] = random.randint(ranges.diameter_min, ranges.diameter_max)
            else:
                params['diameter'] = 30 # Default
                
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
            
            samples.append(params)
            
        return samples

    def suggest_next_parameters(self, method: str = "acquisition") -> Dict[str, float]:
        """
        Suggest the next set of parameters using GP-based optimization.
        
        Args:
            method: Suggestion method ('acquisition', 'pareto', 'hybrid')
            
        Returns:
            Dictionary of parameter values
        """
        try:
            from tui.optimization.gp_suggestion import suggest_next_parameters_gp
            
            # Get suggestions (returns list)
            suggestions = suggest_next_parameters_gp(
                self.project,
                method=method,
                num_suggestions=1
            )
            
            if suggestions:
                return suggestions[0]
        except Exception as e:
            logger.warning(f"GP-based suggestion failed: {e}, falling back to random")
        
        # Fallback to random sampling
        ranges = self.project.parameter_ranges
        best_params = self.project.status.best_parameters
        
        # If no best parameters found yet, random sample
        if not best_params:
            return self.generate_initial_samples(1)[0]
            
        # 30% chance of random exploration, 70% chance of local optimization around best
        if random.random() < 0.3:
             return self.generate_initial_samples(1)[0]
        
        # Local optimization: perturb best params slightly
        new_params = best_params.copy()
        
        if ranges.optimize_diameter:
            delta = random.randint(-5, 5)
            new_params['diameter'] = max(ranges.diameter_min, min(ranges.diameter_max, new_params['diameter'] + delta))
            
        if ranges.optimize_min_size:
            delta = random.randint(-3, 3)
            new_params['min_size'] = max(ranges.min_size_min, min(ranges.min_size_max, new_params['min_size'] + delta))
            
        if ranges.optimize_flow_threshold:
            delta = random.uniform(-0.1, 0.1)
            new_params['flow_threshold'] = max(ranges.flow_threshold_min, min(ranges.flow_threshold_max, new_params['flow_threshold'] + delta))

        return new_params

    def create_iteration_config(self, params: Dict[str, float]) -> str:
        """
        Create a JSON configuration file for the given parameters.
        Saves it to config/iterations/iteration_{id}.json.
        Returns the full path to the created config file.
        """
        # Generate unique iteration ID based on existing files
        iteration_id = self._get_next_iteration_id()
        filename = f"iteration_{iteration_id}.json"
        filepath = os.path.join(self.iterations_dir, filename)
        
        # 1. Load a template config
        # We try to use the first included config in the project as a template
        # If none, we default to a basic structure
        template_config = {}
        template_source = None
        
        if self.project.config_files:
            # Use the first included config
            for cf in self.project.config_files:
                if cf.included and os.path.exists(cf.filepath):
                    try:
                        with open(cf.filepath, 'r') as f:
                            template_config = json.load(f)
                        template_source = cf.filepath
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load template config {cf.filepath}: {e}")
        
        if not template_config:
            # Basic fallback structure
            template_config = {
                "global_segmentation_settings": {
                    "default_log_level": "INFO",
                    "max_processes": 1,
                    "FORCE_GRAYSCALE": True,
                    "USE_GPU_IF_AVAILABLE": True
                },
                "image_configurations": [],
                "cellpose_parameter_configurations": []
            }
            # Add active images from pool if available
            if self.project.image_pool:
                 for img_entry in self.project.image_pool:
                     if img_entry.is_active:
                         template_config["image_configurations"].append({
                             "image_id": os.path.splitext(os.path.basename(img_entry.filepath))[0],
                             "original_image_filename": img_entry.filepath,
                             "is_active": True,
                             "segmentation_options": {"apply_segmentation": True}
                         })

        # 2. Update cellpose_parameter_configurations
        # We replace whatever was there with our single optimized parameter set
        param_set_id = f"iter_{iteration_id}"
        
        # Start with a base set of parameters (from first config if available, else defaults)
        base_cp_params = {
            "MODEL_CHOICE": "cyto3",
            "DIAMETER": 30,
            "MIN_SIZE": 15,
            "CELLPROB_THRESHOLD": 0.0,
            "FORCE_GRAYSCALE": True,
            "USE_GPU": True
        }
        
        if template_config.get("cellpose_parameter_configurations"):
            # Try to inherit other settings from the first config in the template
            first_cp_config = template_config["cellpose_parameter_configurations"][0]
            if "cellpose_parameters" in first_cp_config:
                base_cp_params.update(first_cp_config["cellpose_parameters"])
        
        # Overlay optimization parameters
        # Ensure we map our simplified keys to the actual JSON keys
        if 'diameter' in params:
            base_cp_params['DIAMETER'] = int(params['diameter'])
        if 'min_size' in params:
            base_cp_params['MIN_SIZE'] = int(params['min_size'])
        if 'flow_threshold' in params:
            base_cp_params['FLOW_THRESHOLD'] = params['flow_threshold']
        if 'cellprob_threshold' in params:
            base_cp_params['CELLPROB_THRESHOLD'] = params['cellprob_threshold']
            
        new_cp_config = {
            "param_set_id": param_set_id,
            "is_active": True,
            "cellpose_parameters": base_cp_params
        }
        
        template_config["cellpose_parameter_configurations"] = [new_cp_config]
        
        # 3. Save the new config
        try:
            with open(filepath, 'w') as f:
                json.dump(template_config, f, indent=2)
            logger.info(f"Created iteration config: {filepath}")
            
            # 4. Add to project's config files pool
            from tui.optimization.models import ConfigFileInfo
            config_info = ConfigFileInfo(filepath=filepath, included=True)
            self.project.config_files.append(config_info)
            self.project.save()  # Save project to persist the new config file
            
            return filepath
        except Exception as e:
            logger.error(f"Error saving iteration config {filepath}: {e}")
            raise
    
    def create_batch_iteration_config(self, suggestions: List[Dict[str, float]]) -> str:
        """
        Create a JSON configuration file for multiple parameter suggestions.
        Each suggestion becomes a separate param_set_id in the same config file.
        Saves it to config/iterations/iteration_{id}.json.
        Returns the full path to the created config file.
        
        Args:
            suggestions: List of parameter dictionaries
            
        Returns:
            Full path to the created config file
        """
        if not suggestions:
            raise ValueError("No suggestions provided for batch config generation")
        
        # Generate unique iteration ID based on existing files
        iteration_id = self._get_next_iteration_id()
        filename = f"iteration_{iteration_id}.json"
        filepath = os.path.join(self.iterations_dir, filename)
        
        # 1. Load a template config
        template_config = {}
        template_source = None
        
        if self.project.config_files:
            # Use the first included config
            for cf in self.project.config_files:
                if cf.included and os.path.exists(cf.filepath):
                    try:
                        with open(cf.filepath, 'r') as f:
                            template_config = json.load(f)
                        template_source = cf.filepath
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load template config {cf.filepath}: {e}")
        
        if not template_config:
            # Basic fallback structure
            template_config = {
                "global_segmentation_settings": {
                    "default_log_level": "INFO",
                    "max_processes": 1,
                    "FORCE_GRAYSCALE": True,
                    "USE_GPU_IF_AVAILABLE": True
                },
                "image_configurations": [],
                "cellpose_parameter_configurations": []
            }
            # Add active images from pool if available
            if self.project.image_pool:
                for img_entry in self.project.image_pool:
                    if img_entry.is_active:
                        template_config["image_configurations"].append({
                            "image_id": os.path.splitext(os.path.basename(img_entry.filepath))[0],
                            "original_image_filename": img_entry.filepath,
                            "is_active": True,
                            "segmentation_options": {"apply_segmentation": True}
                        })
        
        # 2. Create parameter configurations from all suggestions
        param_configs = []
        
        # Get base parameters from template if available
        base_cp_params = {
            "MODEL_CHOICE": "cyto3",
            "DIAMETER": 30,
            "MIN_SIZE": 15,
            "CELLPROB_THRESHOLD": 0.0,
            "FORCE_GRAYSCALE": True,
            "USE_GPU": True
        }
        
        if template_config.get("cellpose_parameter_configurations"):
            # Try to inherit other settings from the first config in the template
            first_cp_config = template_config["cellpose_parameter_configurations"][0]
            if "cellpose_parameters" in first_cp_config:
                base_cp_params.update(first_cp_config["cellpose_parameters"])
        
        # Create a param config for each suggestion
        for idx, params in enumerate(suggestions):
            # Start with base parameters
            cp_params = base_cp_params.copy()
            
            # Overlay optimization parameters
            if 'diameter' in params:
                cp_params['DIAMETER'] = int(params['diameter'])
            if 'min_size' in params:
                cp_params['MIN_SIZE'] = int(params['min_size'])
            if 'flow_threshold' in params:
                cp_params['FLOW_THRESHOLD'] = params['flow_threshold']
            if 'cellprob_threshold' in params:
                cp_params['CELLPROB_THRESHOLD'] = params['cellprob_threshold']
            
            param_set_id = f"iter_{iteration_id}_{idx + 1}"
            
            param_config = {
                "param_set_id": param_set_id,
                "is_active": True,
                "cellpose_parameters": cp_params
            }
            
            param_configs.append(param_config)
        
        template_config["cellpose_parameter_configurations"] = param_configs
        
        # 3. Save the new config
        try:
            with open(filepath, 'w') as f:
                json.dump(template_config, f, indent=2)
            logger.info(f"Created batch iteration config with {len(suggestions)} parameter sets: {filepath}")
            
            # 4. Add to project's config files pool
            from tui.optimization.models import ConfigFileInfo
            config_info = ConfigFileInfo(filepath=filepath, included=True)
            self.project.config_files.append(config_info)
            self.project.save()  # Save project to persist the new config file
            
            return filepath
        except Exception as e:
            logger.error(f"Error saving batch iteration config {filepath}: {e}")
            raise
    
    def _get_next_iteration_id(self) -> int:
        """Get the next iteration ID by checking existing iteration files.
        
        Returns:
            Next available iteration ID (starts at 1 if no files exist)
        """
        if not os.path.exists(self.iterations_dir):
            return 1
        
        # Find all existing iteration files
        existing_ids = []
        for filename in os.listdir(self.iterations_dir):
            if filename.startswith("iteration_") and filename.endswith(".json"):
                try:
                    # Extract ID from filename like "iteration_5.json"
                    id_str = filename.replace("iteration_", "").replace(".json", "")
                    existing_ids.append(int(id_str))
                except ValueError:
                    continue
        
        if not existing_ids:
            return 1
        
        # Return next ID (max + 1)
        return max(existing_ids) + 1

    def scan_config_files_for_parameter_sets(self) -> List[ParameterSetEntry]:
        """
        Scan all included config files and extract all active parameter sets.
        Only processes images that are active in the project's image pool.
        
        Returns:
            List of ParameterSetEntry objects, one for each parameter set found.
        """
        parameter_sets = []
        
        # Lazy import to avoid importing ProjectConfig at module level
        from tui.models import ProjectConfig
        
        # Build set of active image paths from project's image pool (normalized)
        active_image_paths_normalized = set()
        for entry in self.project.image_pool:
            if entry.is_active:
                normalized = os.path.normpath(entry.filepath)
                active_image_paths_normalized.add(normalized)
        
        # Helper function to normalize and resolve image paths
        def normalize_image_path(image_path: str, config_file_path: str) -> str:
            """Normalize an image path, resolving relative paths if needed."""
            if os.path.isabs(image_path):
                return os.path.normpath(image_path)
            
            # Try relative to config file location first
            config_dir = os.path.dirname(config_file_path)
            resolved = os.path.normpath(os.path.join(config_dir, image_path))
            if os.path.exists(resolved):
                return resolved
            
            # Try relative to PROJECT_ROOT
            try:
                from src.file_paths import PROJECT_ROOT
                resolved = os.path.normpath(os.path.join(PROJECT_ROOT, image_path))
                if os.path.exists(resolved):
                    return resolved
            except ImportError:
                pass
            
            # Return normalized original path if resolution fails
            return os.path.normpath(image_path)
        
        for config_file_info in self.project.config_files:
            if not config_file_info.included:
                continue
                
            if not os.path.exists(config_file_info.filepath):
                logger.warning(f"Config file not found: {config_file_info.filepath}")
                continue
            
            try:
                config = ProjectConfig.from_json_file(config_file_info.filepath)
                
                # Auto-sync images from config to project pool if not present
                # This ensures new config files are automatically recognized
                images_synced = False
                for img_config in config.image_configurations:
                    if not img_config.is_active:
                        continue
                    
                    image_path = normalize_image_path(
                        img_config.original_image_filename,
                        config_file_info.filepath
                    )
                    
                    # Check if this image is already in the pool (normalized comparison)
                    image_in_pool = False
                    for pool_entry in self.project.image_pool:
                        if os.path.normpath(pool_entry.filepath) == image_path:
                            image_in_pool = True
                            # Update is_active status to match config
                            if pool_entry.is_active != img_config.is_active:
                                pool_entry.is_active = img_config.is_active
                                images_synced = True
                            break
                    
                    # Add to pool if not present
                    if not image_in_pool:
                        from tui.optimization.models import ImagePoolEntry
                        pool_entry = ImagePoolEntry(
                            filepath=image_path,
                            is_active=img_config.is_active
                        )
                        self.project.image_pool.append(pool_entry)
                        active_image_paths_normalized.add(os.path.normpath(image_path))
                        images_synced = True
                
                # Save project if images were synced
                if images_synced:
                    self.project.save()
                    logger.info(f"Synced images from config {os.path.basename(config_file_info.filepath)} to project pool")
                
                # Process all parameter sets from this config
                # Only include parameter sets if the project has an image pool filter,
                # OR if the project has no image pool (process all)
                should_process_config = True
                if active_image_paths_normalized:
                    # Check if config has any images that match the project pool
                    config_has_matching_images = False
                    for img_config in config.image_configurations:
                        if not img_config.is_active:
                            continue
                        image_path = normalize_image_path(
                            img_config.original_image_filename,
                            config_file_info.filepath
                        )
                        if os.path.normpath(image_path) in active_image_paths_normalized:
                            config_has_matching_images = True
                            break
                    should_process_config = config_has_matching_images
                
                if should_process_config:
                    for param_config in config.cellpose_parameter_configurations:
                        if not param_config.is_active:
                            continue
                        
                        param_set_id = param_config.param_set_id
                        cp_params = param_config.cellpose_parameters
                        
                        # Extract parameters in the format expected by the dashboard
                        params = {
                            'diameter': cp_params.DIAMETER if cp_params.DIAMETER is not None else 0,
                            'min_size': cp_params.MIN_SIZE,
                            'flow_threshold': getattr(cp_params, 'FLOW_THRESHOLD', 0.0) if hasattr(cp_params, 'FLOW_THRESHOLD') else 0.0,
                            'cellprob_threshold': cp_params.CELLPROB_THRESHOLD,
                        }
                        
                        # Get or create parameter set entry
                        param_set = self.project.get_or_create_parameter_set(
                            config_file_path=config_file_info.filepath,
                            param_set_id=param_set_id,
                            parameters=params
                        )
                        
                        parameter_sets.append(param_set)
                else:
                    logger.debug(f"Skipping config {os.path.basename(config_file_info.filepath)}: no matching images in project pool")
                    
            except Exception as e:
                logger.error(f"Error scanning config file {config_file_info.filepath}: {e}")
                continue
        
        return parameter_sets

    def check_parameter_set_status(self, param_set: ParameterSetEntry) -> Optional[OptimizationResult]:
        """
        Check if results exist for the given parameter set.
        Only processes images that are active in the project's image pool.
        """
        # Lazy import to avoid importing numpy/tifffile at module level
        from tui.optimization.stats import calculate_stats_for_config
        
        # Normalize config path to handle drive letter mismatches
        config_path = self._normalize_config_path(param_set.config_file_path)
        
        # Build set of active image paths from project's image pool
        active_image_paths = {
            entry.filepath for entry in self.project.image_pool 
            if entry.is_active
        }
        
        return calculate_stats_for_config(
            config_path, 
            param_set_id=param_set.param_set_id,
            active_image_paths=active_image_paths if active_image_paths else None
        )
    
    def _normalize_and_store_path(self, path: str, context: str = "") -> Tuple[str, str]:
        """
        Normalize a path and return both original and normalized versions.
        
        Args:
            path: Path to normalize
            context: Context string for logging (e.g., "config", "image_pool")
        
        Returns:
            Tuple of (original_path, normalized_absolute_path)
        """
        original = path
        
        # Convert to absolute if relative
        if not os.path.isabs(path):
            # Try relative to PROJECT_ROOT
            path = os.path.join(PROJECT_ROOT, path)
        
        # Normalize (resolve .., ., //, etc.)
        normalized = os.path.normpath(path)
        
        # Log for debugging if paths differ significantly
        if os.path.normpath(original) != normalized:
            logger.debug(f"Path normalized ({context}): {original} -> {normalized}")
        
        return original, normalized
    
    def _normalize_config_path(self, config_path: str) -> str:
        """
        Normalize a config file path to handle drive letter mismatches.
        
        If the path doesn't exist as-is, try to construct it relative to PROJECT_ROOT.
        """
        if os.path.exists(config_path):
            return config_path
        
        # Try relative to PROJECT_ROOT
        rel_path = os.path.relpath(config_path, start=PROJECT_ROOT) if os.path.isabs(config_path) else config_path
        normalized = os.path.join(PROJECT_ROOT, rel_path)
        if os.path.exists(normalized):
            return normalized
        
        # Try replacing drive letter with current workspace drive
        if os.path.isabs(config_path) and len(config_path) >= 2 and config_path[1] == ':':
            # Extract path without drive
            path_without_drive = config_path[2:].lstrip('\\/')
            # Try with current workspace drive
            workspace_drive = os.path.splitdrive(PROJECT_ROOT)[0]
            if workspace_drive:
                alt_path = workspace_drive + os.sep + path_without_drive
                if os.path.exists(alt_path):
                    return alt_path
        
        # Return original if we can't normalize
        return config_path
    
    def has_output_files(self, param_set: ParameterSetEntry) -> bool:
        """
        Lightweight check if output files exist for a parameter set without importing numpy or cv2.
        This reads the JSON config directly and constructs expected paths without full job expansion.
        
        Returns:
            True if output mask files exist, False otherwise.
        """
        # #region agent log
        try:
            import json, time
            with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"optimizer.py:has_output_files","message":"ENTRY","data":{"config_path":param_set.config_file_path,"param_set_id":param_set.param_set_id},"timestamp":time.time()*1000})+"\n")
        except: pass
        # #endregion
        from src.file_paths import RESULTS_DIR_BASE
        
        # Normalize config path to handle drive letter mismatches
        config_path = self._normalize_config_path(param_set.config_file_path)
        # #region agent log
        try:
            import json, time
            with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"optimizer.py:has_output_files","message":"normalized config path","data":{"original_path":param_set.config_file_path,"normalized_path":config_path,"exists":os.path.exists(config_path)},"timestamp":time.time()*1000})+"\n")
        except: pass
        # #endregion
        
        if not os.path.exists(config_path):
            return False
        
        try:
            # Read JSON config directly (no heavy imports)
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            image_configs = config_data.get("image_configurations", [])
            cellpose_param_configs = config_data.get("cellpose_parameter_configurations", [])
            
            if not image_configs or not cellpose_param_configs:
                return False
            
            # Find the matching param config
            matching_param_config = None
            for param_config in cellpose_param_configs:
                if not param_config.get("is_active", True):
                    continue
                if param_config.get("param_set_id") == param_set.param_set_id:
                    matching_param_config = param_config
                    break
            
            if not matching_param_config:
                return False
            
            # Build set of active image paths from project's image pool
            active_image_paths = {
                entry.filepath for entry in self.project.image_pool 
                if entry.is_active
            }
            
            # Check for mask files for each active image config that's also in project's active image pool
            masks_found = 0
            for img_config in image_configs:
                if not img_config.get("is_active", True):
                    continue
                
                # Check if this image is in the project's active image pool
                image_path = img_config.get("original_image_filename", "")
                if not image_path:
                    continue
                
                # Normalize BOTH paths using the same method before comparison
                image_path_normalized = os.path.normpath(
                    os.path.join(PROJECT_ROOT, image_path) if not os.path.isabs(image_path) else image_path
                )
                
                # Skip if image is not in project's active image pool
                if active_image_paths:  # Only filter if project has defined image pool
                    image_in_pool = False
                    for active_path in active_image_paths:
                        active_path_normalized = os.path.normpath(
                            os.path.join(PROJECT_ROOT, active_path) if not os.path.isabs(active_path) else active_path
                        )
                        if active_path_normalized == image_path_normalized:
                            image_in_pool = True
                            break
                    if not image_in_pool:
                        continue  # Skip this image - not in project's active pool
                
                image_id = img_config.get("image_id", "")
                if not image_id:
                    # Try to derive from filename
                    original_filename = img_config.get("original_image_filename", "")
                    if original_filename:
                        image_id = os.path.splitext(os.path.basename(original_filename))[0]
                
                if not image_id:
                    continue
                
                # Construct base experiment ID: image_id_param_set_id
                base_experiment_id = f"{image_id}_{param_set.param_set_id}"
                
                # Check for mask files in results directory
                # We check both the base experiment ID and possible scaled/tiled variants
                experiment_dir = os.path.join(RESULTS_DIR_BASE, base_experiment_id)
                if os.path.exists(experiment_dir):
                    # Check if any mask files exist in this directory
                    try:
                        for filename in os.listdir(experiment_dir):
                            if filename.endswith("_mask.tif"):
                                masks_found += 1
                                break  # Found at least one mask for this image
                    except OSError:
                        pass
                
                # Also check for scaled variants (common pattern: base_id_scaled_X_Y)
                # This is a simplified check - we look for directories that start with base_experiment_id
                try:
                    if os.path.exists(RESULTS_DIR_BASE):
                        for dirname in os.listdir(RESULTS_DIR_BASE):
                            if dirname.startswith(base_experiment_id):
                                exp_dir = os.path.join(RESULTS_DIR_BASE, dirname)
                                if os.path.isdir(exp_dir):
                                    try:
                                        for filename in os.listdir(exp_dir):
                                            if filename.endswith("_mask.tif"):
                                                masks_found += 1
                                                break
                                    except OSError:
                                        pass
                except OSError:
                    pass
            
            result = masks_found > 0
            # #region agent log
            try:
                import json, time
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"optimizer.py:has_output_files","message":"RETURN","data":{"masks_found":masks_found,"result":result,"param_set_id":param_set.param_set_id},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            return result
            
        except Exception as e:
            # #region agent log
            try:
                import json, time
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"optimizer.py:has_output_files","message":"EXCEPTION","data":{"error":str(e),"param_set_id":param_set.param_set_id},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            logger.debug(f"Error checking output files for {param_set.param_set_id}: {e}")
            return False

