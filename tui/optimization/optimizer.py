import os
import json
import random
import logging
from pathlib import Path
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
    Each parameter_set x image combination is treated as a separate data point.
    
    Args:
        project: OptimizationProject instance
        method: Method to use ('acquisition', 'pareto', 'hybrid', or 'score')
        
    Returns:
        Tuple of (param_set_id, image_path) for the best set, or None
    """
    from tui.optimization.models import OptimizationResult
    from tui.optimization.gp_multidim import fit_multidim_gp_for_objective, get_active_parameters
    from tui.optimization.acquisition import compute_acquisition_function
    import numpy as np
    
    # Helper to check if result has default metrics
    def _has_default_metrics(result: OptimizationResult) -> bool:
        """Check if result has default/zero quality metrics."""
        return project._result_has_default_metrics(result)
    
    # Get parameter sets with results (each parameter_set x image is a separate entry)
    sets_with_results = [
        ps for ps in project.get_filtered_parameter_sets()
        if ps.result is not None and not _has_default_metrics(ps.result) and ps.image_path
    ]
    
    if not sets_with_results:
        return None
    
    active_params = get_active_parameters(project)
    
    if method == "score":
        # Simple: best score across all parameter_set x image combinations
        best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
        return (best_ps.param_set_id, best_ps.image_path or "")
    
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
            return (best_ps.param_set_id, best_ps.image_path or "")
        
        # Use first enabled objective (or score if available)
        primary_obj = "score" if "score" in enabled_objectives else enabled_objectives[0]
        
        # Get data for primary objective (using get_objective_data which returns per-image results)
        # Format: List[Tuple[Dict[str, float], float, str]] where last element is image_path
        data_with_images = project.get_objective_data(primary_obj)
        
        # Extract just (parameters, value) for GP fitting (GP doesn't need image info)
        data = [(params, value) for params, value, _ in data_with_images]
        
        if len(data) < 3:
            # Not enough data for GP, use score
            best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
            return (best_ps.param_set_id, best_ps.image_path or "")
        
        # Fit GP using all parameter_set x image combinations as training data
        gp_model = fit_multidim_gp_for_objective(
            primary_obj,
            data,
            active_params,
            kernel_type=project.gp_kernel_type
        )
        
        if not gp_model:
            # GP fitting failed, use score
            best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
            return (best_ps.param_set_id, best_ps.image_path or "")
        
        # Prepare parameter matrix for batch prediction (more efficient)
        # Use all parameter_set x image combinations for prediction
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
            return (best_ps.param_set_id, best_ps.image_path or "")
        
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
        
        return (best_ps.param_set_id, best_ps.image_path or "")
    
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
            return (best_ps.param_set_id, best_ps.image_path or "")
        
        # Use first two objectives for Pareto
        obj1, obj2 = enabled_objectives[0], enabled_objectives[1]
        
        # Get directions for normalization
        dir1 = project.objective_directions.get(obj1, OptimizationResult.get_objective_direction(obj1))
        dir2 = project.objective_directions.get(obj2, OptimizationResult.get_objective_direction(obj2))
        maximize = [dir1 is True, dir2 is True]  # True = maximize, False = minimize
        
        # Prepare objective matrix for pymoo (n_points x n_objectives)
        # Each parameter_set x image combination is a separate point
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
            return (best_ps.param_set_id, best_ps.image_path or "")
        
        # Use pymoo's NonDominatedSorting directly
        try:
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            
            F = np.array(objective_matrix)
            nds = NonDominatedSorting()
            fronts = nds.do(F, only_non_dominated_front=True)
            
            if len(fronts) == 0 or len(fronts[0]) == 0:
                # No Pareto front found, use score
                best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
                return (best_ps.param_set_id, best_ps.image_path or "")
            
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
            
            return (best_ps.param_set_id, best_ps.image_path or "")
            
        except ImportError:
            # pymoo not available, fallback to score
            logger.warning("pymoo not available for Pareto analysis, using score-based selection")
            best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
            return (best_ps.param_set_id, best_ps.image_path or "")
        except Exception as e:
            logger.error(f"Error using pymoo for Pareto analysis: {e}")
            # Fallback to score
            best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
            return (best_ps.param_set_id, best_ps.image_path or "")
    
    # Fallback: use score
    best_ps = max(sets_with_results, key=lambda ps: ps.result.score if ps.result else 0.0)
    return (best_ps.param_set_id, best_ps.image_path or "") if best_ps else None

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

    def _normalize_path_for_comparison(self, path: str, project_root: Path, config_filepath: Optional[Path] = None) -> str:
        """Normalize a path for comparison, handling both relative and absolute paths.
        
        Args:
            path: Path to normalize (can be relative or absolute)
            project_root: Project root directory
            config_filepath: Optional config file path for relative resolution
            
        Returns:
            Normalized absolute path string
        """
        if not path:
            return ""
        
        # If absolute, just normalize
        if Path(path).is_absolute():
            return os.path.normpath(path)
        
        # Try relative to config file location first
        if config_filepath:
            config_dir = Path(config_filepath).parent
            resolved = (config_dir / path).resolve()
            if resolved.exists():
                return os.path.normpath(str(resolved))
        
        # Try relative to PROJECT_ROOT
        try:
            resolved = (project_root / path).resolve()
            if resolved.exists():
                return os.path.normpath(str(resolved))
        except Exception:
            pass
        
        # If resolution failed, construct path relative to project_root for comparison
        # (even if file doesn't exist, we can still compare paths)
        try:
            constructed = (project_root / path).resolve()
            return os.path.normpath(str(constructed))
        except Exception:
            # Last resort: just normalize the original path
            return os.path.normpath(path)
    
    def _find_matching_image_config_from_pool(self, pool_entry) -> Optional[Dict]:
        """Find a matching ImageConfiguration from existing config files.
        
        Args:
            pool_entry: ImagePoolEntry from the project's image pool
            
        Returns:
            Dictionary representation of ImageConfiguration if found, None otherwise
        """
        from tui.models import ProjectConfig
        from pathlib import Path
        
        # Get PROJECT_ROOT for path resolution
        try:
            from src.file_paths import PROJECT_ROOT
            project_root = Path(PROJECT_ROOT)
        except (ImportError, Exception):
            if self.project.filepath:
                project_root = Path(self.project.filepath).parent.parent.parent
            else:
                project_root = Path.cwd()
        
        # Normalize the pool entry path for comparison
        pool_path_normalized = self._normalize_path_for_comparison(pool_entry.filepath, project_root)
        
        # Also get a normalized image_id from the pool entry filename for fallback matching
        pool_img_path = Path(pool_entry.filepath)
        pool_image_id_from_filename = pool_img_path.stem
        if pool_image_id_from_filename.endswith('.ome'):
            pool_image_id_from_filename = pool_image_id_from_filename[:-4]
        # Normalize: replace hyphens with underscores for comparison
        pool_image_id_normalized = pool_image_id_from_filename.replace('-', '_')
        
        # Iterate through config files (newest first to get most recent settings)
        config_files_sorted = sorted(
            self.project.config_files,
            key=lambda cf: cf.created_at,
            reverse=True
        )
        
        # Collect all matches and pick the one with most complete settings
        best_match = None
        best_match_score = -1
        
        for config_info in config_files_sorted:
            # NOTE: We search ALL config files (both included and excluded) when looking for image settings
            # to preserve, because the 'included' flag only affects optimization runs, not settings preservation.
            # This ensures we can find complete settings even if a config is temporarily excluded.
            
            # Resolve config file path
            config_filepath = config_info.filepath
            if not Path(config_filepath).is_absolute():
                config_filepath = project_root / config_filepath
            
            if not os.path.exists(config_filepath):
                continue
            
            try:
                existing_config = ProjectConfig.from_json_file(str(config_filepath))
                
                for img_config in existing_config.image_configurations:
                    # Normalize the image path from config
                    img_path_from_config = img_config.original_image_filename
                    config_path_normalized = self._normalize_path_for_comparison(
                        img_path_from_config, project_root, config_filepath
                    )
                    
                    # Check if paths match
                    path_matches = config_path_normalized == pool_path_normalized
                    
                    # Also try matching by image_id (normalized) as fallback
                    config_image_id_normalized = img_config.image_id.replace('-', '_') if img_config.image_id else ""
                    image_id_matches = config_image_id_normalized == pool_image_id_normalized
                    
                    if path_matches or image_id_matches:
                        # Score this match based on completeness of settings
                        score = 0
                        if img_config.mpp_x is not None:
                            score += 1
                        if img_config.mpp_y is not None:
                            score += 1
                        if img_config.segmentation_options and img_config.segmentation_options.rescaling_config is not None:
                            score += 2  # Rescaling config is more important
                        if img_config.segmentation_options and img_config.segmentation_options.tiling_parameters:
                            score += 1
                        
                        # Keep the match with highest score (most complete settings)
                        if score > best_match_score:
                            best_match_score = score
                            # Found a match! Return dict representation preserving all settings
                            img_dict = img_config.model_dump(mode='json', exclude_none=True)
                            
                            # Update original_image_filename to match pool entry format (but preserve relative path format)
                            # Keep the original_image_filename format from config if it was relative
                            if not Path(img_config.original_image_filename).is_absolute():
                                # If config used relative path, try to make pool entry relative too
                                try:
                                    pool_path = Path(pool_entry.filepath)
                                    if pool_path.is_absolute():
                                        relative_path = pool_path.relative_to(project_root)
                                        img_dict['original_image_filename'] = str(relative_path).replace('\\', '/')
                                    else:
                                        img_dict['original_image_filename'] = pool_entry.filepath
                                except ValueError:
                                    img_dict['original_image_filename'] = pool_entry.filepath
                            else:
                                img_dict['original_image_filename'] = pool_entry.filepath
                            img_dict['is_active'] = True
                            
                            best_match = img_dict
            except Exception as e:
                logger.warning(f"Error reading config file {config_filepath} for image lookup: {e}")
                continue
        
        if best_match:
            return best_match
        
        return None
    
    def _ensure_image_configs_from_pool(self, template_config: dict) -> None:
        """Ensure image_configurations in template match project's image_pool, preserving settings.
        
        Args:
            template_config: Template config dictionary to update
        """
        from pathlib import Path
        
        # Get PROJECT_ROOT for path resolution
        try:
            from src.file_paths import PROJECT_ROOT
            project_root = Path(PROJECT_ROOT)
        except (ImportError, Exception):
            if self.project.filepath:
                project_root = Path(self.project.filepath).parent.parent.parent
            else:
                project_root = Path.cwd()
        
        # Build a set of image paths already in template (normalized)
        template_image_paths = set()
        template_image_ids = set()
        for img_config in template_config.get("image_configurations", []):
            img_path = img_config.get("original_image_filename", "")
            if img_path:
                normalized_path = self._normalize_path_for_comparison(img_path, project_root)
                template_image_paths.add(normalized_path)
            # Also track image_ids for fallback matching
            img_id = img_config.get("image_id", "")
            if img_id:
                template_image_ids.add(img_id.replace('-', '_'))
        
        # Ensure all active images from pool are in template
        if self.project.image_pool:
            for img_entry in self.project.image_pool:
                if not img_entry.is_active:
                    continue
                
                pool_path_normalized = self._normalize_path_for_comparison(img_entry.filepath, project_root)
                
                # Also get normalized image_id for fallback check
                pool_img_path = Path(img_entry.filepath)
                pool_image_id_from_filename = pool_img_path.stem
                if pool_image_id_from_filename.endswith('.ome'):
                    pool_image_id_from_filename = pool_image_id_from_filename[:-4]
                pool_image_id_normalized = pool_image_id_from_filename.replace('-', '_')
                
                # Check if already in template (by path or image_id)
                if pool_path_normalized in template_image_paths:
                    continue
                if pool_image_id_normalized in template_image_ids:
                    continue
                
                # Try to find matching config from existing configs
                matching_img_dict = self._find_matching_image_config_from_pool(img_entry)
                
                if matching_img_dict:
                    # Use the matching config with all settings preserved
                    template_config.setdefault("image_configurations", []).append(matching_img_dict)
                else:
                    # Fallback: create default image configuration
                    img_path = Path(img_entry.filepath)
                    image_id = img_path.stem
                    if image_id.endswith('.ome'):
                        image_id = image_id[:-4]
                    
                    template_config.setdefault("image_configurations", []).append({
                        "image_id": image_id,
                        "original_image_filename": img_entry.filepath,
                        "is_active": True,
                        "segmentation_options": {"apply_segmentation": True}
                    })
    
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
        
        # Ensure image_configurations match project's image_pool, preserving settings
        self._ensure_image_configs_from_pool(template_config)

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
        
        # Ensure image_configurations match project's image_pool, preserving settings
        self._ensure_image_configs_from_pool(template_config)
        
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
        Creates one ParameterSetEntry per parameter_set x image combination.
        Only processes images that are active in the project's image pool.
        
        Returns:
            List of ParameterSetEntry objects, one for each parameter_set x image combination.
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
                    # Get active images from config that are also in project's active pool
                    active_images_in_config = []
                    for img_config in config.image_configurations:
                        if not img_config.is_active:
                            continue
                        
                        image_path = normalize_image_path(
                            img_config.original_image_filename,
                            config_file_info.filepath
                        )
                        normalized_image_path = os.path.normpath(image_path)
                        
                        # Only include if in project's active image pool (or if no pool filter)
                        if not active_image_paths_normalized or normalized_image_path in active_image_paths_normalized:
                            # Use original_image_filename from config (may be relative)
                            active_images_in_config.append(img_config.original_image_filename)
                    
                    # If no active images match, skip this config
                    if not active_images_in_config:
                        logger.debug(f"Skipping config {os.path.basename(config_file_info.filepath)}: no matching active images")
                        continue
                    
                    # For each parameter set, create one entry per active image
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
                        
                        # Create one ParameterSetEntry per image
                        for image_filename in active_images_in_config:
                            # Normalize the image path for storage
                            image_path = normalize_image_path(
                                image_filename,
                                config_file_info.filepath
                            )
                            
                            # Get or create parameter set entry (one per param_set x image)
                            param_set = self.project.get_or_create_parameter_set(
                                param_set_id=param_set_id,
                                image_path=image_path,
                                parameters=params,
                                config_file_path=config_file_info.filepath
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
        Check if results exist for the given parameter set and image combination.
        Calculates stats for the specific image in the parameter set entry.
        """
        # Need image_path to calculate stats for specific image
        if not param_set.image_path:
            logger.warning(f"Parameter set {param_set.param_set_id} has no image_path, cannot calculate stats")
            return None
        
        # Need config_file_path to load config
        if not param_set.config_file_path:
            logger.warning(f"Parameter set {param_set.param_set_id} has no config_file_path, cannot calculate stats")
            return None
        
        # Lazy import to avoid importing numpy/tifffile at module level
        from tui.optimization.stats import calculate_stats_for_image
        
        # Normalize config path to handle drive letter mismatches
        config_path = self._normalize_config_path(param_set.config_file_path)
        
        
        try:
            result = calculate_stats_for_image(
                config_path=config_path,
                param_set_id=param_set.param_set_id,
                image_path=param_set.image_path
            )
            return result
        except Exception as e:
            raise
    
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
        
        If param_set.image_path is provided, only checks for that specific image.
        Otherwise, checks if ANY image in the config has masks.
        
        Returns:
            True if output mask files exist, False otherwise.
        """
        from src.file_paths import RESULTS_DIR_BASE, PROJECT_ROOT
        
        # Normalize config path to handle drive letter mismatches
        config_path = self._normalize_config_path(param_set.config_file_path)
        
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
            
            # If param_set has a specific image_path, only check for that image
            # Otherwise, check all images in the config
            target_image_path = None
            if param_set.image_path:
                # Normalize the target image path for comparison
                target_image_path = os.path.normpath(
                    os.path.join(PROJECT_ROOT, param_set.image_path) if not os.path.isabs(param_set.image_path) else param_set.image_path
                )
            
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
                
                # If we're checking for a specific image, skip if this doesn't match
                if target_image_path and image_path_normalized != target_image_path:
                    continue
                
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
                        matching_dirs = []
                        for dirname in os.listdir(RESULTS_DIR_BASE):
                            if dirname.startswith(base_experiment_id):
                                exp_dir = os.path.join(RESULTS_DIR_BASE, dirname)
                                if os.path.isdir(exp_dir):
                                    matching_dirs.append(dirname)
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
            return result
            
        except Exception as e:
            logger.debug(f"Error checking output files for {param_set.param_set_id}: {e}")
            return False

