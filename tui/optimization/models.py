from typing import List, Dict, Optional, Literal, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import json
import os

class ConfigFileInfo(BaseModel):
    """Information about a config file in the project."""
    filepath: str
    included: bool = True  # Whether this config is included in the optimization pool
    created_at: datetime = Field(default_factory=datetime.now)  # When this config file was added to the project

class ImagePoolEntry(BaseModel):
    """Information about an image in the project pool."""
    filepath: str
    is_active: bool = True  # Whether this image is active for optimization

class ParameterRanges(BaseModel):
    """Defines the search space for optimization."""
    diameter_min: int = 15
    diameter_max: int = 100
    min_size_min: int = 5
    min_size_max: int = 50
    flow_threshold_min: float = 0.0
    flow_threshold_max: float = 1.0
    cellprob_threshold_min: float = -6.0
    cellprob_threshold_max: float = 6.0
    
    # Flags to indicate which parameters are active for optimization
    optimize_diameter: bool = True
    optimize_min_size: bool = True
    optimize_flow_threshold: bool = False
    optimize_cellprob_threshold: bool = False

class OptimizationResult(BaseModel):
    """The result of a single optimization run (one set of parameters on one image/batch)."""
    score: float = 0.0  # The "goodness" metric (e.g. 0.0 to 1.0)
    num_cells: int = 0
    mean_area: float = 0.0
    
    # Quality metrics (calculated from segmentation masks)
    mean_solidity: float = 0.0  # Mean solidity of cells (detects under-segmentation)
    mean_circularity: float = 0.0  # Mean circularity of cells (detects over-segmentation)
    euler_integrity_pct: float = 0.0  # Percentage of cells with Euler number = 1 (no holes)
    fcr: float = 0.0  # Foreground Coverage Ratio (Σ Cell Area / ROI Area)
    cv_area: float = 0.0  # Coefficient of Variation of cell areas
    geometric_cv: float = 0.0  # Geometric CV (scale-independent measure)
    lognormal_pvalue: float = 0.0  # Shapiro-Wilk p-value for log-normal distribution test
    mean_eccentricity: float = 0.0  # Mean eccentricity of cells (detects elongation artifacts)

    @classmethod
    def get_objective_fields(cls) -> List[str]:
        """Get the list of objective fields (numeric metrics).
        
        Returns:
            List of field names that can be used as objectives.
        """
        # Manually specify known numeric fields to be safe and ordered
        # In the future, this could be dynamic using model inspection
        return [
            "num_cells", "mean_area",
            "mean_solidity", "mean_circularity", "euler_integrity_pct",
            "fcr", "cv_area", "geometric_cv", "lognormal_pvalue", "mean_eccentricity"
        ]
    
    @classmethod
    def get_objective_direction(cls, objective_name: str) -> Optional[bool]:
        """Get the default optimization direction for an objective.
        
        Args:
            objective_name: Name of the objective field
            
        Returns:
            True if should maximize, False if should minimize, None if undefined
        """
        # Default directions for common objectives
        maximize_objectives = {
            "num_cells",  # More cells is better (usually)
            "mean_solidity",  # Higher solidity = less under-segmentation
            "mean_circularity",  # Higher circularity = less over-segmentation
            "euler_integrity_pct",  # Higher = fewer holes
            "fcr",  # Higher = better coverage
            "lognormal_pvalue",  # Higher p-value = better fit to log-normal
        }
        
        minimize_objectives = {
            "cv_area",  # Lower CV = more uniform cell sizes
            "geometric_cv",  # Lower = more uniform
            "mean_eccentricity",  # Lower = less elongation
        }
        
        if objective_name in maximize_objectives:
            return True
        elif objective_name in minimize_objectives:
            return False
        else:
            return None  # Undefined (e.g., mean_area - depends on use case)

class ParameterSetEntry(BaseModel):
    """Represents a parameter set from a config file with its calculated statistics.
    
    Each entry represents a unique combination of parameter_set x image.
    """
    param_set_id: str  # The param_set_id from the config file
    image_path: Optional[str] = None  # Path to the image file (required for new entries)
    image_id: Optional[str] = None  # Image ID derived from image_path for easier identification
    config_file_path: Optional[str] = None  # Path to the config file (optional, for backward compatibility)
    parameters: Dict[str, float]  # The parameter values (diameter, min_size, flow_threshold, cellprob_threshold)
    result: Optional[OptimizationResult] = None  # Calculated statistics (None if not yet calculated)
    last_updated: Optional[datetime] = None  # When the result was last updated
    
    # Helper to get score safely
    @property
    def score(self) -> float:
        return self.result.score if self.result else 0.0
    
    def _derive_image_id(self) -> Optional[str]:
        """Derive image_id from image_path if not set."""
        if self.image_path and not self.image_id:
            import os
            image_id = os.path.splitext(os.path.basename(self.image_path))[0]
            if image_id.endswith('.ome'):
                image_id = image_id[:-4]
            return image_id
        return self.image_id
    
    def __init__(self, **data):
        super().__init__(**data)
        # Auto-derive image_id if image_path is provided
        if self.image_path and not self.image_id:
            self.image_id = self._derive_image_id()

class OptimizationStatus(BaseModel):
    """Current status of the optimization project."""
    state: Literal["SETUP", "RUNNING", "PAUSED", "COMPLETED"] = "SETUP"
    best_score: float = 0.0
    best_parameters: Optional[Dict[str, float]] = None
    best_config_file_path: Optional[str] = None
    best_param_set_id: Optional[str] = None

class SuggestedParameterPoint(BaseModel):
    """A suggested parameter point from LHS sampling (not yet converted to config)."""
    parameters: Dict[str, float]  # The parameter values
    suggested_at: datetime = Field(default_factory=datetime.now)

class OptimizationProject(BaseModel):
    """Root model for an optimization project.
    
    A project has:
    - An image pool: collection of images that belong to the project
    - Config files: can be included or excluded from the pool
    - Parameter ranges: search space for optimization
    - Suggested points: LHS-sampled parameter points not yet converted to configs
    """
    filepath: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Image pool: list of image pool entries
    image_pool: List[ImagePoolEntry] = Field(default_factory=list, description="Images in the project pool")
    
    # Config files: with include/exclude status
    config_files: List[ConfigFileInfo] = Field(default_factory=list, description="Config files associated with the project")
    
    # Suggested parameter points from LHS sampling
    suggested_points: List[SuggestedParameterPoint] = Field(default_factory=list, description="Suggested parameter points from LHS sampling")
    
    parameter_ranges: ParameterRanges = Field(default_factory=ParameterRanges)
    
    # State
    status: OptimizationStatus = Field(default_factory=OptimizationStatus)
    parameter_sets: List[ParameterSetEntry] = Field(default_factory=list, description="Parameter sets from config files with their calculated statistics")
    
    # Method configuration
    gp_kernel_type: str = Field(default="rbf", description="GP kernel type: 'rbf', 'matern', 'rational_quadratic', 'exp_sine_squared'")
    acquisition_method: str = Field(default="ucb", description="Acquisition function: 'ucb', 'ei', 'pi', 'lcb'")
    pareto_algorithm: str = Field(default="dominance", description="Pareto algorithm: 'dominance', 'nsga2', 'nsga3'")
    objective_directions: Dict[str, Optional[bool]] = Field(
        default_factory=dict,
        description="Optimization direction for each objective: True=maximize, False=minimize, None=undefined"
    )
    
    def save(self) -> None:
        """Save project to JSON file."""
        if not self.filepath:
            raise ValueError("Filepath not set for OptimizationProject")
        
        self.updated_at = datetime.now()
        
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(mode='json'), f, indent=2)
            
    @classmethod
    def load(cls, filepath: str) -> "OptimizationProject":
        """Load project from JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Optimization project file not found: {filepath}")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Backward compatibility: migrate old config_filepaths to config_files
        if 'config_filepaths' in data and 'config_files' not in data:
            config_filepaths = data.pop('config_filepaths', [])
            data['config_files'] = [ConfigFileInfo(filepath=path, included=True) for path in config_filepaths]
        
        # Backward compatibility: add created_at to config files that don't have it
        if 'config_files' in data:
            project_created_at = data.get('created_at')
            if isinstance(project_created_at, str):
                # Already in ISO format
                fallback_time = project_created_at
            elif isinstance(project_created_at, datetime):
                fallback_time = project_created_at.isoformat()
            else:
                fallback_time = datetime.now().isoformat()
            
            for cf_data in data['config_files']:
                if 'created_at' not in cf_data:
                    # Use project created_at as fallback for old config files
                    cf_data['created_at'] = fallback_time
        
        # Backward compatibility: migrate old image_pool (List[str]) to new format (List[ImagePoolEntry])
        if 'image_pool' in data and data['image_pool'] and isinstance(data['image_pool'][0], str):
            # Old format: list of strings
            image_paths = data['image_pool']
            data['image_pool'] = [ImagePoolEntry(filepath=path, is_active=True) for path in image_paths]
        elif 'image_pool' not in data:
            data['image_pool'] = []
        
        # Remove any old history field if present (no longer used)
        data.pop('history', None)
        
        # Ensure parameter_sets exists
        if 'parameter_sets' not in data:
            data['parameter_sets'] = []
        
        # Migration: Convert old format (config_file x param_set) to new format (param_set x image)
        # Old entries have config_file_path and param_set_id but no image_path
        # We need to scan config files to create per-image entries
        needs_migration = False
        old_parameter_sets = []
        for ps_data in data.get('parameter_sets', []):
            if 'image_path' not in ps_data or not ps_data.get('image_path'):
                needs_migration = True
                old_parameter_sets.append(ps_data)
        
        if needs_migration and old_parameter_sets:
            # Create project instance first (without parameter_sets) to access methods
            data_without_ps = data.copy()
            data_without_ps['parameter_sets'] = []
            temp_project = cls(**data_without_ps)
            temp_project.filepath = filepath
            
            # Try to migrate old entries by scanning config files
            try:
                from tui.optimization.optimizer import FileBasedOptimizer
                optimizer = FileBasedOptimizer(temp_project)
                
                # Scan config files to get all parameter_set x image combinations
                new_parameter_sets = optimizer.scan_config_files_for_parameter_sets()
                
                # Try to preserve existing results by matching old entries to new ones
                # Match by config_file_path and param_set_id, then assign to images
                for old_ps in old_parameter_sets:
                    old_config_path = old_ps.get('config_file_path')
                    old_param_set_id = old_ps.get('param_set_id')
                    old_result = old_ps.get('result')
                    old_last_updated = old_ps.get('last_updated')
                    
                    if old_config_path and old_param_set_id and old_result:
                        # Find matching new entries (same param_set_id, from same config)
                        for new_ps in new_parameter_sets:
                            if (new_ps.param_set_id == old_param_set_id and 
                                new_ps.config_file_path == old_config_path and
                                new_ps.image_path):
                                # Copy result to this image-specific entry
                                try:
                                    from tui.optimization.models import OptimizationResult
                                    result_obj = OptimizationResult(**old_result) if isinstance(old_result, dict) else old_result
                                    new_ps.result = result_obj
                                    if old_last_updated:
                                        from datetime import datetime
                                        if isinstance(old_last_updated, str):
                                            new_ps.last_updated = datetime.fromisoformat(old_last_updated)
                                        else:
                                            new_ps.last_updated = old_last_updated
                                except Exception as e:
                                    import logging
                                    logger = logging.getLogger(__name__)
                                    logger.warning(f"Could not migrate result for {old_param_set_id}: {e}")
                
                # Replace old parameter_sets with new ones
                data['parameter_sets'] = [ps.model_dump(mode='json') for ps in new_parameter_sets]
                
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Migration of parameter sets failed: {e}. Old entries will be preserved but may not work correctly.")
                # Keep old entries but mark them as needing migration
                for ps_data in data.get('parameter_sets', []):
                    if 'image_path' not in ps_data:
                        ps_data['_needs_migration'] = True
        
        # Initialize method configuration defaults if missing
        if 'gp_kernel_type' not in data:
            data['gp_kernel_type'] = 'rbf'
        if 'acquisition_method' not in data:
            data['acquisition_method'] = 'ucb'
        if 'pareto_algorithm' not in data:
            data['pareto_algorithm'] = 'dominance'
        
        # Initialize objective_directions with defaults if missing
        if 'objective_directions' not in data or not data['objective_directions']:
            data['objective_directions'] = {}
            for obj_name in OptimizationResult.get_objective_fields():
                default = OptimizationResult.get_objective_direction(obj_name)
                data['objective_directions'][obj_name] = default
            
        project = cls(**data)
        project.filepath = filepath # Ensure filepath is set correctly from load source
        return project
        
    def get_or_create_parameter_set(self, param_set_id: str, image_path: str, parameters: Dict[str, float], config_file_path: Optional[str] = None) -> ParameterSetEntry:
        """Get an existing parameter set or create a new one.
        
        Args:
            param_set_id: The parameter set ID
            image_path: Path to the image file
            parameters: The parameter values
            config_file_path: Optional path to the config file (for backward compatibility)
            
        Returns:
            ParameterSetEntry (existing or newly created)
        """
        # Normalize image path for comparison - handle both absolute and relative paths
        import os
        from src.file_paths import PROJECT_ROOT
        
        def normalize_for_comparison(path: str) -> str:
            """Normalize path for comparison, handling both absolute and relative."""
            if not path:
                return ""
            # Try to make absolute
            if not os.path.isabs(path):
                try:
                    abs_path = os.path.join(PROJECT_ROOT, path)
                    if os.path.exists(abs_path):
                        return os.path.normpath(os.path.abspath(abs_path))
                except:
                    pass
            # Use absolute path if available
            try:
                return os.path.normpath(os.path.abspath(path))
            except:
                return os.path.normpath(path)
        
        normalized_image_path = normalize_for_comparison(image_path) if image_path else None
        
        # Look for existing parameter set (using param_set_id x image_path as unique key)
        for ps in self.parameter_sets:
            if not ps.image_path:
                continue
            ps_normalized = normalize_for_comparison(ps.image_path)
            if ps.param_set_id == param_set_id and ps_normalized == normalized_image_path:
                # Update parameters in case config changed
                ps.parameters = parameters
                if config_file_path and not ps.config_file_path:
                    ps.config_file_path = config_file_path
                return ps
        
        # Create new parameter set
        param_set = ParameterSetEntry(
            param_set_id=param_set_id,
            image_path=image_path,
            parameters=parameters,
            config_file_path=config_file_path
        )
        self.parameter_sets.append(param_set)
        return param_set
        
    def update_parameter_set_result(self, param_set_id: str, image_path: str, result: OptimizationResult, config_file_path: Optional[str] = None):
        """Update the result for a specific parameter set.
        
        Args:
            param_set_id: The parameter set ID
            image_path: Path to the image file
            result: The calculated statistics
            config_file_path: Optional path to the config file (for backward compatibility)
        """
        import os
        normalized_image_path = os.path.normpath(image_path) if image_path else None
        
        for ps in self.parameter_sets:
            ps_normalized = os.path.normpath(ps.image_path) if ps.image_path else None
            if ps.param_set_id == param_set_id and ps_normalized == normalized_image_path:
                ps.result = result
                ps.last_updated = datetime.now()
                
                # Update best score if improved
                if result.score > self.status.best_score:
                    self.status.best_score = result.score
                    self.status.best_parameters = ps.parameters
                    self.status.best_param_set_id = param_set_id
                    self.status.best_config_file_path = config_file_path or ps.config_file_path
                break
    
    def get_config_files_by_order(self) -> List[ConfigFileInfo]:
        """Get config files sorted by creation order (oldest first)."""
        return sorted(self.config_files, key=lambda cf: cf.created_at)
    
    def get_statistics_for_param_set(self, param_set_id: str, image_path: str) -> Optional[OptimizationResult]:
        """Get statistics for a specific parameter set and image combination.
        
        Args:
            param_set_id: The parameter set ID
            image_path: Path to the image file
            
        Returns:
            OptimizationResult if found, None otherwise
        """
        import os
        normalized_image_path = os.path.normpath(image_path) if image_path else None
        
        for ps in self.parameter_sets:
            ps_normalized = os.path.normpath(ps.image_path) if ps.image_path else None
            if ps.param_set_id == param_set_id and ps_normalized == normalized_image_path:
                return ps.result
        return None
    
    def get_all_statistics(self) -> List[Dict]:
        """Get all statistics as a list of dictionaries.
        
        Returns:
            List of dicts with keys: param_set_id, image_path, image_id, config_file_path, parameters, result, last_updated
        """
        stats = []
        for ps in self.parameter_sets:
            stats.append({
                'param_set_id': ps.param_set_id,
                'image_path': ps.image_path,
                'image_id': ps.image_id,
                'config_file_path': ps.config_file_path,
                'parameters': ps.parameters,
                'result': ps.result.model_dump() if ps.result else None,
                'last_updated': ps.last_updated.isoformat() if ps.last_updated else None
            })
        return stats

    def get_filtered_parameter_sets(self) -> List[ParameterSetEntry]:
        """Get parameter sets from included config files and active images.
        
        Returns:
            List of ParameterSetEntry objects from included config files and active images
        """
        import os
        
        # Get included config file paths
        included_config_paths = {
            cf.filepath for cf in self.config_files if cf.included
        }
        
        # Get active image paths (normalized for comparison)
        active_image_paths = {
            os.path.normpath(entry.filepath) for entry in self.image_pool 
            if entry.is_active
        }
        
        filtered = []
        for ps in self.parameter_sets:
            # Filter by config file (if config_file_path is set)
            if ps.config_file_path and ps.config_file_path not in included_config_paths:
                continue
            
            # Filter by active images (if image_path is set)
            if ps.image_path:
                ps_normalized = os.path.normpath(ps.image_path)
                if active_image_paths and ps_normalized not in active_image_paths:
                    continue
            
            # For backward compatibility: if no image_path, include if config is included
            # (old format entries without image_path)
            filtered.append(ps)
        
        return filtered
    
    def _result_has_default_metrics(self, result: OptimizationResult) -> bool:
        """Check if a result has default/zero quality metrics (migrated from old format).
        
        Args:
            result: OptimizationResult to check
            
        Returns:
            True if result has default metrics (all quality metrics are 0.0)
        """
        # Check if all quality metrics are zero (default values)
        # This indicates the result was migrated from old format or calculation failed
        quality_metrics = [
            result.mean_solidity,
            result.mean_circularity,
            result.euler_integrity_pct,
            result.fcr,
            result.cv_area,
            result.geometric_cv,
            result.lognormal_pvalue,
            result.mean_eccentricity
        ]
        return all(m == 0.0 for m in quality_metrics)
    
    def is_result_valid_for_current_pool(self, param_set: ParameterSetEntry) -> bool:
        """Check if a stored result is valid for the current active image pool.
        
        Results are invalidated when the image pool changes, as they were calculated
        for a different set of images. This ensures plots only show data that matches
        the current table view.
        
        Args:
            param_set: ParameterSetEntry to check
            
        Returns:
            True if result is valid for current pool, False otherwise
        """
        if not param_set.result:
            return False
        
        # For now, we consider results valid if they exist
        # In the future, we could add image pool fingerprinting to track changes
        # TODO: Add image pool fingerprint to results to detect pool changes
        return True
    
    def get_objective_data(self, objective_name: str) -> List[Tuple[Dict[str, float], float, str]]:
        """Get data for a specific objective ready for GP regression.
        Only includes data from parameter sets in included config files and active images.
        Excludes results with default/zero quality metrics (migrated from old format).
        
        **CRITICAL**: Only includes results that are valid for the current active image pool.
        Each parameter_set x image combination is treated as a separate data point.
        
        Args:
            objective_name: Name of the objective field in OptimizationResult
            
        Returns:
            List of (parameters, objective_value, image_path) tuples where image_path is optional
        """
        data = []
        for ps in self.get_filtered_parameter_sets():
            # CRITICAL: Only use result if it's valid for current pool
            if ps.result and self.is_result_valid_for_current_pool(ps):
                has_defaults = self._result_has_default_metrics(ps.result)
                if not has_defaults:
                    val = getattr(ps.result, objective_name, None)
                    if val is not None and isinstance(val, (int, float)):
                        # Include image_path in the data (can be None for backward compatibility)
                        data.append((ps.parameters, float(val), ps.image_path or ""))
        return data
