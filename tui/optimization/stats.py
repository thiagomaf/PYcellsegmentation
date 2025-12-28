import os
import logging
from typing import Optional, List, Dict
from tui.optimization.models import OptimizationResult

logger = logging.getLogger(__name__)

def calculate_quality_metrics(mask, roi_area: Optional[float] = None) -> Optional[Dict[str, float]]:
    """
    Calculate quality metrics for a segmentation mask.
    
    Based on research-backed metrics for plant cell segmentation evaluation:
    - Shape metrics (solidity, circularity, eccentricity)
    - Topology metrics (Euler number integrity)
    - Coverage metrics (FCR)
    - Distribution metrics (CV, geometric CV, log-normal fit)
    
    Args:
        mask: Segmentation mask array (numpy array with labeled regions)
        roi_area: Optional ROI area for FCR calculation. If None, FCR will be -1.0
        
    Returns:
        Dictionary with quality metrics, or None if no cells found
    """
    try:
        import numpy as np
        from scipy import stats
        from skimage.measure import regionprops
    except ImportError as e:
        missing_module = str(e).replace("No module named ", "").strip("'\"")
        error_msg = f"Missing dependency for quality metrics: {missing_module}. Please install it with: pip install {missing_module}"
        logger.error(error_msg)
        return None
    
    props = regionprops(mask)
    if len(props) == 0:
        return None  # No cells found
    
    # Extract per-cell properties
    areas = np.array([p.area for p in props])
    solidities = np.array([p.solidity for p in props])
    perimeters = np.array([p.perimeter for p in props])
    eccentricities = np.array([p.eccentricity for p in props])
    euler_numbers = np.array([p.euler_number for p in props])
    
    # 1. Mean Solidity - detects under-segmentation (merged cells create non-convex shapes)
    mean_solidity = float(np.mean(solidities))
    
    # 2. Mean Circularity - detects over-segmentation and irregular boundaries
    # Circularity = 4π·Area/Perimeter², 1.0 = perfect circle
    circularities = np.array([
        4 * np.pi * p.area / (p.perimeter ** 2) if p.perimeter > 0 else 0.0
        for p in props
    ])
    mean_circularity = float(np.mean(circularities))
    
    # 3. Euler Integrity - percentage of cells with χ=1 (no holes)
    # For plant cells, we expect χ=1 (one object, zero holes)
    euler_integrity_pct = 100.0 * float(np.sum(euler_numbers == 1) / len(props))
    
    # 4. Foreground Coverage Ratio (FCR) - detects gaps or overlaps
    # FCR = Σ Cell Area / ROI Area, should be ~0.90-1.00
    if roi_area is not None and roi_area > 0:
        fcr = float(np.sum(areas) / roi_area)
    else:
        fcr = -1.0  # Sentinel value indicating ROI not available
    
    # 5. Coefficient of Variation (CV) of Area - detects segmentation instability
    if np.mean(areas) > 0:
        cv_area = float(np.std(areas) / np.mean(areas))
    else:
        cv_area = 0.0
    
    # 6. Geometric CV - scale-independent measure for log-normal populations
    # More appropriate for multiplicative growth processes
    log_areas = np.log(areas[areas > 0])  # Only positive areas
    if len(log_areas) > 0:
        sigma_log = float(np.std(log_areas))
        geometric_cv = float(np.sqrt(np.exp(sigma_log ** 2) - 1))
    else:
        geometric_cv = 0.0
    
    # 7. Log-Normal p-value - tests biological plausibility
    # Plant cell sizes typically follow log-normal distribution
    # High p-value (>0.05) indicates biologically plausible distribution
    if len(log_areas) >= 3:  # Shapiro-Wilk requires at least 3 samples
        try:
            _, lognormal_pvalue = stats.shapiro(log_areas)
            lognormal_pvalue = float(lognormal_pvalue)
        except Exception as e:
            logger.debug(f"Shapiro-Wilk test failed: {e}")
            lognormal_pvalue = 0.0
    else:
        lognormal_pvalue = 0.0
    
    # 8. Mean Eccentricity - detects elongation artifacts
    # 0 = circle, 1 = line
    mean_eccentricity = float(np.mean(eccentricities))
    
    return {
        'mean_solidity': mean_solidity,
        'mean_circularity': mean_circularity,
        'euler_integrity_pct': euler_integrity_pct,
        'fcr': fcr,
        'cv_area': cv_area,
        'geometric_cv': geometric_cv,
        'lognormal_pvalue': lognormal_pvalue,
        'mean_eccentricity': mean_eccentricity
    }

def calculate_stats_for_config(
    config_path: str, 
    param_set_id: Optional[str] = None,
    active_image_paths: Optional[set] = None
) -> Optional[OptimizationResult]:
    """
    Calculate statistics for a given configuration file by inspecting the output results.
    Only processes images that are in the active_image_paths set (if provided).
    
    Args:
        config_path: Path to the configuration file used for the run.
        param_set_id: Optional parameter set ID to filter results. If provided, only jobs
                     with this param_set_id will be included in the statistics.
        active_image_paths: Optional set of active image file paths from the project's image pool.
                          If provided, only jobs for these images will be included.
        
    Returns:
        OptimizationResult object if results are found and valid, None otherwise.
    """
    # Lazy import heavy dependencies to avoid slow startup
    missing_deps = []
    try:
        import numpy as np
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import tifffile
    except ImportError:
        missing_deps.append("tifffile")
    
    if missing_deps:
        missing_str = " and ".join(missing_deps)
        install_cmd = "pip install " + " ".join(missing_deps)
        error_msg = f"{missing_str} {'is' if len(missing_deps) == 1 else 'are'} required for calculating stats but {'is' if len(missing_deps) == 1 else 'are'} not installed. Please install {'it' if len(missing_deps) == 1 else 'them'} with: {install_cmd}"
        logger.error(f"Cannot calculate stats: {error_msg}")
        raise ImportError(error_msg)
    
    # Check for cv2 (opencv-python) which is needed by pipeline_config_parser
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
        error_msg = "opencv-python is required for calculating stats but is not installed. Please install it with: pip install opencv-python"
        logger.error(f"Cannot calculate stats: {error_msg}")
        raise ImportError(error_msg)
    
    from src.pipeline_config_parser import load_and_expand_configurations
    from src.file_paths import RESULTS_DIR_BASE, PROJECT_ROOT
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return None

    try:
        # Check config structure before calling load_and_expand_configurations
        # #endregion
        
        # Expand the configuration to get the list of expected jobs
        # We pass global_use_gpu_if_available=False as we only need the job details (paths), not to run them
        # We pass skip_image_file_check=True because we don't need image files to exist for stats calculation
        try:
            jobs = load_and_expand_configurations(config_path, global_use_gpu_if_available=False, skip_image_file_check=True)
        except Exception as e:
            import traceback
            logger.error(f"Exception in load_and_expand_configurations: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            jobs = []
        jobs_before_filter = len(jobs) if jobs else 0
        
        if not jobs:
            logger.warning(f"No jobs generated from config: {config_path}")
            return None

        total_cells = 0
        total_area = 0.0
        cell_counts = []
        masks_found = 0
        jobs_after_filter = 0
        
        # Aggregators for quality metrics (weighted by cell count)
        quality_metrics_sum = {
            'mean_solidity': 0.0,
            'mean_circularity': 0.0,
            'euler_integrity_pct': 0.0,
            'fcr': 0.0,
            'cv_area': 0.0,
            'geometric_cv': 0.0,
            'lognormal_pvalue': 0.0,
            'mean_eccentricity': 0.0
        }
        quality_metrics_count = 0  # Number of masks with valid quality metrics

        for job in jobs:
            # Filter by param_set_id if provided
            if param_set_id is not None:
                job_param_set_id = job.get("param_set_id_for_log")
                if job_param_set_id != param_set_id:
                    continue
            
            # Filter by active image paths from project pool if provided
            if active_image_paths:
                # Get the image path from the job
                # The job dictionary uses "original_image_filename_for_log" (see pipeline_config_parser.py line 317)
                original_image_filename = job.get("original_image_filename_for_log", "")
                if not original_image_filename:
                    continue
                
                # Normalize job image path - use absolute path for reliable comparison
                if not os.path.isabs(original_image_filename):
                    job_abs_path = os.path.normpath(os.path.join(PROJECT_ROOT, original_image_filename))
                else:
                    job_abs_path = os.path.normpath(original_image_filename)
                
                # Also try relative path for comparison
                try:
                    job_rel_path = os.path.relpath(job_abs_path, PROJECT_ROOT)
                except ValueError:
                    job_rel_path = None
                
                # Check if this image is in the active image pool
                # Compare against all normalized formats in active_image_paths
                image_in_pool = False
                for active_path in active_image_paths:
                    # Normalize active path to absolute for comparison
                    if os.path.isabs(active_path):
                        active_abs = os.path.normpath(active_path)
                    else:
                        active_abs = os.path.normpath(os.path.join(PROJECT_ROOT, active_path))
                    
                    # Compare absolute paths (most reliable)
                    if active_abs == job_abs_path:
                        image_in_pool = True
                        break
                    
                    # Also try relative path comparison if available
                    if job_rel_path:
                        try:
                            active_rel = os.path.relpath(active_abs, PROJECT_ROOT)
                            if active_rel == job_rel_path:
                                image_in_pool = True
                                break
                        except ValueError:
                            pass
                    
                    # Fallback: compare original strings (case-insensitive on Windows)
                    if os.path.normpath(active_path).lower() == os.path.normpath(original_image_filename).lower():
                        image_in_pool = True
                        break
                
                if not image_in_pool:
                    logger.debug(f"Skipping job - image not in active pool: {job_abs_path} (looking for one of {active_image_paths})")
                    continue  # Skip this job - image not in project's active pool
            
            jobs_after_filter += 1
            
            experiment_id = job.get("experiment_id_final")
            processing_unit_name = job.get("processing_unit_name")
            
            
            if not experiment_id or not processing_unit_name:
                continue

            # Construct the expected mask path
            # The mask filename logic should match segmentation_worker.py/pipeline_utils.py
            # segmentation_worker.py: mask_filename_part = f"{os.path.splitext(processing_unit_name)[0]}_mask.tif"
            # pipeline_utils.py: construct_mask_path handles this
            
            # We construct the path manually here based on known constants to be safe, 
            # or use the helper from pipeline_utils if strictly consistent.
            # Using simple join here as we know the structure: RESULTS_DIR_BASE / experiment_id / filename_mask.tif
            
            base_name = os.path.splitext(processing_unit_name)[0]
            mask_filename = f"{base_name}_mask.tif"
            mask_path = os.path.join(RESULTS_DIR_BASE, experiment_id, mask_filename)
            
            
            # If exact mask path not found, try searching for variant directories
            # (similar to has_output_files logic - check directories that start with base experiment ID)
            if not os.path.exists(mask_path):
                mask_found_in_variant = False
                try:
                    # Construct base experiment ID: image_id_param_set_id (without scale/tile suffixes)
                    image_id = job.get("original_image_id_for_log", "")
                    if image_id and param_set_id:
                        base_experiment_id = f"{image_id}_{param_set_id}"
                        # Search for directories that start with base_experiment_id
                        if os.path.exists(RESULTS_DIR_BASE):
                            for dirname in os.listdir(RESULTS_DIR_BASE):
                                if dirname.startswith(base_experiment_id):
                                    variant_dir = os.path.join(RESULTS_DIR_BASE, dirname)
                                    if os.path.isdir(variant_dir):
                                        # Check if mask file exists in this variant directory
                                        variant_mask_path = os.path.join(variant_dir, mask_filename)
                                        if os.path.exists(variant_mask_path):
                                            # Found mask in variant directory - use it
                                            mask_path = variant_mask_path
                                            mask_found_in_variant = True
                                            break
                except Exception as e:
                    logger.debug(f"Error searching variant directories: {e}")
                
                if not mask_found_in_variant:
                    # If a mask is missing, we might consider the run incomplete
                    # But for now, let's just log it. The optimizer might keep waiting or accept partial results.
                    logger.debug(f"Mask not found: {mask_path}")
                    continue  # Skip this job - no mask found
            
            # Process the mask (either from exact path or variant directory)
            if os.path.exists(mask_path):
                try:
                    mask_data = tifffile.imread(mask_path)
                    
                    # Count cells (assuming background is 0)
                    # For integer masks, max label is usually the count if labels are sequential
                    # But unique - 1 is safer
                    unique_labels = np.unique(mask_data)
                    num_cells = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
                    
                    if num_cells > 0:
                        # efficient area calculation: count non-zero pixels
                        # This gives total area of all cells. Mean area = total_area / num_cells
                        area_pixels = np.count_nonzero(mask_data)
                        
                        total_cells += num_cells
                        total_area += area_pixels
                        cell_counts.append(num_cells)
                        
                        # Calculate quality metrics for this mask
                        # For FCR, we could use the mask bounding box or actual image size
                        # For now, we'll use the mask size as ROI approximation
                        roi_area_approx = float(mask_data.size)  # Total pixels in mask
                        try:
                            quality_metrics = calculate_quality_metrics(mask_data, roi_area=roi_area_approx)
                            
                            if quality_metrics:
                                # Weight metrics by cell count (more cells = more reliable)
                                weight = num_cells
                                for key in quality_metrics_sum:
                                    if key in quality_metrics:
                                        quality_metrics_sum[key] += quality_metrics[key] * weight
                                quality_metrics_count += weight
                            else:
                                logger.debug(f"calculate_quality_metrics returned None for mask {mask_path}")
                        except Exception as e:
                            logger.warning(f"Error calculating quality metrics for {mask_path}: {e}", exc_info=True)
                    
                    masks_found += 1
                    
                except Exception as e:
                    logger.error(f"Error reading mask {mask_path}: {e}")

        if masks_found == 0:
            return None
        
        # Log how many jobs were processed to help debug aggregation issues
        if active_image_paths:
            logger.debug(f"Processed {jobs_after_filter} jobs for {len(active_image_paths)} image path(s), found {masks_found} mask(s)")

        # Aggregate stats
        # Mean area across all processed images (weighted by cell count essentially)
        # If we want mean cell area: total_area_of_all_cells / total_number_of_cells
        avg_mean_area = (total_area / total_cells) if total_cells > 0 else 0.0
        
        # Aggregate quality metrics (weighted average by cell count)
        if quality_metrics_count > 0:
            aggregated_metrics = {
                key: quality_metrics_sum[key] / quality_metrics_count
                for key in quality_metrics_sum
            }
            logger.debug(f"Aggregated quality metrics: {aggregated_metrics}, count={quality_metrics_count}")
        else:
            # Default values if no valid metrics - this should not happen if masks were processed
            logger.warning(f"No quality metrics calculated (quality_metrics_count=0, masks_found={masks_found}, total_cells={total_cells})")
            aggregated_metrics = {
                'mean_solidity': 0.0,
                'mean_circularity': 0.0,
                'euler_integrity_pct': 0.0,
                'fcr': -1.0,  # Sentinel value
                'cv_area': 0.0,
                'geometric_cv': 0.0,
                'lognormal_pvalue': 0.0,
                'mean_eccentricity': 0.0
            }
        
        # Create result object with all quality metrics
        # Log a warning if we processed masks but got no quality metrics (indicates a problem)
        if masks_found > 0 and total_cells > 0 and quality_metrics_count == 0:
            logger.warning(
                f"Processed {masks_found} mask(s) with {total_cells} total cells, "
                f"but quality_metrics_count=0. This may indicate an issue with calculate_quality_metrics()."
            )
        
        # Compute composite score from quality metrics
        # Higher score = better segmentation quality
        # Uses weighted combination of key metrics
        if quality_metrics_count > 0:
            # Normalize metrics to 0-1 range and combine
            # Solidity (0-1): higher is better, weight 0.3
            solidity_score = aggregated_metrics['mean_solidity']
            
            # Euler integrity (0-100%): higher is better, weight 0.2
            euler_score = aggregated_metrics['euler_integrity_pct'] / 100.0
            
            # Circularity (0-1): higher is better (but can be low for irregular cells), weight 0.15
            circularity_score = aggregated_metrics['mean_circularity']
            
            # FCR (0.9-1.0 ideal): penalize if too far from 1.0, weight 0.2
            if aggregated_metrics['fcr'] > 0:
                fcr_score = 1.0 - abs(aggregated_metrics['fcr'] - 1.0)  # Penalize deviation from 1.0
                fcr_score = max(0.0, min(1.0, fcr_score))
            else:
                fcr_score = 0.5  # Neutral if FCR not available
            
            # Log-normal p-value (>0.05 is good): weight 0.15
            lognorm_score = min(1.0, aggregated_metrics['lognormal_pvalue'] / 0.05) if aggregated_metrics['lognormal_pvalue'] > 0 else 0.0
            
            # Composite score: weighted average
            composite_score = (
                0.30 * solidity_score +
                0.20 * euler_score +
                0.15 * circularity_score +
                0.20 * fcr_score +
                0.15 * lognorm_score
            )
        else:
            # No quality metrics available - use default score
            composite_score = 0.0
        
        result = OptimizationResult(
            score=composite_score,
            num_cells=total_cells,
            mean_area=avg_mean_area,
            **aggregated_metrics
        )
        return result

    except Exception as e:
        import traceback
        logger.error(f"Error calculating stats for {config_path}: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        # Re-raise the exception so the caller can see what went wrong
        # This will be caught by the dashboard and shown to the user
        raise

def calculate_stats_for_image(
    config_path: str,
    param_set_id: str,
    image_path: str
) -> Optional[OptimizationResult]:
    """
    Calculate statistics for a single image with a specific parameter set.
    
    This function calculates stats for one image only, unlike calculate_stats_for_config
    which aggregates across multiple images. Each parameter_set x image combination
    gets its own result.
    
    Args:
        config_path: Path to the configuration file used for the run
        param_set_id: Parameter set ID to filter results
        image_path: Path to the specific image file to calculate stats for (can be absolute or relative)
        
    Returns:
        OptimizationResult object if results are found and valid, None otherwise
    """
    import os
    from src.file_paths import PROJECT_ROOT
    
    # Normalize image path - try both absolute and relative formats
    # The job's original_image_filename_for_log might be in either format
    normalized_abs = None
    normalized_rel = None
    
    if os.path.isabs(image_path):
        normalized_abs = os.path.normpath(image_path)
        # Try to get relative path too
        try:
            normalized_rel = os.path.relpath(image_path, PROJECT_ROOT)
        except ValueError:
            normalized_rel = None
    else:
        # Try relative to PROJECT_ROOT
        normalized_rel = os.path.normpath(image_path)
        normalized_abs = os.path.normpath(os.path.join(PROJECT_ROOT, image_path))
    
    # Create set with both formats to ensure matching
    # calculate_stats_for_config will normalize and compare, so we provide both possibilities
    active_image_paths = set()
    if normalized_abs:
        active_image_paths.add(normalized_abs)
    if normalized_rel:
        active_image_paths.add(normalized_rel)
    # Also add original in case it matches exactly
    active_image_paths.add(image_path)
    
    # Ensure we're only calculating for a single image
    # If active_image_paths has multiple entries, they should all refer to the same image (just different path formats)
    # But log a warning if we have too many distinct paths
    if len(active_image_paths) > 3:  # Allow for abs, rel, and original formats
        logger.warning(f"calculate_stats_for_image called with {len(active_image_paths)} path formats for image_path={image_path}. This might indicate a path normalization issue.")
    
    result = calculate_stats_for_config(
        config_path=config_path,
        param_set_id=param_set_id,
        active_image_paths=active_image_paths if active_image_paths else None
    )
    
    # Log the result to help debug if scores are being shared incorrectly
    if result:
        logger.debug(f"Calculated stats for image {image_path}, param_set {param_set_id}: score={result.score:.4f}, cells={result.num_cells}")
    
    return result

