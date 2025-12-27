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
    # #region agent log
    try:
        import json, time
        with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A,B,C,E","location":"stats.py:calculate_stats_for_config","message":"ENTRY","data":{"config_path":config_path,"param_set_id":param_set_id},"timestamp":time.time()*1000})+"\n")
    except: pass
    # #endregion
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
    
    # #region agent log
    try:
        import json, time
        with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"stats.py:calculate_stats_for_config","message":"BEFORE imports from src","data":{"config_path":config_path,"param_set_id":param_set_id},"timestamp":time.time()*1000})+"\n")
    except: pass
    # #endregion
    # #region agent log
    try:
        import json, time
        with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"stats.py:calculate_stats_for_config","message":"BEFORE_IMPORT_pipeline_config_parser","data":{"config_path":config_path,"param_set_id":param_set_id},"timestamp":time.time()*1000})+"\n")
    except: pass
    # #endregion
    from src.pipeline_config_parser import load_and_expand_configurations
    from src.file_paths import RESULTS_DIR_BASE, PROJECT_ROOT
    # #region agent log
    try:
        import json, time
        with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"stats.py:calculate_stats_for_config","message":"AFTER_IMPORT_pipeline_config_parser","data":{"config_path":config_path,"param_set_id":param_set_id,"has_function":hasattr(load_and_expand_configurations, '__call__')},"timestamp":time.time()*1000})+"\n")
    except: pass
    # #endregion
    # #region agent log
    try:
        import json, time
        with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"stats.py:calculate_stats_for_config","message":"AFTER imports from src","data":{"config_path":config_path,"param_set_id":param_set_id},"timestamp":time.time()*1000})+"\n")
    except: pass
    # #endregion
    
    # #region agent log
    try:
        import json, time
        config_exists = os.path.exists(config_path)
        with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"stats.py:calculate_stats_for_config","message":"BEFORE config file check","data":{"config_path":config_path,"config_exists":config_exists,"param_set_id":param_set_id},"timestamp":time.time()*1000})+"\n")
    except: pass
    # #endregion
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        # #region agent log
        try:
            import json, time
            with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"stats.py:calculate_stats_for_config","message":"RETURN None - config not found","data":{"config_path":config_path},"timestamp":time.time()*1000})+"\n")
        except: pass
        # #endregion
        return None

    try:
        # Check config structure before calling load_and_expand_configurations
        # #region agent log
        try:
            import json, time
            with open(config_path, 'r') as f:
                config_preview = json.load(f)
            image_configs_count = len(config_preview.get("image_configurations", []))
            active_image_configs = sum(1 for img in config_preview.get("image_configurations", []) if img.get("is_active", True))
            param_configs_count = len(config_preview.get("cellpose_parameter_configurations", []))
            active_param_configs = sum(1 for p in config_preview.get("cellpose_parameter_configurations", []) if p.get("is_active", True))
            matching_param_configs = sum(1 for p in config_preview.get("cellpose_parameter_configurations", []) if p.get("is_active", True) and p.get("param_set_id") == param_set_id)
            with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"stats.py:calculate_stats_for_config","message":"BEFORE load_and_expand_configurations - config structure","data":{"config_path":config_path,"param_set_id":param_set_id,"image_configs_count":image_configs_count,"active_image_configs":active_image_configs,"param_configs_count":param_configs_count,"active_param_configs":active_param_configs,"matching_param_configs":matching_param_configs},"timestamp":time.time()*1000})+"\n")
        except Exception as e:
            # #region agent log
            try:
                import json, time
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"stats.py:calculate_stats_for_config","message":"ERROR reading config preview","data":{"config_path":config_path,"error":str(e)},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
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
            # #region agent log
            try:
                import json, time
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"stats.py:calculate_stats_for_config","message":"EXCEPTION_IN_LOAD_AND_EXPAND","data":{"config_path":config_path,"error":str(e),"traceback":traceback.format_exc()},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            jobs = []
        jobs_before_filter = len(jobs) if jobs else 0
        # #region agent log
        try:
            import json, time
            job_param_set_ids = [job.get("param_set_id_for_log") for job in jobs[:10]] if jobs else []  # Sample first 10
            with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"stats.py:calculate_stats_for_config","message":"AFTER load_and_expand_configurations","data":{"jobs_count":jobs_before_filter,"param_set_id":param_set_id,"sample_param_set_ids":job_param_set_ids},"timestamp":time.time()*1000})+"\n")
        except: pass
        # #endregion
        
        if not jobs:
            logger.warning(f"No jobs generated from config: {config_path}")
            # #region agent log
            try:
                import json, time
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"stats.py:calculate_stats_for_config","message":"RETURN None - no jobs","data":{"config_path":config_path},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
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
                # #region agent log
                try:
                    import json, time
                    with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A,C","location":"stats.py:calculate_stats_for_config","message":"param_set_id filter check","data":{"param_set_id":param_set_id,"job_param_set_id":job_param_set_id,"match":job_param_set_id == param_set_id},"timestamp":time.time()*1000})+"\n")
                except: pass
                # #endregion
                if job_param_set_id != param_set_id:
                    continue
            
            # Filter by active image paths from project pool if provided
            if active_image_paths:
                # Get the image path from the job
                # The job dictionary uses "original_image_filename_for_log" (see pipeline_config_parser.py line 317)
                original_image_filename = job.get("original_image_filename_for_log", "")
                if not original_image_filename:
                    continue
                
                # Normalize job image path
                if not os.path.isabs(original_image_filename):
                    original_image_filename = os.path.join(PROJECT_ROOT, original_image_filename)
                normalized_job_path = os.path.normpath(original_image_filename)
                
                # Check if this image is in the active image pool
                image_in_pool = False
                for active_path in active_image_paths:
                    # Normalize active path using SAME logic
                    normalized_active = os.path.normpath(
                        os.path.join(PROJECT_ROOT, active_path) if not os.path.isabs(active_path) else active_path
                    )
                    if normalized_active == normalized_job_path:
                        image_in_pool = True
                        break
                
                if not image_in_pool:
                    logger.debug(f"Skipping job - image not in active pool: {normalized_job_path}")
                    continue  # Skip this job - image not in project's active pool
            
            jobs_after_filter += 1
            
            experiment_id = job.get("experiment_id_final")
            processing_unit_name = job.get("processing_unit_name")
            
            # #region agent log
            try:
                import json, time
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"stats.py:calculate_stats_for_config","message":"PROCESSING_JOB","data":{"param_set_id":param_set_id,"job_param_set_id":job.get("param_set_id_for_log"),"experiment_id":experiment_id,"processing_unit_name":processing_unit_name},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            
            if not experiment_id or not processing_unit_name:
                # #region agent log
                try:
                    import json, time
                    with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"stats.py:calculate_stats_for_config","message":"SKIP_JOB_NO_EXPERIMENT_ID","data":{"param_set_id":param_set_id,"job_param_set_id":job.get("param_set_id_for_log")},"timestamp":time.time()*1000})+"\n")
                except: pass
                # #endregion
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
            
            # #region agent log
            try:
                import json, time
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"stats.py:calculate_stats_for_config","message":"CHECKING_MASK_PATH","data":{"param_set_id":param_set_id,"experiment_id":experiment_id,"processing_unit_name":processing_unit_name,"mask_path":mask_path,"mask_exists":os.path.exists(mask_path)},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            
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
                    # #region agent log
                    try:
                        import json, time
                        with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"stats.py:calculate_stats_for_config","message":"MASK_READ_ERROR","data":{"param_set_id":param_set_id,"mask_path":mask_path,"error":str(e)},"timestamp":time.time()*1000})+"\n")
                    except: pass
                    # #endregion
            else:
                # If a mask is missing, we might consider the run incomplete
                # But for now, let's just log it. The optimizer might keep waiting or accept partial results.
                logger.debug(f"Mask not found: {mask_path}")
                # #region agent log
                try:
                    import json, time
                    with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                        # Check if the directory exists
                        mask_dir = os.path.dirname(mask_path)
                        dir_exists = os.path.exists(mask_dir)
                        dir_contents = []
                        if dir_exists:
                            try:
                                dir_contents = os.listdir(mask_dir)[:10]  # First 10 files
                            except:
                                pass
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"stats.py:calculate_stats_for_config","message":"MASK_NOT_FOUND","data":{"param_set_id":param_set_id,"mask_path":mask_path,"mask_dir":mask_dir,"dir_exists":dir_exists,"dir_contents":dir_contents},"timestamp":time.time()*1000})+"\n")
                except: pass
                # #endregion

        if masks_found == 0:
            # #region agent log
            try:
                import json, time
                with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A,E","location":"stats.py:calculate_stats_for_config","message":"RETURN None - masks_found==0","data":{"config_path":config_path,"param_set_id":param_set_id,"jobs_before_filter":jobs_before_filter,"jobs_after_filter":jobs_after_filter},"timestamp":time.time()*1000})+"\n")
            except: pass
            # #endregion
            return None

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
        # #region agent log
        try:
            import json, time
            with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A,B,C,E","location":"stats.py:calculate_stats_for_config","message":"RETURN result","data":{"num_cells":total_cells,"mean_area":avg_mean_area,"masks_found":masks_found,"param_set_id":param_set_id},"timestamp":time.time()*1000})+"\n")
        except: pass
        # #endregion
        return result

    except Exception as e:
        import traceback
        # #region agent log
        try:
            import json, time
            with open(r"c:\Users\Thiago\My Drive\Github\PYcellsegmentation\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"stats.py:calculate_stats_for_config","message":"EXCEPTION","data":{"error":str(e),"error_type":type(e).__name__,"traceback":traceback.format_exc(),"config_path":config_path,"param_set_id":param_set_id},"timestamp":time.time()*1000})+"\n")
        except: pass
        # #endregion
        logger.error(f"Error calculating stats for {config_path}: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        # Re-raise the exception so the caller can see what went wrong
        # This will be caught by the dashboard and shown to the user
        raise

