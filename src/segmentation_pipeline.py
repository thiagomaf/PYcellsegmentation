# src/segmentation_pipeline.py
import os
import json
import time
import argparse
import logging
from multiprocessing import Pool, cpu_count, freeze_support

from .pipeline_config_parser import load_and_expand_configurations
from .segmentation_worker import segment_image_worker
from .file_paths import (
    IMAGE_DIR_BASE, RESCALED_IMAGE_CACHE_DIR, TILED_IMAGE_OUTPUT_BASE, RESULTS_DIR_BASE, PROJECT_ROOT
) # Import from new file_paths, added PROJECT_ROOT for config path

# --- Global Configuration (mostly paths now) ---
# Paths are now imported from file_paths.py
# IMAGE_DIR_BASE = "images"
# RESCALED_IMAGE_CACHE_DIR = os.path.join(IMAGE_DIR_BASE, "rescaled_cache")
# TILED_IMAGE_OUTPUT_BASE = os.path.join(IMAGE_DIR_BASE, "tiled_outputs")
# RESULTS_DIR_BASE = "results"

# --- Logger Setup ---
logger = logging.getLogger(__name__)

def setup_logging(log_level_str="INFO"): # Default to INFO string
    # Convert string log level to logging constant
    level = getattr(logging, str(log_level_str).upper(), logging.INFO)
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger.info(f"Logging initialized at level: {logging.getLevelName(level)}")

def ensure_directories(base_dirs):
    logger.info("Ensuring base directories exist...")
    for base_dir_name in base_dirs:
        if not os.path.exists(base_dir_name):
            try:
                os.makedirs(base_dir_name)
                logger.info(f"Created directory: {base_dir_name}")
            except OSError as e:
                logger.error(f"Fatal: Could not create base directory {base_dir_name}: {e}")
                exit(1)
        else:
            logger.debug(f"Directory already exists: {base_dir_name}")

def run_segmentation_jobs(jobs_to_run, num_processes):
    if not jobs_to_run:
        logger.info("No active jobs to process based on configuration. Exiting.")
        return []

    logger.info(f"Total active jobs to process after expansion: {len(jobs_to_run)}")

    any_gpu_run = any(job.get("USE_GPU", False) for job in jobs_to_run)
    if any_gpu_run and num_processes > 1:
        logger.warning("GPU usage detected with num_processes > 1. "
                       "Consider setting num_processes = 1 for stability if using a single GPU.")

    logger.info(f"Attempting to use up to {num_processes} parallel processes for job execution.")
    logger.info("--------------------------------------------------------------------")
    
    job_results_list = []
    if num_processes > 1 and len(jobs_to_run) > 1:
        with Pool(processes=num_processes) as pool:
            job_results_list = pool.map(segment_image_worker, jobs_to_run)
    else:
        logger.info("Running jobs sequentially (due to num_processes=1 or only 1 job).")
        for job_params in jobs_to_run:
            job_results_list.append(segment_image_worker(job_params))
    return job_results_list

def save_run_log(results, log_file_path):
    logger.info(f"Attempting to save run log to: {log_file_path}")
    try:
        with open(log_file_path, 'w') as f_log:
            json.dump(results, f_log, indent=4)
        logger.info(f"Run log saved successfully to: {log_file_path}")
    except Exception as e:
        logger.error(f"Error saving run log to {log_file_path}: {e}")

def summarize_job_results(job_results_list, total_jobs_configured):
    successful_runs, failed_runs = 0, 0
    logger.info("--- Individual Job Status Summary ---")
    for i, result_dict in enumerate(job_results_list):
        if result_dict is None:
            logger.error(f"  Job {i+1}/{total_jobs_configured}: Error - Worker returned None (unexpected).")
            failed_runs += 1
            continue
        
        status = result_dict.get("status", "unknown")
        exp_id = result_dict.get("experiment_id", f"unknown_job_{i+1}")
        unit_name = result_dict.get("unit", "unknown_image")
        
        if status == "success" or status == "success_segmentation_skipped":
            num_cells_found = result_dict.get('num_cells', 'N/A')
            duration_str = f"(Duration: {result_dict.get('duration_seconds', -1):.2f}s)" if result_dict.get('duration_seconds') is not None else ""
            
            success_message_parts = []
            if status == "success_segmentation_skipped":
                success_message_parts.append("SUCCEEDED (segmentation skipped, mask found)")
            else:
                success_message_parts.append("SUCCEEDED (segmentation performed)")
            
            if num_cells_found != 'N/A':
                 success_message_parts.append(f"(Found {num_cells_found} cells)")
            
            if duration_str:
                success_message_parts.append(duration_str)
            
            full_success_message = " ".join(success_message_parts)
            logger.info(f"  Job {i+1}/{total_jobs_configured}: {exp_id} (Unit: {unit_name}) - {full_success_message}")
            mask_path = result_dict.get("mask_path")
            if mask_path: logger.info(f"    Mask path: {mask_path}")
            summary_path = result_dict.get("summary_path")
            if summary_path: logger.info(f"    Summary JSON: {summary_path}")
            successful_runs += 1
        elif status == "error" or status == "error_mask_not_found":
            error_detail = result_dict.get("error", "No error detail provided.")
            logger.error(f"  Job {i+1}/{total_jobs_configured}: {exp_id} (Unit: {unit_name}) - FAILED")
            failed_runs += 1
    
    logger.info("--- Overall Batch Summary ---")
    logger.info(f"  Total Jobs Attempted: {total_jobs_configured}") # Based on initial count from config
    logger.info(f"  Jobs Processed by Workers: {len(job_results_list)}") # Actual worker results
    logger.info(f"  Successful Jobs     : {successful_runs}")
    logger.info(f"  Failed Jobs         : {failed_runs}")
    logger.info("====================================================================")
    if failed_runs > 0:
        logger.warning("Check individual experiment folders and 'error_log.txt' files for details on failures.")
        logger.warning("Also review the full console output above for detailed tracebacks if errors occurred outside worker.")
    logger.info("====================================================================")
    return successful_runs, failed_runs

def main():
    freeze_support() # For multiprocessing on Windows
    
    parser = argparse.ArgumentParser(description="Image Segmentation Pipeline")
    parser.add_argument("--config", default="parameter_sets.json",
                        help="Path to the parameter sets JSON file (default: parameter_sets.json relative to project root).")
    parser.add_argument("--max_processes", type=int, default=None, # Changed default to None to detect if set
                        help="Maximum number of parallel processes to run. "
                             "Overrides JSON config. Default from JSON or 1 if not set anywhere. "
                             "Be cautious with >1 if using GPU.")
    parser.add_argument("--log_level", type=str, default=None, # Changed from --verbose
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level (e.g., DEBUG, INFO, WARNING). Overrides JSON config.")
    args = parser.parse_args()

    # Determine config file path (absolute or relative to project root)
    config_file_path = args.config
    if not os.path.isabs(config_file_path):
        config_file_path = os.path.join(PROJECT_ROOT, config_file_path)

    # Default log level and max_processes
    effective_log_level = "INFO" # Hardcoded default for log level
    hardcoded_default_max_processes = 1
    effective_max_processes = hardcoded_default_max_processes # Hardcoded default for max_processes
    # Default for GPU usage
    effective_use_gpu = False # Default to False

    # Try to load default_log_level and max_processes from JSON config
    pipeline_params = {} # Initialize to empty dict
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r') as f:
                pipeline_params = json.load(f)
            global_settings = pipeline_params.get("global_segmentation_settings", {})
            
            # Handle log level from JSON
            json_log_level = global_settings.get("default_log_level")
            if json_log_level and isinstance(json_log_level, str) and json_log_level.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                effective_log_level = json_log_level.upper()
            
            # Handle max_processes from JSON
            json_max_processes = global_settings.get("max_processes")
            if json_max_processes is not None and isinstance(json_max_processes, int) and json_max_processes > 0:
                effective_max_processes = json_max_processes
            elif json_max_processes is not None: # Invalid value in JSON
                 print(f"[STARTUP_WARNING] Invalid 'max_processes' value ({json_max_processes}) in {config_file_path}. Using default/CLI.")

            # Handle USE_GPU_IF_AVAILABLE from JSON
            json_use_gpu = global_settings.get("USE_GPU_IF_AVAILABLE")
            if isinstance(json_use_gpu, bool):
                effective_use_gpu = json_use_gpu


        except Exception as e:
            # Can't use logger yet, print a startup error if config parsing for log level fails
            print(f"[STARTUP_WARNING] Could not read settings from {config_file_path}: {e}. Using fallback settings.")
    else:
        print(f"[STARTUP_WARNING] Config file {config_file_path} not found. Using fallback settings.")


    # Command-line argument for log_level overrides JSON and hardcoded default
    if args.log_level:
        effective_log_level = args.log_level.upper()

    # Command-line argument for max_processes overrides JSON and hardcoded default
    if args.max_processes is not None:
        if args.max_processes > 0:
            effective_max_processes = args.max_processes
        else: # Invalid CLI value
            print(f"[STARTUP_WARNING] Invalid '--max_processes' value ({args.max_processes}). Using default/JSON.")
            # effective_max_processes remains what it was (JSON or hardcoded default)

    setup_logging(effective_log_level) # Setup logging with the determined level

    logger.info("====================================================================")
    logger.info(f" INITIALIZING BATCH SEGMENTATION RUN @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f" Using configuration file: {config_file_path}") # Use resolved path
    logger.info(f" Effective logging level: {effective_log_level}")
    logger.info(f" Effective max parallel processes: {effective_max_processes}") # Updated log message
    logger.info(f" Effective GPU usage intent: {effective_use_gpu}") # Log GPU setting
    logger.info("====================================================================")

    # run_log_file path depends on RESULTS_DIR_BASE which is now imported
    run_log_file = os.path.join(RESULTS_DIR_BASE, "run_log.json")
    
    # base_directories list uses imported constants
    base_directories_to_ensure = [IMAGE_DIR_BASE, TILED_IMAGE_OUTPUT_BASE, RESCALED_IMAGE_CACHE_DIR, RESULTS_DIR_BASE]
    ensure_directories(base_directories_to_ensure)

    active_jobs_to_run = load_and_expand_configurations(config_file_path, effective_use_gpu) # Pass GPU setting
    initial_job_count = len(active_jobs_to_run)

    if not active_jobs_to_run:
        logger.info("No active jobs found in configuration. Exiting.")
        exit(0)
        
    start_time_all = time.time()
    
    job_results_list = run_segmentation_jobs(active_jobs_to_run, effective_max_processes) # Use effective_max_processes
    
    save_run_log(job_results_list, run_log_file)

    end_time_all = time.time()
    total_duration = end_time_all - start_time_all
    
    logger.info("====================================================================")
    logger.info(f" ALL JOBS FINISHED @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("====================================================================")
    logger.info(f"Total processing time for {initial_job_count} configured jobs: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes).")

    summarize_job_results(job_results_list, initial_job_count)


if __name__ == "__main__":
    main()

