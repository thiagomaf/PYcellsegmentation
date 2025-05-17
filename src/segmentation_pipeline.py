# src/segmentation_pipeline.py
import os
import json
import time
from multiprocessing import Pool, cpu_count, freeze_support
# Relative imports for our new modules
from .pipeline_config_parser import load_and_expand_configurations
from .segmentation_worker import segment_image_worker
# pipeline_utils and tile_large_image are used by pipeline_config_parser

# --- Global Configuration (mostly paths now) ---
IMAGE_DIR_BASE = "images" 
RESCALED_IMAGE_CACHE_DIR = os.path.join(IMAGE_DIR_BASE, "rescaled_cache")
TILED_IMAGE_OUTPUT_BASE = os.path.join(IMAGE_DIR_BASE, "tiled_outputs")
RESULTS_DIR_BASE = "results"
RUN_LOG_FILE = os.path.join(RESULTS_DIR_BASE, "run_log.json")
PARAMETER_SETS_JSON_FILE = "parameter_sets.json" # Path relative to project root

MAX_PARALLEL_PROCESSES = max(1, cpu_count() // 2)
#MAX_PARALLEL_PROCESSES = 1 # Safer option if GPU is involved or for debugging

if __name__ == "__main__":
    freeze_support() # For multiprocessing on Windows
    
    print("====================================================================")
    print(f" INITIALIZING BATCH SEGMENTATION RUN @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("====================================================================")

    # Ensure necessary base directories exist (worker also checks its output dir)
    for base_dir_name in [IMAGE_DIR_BASE, TILED_IMAGE_OUTPUT_BASE, RESCALED_IMAGE_CACHE_DIR, RESULTS_DIR_BASE]:
        if not os.path.exists(base_dir_name):
            try:
                os.makedirs(base_dir_name)
                print(f"Created directory: {base_dir_name}")
            except OSError as e:
                print(f"Fatal: Could not create base directory {base_dir_name}: {e}")
                exit(1)

    active_jobs_to_run = load_and_expand_configurations(PARAMETER_SETS_JSON_FILE)

    if not active_jobs_to_run:
        print("No active jobs to process based on configuration. Exiting.")
        exit(0)
    
    print(f"Total active jobs to process after expansion: {len(active_jobs_to_run)}")
        
    num_processes_to_use = MAX_PARALLEL_PROCESSES
    any_gpu_run = any(job.get("USE_GPU", False) for job in active_jobs_to_run)
    if any_gpu_run and MAX_PARALLEL_PROCESSES > 1:
        print("WARNING: GPU usage detected with MAX_PARALLEL_PROCESSES > 1. "
              "Consider setting MAX_PARALLEL_PROCESSES = 1 at the top of this script "
              "for stability if using a single GPU.")

    print(f"Attempting to use up to {num_processes_to_use} parallel processes for job execution.")
    print("--------------------------------------------------------------------")
    start_time_all = time.time()
    
    job_results_list = [] 
    if num_processes_to_use > 1 and len(active_jobs_to_run) > 1:
        with Pool(processes=num_processes_to_use) as pool:
            job_results_list = pool.map(segment_image_worker, active_jobs_to_run)
    else:
        print("Running jobs sequentially (due to MAX_PARALLEL_PROCESSES=1 or only 1 job).")
        for job_params in active_jobs_to_run:
            job_results_list.append(segment_image_worker(job_params))

    try:
        with open(RUN_LOG_FILE, 'w') as f_log:
            json.dump(job_results_list, f_log, indent=4)
        print(f"Run log saved to: {RUN_LOG_FILE}")
    except Exception as e:
        print(f"Error saving run log: {e}")

    end_time_all = time.time()
    print("====================================================================")
    print(f" ALL JOBS FINISHED @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("====================================================================")
    total_duration = end_time_all - start_time_all
    print(f"Total processing time for {len(active_jobs_to_run)} jobs: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes).")

    successful_runs, failed_runs = 0, 0
    print("--- Individual Job Status Summary ---")
    for i, result_dict in enumerate(job_results_list): 
        if result_dict is None: 
            print(f"  Job {i+1}: Error - Worker returned None (unexpected).")
            failed_runs +=1
            continue
        
        status = result_dict.get("status", "unknown")
        exp_id = result_dict.get("experiment_id_final", f"unknown_job_{i+1}")
        unit_name = result_dict.get("processing_unit_name", "unknown_image")
        
        if status == "succeeded":
            num_cells_found = result_dict.get('num_cells', 'N/A')
            print(f"  Job {i+1}/{len(active_jobs_to_run)}: {exp_id} (Unit: {unit_name}) - SUCCEEDED (Found {num_cells_found} cells)")
            successful_runs +=1
        else:
            error_msg_short = result_dict.get('message','Unknown error') 
            print(f"  Job {i+1}/{len(active_jobs_to_run)}: {exp_id} (Unit: {unit_name}) - FAILED ({error_msg_short})")
            failed_runs +=1
    
    print("--- Overall Batch Summary ---")
    print(f"  Total Jobs Attempted: {len(active_jobs_to_run)}")
    print(f"  Successful Jobs     : {successful_runs}")
    print(f"  Failed Jobs         : {failed_runs}")
    print("====================================================================")
    if failed_runs > 0:
        print("Check individual experiment folders and 'error_log.txt' files for details on failures.")
        print("Also review the full console output above for detailed tracebacks if errors occurred outside worker.")
    print("====================================================================")

