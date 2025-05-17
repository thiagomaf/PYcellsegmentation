import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from cellpose import models, io
import json
import traceback
import time
from multiprocessing import Pool, cpu_count, freeze_support
import re

# --- Global Configuration ---
IMAGE_DIR_BASE = "images"
RESULTS_DIR_BASE = "results"
RUN_LOG_FILE = os.path.join(RESULTS_DIR_BASE, "run_log.json")
MAX_PARALLEL_PROCESSES = max(1, cpu_count() // 2)
#MAX_PARALLEL_PROCESSES = 1

def clean_filename_for_dir(filename):
    name_without_ext = os.path.splitext(filename)[0]
    cleaned_name = re.sub(r'[^\w\.-]', '_', name_without_ext)
    return cleaned_name

def segment_image_worker(job_params_dict):
    experiment_id = job_params_dict["experiment_id"]
    model_choice = job_params_dict["MODEL_CHOICE"]
    diameter_val = job_params_dict["DIAMETER"]
    flow_thresh_val = job_params_dict["FLOW_THRESHOLD"]
    min_size_val = job_params_dict["MIN_SIZE"]
    cellprob_thresh = job_params_dict["CELLPROB_THRESHOLD"]
    force_grayscale_flag = job_params_dict["FORCE_GRAYSCALE"]
    use_gpu_flag = job_params_dict["USE_GPU"]
    image_filename = job_params_dict["image_filename"]

    image_path = os.path.join(IMAGE_DIR_BASE, image_filename)
    output_dir_experiment = os.path.join(RESULTS_DIR_BASE, experiment_id)

    if not os.path.exists(output_dir_experiment):
        try: os.makedirs(output_dir_experiment)
        except OSError as e:
            error_msg = f"[{experiment_id}] Error creating output dir {output_dir_experiment}: {e}"
            print(error_msg)
            return {**job_params_dict, "status": "failed", "error_message": error_msg}

    print(f"--- [{experiment_id}] Starting (Image: {image_filename}) ---")
    log_diameter_eff = 0 if diameter_val is None else diameter_val
    log_flow_eff = 'Cellpose default' if flow_thresh_val is None else flow_thresh_val
    log_min_size_eff = 15 if min_size_val is None else min_size_val
    print(f"[{experiment_id}] Params: Model={model_choice}, GPU={use_gpu_flag}, Gray={force_grayscale_flag}, "
          f"CellProb={cellprob_thresh}, Diam(eff)={log_diameter_eff if log_diameter_eff!=0 else 'auto'}, "
          f"Flow(eff)={log_flow_eff}, MinSize(eff)={log_min_size_eff}")

    if not os.path.exists(image_path):
        error_msg = f"[{experiment_id}] Error: Image not found: {image_path}"
        print(error_msg)
        return {**job_params_dict, "status": "failed", "error_message": error_msg}

    try:
        print(f"[{experiment_id}] Loading: {image_path}")
        img_for_cellpose = io.imread(image_path)
        print(f"[{experiment_id}] Loaded. Shape: {img_for_cellpose.shape}, Dtype: {img_for_cellpose.dtype}")

        print(f"[{experiment_id}] Initializing Model: {model_choice}...")
        model = None
        try: model = models.CellposeModel(gpu=use_gpu_flag, model_type=model_choice)
        except AttributeError:
            print(f"[{experiment_id}] CellposeModel error, trying models.Cellpose...")
            model = models.Cellpose(gpu=use_gpu_flag, model_type=model_choice)
        if model is None: raise ValueError("Model init failed.")

        print(f"[{experiment_id}] Running segmentation...")
        eval_params = {"cellprob_threshold": cellprob_thresh}
        eval_params["diameter"] = 0 if diameter_val is None else diameter_val
        if flow_thresh_val is not None: eval_params["flow_threshold"] = flow_thresh_val
        eval_params["min_size"] = 15 if min_size_val is None else min_size_val
        if force_grayscale_flag: eval_params["channels"] = [0,0]
        
        masks, flows, styles = model.eval(img_for_cellpose, **eval_params)
        print(f"[{experiment_id}] Segmentation done. Masks shape: {masks.shape}, Unique (top 20): {np.unique(masks)[:20]}")
        if eval_params['diameter'] == 0:
             if hasattr(model, 'sz_estimate') and model.sz_estimate is not None: print(f"[{experiment_id}] Cellpose size estimate: {model.sz_estimate:.2f}")
             elif hasattr(model, 'diam_mean'): print(f"[{experiment_id}] Model training diameter: {model.diam_mean:.2f}")

        mask_filename = os.path.join(output_dir_experiment, os.path.splitext(image_filename)[0] + "_mask.tif")
        io.imsave(mask_filename, masks.astype(np.uint16))
        print(f"[{experiment_id}] Mask saved: {mask_filename}")

        num_cells = masks.max()
        print(f"[{experiment_id}] Found {num_cells} cells.")
        coordinates_list = []
        if num_cells > 0:
            for i in range(1, num_cells + 1):
                cell_pixels = (masks == i)
                if np.any(cell_pixels):
                    y, x = np.where(cell_pixels)
                    coordinates_list.append({"cell_id": int(i), "centroid_x": float(f"{np.mean(x):.2f}"), "centroid_y": float(f"{np.mean(y):.2f}")})
        
        coord_json_filename = os.path.join(output_dir_experiment, os.path.splitext(image_filename)[0] + "_coords.json")
        with open(coord_json_filename, 'w') as f_json: json.dump(coordinates_list, f_json, indent=4)
        print(f"[{experiment_id}] Coords saved: {coord_json_filename}")
        
        return {**job_params_dict, "status": "succeeded", "num_cells": int(num_cells)}

    except Exception as e:
        error_full_msg = f"Error in {experiment_id} (Image: {image_filename}): {e}{traceback.format_exc()}"
        print(error_full_msg)
        if os.path.exists(output_dir_experiment):
            with open(os.path.join(output_dir_experiment, "error_log.txt"), "w") as f_err: f_err.write(error_full_msg)
            print(f"[{experiment_id}] Detailed error saved to error_log.txt")
        return {**job_params_dict, "status": "failed", "error_message": error_full_msg}

if __name__ == "__main__":
    freeze_support()
    for base_dir in [IMAGE_DIR_BASE, RESULTS_DIR_BASE]:
        if not os.path.exists(base_dir):
            try: os.makedirs(base_dir); print(f"Created directory: {base_dir}.")
            except OSError as e: print(f"Fatal: Could not create {base_dir}: {e}"); exit()

    param_file = "parameter_sets.json"
    if not os.path.exists(param_file): print(f"Error: Config file '{param_file}' not found."); exit()

    config_data = {}
    try:
        with open(param_file, 'r') as f: config_data = json.load(f)
    except Exception as e: print(f"Error reading/parsing {param_file}: {e}"); exit()

    global_image_list = config_data.get("global_image_filename_list", [])
    parameter_configurations = config_data.get("parameter_configurations", [])

    if not global_image_list: print("No images in 'global_image_filename_list' in JSON."); exit()
    if not parameter_configurations: print("No 'parameter_configurations' in JSON."); exit()

    jobs_to_create = []
    for config_entry in parameter_configurations:
        if not config_entry.get("is_active", True):
            print(f"Skipping inactive parameter configuration: {config_entry.get('experiment_id_base', 'Unnamed_Config')}")
            continue

        base_id = config_entry.get("experiment_id_base", f"params_{time.strftime('%Y%m%d%H%M%S')}")
        for img_fn in global_image_list:
            job = {} 
            for key, value in config_entry.items():
                if key != "experiment_id_base" and key != "is_active":
                    job[key] = value
            
            job["image_filename"] = img_fn
            job["experiment_id"] = f"{base_id}_{clean_filename_for_dir(img_fn)}"
            job.setdefault("MODEL_CHOICE", "cyto3")
            job.setdefault("DIAMETER", None)
            job.setdefault("FLOW_THRESHOLD", None)
            job.setdefault("MIN_SIZE", None)
            job.setdefault("CELLPROB_THRESHOLD", 0.0)
            job.setdefault("FORCE_GRAYSCALE", True)
            job.setdefault("USE_GPU", False)
            
            jobs_to_create.append(job)
            
    if not jobs_to_create: print("No active jobs to process after expansion."); exit()
    print(f"Expanded to {len(jobs_to_create)} active jobs across all images and active parameter configurations.")
        
    num_processes_to_use = MAX_PARALLEL_PROCESSES
    any_gpu_run = any(job.get("USE_GPU", False) for job in jobs_to_create)
    if any_gpu_run and MAX_PARALLEL_PROCESSES > 1:
        print("WARNING: GPU usage with MAX_PARALLEL_PROCESSES > 1. Consider setting to 1 for stability.")

    print(f"Using up to {num_processes_to_use} parallel processes.")
    start_time_all = time.time()
    job_results = []

    if num_processes_to_use > 1 and len(jobs_to_create) > 1:
        with Pool(processes=num_processes_to_use) as pool:
            job_results = pool.map(segment_image_worker, jobs_to_create)
    else:
        print("Running jobs sequentially.")
        for job_params in jobs_to_create: job_results.append(segment_image_worker(job_params))

    try:
        with open(RUN_LOG_FILE, 'w') as f_log: json.dump(job_results, f_log, indent=4)
        print(f"Run log saved to: {RUN_LOG_FILE}")
    except Exception as e: print(f"Error saving run log: {e}")

    end_time_all = time.time()
    print(f"--- All Jobs Finished ---")
    total_duration = end_time_all - start_time_all
    print(f"Total processing time for {len(jobs_to_create)} jobs: {total_duration:.2f} seconds.")

    successful_runs, failed_runs = 0, 0
    for result in job_results:
        if result.get("status") == "succeeded": successful_runs += 1
        else: failed_runs += 1
    print(f"Summary: {successful_runs} successful, {failed_runs} failed out of {len(jobs_to_create)} jobs.")
    if failed_runs > 0: print("Check experiment folders and 'error_log.txt' for details.")
