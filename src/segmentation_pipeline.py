import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from cellpose import models, io
import json
import traceback
import time
from multiprocessing import Pool, cpu_count, freeze_support
import re
from src.tile_large_image import tile_image # Import the tiling function

# --- Global Configuration ---
IMAGE_DIR_BASE = "images" # Original images here
TILED_IMAGE_OUTPUT_BASE = os.path.join(IMAGE_DIR_BASE, "tiled_outputs") # Store generated tiles here
RESULTS_DIR_BASE = "results"
RUN_LOG_FILE = os.path.join(RESULTS_DIR_BASE, "run_log.json")
MAX_PARALLEL_PROCESSES = max(1, cpu_count() // 2)
#MAX_PARALLEL_PROCESSES = 1

def clean_filename_for_dir(filename):
    name_without_ext = os.path.splitext(filename)[0]
    cleaned_name = re.sub(r'[^\w\.-]', '_', name_without_ext)
    return cleaned_name

def segment_image_worker(job_params_dict):
    experiment_id_final = job_params_dict["experiment_id_final"]
    model_choice = job_params_dict["MODEL_CHOICE"]
    diameter_val = job_params_dict["DIAMETER"]
    flow_thresh_val = job_params_dict["FLOW_THRESHOLD"]
    min_size_val = job_params_dict["MIN_SIZE"]
    cellprob_thresh = job_params_dict["CELLPROB_THRESHOLD"]
    force_grayscale_flag = job_params_dict["FORCE_GRAYSCALE"]
    use_gpu_flag = job_params_dict["USE_GPU"]
    actual_image_path_to_process = job_params_dict["actual_image_path_to_process"]
    processing_unit_name = job_params_dict["processing_unit_name"]

    output_dir_job = os.path.join(RESULTS_DIR_BASE, experiment_id_final)

    if not os.path.exists(output_dir_job):
        try: os.makedirs(output_dir_job)
        except OSError as e:
            error_msg = f"[{experiment_id_final}] Error creating output dir {output_dir_job}: {e}"
            print(error_msg); return {**job_params_dict, "status": "failed", "error_message": error_msg}

    print(f"--- [{experiment_id_final}] Starting (Processing Unit: {processing_unit_name}) ---")
    log_diameter_eff = 0 if diameter_val is None else diameter_val
    log_flow_eff = 'Cellpose default' if flow_thresh_val is None else flow_thresh_val
    log_min_size_eff = 15 if min_size_val is None else min_size_val
    print(f"[{experiment_id_final}] Params: Model={model_choice}, GPU={use_gpu_flag}, Gray={force_grayscale_flag}, "
          f"CellProb={cellprob_thresh}, Diam(eff)={log_diameter_eff if log_diameter_eff!=0 else 'auto'}, "
          f"Flow(eff)={log_flow_eff}, MinSize(eff)={log_min_size_eff}")

    if not os.path.exists(actual_image_path_to_process):
        error_msg = f"[{experiment_id_final}] Error: Image not found: {actual_image_path_to_process}"
        print(error_msg); return {**job_params_dict, "status": "failed", "error_message": error_msg}

    try:
        print(f"[{experiment_id_final}] Loading: {actual_image_path_to_process}")
        img_for_cellpose = io.imread(actual_image_path_to_process)
        print(f"[{experiment_id_final}] Loaded. Shape: {img_for_cellpose.shape}, Dtype: {img_for_cellpose.dtype}")

        print(f"[{experiment_id_final}] Initializing Model: {model_choice}...")
        model = None
        try: model = models.CellposeModel(gpu=use_gpu_flag, model_type=model_choice)
        except AttributeError:
            print(f"[{experiment_id_final}] CellposeModel error, trying models.Cellpose...")
            model = models.Cellpose(gpu=use_gpu_flag, model_type=model_choice)
        if model is None: raise ValueError("Model init failed.")

        print(f"[{experiment_id_final}] Running segmentation...")
        eval_params = {"cellprob_threshold": cellprob_thresh}
        eval_params["diameter"] = 0 if diameter_val is None else diameter_val
        if flow_thresh_val is not None: eval_params["flow_threshold"] = flow_thresh_val
        eval_params["min_size"] = 15 if min_size_val is None else min_size_val
        if force_grayscale_flag: eval_params["channels"] = [0,0]
        
        masks, flows, styles = model.eval(img_for_cellpose, **eval_params)
        print(f"[{experiment_id_final}] Seg done. Masks: {masks.shape}, Unique (top 20): {np.unique(masks)[:20]}")
        if eval_params['diameter'] == 0:
             if hasattr(model, 'sz_estimate') and model.sz_estimate is not None: print(f"[{experiment_id_final}] Cellpose size est: {model.sz_estimate:.2f}")
             elif hasattr(model, 'diam_mean'): print(f"[{experiment_id_final}] Model training diam: {model.diam_mean:.2f}")

        base_output_filename = os.path.splitext(processing_unit_name)[0]
        mask_filename = os.path.join(output_dir_job, base_output_filename + "_mask.tif")
        io.imsave(mask_filename, masks.astype(np.uint16))
        print(f"[{experiment_id_final}] Mask saved: {mask_filename}")

        num_cells = masks.max()
        print(f"[{experiment_id_final}] Found {num_cells} cells.")
        coordinates_list = []
        if num_cells > 0:
            for i in range(1, num_cells + 1):
                cell_pixels = (masks == i)
                if np.any(cell_pixels):
                    y, x = np.where(cell_pixels)
                    coordinates_list.append({"cell_id": int(i), "centroid_x": float(f"{np.mean(x):.2f}"), "centroid_y": float(f"{np.mean(y):.2f}")})
        
        coord_json_filename = os.path.join(output_dir_job, base_output_filename + "_coords.json")
        with open(coord_json_filename, 'w') as f_json: json.dump(coordinates_list, f_json, indent=4)
        print(f"[{experiment_id_final}] Coords saved: {coord_json_filename}")
        
        return {**job_params_dict, "status": "succeeded", "num_cells": int(num_cells), "output_mask_path": mask_filename}

    except Exception as e:
        error_full_msg = f"Error in {experiment_id_final} (Unit: {processing_unit_name}): {e} {traceback.format_exc()}"
        print(error_full_msg)
        if os.path.exists(output_dir_job):
            with open(os.path.join(output_dir_job, "error_log.txt"), "w") as f_err: f_err.write(error_full_msg)
            print(f"[{experiment_id_final}] Detailed error saved to error_log.txt")
        return {**job_params_dict, "status": "failed", "error_message": error_full_msg}

if __name__ == "__main__":
    freeze_support()
    for base_dir in [IMAGE_DIR_BASE, TILED_IMAGE_OUTPUT_BASE, RESULTS_DIR_BASE]:
        if not os.path.exists(base_dir):
            try: os.makedirs(base_dir); print(f"Created directory: {base_dir}.")
            except OSError as e: print(f"Fatal: Could not create {base_dir}: {e}"); exit()

    param_file = "parameter_sets.json"
    if not os.path.exists(param_file): print(f"Error: Config file '{param_file}' not found."); exit()

    config_data = {}
    try:
        with open(param_file, 'r') as f: config_data = json.load(f)
    except Exception as e: print(f"Error reading/parsing {param_file}: {e}"); exit()

    image_configs = config_data.get("image_configurations", [])
    cellpose_param_configs = config_data.get("cellpose_parameter_configurations", [])

    if not image_configs: print("No 'image_configurations' in JSON."); exit()
    if not cellpose_param_configs: print("No 'cellpose_parameter_configurations' in JSON."); exit()

    all_jobs_to_create = []
    for img_config in image_configs:
        if not img_config.get("is_active", True):
            print(f"Skipping inactive image configuration: {img_config.get('image_id', 'UnnamedImageConfig')}")
            continue

        original_image_filename = img_config.get("original_image_filename")
        image_id_from_config = img_config.get("image_id", clean_filename_for_dir(original_image_filename) if original_image_filename else "unknown_image")
        
        if not original_image_filename:
            print(f"Warning: Image config '{image_id_from_config}' missing 'original_image_filename'. Skipping.")
            continue
        
        original_image_path = os.path.join(IMAGE_DIR_BASE, original_image_filename)
        if not os.path.exists(original_image_path):
            print(f"Warning: Original image {original_image_path} for config '{image_id_from_config}' not found. Skipping.")
            continue

        images_to_process_for_this_img_config = [] 

        tiling_cfg = img_config.get("tiling_config")
        if tiling_cfg and tiling_cfg.get("tile_size"):
            print(f"Tiling configured for {original_image_filename} (Image ID: {image_id_from_config}).")
            # Define a specific output directory for tiles of this original image
            tile_storage_dir = os.path.join(TILED_IMAGE_OUTPUT_BASE, image_id_from_config + "_tiles")
            tile_prefix = tiling_cfg.get("tile_output_prefix_base", clean_filename_for_dir(original_image_filename) + "_tile")
            
            if not os.path.exists(tile_storage_dir):
                 try: os.makedirs(tile_storage_dir); print(f"Created tile storage directory: {tile_storage_dir}")
                 except OSError as e: print(f"Error creating tile storage dir {tile_storage_dir}: {e}"); continue

            tile_manifest_data = tile_image(
                original_image_path, 
                tile_storage_dir, # Tiles are saved here
                tile_size=tiling_cfg["tile_size"],
                overlap=tiling_cfg.get("overlap", 100),
                output_prefix=tile_prefix
            )
            if tile_manifest_data and tile_manifest_data.get("tiles"):
                for tile_info in tile_manifest_data["tiles"]:
                    images_to_process_for_this_img_config.append({
                        "path": tile_info["path"], 
                        "name": tile_info["filename"], 
                        "original_image_id": image_id_from_config,
                        "original_image_filename": original_image_filename,
                        "is_tile": True,
                        "tile_info": tile_info # Store full tile info if needed later for stitching
                    })
                print(f"Generated {len(tile_manifest_data['tiles'])} tiles for {original_image_filename}.")
            else:
                print(f"Warning: Tiling failed or produced no tiles for {original_image_filename}. Will attempt to process as full image.")
                images_to_process_for_this_img_config.append({
                    "path": original_image_path, "name": original_image_filename,
                    "original_image_id": image_id_from_config, "original_image_filename": original_image_filename,
                    "is_tile": False
                })
        else: 
            images_to_process_for_this_img_config.append({
                "path": original_image_path, "name": original_image_filename,
                "original_image_id": image_id_from_config, "original_image_filename": original_image_filename,
                "is_tile": False
            })

        for img_proc_info in images_to_process_for_this_img_config:
            for cp_config in cellpose_param_configs:
                if not cp_config.get("is_active", True):
                    continue
                
                param_set_id = cp_config.get("param_set_id", "params_unknown")
                job = {}
                for key, value in cp_config.items():
                    if key not in ["param_set_id", "is_active"]:
                        job[key] = value
                
                job["actual_image_path_to_process"] = img_proc_info["path"]
                job["processing_unit_name"] = img_proc_info["name"]
                job["original_image_id_for_log"] = img_proc_info["original_image_id"]
                job["original_image_filename_for_log"] = img_proc_info["original_image_filename"]
                job["is_tile_for_log"] = img_proc_info["is_tile"]
                if img_proc_info["is_tile"]:
                    job["tile_details_for_log"] = img_proc_info["tile_info"]

                cleaned_processing_unit_name = clean_filename_for_dir(img_proc_info["name"]) # Used for unique folder part
                if img_proc_info["is_tile"]:
                    job["experiment_id_final"] = f"{img_proc_info['original_image_id']}_{param_set_id}_{cleaned_processing_unit_name}"
                else: 
                    job["experiment_id_final"] = f"{img_proc_info['original_image_id']}_{param_set_id}"

                job.setdefault("MODEL_CHOICE", "cyto3")
                job.setdefault("DIAMETER", None)
                job.setdefault("FLOW_THRESHOLD", None)
                job.setdefault("MIN_SIZE", 15) 
                job.setdefault("CELLPROB_THRESHOLD", 0.0)
                job.setdefault("FORCE_GRAYSCALE", True)
                job.setdefault("USE_GPU", False)
                
                all_jobs_to_create.append(job)

    if not all_jobs_to_create: print("No active jobs to process after expansion."); exit()
    print(f"Total active jobs to process: {len(all_jobs_to_create)}")
        
    num_processes_to_use = MAX_PARALLEL_PROCESSES
    any_gpu_run = any(job.get("USE_GPU", False) for job in all_jobs_to_create)
    if any_gpu_run and MAX_PARALLEL_PROCESSES > 1:
        print("WARNING: GPU usage with MAX_PARALLEL_PROCESSES > 1. Consider setting to 1 for stability.")

    print(f"Using up to {num_processes_to_use} parallel processes.")
    start_time_all = time.time()
    job_results = []

    if num_processes_to_use > 1 and len(all_jobs_to_create) > 1:
        with Pool(processes=num_processes_to_use) as pool:
            job_results = pool.map(segment_image_worker, all_jobs_to_create)
    else:
        print("Running jobs sequentially.")
        for job_params in all_jobs_to_create: job_results.append(segment_image_worker(job_params))

    try:
        with open(RUN_LOG_FILE, 'w') as f_log: json.dump(job_results, f_log, indent=4)
        print(f"Run log saved to: {RUN_LOG_FILE}")
    except Exception as e: print(f"Error saving run log: {e}")

    end_time_all = time.time()
    print(f"--- All Jobs Finished ---")
    total_duration = end_time_all - start_time_all
    print(f"Total processing time for {len(all_jobs_to_create)} jobs: {total_duration:.2f} seconds.")

    successful_runs, failed_runs = 0, 0
    for result in job_results:
        if result.get("status") == "succeeded": successful_runs += 1
        else: failed_runs += 1
    print(f"Summary: {successful_runs} successful, {failed_runs} failed out of {len(all_jobs_to_create)} jobs.")
    if failed_runs > 0: print("Check experiment folders and 'error_log.txt' for details.")

