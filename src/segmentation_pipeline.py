import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import cv2 # For resizing
from cellpose import models, io
import json
import traceback
import time
from multiprocessing import Pool, cpu_count, freeze_support
import re
from .tile_large_image import tile_image

# --- Global Configuration ---
IMAGE_DIR_BASE = "images" 
RESCALED_IMAGE_CACHE_DIR = os.path.join(IMAGE_DIR_BASE, "rescaled_cache") # Store rescaled images
TILED_IMAGE_OUTPUT_BASE = os.path.join(IMAGE_DIR_BASE, "tiled_outputs") 
RESULTS_DIR_BASE = "results"
RUN_LOG_FILE = os.path.join(RESULTS_DIR_BASE, "run_log.json")
MAX_PARALLEL_PROCESSES = max(1, cpu_count() // 2)
#MAX_PARALLEL_PROCESSES = 1

def clean_filename_for_dir(filename):
    name_without_ext = os.path.splitext(filename)[0]
    cleaned_name = re.sub(r'[^\w\.-]', '_', name_without_ext)
    return cleaned_name

def get_cv2_interpolation_method(method_str="INTER_AREA"):
    methods = {
        "INTER_NEAREST": cv2.INTER_NEAREST,
        "INTER_LINEAR": cv2.INTER_LINEAR,
        "INTER_AREA": cv2.INTER_AREA, 
        "INTER_CUBIC": cv2.INTER_CUBIC, 
        "INTER_LANCZOS4": cv2.INTER_LANCZOS4
    }
    return methods.get(method_str.upper(), cv2.INTER_AREA)

def rescale_image_and_save(original_image_path, image_id, rescaling_config):
    if not rescaling_config or "scale_factor" not in rescaling_config:
        return original_image_path, 1.0

    scale_factor = rescaling_config["scale_factor"]
    interpolation_str = rescaling_config.get("interpolation", "INTER_AREA")
    interpolation_method = get_cv2_interpolation_method(interpolation_str)

    rescaled_dir = os.path.join(RESCALED_IMAGE_CACHE_DIR, image_id)
    if not os.path.exists(rescaled_dir):
        try: os.makedirs(rescaled_dir)
        except OSError: pass 

    original_filename = os.path.basename(original_image_path)
    rescaled_image_filename = f"{os.path.splitext(original_filename)[0]}_scaled_{scale_factor:.2f}.tif"
    rescaled_image_path = os.path.join(rescaled_dir, rescaled_image_filename)

    if os.path.exists(rescaled_image_path):
        print(f"  Found cached rescaled image: {rescaled_image_path}")
        return rescaled_image_path, scale_factor

    try:
        print(f"  Rescaling {original_image_path} by factor {scale_factor} using {interpolation_str}...")
        img = cv2.imread(original_image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if img is None:
            print(f"  Error: Could not load original image {original_image_path} for rescaling.")
            return original_image_path, 1.0 

        new_width = int(img.shape[1] * scale_factor)
        new_height = int(img.shape[0] * scale_factor)
        
        if new_width == 0 or new_height == 0:
            print(f"  Error: Rescaled dimensions are zero ({new_width}x{new_height}). Check scale_factor {scale_factor}.")
            return original_image_path, 1.0

        rescaled_img = cv2.resize(img, (new_width, new_height), interpolation=interpolation_method)
        
        tifffile.imwrite(rescaled_image_path, rescaled_img)
        print(f"  Saved rescaled image to: {rescaled_image_path}")
        return rescaled_image_path, scale_factor
    except Exception as e:
        print(f"  Error during rescaling or saving rescaled image {original_image_path}: {e}")
        traceback.print_exc()
        return original_image_path, 1.0 

def segment_image_worker(job_params_dict):
    experiment_id_final = job_params_dict["experiment_id_final"] 
    model_choice = job_params_dict["MODEL_CHOICE"]
    diameter_val_for_cp = job_params_dict["DIAMETER_FOR_CELLPOSE"] 
    flow_thresh_val = job_params_dict["FLOW_THRESHOLD"]
    min_size_val = job_params_dict["MIN_SIZE"]
    cellprob_thresh = job_params_dict["CELLPROB_THRESHOLD"]
    force_grayscale_flag = job_params_dict["FORCE_GRAYSCALE"]
    use_gpu_flag = job_params_dict["USE_GPU"]
    actual_image_path_to_process = job_params_dict["actual_image_path_to_process"] 
    processing_unit_name = job_params_dict["processing_unit_name"] 

    output_dir_job = os.path.join(RESULTS_DIR_BASE, experiment_id_final)

    log_diameter_for_eval = 0 if diameter_val_for_cp is None else diameter_val_for_cp
    log_flow_for_eval = 'Cellpose default' if flow_thresh_val is None else flow_thresh_val
    log_min_size_for_eval = 15 if min_size_val is None else min_size_val

    print(f">>> [START JOB] ID: {experiment_id_final} (Unit: {processing_unit_name}) ---")
    param_log_lines = [
        f"    Image Path    : {actual_image_path_to_process}",
        f"    Model         : {model_choice}",
        f"    USE_GPU       : {use_gpu_flag}",
        f"    ForceGrayscale: {force_grayscale_flag}",
        f"    CellProbThr   : {cellprob_thresh}",
        f"    Diameter(eval): {log_diameter_for_eval if log_diameter_for_eval!=0 else 'auto'}",
        f"    FlowThr(eval) : {log_flow_for_eval}",
        f"    MinSize(eval) : {log_min_size_for_eval}"
    ]
    print(f"[{experiment_id_final}] Parameters:")
    for line in param_log_lines: print(f"[{experiment_id_final}] {line}")

    if not os.path.exists(output_dir_job):
        try: os.makedirs(output_dir_job)
        except OSError as e:
            error_msg = f"Error creating output dir {output_dir_job}: {e}"
            print(f"<<< [END JOB] ID: {experiment_id_final}. Status: FAILED. ({error_msg})")
            return {**job_params_dict, "status": "failed", "error_message": error_msg, "message": error_msg}

    if not os.path.exists(actual_image_path_to_process):
        error_msg = f"Error: Image not found: {actual_image_path_to_process}"
        print(f"<<< [END JOB] ID: {experiment_id_final}. Status: FAILED. ({error_msg})")
        return {**job_params_dict, "status": "failed", "error_message": error_msg, "message": error_msg}

    try:
        print(f"[{experiment_id_final}] Loading image...")
        img_for_cellpose = io.imread(actual_image_path_to_process)
        print(f"[{experiment_id_final}] Loaded. Shape: {img_for_cellpose.shape}, Dtype: {img_for_cellpose.dtype}")

        print(f"[{experiment_id_final}] Initializing Cellpose Model: {model_choice}...")
        model = None
        try: model = models.CellposeModel(gpu=use_gpu_flag, model_type=model_choice)
        except AttributeError:
            print(f"[{experiment_id_final}] CellposeModel error, trying models.Cellpose...")
            model = models.Cellpose(gpu=use_gpu_flag, model_type=model_choice)
        if model is None: raise ValueError("Model initialization failed.")

        print(f"[{experiment_id_final}] Running Cellpose segmentation...")
        eval_params = {"cellprob_threshold": cellprob_thresh}
        eval_params["diameter"] = 0 if diameter_val_for_cp is None else diameter_val_for_cp
        if flow_thresh_val is not None: eval_params["flow_threshold"] = flow_thresh_val
        eval_params["min_size"] = 15 if min_size_val is None else min_size_val 
        if force_grayscale_flag: eval_params["channels"] = [0,0]
        
        masks, flows, styles = model.eval(img_for_cellpose, **eval_params)
        print(f"[{experiment_id_final}] Segmentation done. Masks shape: {masks.shape}, Unique (top 20): {np.unique(masks)[:20]}")
        if eval_params['diameter'] == 0: 
             if hasattr(model, 'sz_estimate') and model.sz_estimate is not None: print(f"[{experiment_id_final}] Cellpose size est: {model.sz_estimate:.2f}")
             elif hasattr(model, 'diam_mean'): print(f"[{experiment_id_final}] Model training diam: {model.diam_mean:.2f}")

        base_output_filename = os.path.splitext(processing_unit_name)[0]
        mask_filename = os.path.join(output_dir_job, base_output_filename + "_mask.tif")
        io.imsave(mask_filename, masks.astype(np.uint16))
        print(f"[{experiment_id_final}] Mask saved: {mask_filename}")

        num_cells = masks.max()
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
        
        success_msg = f"Successfully processed. Found {num_cells} cells."
        print(f"<<< [END JOB] ID: {experiment_id_final}. Status: SUCCEEDED. ({success_msg})")
        return {**job_params_dict, "status": "succeeded", "num_cells": int(num_cells), "output_mask_path": mask_filename, "message": success_msg}

    except Exception as e:
        error_full_msg = f"Error in {experiment_id_final} (Unit: {processing_unit_name}): {e} {traceback.format_exc()}"
        print(f"[{experiment_id_final}] {error_full_msg}") 
        if os.path.exists(output_dir_job):
            with open(os.path.join(output_dir_job, "error_log.txt"), "w") as f_err: f_err.write(error_full_msg)
            print(f"[{experiment_id_final}] Detailed error saved to error_log.txt")
        short_error_msg = error_full_msg.splitlines()[0]
        print(f"<<< [END JOB] ID: {experiment_id_final}. Status: FAILED. ({short_error_msg})")
        return {**job_params_dict, "status": "failed", "error_message": error_full_msg, "message": short_error_msg}


if __name__ == "__main__":
    freeze_support()
    
    print("====================================================================")
    print(f" INITIALIZING BATCH SEGMENTATION RUN @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("====================================================================")

    for base_dir in [IMAGE_DIR_BASE, TILED_IMAGE_OUTPUT_BASE, RESCALED_IMAGE_CACHE_DIR, RESULTS_DIR_BASE]:
        if not os.path.exists(base_dir):
            try: os.makedirs(base_dir); print(f"Created directory: {base_dir}.")
            except OSError as e: print(f"Fatal: Could not create {base_dir}: {e}"); exit()

    param_file = "parameter_sets.json"
    if not os.path.exists(param_file): print(f"Error: Config file '{param_file}' not found."); exit()

    config_data = {}
    try:
        with open(param_file, 'r') as f: config_data = json.load(f)
    except Exception as e: print(f"Error reading/parsing {param_file}: {e}"); exit()

    image_configs_from_json = config_data.get("image_configurations", [])
    cellpose_param_configs = config_data.get("cellpose_parameter_configurations", [])

    if not image_configs_from_json: print("No 'image_configurations' in JSON. Exiting."); exit()
    if not cellpose_param_configs: print("No 'cellpose_parameter_configurations' in JSON. Exiting."); exit()

    all_jobs_to_create = []
    print("--- Generating Job List ---")
    for img_config in image_configs_from_json:
        if not img_config.get("is_active", True):
            print(f"Skipping inactive image configuration: {img_config.get('image_id', 'UnnamedImageConfig')}")
            continue

        original_image_filename = img_config.get("original_image_filename")
        image_id_base = img_config.get("image_id", clean_filename_for_dir(original_image_filename) if original_image_filename else "unknown_image")
        
        if not original_image_filename:
            print(f"Warning: Image config '{image_id_base}' missing 'original_image_filename'. Skipping.")
            continue
        
        original_image_path = os.path.join(IMAGE_DIR_BASE, original_image_filename)
        if not os.path.exists(original_image_path):
            print(f"Warning: Original image {original_image_path} for config '{image_id_base}' not found. Skipping this image config.")
            continue

        current_image_path_for_processing = original_image_path
        current_image_name_for_processing = original_image_filename
        applied_scale_factor = 1.0
        rescaling_cfg = img_config.get("rescaling_config")
        if rescaling_cfg and "scale_factor" in rescaling_cfg:
            print(f"Processing Image Config: '{image_id_base}' (Rescaling enabled for {original_image_filename}) -> Scale: {rescaling_cfg['scale_factor']}")
            current_image_path_for_processing, applied_scale_factor = rescale_image_and_save(
                original_image_path, 
                image_id_base, 
                rescaling_cfg
            )
            current_image_name_for_processing = os.path.basename(current_image_path_for_processing)
            if applied_scale_factor == 1.0 and current_image_path_for_processing == original_image_path:
                print(f"  Rescaling resulted in no change or failed, using original image: {original_image_path}")
        else:
            print(f"Processing Image Config: '{image_id_base}' (No rescaling for {original_image_filename})")

        images_to_segment_this_round = [] 
        tiling_cfg = img_config.get("tiling_config")

        if tiling_cfg and tiling_cfg.get("tile_size"):
            print(f"  Tiling configured for processed image: {current_image_name_for_processing} (Original: {original_image_filename})")
            tile_storage_dir_suffix = f"{image_id_base}"
            if applied_scale_factor != 1.0:
                tile_storage_dir_suffix += f"_scaled{applied_scale_factor:.2f}"
            tile_storage_dir_suffix += "_tiles"
            tile_storage_dir = os.path.join(TILED_IMAGE_OUTPUT_BASE, tile_storage_dir_suffix)
            
            tile_prefix = tiling_cfg.get("tile_output_prefix_base", clean_filename_for_dir(current_image_name_for_processing) + "_tile")
            
            if not os.path.exists(tile_storage_dir):
                 try: os.makedirs(tile_storage_dir); print(f"    Created tile storage directory: {tile_storage_dir}")
                 except OSError as e: print(f"    Error creating tile storage dir {tile_storage_dir}: {e}. Skipping tiling for this image."); continue

            tile_manifest_data = tile_image(
                current_image_path_for_processing, tile_storage_dir, 
                tile_size=tiling_cfg["tile_size"],
                overlap=tiling_cfg.get("overlap", 100),
                output_prefix=tile_prefix
            )
            if tile_manifest_data and tile_manifest_data.get("tiles"):
                for tile_info in tile_manifest_data["tiles"]:
                    images_to_segment_this_round.append({
                        "path": tile_info["path"], "name": tile_info["filename"], 
                        "original_image_id": image_id_base, 
                        "original_image_filename": original_image_filename, 
                        "applied_scale_factor": applied_scale_factor,
                        "is_tile": True, "tile_info": tile_info
                    })
                print(f"    Generated {len(tile_manifest_data['tiles'])} tiles for {current_image_name_for_processing}.")
            else: 
                print(f"    Warning: Tiling failed or no tiles for {current_image_name_for_processing}. Using this image directly.")
                images_to_segment_this_round.append({
                    "path": current_image_path_for_processing, "name": current_image_name_for_processing,
                    "original_image_id": image_id_base, "original_image_filename": original_image_filename,
                    "applied_scale_factor": applied_scale_factor, "is_tile": False
                })
        else: 
            images_to_segment_this_round.append({
                "path": current_image_path_for_processing, "name": current_image_name_for_processing,
                "original_image_id": image_id_base, "original_image_filename": original_image_filename,
                "applied_scale_factor": applied_scale_factor, "is_tile": False
            })

        for img_proc_info in images_to_segment_this_round:
            for cp_config in cellpose_param_configs:
                if not cp_config.get("is_active", True): continue
                
                param_set_id = cp_config.get("param_set_id", "params_unknown")
                job = {}
                for key, value in cp_config.items(): 
                    if key not in ["param_set_id", "is_active"]: job[key] = value
                
                job["actual_image_path_to_process"] = img_proc_info["path"]
                job["processing_unit_name"] = img_proc_info["name"] 
                job["original_image_id_for_log"] = img_proc_info["original_image_id"]
                job["original_image_filename_for_log"] = img_proc_info["original_image_filename"]
                job["scale_factor_applied_for_log"] = img_proc_info["applied_scale_factor"]
                job["is_tile_for_log"] = img_proc_info["is_tile"]
                if img_proc_info["is_tile"]: job["tile_details_for_log"] = img_proc_info["tile_info"]

                current_diameter = job.get("DIAMETER")
                if img_proc_info["applied_scale_factor"] != 1.0 and current_diameter is not None and current_diameter > 0:
                    job["DIAMETER_FOR_CELLPOSE"] = int(round(current_diameter * img_proc_info["applied_scale_factor"]))
                    # print(f"    Adjusted DIAMETER from {current_diameter} to {job['DIAMETER_FOR_CELLPOSE']} due to scale factor {img_proc_info['applied_scale_factor']}")
                else:
                    job["DIAMETER_FOR_CELLPOSE"] = current_diameter 

                cleaned_proc_unit_name = clean_filename_for_dir(img_proc_info["name"])
                if img_proc_info["is_tile"]:
                    job["experiment_id_final"] = f"{img_proc_info['original_image_id']}_{param_set_id}_{cleaned_proc_unit_name}"
                else: 
                    job["experiment_id_final"] = f"{img_proc_info['original_image_id']}_{param_set_id}"
                    if img_proc_info["applied_scale_factor"] != 1.0: 
                         job["experiment_id_final"] += f"_scaled{img_proc_info['applied_scale_factor']:.2f}"

                job.setdefault("MODEL_CHOICE", "cyto3")
                job.setdefault("FLOW_THRESHOLD", None) 
                job.setdefault("MIN_SIZE", None) 
                job.setdefault("CELLPROB_THRESHOLD", 0.0)
                job.setdefault("FORCE_GRAYSCALE", True)
                job.setdefault("USE_GPU", False)
                all_jobs_to_create.append(job)

    if not all_jobs_to_create: print("No active jobs to process after expansion. Exiting."); exit()
    print(f"Total active jobs to process: {len(all_jobs_to_create)}")
        
    num_processes_to_use = MAX_PARALLEL_PROCESSES
    any_gpu_run = any(job.get("USE_GPU", False) for job in all_jobs_to_create)
    if any_gpu_run and MAX_PARALLEL_PROCESSES > 1:
        print("WARNING: GPU usage with MAX_PARALLEL_PROCESSES > 1. Consider setting to 1 for stability.")

    print(f"Attempting to use up to {num_processes_to_use} parallel processes for job execution.")
    print("--------------------------------------------------------------------")
    start_time_all = time.time()
    job_results = []

    if num_processes_to_use > 1 and len(all_jobs_to_create) > 1:
        with Pool(processes=num_processes_to_use) as pool:
            job_results = pool.map(segment_image_worker, all_jobs_to_create)
    else:
        print("Running jobs sequentially (due to MAX_PARALLEL_PROCESSES=1 or only 1 job).")
        for job_params in all_jobs_to_create: job_results.append(segment_image_worker(job_params))

    try:
        with open(RUN_LOG_FILE, 'w') as f_log: json.dump(job_results, f_log, indent=4)
        print(f"Run log saved to: {RUN_LOG_FILE}")
    except Exception as e: print(f"Error saving run log: {e}")

    end_time_all = time.time()
    print("====================================================================")
    print(f" ALL JOBS FINISHED @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("====================================================================")
    total_duration = end_time_all - start_time_all
    print(f"Total processing time for {len(all_jobs_to_create)} jobs: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes).")

    successful_runs, failed_runs = 0, 0
    print("--- Individual Job Status Summary ---")
    for i, result in enumerate(job_results):
        if result is None: 
            print(f"  Job {i+1}: Error - Worker returned None (unexpected).")
            failed_runs +=1; continue
        
        status = result.get("status", "unknown")
        exp_id = result.get("experiment_id_final", f"unknown_job_{i+1}")
        unit_name = result.get("processing_unit_name", "unknown_image")
        
        if status == "succeeded":
            num_cells_found = result.get('num_cells', 'N/A')
            print(f"  Job {i+1}/{len(all_jobs_to_create)}: {exp_id} (Unit: {unit_name}) - SUCCEEDED (Found {num_cells_found} cells)")
            successful_runs +=1
        else:
            error_msg_short = result.get('message','Unknown error').splitlines()[0] 
            print(f"  Job {i+1}/{len(all_jobs_to_create)}: {exp_id} (Unit: {unit_name}) - FAILED ({error_msg_short})")
            failed_runs +=1
    
    print("--- Overall Batch Summary ---")
    print(f"  Total Jobs Attempted: {len(all_jobs_to_create)}")
    print(f"  Successful Jobs     : {successful_runs}")
    print(f"  Failed Jobs         : {failed_runs}")
    print("====================================================================")
    if failed_runs > 0:
        print("Check individual experiment folders and 'error_log.txt' files for details on failures.")
    print("====================================================================")

