# src/segmentation_worker.py
import os
import numpy as np
from cellpose import models, io
import json
import traceback

RESULTS_DIR_BASE = "results" # Relative to project root

def segment_image_worker(job_params_dict):
    # Assumes job_params_dict contains all necessary resolved parameters
    experiment_id_final = job_params_dict["experiment_id_final"]
    model_choice = job_params_dict["MODEL_CHOICE"]
    diameter_val_for_cp = job_params_dict["DIAMETER_FOR_CELLPOSE"]
    flow_thresh_val = job_params_dict["FLOW_THRESHOLD"]
    min_size_val = job_params_dict["MIN_SIZE"] # This is now always a number (e.g. 15 if was None)
    cellprob_thresh = job_params_dict["CELLPROB_THRESHOLD"]
    force_grayscale_flag = job_params_dict["FORCE_GRAYSCALE"]
    use_gpu_flag = job_params_dict["USE_GPU"]
    actual_image_path_to_process = job_params_dict["actual_image_path_to_process"]
    processing_unit_name = job_params_dict["processing_unit_name"]

    output_dir_job = os.path.join(RESULTS_DIR_BASE, experiment_id_final)

    log_diameter_for_eval = 0 if diameter_val_for_cp is None else diameter_val_for_cp
    log_flow_for_eval = 'Cellpose default' if flow_thresh_val is None else flow_thresh_val
    log_min_size_for_eval = min_size_val 

    print(f">>> [START JOB] ID: {experiment_id_final} (Unit: {processing_unit_name})")
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
        eval_params["min_size"] = min_size_val 
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
