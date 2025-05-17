import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Must be before other imports that might use OpenMP
import cv2
import numpy as np
from cellpose import models, io, plot
import json
import traceback # For detailed error printing
import time
from multiprocessing import Pool, cpu_count, freeze_support

# --- Global Configuration ---
IMAGE_DIR_BASE = "images"       # Base directory for input images
RESULTS_DIR_BASE = "results"    # Base directory for all experiment outputs

# Max number of parallel processes to use.
# Be cautious if all jobs use GPU; might be better to set to 1 or manage GPU assignment.
# Using MAX_PARALLEL_PROCESSES = 1 is safest if unsure or if using GPU.
MAX_PARALLEL_PROCESSES = max(1, cpu_count() // 2) # Default: Use half of CPU cores, at least 1
# MAX_PARALLEL_PROCESSES = 1 # Safer option if GPU is involved or to debug multiprocessing issues

def segment_image_worker(params_dict):
    """
    Worker function for a single segmentation experiment.
    params_dict is a dictionary containing all parameters for one experiment.
    """
    experiment_id = params_dict.get("experiment_id", f"exp_{time.strftime('%Y%m%d_%H%M%S')}")
    model_choice = params_dict.get("MODEL_CHOICE", "cyto3")
    diameter_val = params_dict.get("DIAMETER", None)
    flow_thresh_val = params_dict.get("FLOW_THRESHOLD", None) # Cellpose uses its default if None (e.g., 0.4)
    min_size_val = params_dict.get("MIN_SIZE", None)       # Cellpose uses its default if None (e.g., 15)
    cellprob_thresh = params_dict.get("CELLPROB_THRESHOLD", 0.0)
    force_grayscale_flag = params_dict.get("FORCE_GRAYSCALE", True)
    use_gpu_flag = params_dict.get("USE_GPU", False)
    image_filename = params_dict.get("image_filename", "test_image.tif")

    # Construct full image path
    image_path = os.path.join(IMAGE_DIR_BASE, image_filename)

    # Create a unique output directory for this experiment
    output_dir_experiment = os.path.join(RESULTS_DIR_BASE, experiment_id)
    if not os.path.exists(output_dir_experiment):
        try:
            os.makedirs(output_dir_experiment)
        except OSError as e:
            error_msg = f"[{experiment_id}] Error creating experiment output directory {output_dir_experiment}: {e}"
            print(error_msg)
            return experiment_id, False, error_msg
    
    print(f"\n--- [{experiment_id}] Starting ---")
    print(f"[{experiment_id}] Image: {image_path}")
    print(f"[{experiment_id}] Parameters: Model={model_choice}, GPU={use_gpu_flag}, Grayscale={force_grayscale_flag}, "
          f"CellProb={cellprob_thresh}, Diameter={diameter_val}, "
          f"Flow={flow_thresh_val if flow_thresh_val is not None else 'default'}, "
          f"MinSize={min_size_val if min_size_val is not None else 'default'}")

    if not os.path.exists(image_path):
        error_msg = f"[{experiment_id}] Error: Image not found at {image_path}"
        print(error_msg)
        return experiment_id, False, error_msg

    try:
        print(f"[{experiment_id}] Loading image for Cellpose: {image_path}")
        img_for_cellpose = io.imread(image_path)
        print(f"[{experiment_id}] Image for Cellpose loaded. Shape: {img_for_cellpose.shape}, Data type: {img_for_cellpose.dtype}")

        print(f"[{experiment_id}] Initializing Cellpose model (Model: {model_choice})...")
        model = None
        try:
            model = models.CellposeModel(gpu=use_gpu_flag, model_type=model_choice)
        except AttributeError:
            print(f"[{experiment_id}] AttributeError with CellposeModel, trying models.Cellpose...")
            model = models.Cellpose(gpu=use_gpu_flag, model_type=model_choice)
        
        if model is None: # Should be caught by exceptions above, but as a safeguard
            raise ValueError("Model initialization returned None.")

        print(f"[{experiment_id}] Running Cellpose segmentation...")
        eval_params = {
            "diameter": diameter_val,
            "flow_threshold": flow_thresh_val,
            "cellprob_threshold": cellprob_thresh,
            "min_size": min_size_val
        }
        if force_grayscale_flag:
            eval_params["channels"] = [0,0]
        
        masks, flows, styles = model.eval(img_for_cellpose, **eval_params)
        print(f"[{experiment_id}] Segmentation complete. Masks shape: {masks.shape}, Unique mask values (first 20): {np.unique(masks)[:20]}")
        if (diameter_val is None or diameter_val == 0):
             if hasattr(model, 'sz_estimate') and model.sz_estimate is not None:
                 print(f"[{experiment_id}] Cellpose internal size estimate for current image (model.sz_estimate): {model.sz_estimate:.2f}")
             elif hasattr(model, 'diam_mean'): # diam_mean is the model's training diameter
                 print(f"[{experiment_id}] Model's mean training diameter (model.diam_mean): {model.diam_mean:.2f} (used if diameter not set)")


        mask_filename = os.path.join(output_dir_experiment, os.path.splitext(image_filename)[0] + "_mask.tif")
        io.imsave(mask_filename, masks.astype(np.uint16))
        print(f"[{experiment_id}] Segmentation mask saved to: {mask_filename}")

        # Overlay generation
        overlay_filename_rel = os.path.splitext(image_filename)[0] + "_overlay.png"
        overlay_filename_abs = os.path.join(output_dir_experiment, overlay_filename_rel)
        print(f"[{experiment_id}] Attempting to create overlay for: {image_path}")
        
        img_for_display_cv = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img_for_display_normalized = img_for_display_cv.copy() if img_for_display_cv is not None else img_for_cellpose.copy()
        
        if img_for_display_normalized.dtype == np.uint16:
            img_max_val = np.iinfo(np.uint16).max
            img_for_display_normalized = (img_for_display_normalized / img_max_val * 255.0).astype(np.uint8) if img_max_val > 0 else img_for_display_normalized.astype(np.uint8)
        elif img_for_display_normalized.dtype not in [np.uint8, np.float32, np.float64]:
             img_actual_max = np.max(img_for_display_normalized)
             img_for_display_normalized = (img_for_display_normalized / img_actual_max * 255.0).astype(np.uint8) if img_actual_max > 0 else np.zeros_like(img_for_display_normalized, dtype=np.uint8)
        elif np.issubdtype(img_for_display_normalized.dtype, np.floating):
            img_min, img_max = np.min(img_for_display_normalized), np.max(img_for_display_normalized)
            img_for_display_normalized = ((img_for_display_normalized - img_min) / (img_max - img_min) * 255.0).astype(np.uint8) if img_max > img_min else (np.zeros_like(img_for_display_normalized) + 128).astype(np.uint8)
        
        img_to_plot_for_overlay = None
        if force_grayscale_flag:
            img_to_plot_for_overlay = cv2.cvtColor(img_for_display_normalized, cv2.COLOR_BGR2GRAY) if img_for_display_normalized.ndim == 3 and img_for_display_normalized.shape[-1] >= 3 else img_for_display_normalized
        else:
            if img_for_display_normalized.ndim == 2: img_to_plot_for_overlay = cv2.cvtColor(img_for_display_normalized, cv2.COLOR_GRAY2BGR)
            elif img_for_display_normalized.ndim == 3 and img_for_display_normalized.shape[-1] == 4: img_to_plot_for_overlay = cv2.cvtColor(img_for_display_normalized, cv2.COLOR_BGRA2BGR)
            elif img_for_display_normalized.ndim == 3 and img_for_display_normalized.shape[-1] == 3: img_to_plot_for_overlay = img_for_display_normalized
            else:
                current_img = img_for_display_normalized
                if current_img.ndim > 2 : img_to_plot_for_overlay = current_img[:,:,0]
                else : img_to_plot_for_overlay = current_img
                if img_to_plot_for_overlay.ndim == 3: img_to_plot_for_overlay = cv2.cvtColor(img_to_plot_for_overlay, cv2.COLOR_BGR2GRAY)
        
        if img_to_plot_for_overlay is not None:
            overlay_input_img_rgb = cv2.cvtColor(img_to_plot_for_overlay, cv2.COLOR_BGR2RGB) if img_to_plot_for_overlay.ndim == 3 and img_to_plot_for_overlay.shape[-1] == 3 else img_to_plot_for_overlay
            overlay_rgb_output = plot.mask_overlay(overlay_input_img_rgb, masks)
            overlay_bgr_to_save = cv2.cvtColor(overlay_rgb_output, cv2.COLOR_RGB2BGR)
            cv2.imwrite(overlay_filename_abs, overlay_bgr_to_save)
            print(f"[{experiment_id}] Overlay image saved to: {overlay_filename_abs}")
        else:
            print(f"[{experiment_id}] Error: Image for overlay plotting is None. Skipping overlay.")

        # JSON coordinates
        num_cells = masks.max()
        print(f"[{experiment_id}] Found {num_cells} cells.")
        coordinates_list = []
        if num_cells > 0:
            for i in range(1, num_cells + 1):
                cell_mask_pixels = (masks == i)
                if np.any(cell_mask_pixels):
                    y_coords, x_coords = np.where(cell_mask_pixels)
                    centroid_y, centroid_x = np.mean(y_coords), np.mean(x_coords)
                    coordinates_list.append({"cell_id": int(i), "centroid_x": float(f"{centroid_x:.2f}"), "centroid_y": float(f"{centroid_y:.2f}")})
        
        coord_json_filename_rel = os.path.splitext(image_filename)[0] + "_coords.json"
        coord_json_filename_abs = os.path.join(output_dir_experiment, coord_json_filename_rel)
        with open(coord_json_filename_abs, 'w') as f_json:
            json.dump(coordinates_list, f_json, indent=4)
        print(f"[{experiment_id}] Cell coordinates saved to: {coord_json_filename_abs}")
        
        return experiment_id, True, f"Successfully processed. Found {num_cells} cells."

    except Exception as e:
        error_full_msg = f"Error during experiment {experiment_id}: {e}\n{traceback.format_exc()}"
        print(error_full_msg)
        # Ensure output_dir_experiment exists before trying to write an error file there
        if not os.path.exists(output_dir_experiment):
            try: # Attempt to create it one last time for the error log
                os.makedirs(output_dir_experiment)
            except OSError:
                pass # If it fails, error log will just print to console

        if os.path.exists(output_dir_experiment):
            error_log_file = os.path.join(output_dir_experiment, "error_log.txt")
            with open(error_log_file, "w") as f_err:
                f_err.write(error_full_msg)
            print(f"[{experiment_id}] Detailed error saved to {error_log_file}")
        return experiment_id, False, error_full_msg


if __name__ == "__main__":
    freeze_support() # Recommended for multiprocessing on Windows when bundled

    # Ensure base directories exist
    if not os.path.exists(IMAGE_DIR_BASE):
        try:
            os.makedirs(IMAGE_DIR_BASE)
            print(f"Created base image directory: {IMAGE_DIR_BASE}. Please add images here.")
        except OSError as e:
            print(f"Fatal: Could not create base image directory {IMAGE_DIR_BASE}: {e}")
            exit()
    
    if not os.path.exists(RESULTS_DIR_BASE):
        try:
            os.makedirs(RESULTS_DIR_BASE)
            print(f"Created base results directory: {RESULTS_DIR_BASE}")
        except OSError as e:
            print(f"Fatal: Could not create base results directory {RESULTS_DIR_BASE}: {e}")
            exit()


    # Load parameter sets from JSON
    param_file = "parameter_sets.json"
    if not os.path.exists(param_file):
        print(f"Error: Parameter configuration file '{param_file}' not found in project root.")
        exit()

    parameter_sets = []
    try:
        with open(param_file, 'r') as f:
            parameter_sets = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {param_file}: {e}")
        exit()
    except Exception as e:
        print(f"Error reading {param_file}: {e}")
        exit()


    if not parameter_sets:
        print("No parameter sets found in JSON file or file is empty.")
        exit()

    print(f"Found {len(parameter_sets)} parameter sets to process.")
    
    num_processes_to_use = MAX_PARALLEL_PROCESSES
    
    # Check if any run uses GPU for multiprocessing warning
    any_gpu_run = any(params.get("USE_GPU", False) for params in parameter_sets)
    if any_gpu_run and MAX_PARALLEL_PROCESSES > 1:
        print("\nWARNING: One or more parameter sets specify USE_GPU=True.")
        print("Running multiple GPU-accelerated Cellpose instances in parallel on a single GPU")
        print("can lead to memory issues or crashes. It is highly recommended to set")
        print("MAX_PARALLEL_PROCESSES = 1 at the top of the script if using GPU for any run,")
        print("or ensure you have multiple GPUs and are managing their assignment (not handled by this script).\n")
        # Optionally, force num_processes_to_use to 1 if GPU is detected
        # num_processes_to_use = 1 
        # print("Overriding MAX_PARALLEL_PROCESSES to 1 due to GPU usage.")


    print(f"Using up to {num_processes_to_use} parallel processes.")
    start_time_all = time.time()

    results_summary = []
    if num_processes_to_use > 1 and len(parameter_sets) > 1 :
        with Pool(processes=num_processes_to_use) as pool:
            results_summary = pool.map(segment_image_worker, parameter_sets)
    else: # Run sequentially if only 1 process or 1 parameter set
        print("Running experiments sequentially (MAX_PARALLEL_PROCESSES=1 or only 1 experiment).")
        for params in parameter_sets:
            results_summary.append(segment_image_worker(params))


    end_time_all = time.time()
    print(f"\n--- All Experiments Finished ---")
    total_duration = end_time_all - start_time_all
    print(f"Total processing time for {len(parameter_sets)} experiments: {total_duration:.2f} seconds.")

    successful_runs = 0
    failed_runs = 0
    for res_tuple in results_summary:
        if res_tuple is None: # Should not happen with current worker logic but safeguard
            print("Error: A worker returned None (unexpected).")
            failed_runs +=1
            continue
        exp_id, success, message = res_tuple
        if success:
            print(f"Experiment {exp_id}: Succeeded. {message}")
            successful_runs +=1
        else:
            # Only print the first line of a potentially long error message in summary
            print(f"Experiment {exp_id}: Failed. {message.splitlines()[0]}")
            failed_runs +=1
    
    print(f"\nSummary: {successful_runs} successful, {failed_runs} failed.")
    if failed_runs > 0:
        print("Check individual experiment folders and 'error_log.txt' files for details on failures.")
