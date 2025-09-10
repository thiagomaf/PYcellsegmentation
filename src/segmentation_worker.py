# src/segmentation_worker.py
import os
import numpy as np
from cellpose import models, io
import json
import traceback
import logging
from .file_paths import RESULTS_DIR_BASE
import time
import tifffile
import torch

logger = logging.getLogger(__name__)

# RESULTS_DIR_BASE = "results" # Removed

def segment_image_worker(job_params_dict):
    """
    Worker function to perform segmentation on a single image or tile.
    job_params_dict should contain all necessary parameters including:
    - actual_image_path_to_process: Full path to the image file.
    - processing_unit_name: Filename of the image/tile being processed (for naming outputs).
    - experiment_id_final: Unique ID for this specific job run (image_paramSet_scale_tile).
    - MODEL_CHOICE: e.g., "cyto3", "nuclei".
    - DIAMETER_FOR_CELLPOSE: Diameter for Cellpose.
    - USE_GPU: Boolean.
    - Other Cellpose specific params like FLOW_THRESHOLD, CELLPROB_THRESHOLD, MIN_SIZE etc.
    """
    start_time_worker = time.time()
    
    image_path = job_params_dict.get("actual_image_path_to_process")
    processing_unit_name = job_params_dict.get("processing_unit_name")
    experiment_id = job_params_dict.get("experiment_id_final")
    
    segmentation_should_run = job_params_dict.get("segmentation_step_should_run", True) # Default to True if key is missing

    # Log the start of processing for this worker
    logger.info(f"Worker starting for Experiment ID: {experiment_id}, Image Unit: {processing_unit_name}")
    logger.info(f"  Image path: {image_path}")
    logger.info(f"  Segmentation step should run: {segmentation_should_run}")

    if not segmentation_should_run:
        logger.info(f"  Segmentation step is SKIPPED for {processing_unit_name} (Exp ID: {experiment_id}) as per configuration.")
        # Construct the expected mask path based on how an active segmentation would save it.
        output_dir_for_experiment = os.path.join(RESULTS_DIR_BASE, experiment_id)
        mask_filename_part = f"{os.path.splitext(processing_unit_name)[0]}_mask.tif" # Based on the (potentially scaled/tiled) processing unit name
        expected_mask_path = os.path.join(output_dir_for_experiment, mask_filename_part)

        if os.path.exists(expected_mask_path):
            logger.info(f"  Found pre-existing mask at expected location: {expected_mask_path}")
            # We can attempt to get num_cells if useful for downstream, otherwise just report success.
            try:
                mask_data = tifffile.imread(expected_mask_path)
                num_cells_in_mask = len(np.unique(mask_data)) - 1 # -1 for background
                logger.info(f"    Mask contains {num_cells_in_mask} cells.")
            except Exception as e_read_mask:
                logger.warning(f"    Could not read pre-existing mask {expected_mask_path} to count cells: {e_read_mask}")
                num_cells_in_mask = -1 # Indicate unknown
            
            return {
                "status": "success_segmentation_skipped", 
                "experiment_id": experiment_id, 
                "unit": processing_unit_name, 
                "mask_path": expected_mask_path, 
                "num_cells": num_cells_in_mask,
                "message": "Segmentation step skipped; pre-existing mask found at standard location."
            }
        else:
            error_msg = f"Segmentation step skipped, but pre-existing mask NOT FOUND at expected location: {expected_mask_path}"
            logger.error(error_msg)
            return {"status": "error_mask_not_found", "experiment_id": experiment_id, "unit": processing_unit_name, "error": error_msg}

    if not all([image_path, processing_unit_name, experiment_id]):
        error_msg = "Worker error: Missing critical parameters (image_path, processing_unit_name, or experiment_id)."
        logger.error(error_msg)
        return {"status": "error", "experiment_id": experiment_id, "unit": processing_unit_name, "error": error_msg, "traceback": ""}

    try:
        if not os.path.exists(image_path):
            error_msg = f"Image file not found: {image_path}"
            logger.error(error_msg)
            return {"status": "error", "experiment_id": experiment_id, "unit": processing_unit_name, "error": error_msg, "traceback": ""}

        img = io.imread(image_path)
        if img is None:
            error_msg = f"Failed to read image: {image_path}"
            logger.error(error_msg)
            return {"status": "error", "experiment_id": experiment_id, "unit": processing_unit_name, "error": error_msg, "traceback": ""}

        model_choice = job_params_dict.get("MODEL_CHOICE", "cyto3")
        # Get DIAMETER_FOR_CELLPOSE, which should have been calculated by the parser
        diameter_for_eval = job_params_dict.get("DIAMETER_FOR_CELLPOSE") 
        use_gpu = job_params_dict.get("USE_GPU", False) # Default to False

        # Optional Cellpose parameters
        flow_threshold = job_params_dict.get("FLOW_THRESHOLD") 
        cellprob_threshold = job_params_dict.get("CELLPROB_THRESHOLD", 0.0) 
        min_size_from_config = job_params_dict.get("MIN_SIZE") 
        force_grayscale = job_params_dict.get("FORCE_GRAYSCALE", True)
        
        logger.info(f"  Model: {model_choice}, DIAMETER_FOR_CELLPOSE received by worker: {diameter_for_eval}, GPU: {use_gpu}")
        # Log the original diameter from config as well, if available in job_params
        original_diameter_from_config = job_params_dict.get("DIAMETER") # This is the unscaled one
        if original_diameter_from_config is not None:
            logger.info(f"    (Original DIAMETER from config was: {original_diameter_from_config})")
        logger.info(f"  Optional params - FlowThresh: {flow_threshold}, CellProbThresh: {cellprob_threshold}, MinSize (from config): {min_size_from_config}, ForceGrayscale: {force_grayscale}")

        # Force model reloading to avoid caching issues between different model types
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache if using GPU
        
        model = models.CellposeModel(gpu=use_gpu, model_type=model_choice)
        logger.info(f"  Created new CellposeModel instance for model_type: {model_choice}")
        
        # Verify model was loaded correctly
        if hasattr(model, 'diam_mean'):
            logger.info(f"  Model diam_mean: {model.diam_mean}")
        if hasattr(model, 'net') and hasattr(model.net, 'state_dict'):
            # Log a hash of the first few parameters to verify different models
            state_dict = model.net.state_dict()
            if state_dict:
                first_key = next(iter(state_dict.keys()))
                first_param = state_dict[first_key]
                # Cast to a more common dtype like float32 before converting to numpy to avoid BFloat16 error
                param_hash = hash(first_param.cpu().to(torch.float32).numpy().tobytes()) if hasattr(first_param, 'cpu') else hash(str(first_param))
                logger.info(f"  Model parameter hash (first layer): {param_hash}")
        logger.info(f"  Model type verification: {getattr(model, 'model_type', 'unknown')}")
        
        # Prepare channels: if grayscale and 3D, expand dims for Cellpose
        channels = [0,0] # Default for grayscale
        if force_grayscale and img.ndim == 2:
            pass # Standard 2D grayscale
        elif force_grayscale and img.ndim == 3: # Potentially Z-stack or RGB-like
            if img.shape[-1] in [3,4]: # Likely RGB/A, take first channel (or mean/Luminosity)
                logger.info(f"  Image has {img.shape[-1]} channels, using first channel for grayscale.")
                img = img[..., 0]
            # If it's already a 3D grayscale (Z, H, W), Cellpose might handle it or expect (num_planes, H, W)
            # For safety, if it is (H,W,Z), it might need reordering. Assuming (Z,H,W) or (H,W) for now.
        elif not force_grayscale: # Color
             # Expects (H, W, C) where C is R,G,B. Cellpose channel args might be needed e.g. [R_chan, G_chan]
            logger.warning("  Processing in color mode. Ensure image is (H,W,C) and channels are set if not standard RGB.")
            # channels = [R,G] # Example: channels = [1,2] for R=1, G=2 if image is R,G,B
            # This part might need more sophisticated channel handling based on image specifics if not simple RGB

        logger.info(f"  Running Cellpose segmentation for Experiment ID: {experiment_id} on {processing_unit_name}...")
        
        eval_params = {
            "diameter": diameter_for_eval, # Use the (potentially scaled) diameter
            "channels": channels,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold
        }
        if min_size_from_config is not None:
            eval_params["min_size"] = min_size_from_config
            logger.info(f"  Explicitly setting min_size to {min_size_from_config} for Cellpose model.eval().")
        else:
            # If min_size is not in config, use a Cellpose-friendly default (e.g., 15)
            # rather than letting it be None implicitly passed to Cellpose internals.
            eval_params["min_size"] = 15 
            logger.info(f"  min_size not specified in config or is None; explicitly setting to {eval_params['min_size']} for Cellpose.")

        masks, flows, styles = model.eval(img, **eval_params)
        
        # Output paths
        # Masks are saved relative to a folder named after the experiment_id_final, under RESULTS_DIR_BASE
        # The mask filename itself is based on the processing_unit_name to ensure uniqueness if an experiment_id yields multiple files (e.g. tiles)
        
        output_dir_for_experiment = os.path.join(RESULTS_DIR_BASE, experiment_id)
        if not os.path.exists(output_dir_for_experiment):
            os.makedirs(output_dir_for_experiment)
            logger.info(f"  Created results directory: {output_dir_for_experiment}")

        mask_filename_part = f"{os.path.splitext(processing_unit_name)[0]}_mask.tif"
        mask_output_path = os.path.join(output_dir_for_experiment, mask_filename_part)
        
        io.imsave(mask_output_path, masks)
        logger.info(f"  Mask saved to: {mask_output_path}")

        # Save flows and other metadata (optional, can be large)
        # flow_output_path = os.path.join(output_dir_for_experiment, f"{os.path.splitext(processing_unit_name)[0]}_flows.tif")
        # io.imsave(flow_output_path, flows[0]) # Save flows if needed, flows[0] is usually the relevant one
        # logger.info(f"  Flows saved to: {flow_output_path}")


        # Save a small JSON with info about the run for this specific unit
        job_summary = {
            "experiment_id": experiment_id,
            "source_image_unit": processing_unit_name,
            "image_path_processed": image_path,
            "mask_output_path": mask_output_path,
            "cellpose_model": model_choice,
            "input_diameter_arg": job_params_dict.get("DIAMETER"), # Log the original diameter from config
            "diameter_used_for_eval": diameter_for_eval, # Log the actual diameter passed to model.eval()
            "params_used": {
                "flow_threshold": flow_threshold,
                "cellprob_threshold": cellprob_threshold,
                "min_size": eval_params.get("min_size"), # Log what was actually used (None if Cellpose default was used)
                "channels": channels, # Log what channels were actually used.
                "USE_GPU": use_gpu,
                "FORCE_GRAYSCALE": force_grayscale
            },
            "job_params_received": job_params_dict # Log the full input for traceability
        }
        summary_filename = f"{os.path.splitext(processing_unit_name)[0]}_segmentation_summary.json"
        summary_output_path = os.path.join(output_dir_for_experiment, summary_filename)
        with open(summary_output_path, 'w') as f_json:
            json.dump(job_summary, f_json, indent=4, default=lambda o: '<not serializable>')
        logger.info(f"  Segmentation summary saved to: {summary_output_path}")
        
        duration_worker = time.time() - start_time_worker
        logger.info(f"Worker finished for Experiment ID: {experiment_id}, Unit: {processing_unit_name}. Duration: {duration_worker:.2f}s")

        return {"status": "success", "experiment_id": experiment_id, "unit": processing_unit_name, "mask_path": mask_output_path, "summary_path": summary_output_path, "duration_seconds": duration_worker}

    except Exception as e:
        tb_str = traceback.format_exc()
        error_msg = f"Error during segmentation of {processing_unit_name} (Exp ID: {experiment_id}): {e}"
        logger.error(error_msg)
        logger.error(f"Traceback: {tb_str}")
        return {"status": "error", "experiment_id": experiment_id, "unit": processing_unit_name, "error": error_msg, "traceback": tb_str}

# Example of how this worker might be called (for testing, not actual use in pipeline)
if __name__ == '__main__':
    # This block is for direct testing of the worker.
    # Ensure logging is configured if you run this directly.
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Testing segmentation_worker.py directly...")

    # Create dummy files and directories for testing
    # IMPORTANT: This test setup assumes it's run from the project root or paths are adjusted.
    # For simplicity, it assumes 'test_images' and 'test_results' in the current dir if run directly.
    
    # Relative paths for direct script execution test
    test_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Assuming src is parent of this file
    
    # Update RESULTS_DIR_BASE for the test
    RESULTS_DIR_BASE = os.path.join(test_project_root, "test_results_worker") # Override for test
    if not os.path.exists(RESULTS_DIR_BASE):
        os.makedirs(RESULTS_DIR_BASE)

    dummy_image_dir = os.path.join(test_project_root, "test_images_worker")
    if not os.path.exists(dummy_image_dir):
        os.makedirs(dummy_image_dir)
    
    dummy_image_name = "test_img_worker.tif"
    dummy_image_path = os.path.join(dummy_image_dir, dummy_image_name)

    # Create a small dummy TIFF image using tifffile
    try:
        import tifffile
        dummy_array = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
        tifffile.imwrite(dummy_image_path, dummy_array)
        logger.info(f"Created dummy image for testing: {dummy_image_path}")
    except ImportError:
        logger.error("tifffile not installed. Cannot create dummy image for worker test. Skipping test.")
        exit()
    except Exception as e:
        logger.error(f"Failed to create dummy TIFF: {e}")
        exit()

    test_job_params = {
        "actual_image_path_to_process": dummy_image_path,
        "processing_unit_name": dummy_image_name, # Typically the filename of the unit
        "experiment_id_final": "testExp_cyto2_scale1_unitWorker",
        "MODEL_CHOICE": "cyto2", # ensure you have this model or use a default like 'cyto' / 'nuclei'
        "DIAMETER_FOR_CELLPOSE": 0, # 0 to estimate
        "USE_GPU": False, # Set to True if you have a GPU and want to test
        "FLOW_THRESHOLD": 0.4, # Example
        "CELLPROB_THRESHOLD": 0.0, # Example
        "MIN_SIZE": 15, # Example
        "FORCE_GRAYSCALE": True,
        "mpp_x_original_for_log": 0.5, # Example metadata
        "mpp_y_original_for_log": 0.5,  # Example metadata
        "experiment_id_for_mask_folder": "testExp_cyto2_scale1_unitWorker",
        "mask_filename_base": "testExp_cyto2_scale1_unitWorker"
    }
    result = segment_image_worker(test_job_params)
    logger.info(f"Test worker result: {result}")

    # Clean up dummy files (optional)
    # if os.path.exists(dummy_image_path): os.remove(dummy_image_path)
    # if os.path.exists(dummy_image_dir): os.rmdir(dummy_image_dir) # only if empty
    # Can also clean up results dir if desired.
    logger.info("Test finished. Check 'test_results_worker' directory.")

