# src/pipeline_config_parser.py
import os
import json
import time
import traceback
import logging
from .pipeline_utils import clean_filename_for_dir, rescale_image_and_save, construct_full_experiment_id
from .tile_large_image import tile_image
from .file_paths import IMAGE_DIR_BASE, TILED_IMAGE_OUTPUT_BASE, RESCALED_IMAGE_CACHE_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

def load_and_expand_configurations(param_json_file_path, global_use_gpu_if_available: bool):
    """
    Loads configurations from parameter_sets.json, handles rescaling and tiling,
    and generates a flat list of all individual jobs to be run.
    Each job in the list is a dictionary fully resolved with all parameters.
    """
    if not os.path.exists(param_json_file_path):
        logger.error(f"Error: Config file '{param_json_file_path}' not found.")
        return []

    config_data = {}
    try:
        with open(param_json_file_path, 'r') as f:
            config_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading/parsing {param_json_file_path}: {e}")
        logger.error(traceback.format_exc())
        return []

    image_configs_from_json = config_data.get("image_configurations", [])
    cellpose_param_configs = config_data.get("cellpose_parameter_configurations", [])

    if not image_configs_from_json:
        logger.warning("No 'image_configurations' found in JSON.")
        return []
    if not cellpose_param_configs:
        logger.warning("No 'cellpose_parameter_configurations' found in JSON.")
        return []

    all_jobs_to_create = []
    logger.info("--- Generating Job List (from pipeline_config_parser) ---")

    # Create cache and tiled output base directories if they don't exist
    if not os.path.exists(RESCALED_IMAGE_CACHE_DIR):
        try: 
            os.makedirs(RESCALED_IMAGE_CACHE_DIR)
            logger.info(f"Created cache directory: {RESCALED_IMAGE_CACHE_DIR}")
        except OSError as e:
            logger.error(f"OSError creating cache directory {RESCALED_IMAGE_CACHE_DIR}: {e}. Some operations might fail.")
    if not os.path.exists(TILED_IMAGE_OUTPUT_BASE):
        try:
            os.makedirs(TILED_IMAGE_OUTPUT_BASE)
            logger.info(f"Created tiled image output base directory: {TILED_IMAGE_OUTPUT_BASE}")
        except OSError as e:
            logger.error(f"OSError creating tiled image base directory {TILED_IMAGE_OUTPUT_BASE}: {e}. Tiling might fail.")

    for img_config in image_configs_from_json:
        if not img_config.get("is_active", True):
            logger.info(f"Skipping inactive image configuration: {img_config.get('image_id', 'UnnamedImageConfig')}")
            continue

        original_image_filename = img_config.get("original_image_filename")
        image_id_base = img_config.get("image_id", clean_filename_for_dir(original_image_filename) if original_image_filename else f"img_unknown_{time.strftime('%Y%m%d%H%M%S')}")
        
        if not original_image_filename:
            logger.warning(f"Image config '{image_id_base}' missing 'original_image_filename'. Skipping.")
            continue
        
        original_image_path = os.path.join(PROJECT_ROOT, original_image_filename)
        if not os.path.exists(original_image_path):
            logger.warning(f"Original image {original_image_path} for config '{image_id_base}' not found. Skipping this image config.")
            continue

        current_image_path_for_processing = original_image_path
        current_image_name_for_processing = os.path.basename(original_image_filename)
        applied_scale_factor = 1.0
        segmentation_options = img_config.get("segmentation_options", {})
        rescaling_cfg = segmentation_options.get("rescaling_config")

        if rescaling_cfg and "scale_factor" in rescaling_cfg and rescaling_cfg["scale_factor"] != 1.0 :
            logger.info(f"  Image Config: '{image_id_base}' (Rescaling {original_image_filename} by factor {rescaling_cfg['scale_factor']})")
            current_image_path_for_processing, applied_scale_factor = rescale_image_and_save(
                original_image_path, 
                image_id_base, 
                rescaling_cfg
            )
            current_image_name_for_processing = os.path.basename(current_image_path_for_processing)
            if applied_scale_factor == 1.0 and current_image_path_for_processing == original_image_path:
                 logger.info(f"    Rescaling resulted in no change or failed, using original image: {original_image_path}")
        else:
            logger.info(f"  Image Config: '{image_id_base}' (No rescaling for {original_image_filename})")

        images_to_segment_this_round = [] 
        tiling_cfg = segmentation_options.get("tiling_parameters")

        if tiling_cfg and tiling_cfg.get("apply_tiling") and tiling_cfg.get("tile_size"):
            logger.info(f"    Tiling configured for: {current_image_name_for_processing} (orig: {original_image_filename})")
            
            tile_storage_dir_suffix = image_id_base 
            if applied_scale_factor != 1.0:
                scale_factor_str = str(applied_scale_factor).replace('.', '_')
                tile_storage_dir_suffix += f"_scaled{scale_factor_str}"
            
            tile_storage_dir = os.path.join(TILED_IMAGE_OUTPUT_BASE, tile_storage_dir_suffix)
            tile_prefix = tiling_cfg.get("tile_output_prefix_base", clean_filename_for_dir(current_image_name_for_processing) + "_tile")
            
            if not os.path.exists(tile_storage_dir):
                 try: os.makedirs(tile_storage_dir); logger.info(f"      Created tile storage directory: {tile_storage_dir}")
                 except OSError as e: 
                     logger.error(f"      Error creating tile storage dir {tile_storage_dir}: {e}. Skipping tiling for this image config."); continue

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
                logger.info(f"      Generated {len(tile_manifest_data['tiles'])} tiles for {current_image_name_for_processing}.")
            else: 
                logger.warning(f"      Warning: Tiling failed or no tiles for {current_image_name_for_processing}. Using this image directly.")
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
            segmentation_options_for_image = img_config.get("segmentation_options", {}) # Get it from the original img_config
            apply_segmentation_for_this_image = segmentation_options_for_image.get("apply_segmentation", True)

            if apply_segmentation_for_this_image:
                for cp_config in cellpose_param_configs:
                    if not cp_config.get("is_active", True): continue
                    
                    param_set_id = cp_config.get("param_set_id", f"cp_params_unknown_{time.strftime('%H%M%S')}")
                    job = {} 

                    if "cellpose_parameters" in cp_config and isinstance(cp_config["cellpose_parameters"], dict):
                        for cp_key, cp_value in cp_config["cellpose_parameters"].items():
                            job[cp_key] = cp_value
                    
                    job["actual_image_path_to_process"] = img_proc_info["path"]
                    job["processing_unit_name"] = img_proc_info["name"]
                    job["original_image_id_for_log"] = img_proc_info["original_image_id"]
                    job["original_image_filename_for_log"] = img_proc_info["original_image_filename"]
                    job["param_set_id_for_log"] = param_set_id
                    job["scale_factor_applied_for_log"] = img_proc_info["applied_scale_factor"]
                    job["is_tile_for_log"] = img_proc_info["is_tile"]
                    if img_proc_info["is_tile"]:
                        job["tile_details_for_log"] = img_proc_info["tile_info"]

                    cp_diameter = job.get("DIAMETER") 
                    if img_proc_info["applied_scale_factor"] != 1.0 and cp_diameter is not None and cp_diameter > 0:
                        job["DIAMETER_FOR_CELLPOSE"] = int(round(cp_diameter * img_proc_info["applied_scale_factor"]))
                        logger.info(f"    Adjusted diameter for scaled image: {job['DIAMETER_FOR_CELLPOSE']} (original: {cp_diameter}, scale: {img_proc_info['applied_scale_factor']})")
                    else:
                        job["DIAMETER_FOR_CELLPOSE"] = cp_diameter 
                    
                    job["experiment_id_final"] = construct_full_experiment_id(
                        image_id=img_proc_info["original_image_id"],
                        param_set_id=param_set_id,
                        scale_factor=img_proc_info["applied_scale_factor"],
                        processing_unit_name_for_tile=img_proc_info["name"],
                        is_tile=img_proc_info["is_tile"]
                    )
                    job.setdefault("MODEL_CHOICE", "cyto3")
                    job.setdefault("MIN_SIZE", 15)
                    job.setdefault("CELLPROB_THRESHOLD", 0.0)
                    job.setdefault("FORCE_GRAYSCALE", True)
                    job["USE_GPU"] = cp_config.get("cellpose_parameters", {}).get("USE_GPU", global_use_gpu_if_available)
                    job["segmentation_skipped"] = False # Explicitly set for segmentation jobs

                    img_mpp_x = img_config.get("mpp_x")
                    img_mpp_y = img_config.get("mpp_y")
                    if img_mpp_x is not None: job["mpp_x_original_for_log"] = img_mpp_x
                    if img_mpp_y is not None: job["mpp_y_original_for_log"] = img_mpp_y

                    all_jobs_to_create.append(job)
                    logger.info(f"    Created segmentation job: ExpID '{job['experiment_id_final']}' for unit '{job['processing_unit_name']}'")
            else:
                # Segmentation is OFF for this image unit.
                job = {}
                
                # Information about the locally scaled intensity image for this run
                job["scaled_image_path_for_current_run"] = img_proc_info["path"]
                job["scaled_unit_name_for_current_run"] = img_proc_info["name"]
                job["current_run_scale_factor"] = img_proc_info["applied_scale_factor"] # Local scaling

                # Information for logging and general reference
                job["original_image_id_for_log"] = img_proc_info["original_image_id"]
                job["original_image_filename_for_log"] = img_proc_info["original_image_filename"]
                job["is_tile_for_log"] = img_proc_info["is_tile"]
                if img_proc_info["is_tile"]:
                    job["tile_details_for_log"] = img_proc_info["tile_info"]

                param_set_id_for_existing_mask = segmentation_options_for_image.get("use_existing_mask_param_set_id")
                if not param_set_id_for_existing_mask:
                    logger.warning(f"  Image config '{img_proc_info['original_image_id']}' has apply_segmentation=false but is missing 'use_existing_mask_param_set_id' in segmentation_options. Cannot generate job for pre-existing mask. Skipping this job.")
                    continue
                
                job["param_set_id_for_log"] = param_set_id_for_existing_mask

                # Construct Experiment ID for the MASK FOLDER (assumes mask from original scale)
                job["experiment_id_for_mask_folder"] = construct_full_experiment_id(
                    image_id=img_proc_info["original_image_id"],
                    param_set_id=param_set_id_for_existing_mask,
                    scale_factor=1.0, # CRITICAL: Assumes masks were generated from original scale
                    # For non-tiled original masks, processing_unit_name_for_tile might be original filename
                    processing_unit_name_for_tile=os.path.basename(img_proc_info["original_image_filename"]), 
                    is_tile=False # Assuming pre-existing masks are not from a tiled version of original for simplicity.
                                  # If they *could* be, this needs to be more nuanced, possibly another config.
                )

                # Base name for the MASK FILE itself (from original image filename)
                job["mask_filename_base"] = os.path.splitext(os.path.basename(img_proc_info["original_image_filename"]))[0]
                
                # The main "experiment_id_final" for this job entry can be the same as for mask folder,
                # or could be made more unique if needed, but worker uses the specific ones above.
                # For clarity in run_log.json and logs, let's use the mask folder ID.
                job["experiment_id_final"] = job["experiment_id_for_mask_folder"]
                
                # For summarize_job_results, it expects "actual_image_path_to_process" and "processing_unit_name"
                # to log about the job. Let's provide the locally scaled ones here for that summary.
                # The worker will know to use the specific mask-related keys.
                job["actual_image_path_to_process"] = job["scaled_image_path_for_current_run"]
                job["processing_unit_name"] = job["scaled_unit_name_for_current_run"]
                job["scale_factor_applied_for_log"] = job["current_run_scale_factor"]


                img_mpp_x = img_config.get("mpp_x")
                img_mpp_y = img_config.get("mpp_y")
                if img_mpp_x is not None: job["mpp_x_original_for_log"] = img_mpp_x
                if img_mpp_y is not None: job["mpp_y_original_for_log"] = img_mpp_y
                
                job["segmentation_skipped"] = True
                job["USE_GPU"] = global_use_gpu_if_available # Default, not used by worker if skipped

                all_jobs_to_create.append(job)
                logger.info(f"    Created job for pre-existing mask: ExpID for mask folder '{job['experiment_id_for_mask_folder']}', Mask base name '{job['mask_filename_base']}'. Current run unit '{job['scaled_unit_name_for_current_run']}'.")

    logger.info(f"--- Total jobs generated: {len(all_jobs_to_create)} ---")
    return all_jobs_to_create

