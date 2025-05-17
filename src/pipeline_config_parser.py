# src/pipeline_config_parser.py
import os
import json
import time
import traceback
from .pipeline_utils import clean_filename_for_dir, rescale_image_and_save # Relative import
from .tile_large_image import tile_image # Relative import

IMAGE_DIR_BASE = "images"
TILED_IMAGE_OUTPUT_BASE = os.path.join(IMAGE_DIR_BASE, "tiled_outputs")

def load_and_expand_configurations(param_json_file_path):
    """
    Loads configurations from parameter_sets.json, handles rescaling and tiling,
    and generates a flat list of all individual jobs to be run.
    Each job in the list is a dictionary fully resolved with all parameters.
    """
    if not os.path.exists(param_json_file_path):
        print(f"Error: Config file '{param_json_file_path}' not found.")
        return []

    config_data = {}
    try:
        with open(param_json_file_path, 'r') as f:
            config_data = json.load(f)
    except Exception as e:
        print(f"Error reading/parsing {param_json_file_path}: {e}")
        traceback.print_exc()
        return []

    image_configs_from_json = config_data.get("image_configurations", [])
    cellpose_param_configs = config_data.get("cellpose_parameter_configurations", [])

    if not image_configs_from_json:
        print("No 'image_configurations' found in JSON.")
        return []
    if not cellpose_param_configs:
        print("No 'cellpose_parameter_configurations' found in JSON.")
        return []

    all_jobs_to_create = []
    print("--- Generating Job List (from pipeline_config_parser) ---")

    for img_config in image_configs_from_json:
        if not img_config.get("is_active", True):
            print(f"Skipping inactive image configuration: {img_config.get('image_id', 'UnnamedImageConfig')}")
            continue

        original_image_filename = img_config.get("original_image_filename")
        image_id_base = img_config.get("image_id", clean_filename_for_dir(original_image_filename) if original_image_filename else f"img_unknown_{time.strftime('%Y%m%d%H%M%S')}")
        
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

        if rescaling_cfg and "scale_factor" in rescaling_cfg and rescaling_cfg["scale_factor"] != 1.0 :
            print(f"  Image Config: '{image_id_base}' (Rescaling {original_image_filename} by factor {rescaling_cfg['scale_factor']})")
            current_image_path_for_processing, applied_scale_factor = rescale_image_and_save(
                original_image_path, 
                image_id_base, 
                rescaling_cfg
            )
            current_image_name_for_processing = os.path.basename(current_image_path_for_processing)
            if applied_scale_factor == 1.0 and current_image_path_for_processing == original_image_path:
                 print(f"    Rescaling resulted in no change or failed, using original image: {original_image_path}")
        else:
            print(f"  Image Config: '{image_id_base}' (No rescaling for {original_image_filename})")

        images_to_segment_this_round = [] 
        tiling_cfg = img_config.get("tiling_config")

        if tiling_cfg and tiling_cfg.get("tile_size"):
            print(f"    Tiling configured for: {current_image_name_for_processing} (orig: {original_image_filename})")
            
            tile_storage_dir_suffix = image_id_base 
            if applied_scale_factor != 1.0:
                scale_factor_str = str(applied_scale_factor).replace('.', '_')
                tile_storage_dir_suffix += f"_scaled{scale_factor_str}"
            
            tile_storage_dir = os.path.join(TILED_IMAGE_OUTPUT_BASE, tile_storage_dir_suffix)
            tile_prefix = tiling_cfg.get("tile_output_prefix_base", clean_filename_for_dir(current_image_name_for_processing) + "_tile")
            
            if not os.path.exists(tile_storage_dir):
                 try: os.makedirs(tile_storage_dir); print(f"      Created tile storage directory: {tile_storage_dir}")
                 except OSError as e: 
                     print(f"      Error creating tile storage dir {tile_storage_dir}: {e}. Skipping tiling for this image config."); continue

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
                print(f"      Generated {len(tile_manifest_data['tiles'])} tiles for {current_image_name_for_processing}.")
            else: 
                print(f"      Warning: Tiling failed or no tiles for {current_image_name_for_processing}. Using this image directly.")
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
                
                param_set_id = cp_config.get("param_set_id", f"cp_params_unknown_{time.strftime('%H%M%S')}")
                job = {} 
                
                for key, value in cp_config.items():
                    if key not in ["param_set_id", "is_active"]:
                        job[key] = value
                
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
                else:
                    job["DIAMETER_FOR_CELLPOSE"] = cp_diameter 
                
                cleaned_proc_unit_name = clean_filename_for_dir(img_proc_info["name"])
                if img_proc_info["is_tile"]:
                    job["experiment_id_final"] = f"{img_proc_info['original_image_id']}_{param_set_id}_{cleaned_proc_unit_name}"
                else: 
                    job["experiment_id_final"] = f"{img_proc_info['original_image_id']}_{param_set_id}"
                    if img_proc_info["applied_scale_factor"] != 1.0: 
                         scale_factor_str = str(img_proc_info['applied_scale_factor']).replace('.','_')
                         job["experiment_id_final"] += f"_scaled{scale_factor_str}"

                job.setdefault("MODEL_CHOICE", "cyto3")
                job.setdefault("FLOW_THRESHOLD", None) 
                job.setdefault("MIN_SIZE", None) 
                job.setdefault("CELLPROB_THRESHOLD", 0.0)
                job.setdefault("FORCE_GRAYSCALE", True)
                job.setdefault("USE_GPU", False)
                
                all_jobs_to_create.append(job)

    return all_jobs_to_create

