# src/pipeline_utils.py
import os
import re
import cv2
import tifffile
import numpy as np
import traceback

RESCALED_IMAGE_CACHE_DIR = os.path.join("images", "rescaled_cache") # Relative to project root

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

def rescale_image_and_save(original_image_path, image_id_for_cache, rescaling_config):
    """
    Loads an image, rescales it, and saves it to a cache directory.
    Returns the path to the rescaled image and the actual scale factor used.
    Assumes RESCALED_IMAGE_CACHE_DIR is accessible from project root.
    """
    if not os.path.exists(original_image_path):
        print(f"  Error: Original image for rescaling not found: {original_image_path}")
        return original_image_path, 1.0

    if not rescaling_config or "scale_factor" not in rescaling_config:
        return original_image_path, 1.0

    scale_factor = rescaling_config["scale_factor"]
    if not (isinstance(scale_factor, (int, float)) and 0 < scale_factor <= 1.0):
        print(f"  Warning: Invalid scale_factor {scale_factor} for {original_image_path}. Must be float between 0 (excl) and 1.0. Skipping rescale.")
        return original_image_path, 1.0
    if scale_factor == 1.0: # No actual rescaling
        return original_image_path, 1.0

    interpolation_str = rescaling_config.get("interpolation", "INTER_AREA")
    interpolation_method = get_cv2_interpolation_method(interpolation_str)
    
    if not os.path.exists(RESCALED_IMAGE_CACHE_DIR):
        try: os.makedirs(RESCALED_IMAGE_CACHE_DIR)
        except OSError as e: print(f"Could not create base rescaled cache dir {RESCALED_IMAGE_CACHE_DIR}: {e}")
    
    rescaled_image_specific_dir = os.path.join(RESCALED_IMAGE_CACHE_DIR, image_id_for_cache)
    if not os.path.exists(rescaled_image_specific_dir):
        try: os.makedirs(rescaled_image_specific_dir)
        except OSError as e:
            print(f"  Error creating cache subdir {rescaled_image_specific_dir}: {e}. Cannot cache rescaled image.")
            return original_image_path, 1.0

    original_filename = os.path.basename(original_image_path)
    scale_factor_str_file = str(scale_factor).replace('.', '_')
    rescaled_image_filename = f"{os.path.splitext(original_filename)[0]}_scaled_{scale_factor_str_file}.tif"
    rescaled_image_path = os.path.join(rescaled_image_specific_dir, rescaled_image_filename)

    if os.path.exists(rescaled_image_path):
        print(f"  Found cached rescaled image: {rescaled_image_path}")
        return rescaled_image_path, scale_factor

    try:
        print(f"  Rescaling {original_image_path} by factor {scale_factor} using {interpolation_str}...")
        img = cv2.imread(original_image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if img is None:
            print(f"  Error: Could not load original image {original_image_path} for rescaling.")
            return original_image_path, 1.0

        new_width = int(round(img.shape[1] * scale_factor))
        new_height = int(round(img.shape[0] * scale_factor))
        
        if new_width == 0 or new_height == 0:
            print(f"  Error: Rescaled dimensions are zero or near-zero ({new_width}x{new_height}). Scale factor {scale_factor} too small for image size {img.shape[:2]}.")
            return original_image_path, 1.0

        rescaled_img = cv2.resize(img, (new_width, new_height), interpolation=interpolation_method)
        
        tifffile.imwrite(rescaled_image_path, rescaled_img)
        print(f"  Saved rescaled image to: {rescaled_image_path} (Shape: {rescaled_img.shape})")
        return rescaled_image_path, scale_factor
    except Exception as e:
        print(f"  Error during rescaling or saving {original_image_path}: {e}")
        traceback.print_exc()
        return original_image_path, 1.0

def determine_image_unit_for_segmentation_and_mask_path(config, RESULTS_DIR_BASE, TILED_IMAGE_DIR_BASE, original_source_image_full_path, target_image_id, target_processing_unit_name, is_tiled_job, tile_job_params=None, rescaling_config_for_image=None):
    """
    Determines the actual image path to be used for segmentation (could be original, rescaled, or a tile)
    and the corresponding mask path where segmentation results should be stored or found.
    """
    applied_scale_factor = 1.0
    path_of_image_unit_for_segmentation = original_source_image_full_path 

    # 1. Handle potential rescaling (primary mechanism via rescaling_config_for_image)
    if rescaling_config_for_image and rescaling_config_for_image.get("scale_factor") != 1.0:
        print(f"  Rescaling config found for {target_image_id}: factor {rescaling_config_for_image.get('scale_factor')}")
        image_path_for_potential_rescaling = original_source_image_full_path
        rescaled_path, actual_sf = rescale_image_and_save(
            image_path_for_potential_rescaling,
            target_image_id, 
            rescaling_config_for_image
        )
        if actual_sf != 1.0 and os.path.exists(rescaled_path):
            path_of_image_unit_for_segmentation = rescaled_path
            applied_scale_factor = actual_sf
            print(f"  Path after (potential) rescaling: {path_of_image_unit_for_segmentation}, Scale factor: {applied_scale_factor}")
        else:
            print(f"  Rescaling did not change path or failed. Using: {path_of_image_unit_for_segmentation}")

    # 2. Handle tiling
    if is_tiled_job and tile_job_params:
        tile_parent_dir_name = tile_job_params.get("tile_parent_dir_name", target_image_id)
        # If the original image was rescaled *before* tiling, tile_parent_dir_name should reflect that.
        # This part assumes tile_parent_dir_name is correctly set by the tiling process.
        path_of_image_unit_for_segmentation = os.path.join(TILED_IMAGE_DIR_BASE, tile_parent_dir_name, target_processing_unit_name)
        print(f"  This is a tiled job. Segmentation will use tile: {path_of_image_unit_for_segmentation}")
        # If tiling is active, the `applied_scale_factor` should ideally be known from the `rescaling_config_for_image`
        # that was applied *before* tiling, or be 1.0 if no pre-tiling rescale happened.

    # 2.5. Infer applied_scale_factor from target_processing_unit_name if not already set by rescaling_config
    # This is a fallback, particularly useful when locating existing processed files (like in visualization scripts)
    # where the filename itself indicates scaling.
    if applied_scale_factor == 1.0 and "_scaled_" in target_processing_unit_name:
        try:
            # Attempt to extract scale factor like "0_25" from "...._scaled_0_25.tif"
            match = re.search(r"_scaled_([0-9]+(?:_[0-9]+)?)\.[^.]+$", os.path.basename(target_processing_unit_name))
            if match:
                scale_str = match.group(1).replace('_', '.')
                parsed_sf = float(scale_str)
                if 0 < parsed_sf <= 1.0:
                    applied_scale_factor = parsed_sf
                    print(f"  Inferred applied_scale_factor {applied_scale_factor} from target_processing_unit_name '{target_processing_unit_name}'")
                else:
                    print(f"  Warning: Parsed scale factor {parsed_sf} from filename is invalid.")
            else: # Fallback for names like "..._scaled_0_25_mask.tif" (if mask name is passed)
                match_mask = re.search(r"_scaled_([0-9]+(?:_[0-9]+)?).*_mask\.[^.]+$", os.path.basename(target_processing_unit_name))
                if match_mask:
                    scale_str = match_mask.group(1).replace('_', '.')
                    parsed_sf = float(scale_str)
                    if 0 < parsed_sf <= 1.0:
                        applied_scale_factor = parsed_sf
                        print(f"  Inferred applied_scale_factor {applied_scale_factor} from target_processing_unit_name (mask pattern) '{target_processing_unit_name}'")
                    else:
                        print(f"  Warning: Parsed scale factor {parsed_sf} from mask filename is invalid.")
        except ValueError:
            print(f"  Could not parse scale factor from filename: {target_processing_unit_name}")


    # 3. Determine the mask path folder and filename
    params_name_suffix = config.get("segmentation_params_name", "Cellpose_DefaultDiam")
    experiment_id_final_for_mask_folder = f"{target_image_id}_{params_name_suffix}"

    # Append scaling factor to folder name if applicable
    if applied_scale_factor != 1.0:
        scale_factor_str_folder = str(applied_scale_factor).replace('.', '_')
        experiment_id_final_for_mask_folder += f"_scaled{scale_factor_str_folder}"

    # The mask filename part is based on the specific processing unit name
    # (e.g., "image_scaled_0_25_mask.tif" or "tile_X_Y_mask.tif")
    mask_filename_base = os.path.splitext(os.path.basename(target_processing_unit_name))[0]
    # Ensure we don't double-append "_mask" if it's already in target_processing_unit_name (e.g. if a mask path was passed)
    if "_mask" not in mask_filename_base: # Check if original name had _mask
        mask_filename_part = f"{mask_filename_base}_mask.tif"
    else: # If target_processing_unit_name already describes a mask
        mask_filename_part = os.path.basename(target_processing_unit_name)


    mask_path = os.path.join(RESULTS_DIR_BASE, experiment_id_final_for_mask_folder, mask_filename_part)
            
    return path_of_image_unit_for_segmentation, mask_path, applied_scale_factor, original_source_image_full_path
