# src/pipeline_utils.py
import os
import re
import cv2
import tifffile
import numpy as np
import traceback
import logging

from .file_paths import RESCALED_IMAGE_CACHE_DIR, IMAGE_DIR_BASE, RESULTS_DIR_BASE, PROJECT_ROOT

logger = logging.getLogger(__name__)

RESCALED_IMAGE_CACHE_DIR = os.path.join("images", "rescaled_cache") # Relative to project root

def clean_filename_for_dir(filename):
    # Get the filename without the directory path
    base_filename = os.path.basename(filename)

    # Remove the very last extension (e.g., .tif, .ext, .gz)
    name_without_final_ext = os.path.splitext(base_filename)[0]

    # If the original filename was something like ".bashrc", splitext gives (".bashrc", "").
    # In this case, name_without_final_ext is ".bashrc". We want "bashrc".
    if name_without_final_ext.startswith(".") and not os.path.splitext(name_without_final_ext)[1]:
         # Check if there's actually a name part after the dot
        if len(name_without_final_ext) > 1:
            name_to_clean = name_without_final_ext[1:]
        else: # it's just "."
            name_to_clean = "_" # or handle as an error/empty string
    else:
        name_to_clean = name_without_final_ext

    # Replace hyphens with a temporary placeholder to distinguish them
    # This is because the test 'complex-name.v1.2.ext' -> 'complex-name_v1_2'
    # suggests hyphens should be preserved if they are not part of a sequence to be replaced by a single underscore.
    # However, other tests imply hyphens become underscores. This is contradictory.
    # Given 'complex-name_v1_2', it seems dots become underscores, hyphens are kept.
    # Given 'my image with spaces.ome.tiff' -> 'my_image_with_spaces_ome', spaces and dots become _.
    # Given 'a!b@c#d$.tif' -> 'a_b_c_d_', non-alphanum become _.

    # Strategy:
    # 1. Replace one or more dots with a single underscore.
    # 2. Replace one or more spaces with a single underscore.
    # 3. Replace any remaining character that is not alphanumeric, not an underscore, and not a hyphen, with a single underscore.
    # 4. Consolidate multiple underscores.
    # 5. Strip leading/trailing underscores UNLESS the test 'a!b@c#d$.tif' -> 'a_b_c_d_' implies a trailing underscore is sometimes desired.

    # Let's try a simpler approach first, matching most general cases.
    # Replace sequences of dots, spaces, or hyphens with a single underscore.
    cleaned_name = re.sub(r'[.\s-]+', '_', name_to_clean)
    # Replace any remaining non-alphanumeric character (that's not already an underscore) with an underscore.
    cleaned_name = re.sub(r'[^a-zA-Z0-9_]', '_', cleaned_name)
    # Consolidate multiple underscores.
    cleaned_name = re.sub(r'_+', '_', cleaned_name)
    # Strip leading/trailing underscores. This might be too aggressive for 'a!b@c#d$.tif'.
    cleaned_name = cleaned_name.strip('_')
    
    # Special case for the 'a!b@c#d$.tif' -> 'a_b_c_d_' test: if the original name_without_final_ext ended with a non-alphanum
    # character which became an underscore, and strip('_') removed it, add it back.
    # This is getting very specific. Let's see if the tests pass without this first.

    # The test 'complex-name.v1.2.ext' -> 'complex-name_v1_2' is the trickiest.
    # It implies that '.' should become '_' BUT '-' should be preserved.
    # My current regex [.\s-]+ makes '-' become '_'.

    # Revised strategy for 'complex-name' and general cases:
    # Preserve hyphens, make dots and spaces underscores, then clean other chars.
    
    name_with_hyphens_preserved = name_to_clean.replace('-', '---HYPHEN---')
    name_dots_spaces_to_underscore = re.sub(r'[.\s]+', '_', name_with_hyphens_preserved)
    name_reverted_hyphens = name_dots_spaces_to_underscore.replace('---HYPHEN---', '-')
    
    # Now, clean any remaining non-alphanumeric chars (excluding '-', '_')
    cleaned_name_final = re.sub(r'[^a-zA-Z0-9_-]+', '_', name_reverted_hyphens)
    
    # Strip leading/trailing underscores
    cleaned_name_final = cleaned_name_final.strip('_')

    # Handle the 'a!b@c#d$.tif' -> 'a_b_c_d_' case which wants a trailing underscore.
    # If the original name_without_final_ext ended with a character that became an underscore
    # and was then stripped, we might need to add it back.
    # The character '$' in 'a!b@c#d$' becomes '_' . 'a_b_c_d_'.
    # If name_without_final_ext[-1] is not alphanumeric and not a hyphen, and cleaned_name_final does not end with '_',
    # it suggests a trailing underscore might be expected.
    if name_without_final_ext and not name_without_final_ext[-1].isalnum() and name_without_final_ext[-1] not in ['-', '_'] and not cleaned_name_final.endswith('_'):
         cleaned_name_final += '_'
         # And if this created a double underscore at the end, fix it
         if cleaned_name_final.endswith('__'):
             cleaned_name_final = cleaned_name_final[:-1]


    # If, after all this, the name is empty (e.g. input was "."), return "_"
    if not cleaned_name_final and base_filename: # check base_filename to avoid issues with empty input
        return "_"
        
    return cleaned_name_final

def get_cv2_interpolation_method(method_str="INTER_AREA"):
    methods = {
        "INTER_NEAREST": cv2.INTER_NEAREST,
        "INTER_LINEAR": cv2.INTER_LINEAR,
        "INTER_AREA": cv2.INTER_AREA,
        "INTER_CUBIC": cv2.INTER_CUBIC,
        "INTER_LANCZOS4": cv2.INTER_LANCZOS4
    }
    return methods.get(method_str.upper(), cv2.INTER_AREA)

def rescale_image_and_save(original_image_path, image_id_base, rescaling_config):
    scale_factor = rescaling_config.get("scale_factor", 1.0)
    interpolation_str = rescaling_config.get("interpolation", "INTER_LINEAR")
    interpolation_method = getattr(cv2, interpolation_str, cv2.INTER_LINEAR)

    if scale_factor == 1.0:
        logger.info(f"  Scale factor is 1.0, no rescaling needed for {original_image_path}.")
        return original_image_path, 1.0

    # Construct path for cached rescaled image
    original_filename_base = os.path.splitext(os.path.basename(original_image_path))[0]
    scaled_filename = f"{original_filename_base}_scaled_{str(scale_factor).replace('.', '_')}{os.path.splitext(original_image_path)[1]}"
    
    # Ensure RESCALED_IMAGE_CACHE_DIR is an absolute path if it's defined relative to PROJECT_ROOT
    # This assumes RESCALED_IMAGE_CACHE_DIR is already correctly defined (e.g. using os.path.join(PROJECT_ROOT, "images/rescaled_cache"))
    # Create a subdirectory within the cache for this specific image_id_base to avoid filename clashes if multiple original images have same basename.
    image_specific_cache_dir = os.path.join(RESCALED_IMAGE_CACHE_DIR, image_id_base)
    if not os.path.exists(image_specific_cache_dir):
        try:
            os.makedirs(image_specific_cache_dir)
        except OSError as e:
            logger.error(f"  Error creating image-specific cache directory {image_specific_cache_dir}: {e}. Cannot save rescaled image.")
            return original_image_path, 1.0


    cached_rescaled_image_path = os.path.join(image_specific_cache_dir, scaled_filename)

    if os.path.exists(cached_rescaled_image_path):
        logger.info(f"  Found cached rescaled image: {cached_rescaled_image_path}")
        return cached_rescaled_image_path, scale_factor

    try:
        logger.info(f"  Rescaling {original_image_path} by factor {scale_factor} using {interpolation_str}...")
        img = tifffile.imread(original_image_path)
        
        if img is None:
            logger.error(f"  Error: Could not load original image {original_image_path} for rescaling using tifffile.")
            return original_image_path, 1.0

        logger.debug(f"    Original image shape: {img.shape}, dtype: {img.dtype}")

        img_to_resize = None
        if img.ndim == 2: # Grayscale HxW
            img_to_resize = img
        elif img.ndim == 3: # Could be HxWxC or ZxHxW
            # Assuming for simple rescaling, if it's ZxHxW, we take the middle slice or first slice.
            # Or if HxWxC, cv2.resize handles it. Let's assume HxW or HxWxC for cv2.
            # A common case for OME-TIFFs read by tifffile might be (Series, Z, C, H, W, S) or simpler forms.
            # For now, let's try to be robust for common 2D/3D TIFFs (H,W), (H,W,C), (Z,H,W)
            if img.shape[0] < 10 and img.ndim == 3 : # Probably (Z, H, W) or (C, H, W) with few Z/C
                 logger.info(f"    Input is 3D with shape {img.shape}. Attempting to rescale the first slice/channel as 2D.")
                 img_to_resize = img[0, :, :] # Take the first slice/channel
            elif img.shape[-1] < 10 and img.ndim == 3: # Probably (H,W,C)
                 img_to_resize = img
            else: # Default to first slice if unsure for ZxHxW, or if it's truly volumetric and needs specific handling
                 logger.warning(f"    Input image is {img.ndim}D with shape {img.shape}. Taking first slice/channel if it's the first dimension, otherwise attempting direct resize. This might not be suitable for all volumetric images.")
                 if img.ndim > 2 and img.shape[0] > 1 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]: # Heuristic for (Z,H,W)
                    img_to_resize = img[0,:,:]
                 else: # Hope cv2.resize can handle it or it's (H,W) after all
                    img_to_resize = img


        if img_to_resize is None or img_to_resize.ndim < 2: # Check if img_to_resize is valid
            logger.error(f"    Error: Could not extract a suitable 2D/3D array for resizing from shape {img.shape}. Original path: {original_image_path}")
            return original_image_path, 1.0
        
        logger.debug(f"    Shape of array being sent to cv2.resize: {img_to_resize.shape}")

        original_h, original_w = img_to_resize.shape[0], img_to_resize.shape[1]
        new_w = int(round(original_w * scale_factor))
        new_h = int(round(original_h * scale_factor))

        if new_w <= 0 or new_h <= 0:
            logger.error(f"  Error: Calculated new dimensions ({new_w}x{new_h}) are invalid for {original_image_path}.")
            return original_image_path, 1.0
            
        logger.info(f"    Original dimensions: {original_w}x{original_h}. Target scaled dimensions: {new_w}x{new_h}.")
        
        rescaled_img = cv2.resize(img_to_resize, (new_w, new_h), interpolation=interpolation_method)
        logger.info(f"    Shape after cv2.resize: {rescaled_img.shape}, dtype: {rescaled_img.dtype}")

        try:
            tifffile.imwrite(cached_rescaled_image_path, rescaled_img)
            logger.info(f"    Successfully rescaled and cached to: {cached_rescaled_image_path}")
            return cached_rescaled_image_path, scale_factor
        except Exception as e_write:
            logger.error(f"  CRITICAL: Failed to write rescaled image to {cached_rescaled_image_path}: {e_write}")
            logger.error(traceback.format_exc())
            return original_image_path, 1.0 # Fallback if write fails

    except Exception as e:
        logger.error(f"  Error during rescaling of {original_image_path}: {e}")
        logger.error(traceback.format_exc())
        return original_image_path, 1.0 # Fallback

def determine_image_unit_for_segmentation_and_mask_path(config, RESULTS_DIR_BASE, TILED_IMAGE_DIR_BASE, original_source_image_full_path, target_image_id, target_processing_unit_name, is_tiled_job, tile_job_params=None, rescaling_config_for_image=None):
    """
    Determines the actual image path to be used for segmentation (could be original, rescaled, or a tile)
    and the corresponding mask path where segmentation results should be stored or found.
    """
    applied_scale_factor = 1.0
    path_of_image_unit_for_segmentation = original_source_image_full_path 

    # 1. Handle potential rescaling (primary mechanism via rescaling_config_for_image)
    if rescaling_config_for_image and rescaling_config_for_image.get("scale_factor") != 1.0:
        logger.info(f"  Rescaling config found for {target_image_id}: factor {rescaling_config_for_image.get('scale_factor')}")
        image_path_for_potential_rescaling = original_source_image_full_path
        rescaled_path, actual_sf = rescale_image_and_save(
            image_path_for_potential_rescaling,
            target_image_id, 
            rescaling_config_for_image
        )
        if actual_sf != 1.0 and os.path.exists(rescaled_path):
            path_of_image_unit_for_segmentation = rescaled_path
            applied_scale_factor = actual_sf
            logger.info(f"  Path after (potential) rescaling: {path_of_image_unit_for_segmentation}, Scale factor: {applied_scale_factor}")
        else:
            logger.info(f"  Rescaling did not change path or failed. Using: {path_of_image_unit_for_segmentation}")

    # 2. Handle tiling
    if is_tiled_job and tile_job_params:
        tile_parent_dir_name = tile_job_params.get("tile_parent_dir_name", target_image_id)
        # If the original image was rescaled *before* tiling, tile_parent_dir_name should reflect that.
        # This part assumes tile_parent_dir_name is correctly set by the tiling process.
        path_of_image_unit_for_segmentation = os.path.join(TILED_IMAGE_DIR_BASE, tile_parent_dir_name, target_processing_unit_name)
        logger.info(f"  This is a tiled job. Segmentation will use tile: {path_of_image_unit_for_segmentation}")
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
                    logger.info(f"  Inferred applied_scale_factor {applied_scale_factor} from target_processing_unit_name '{target_processing_unit_name}'")
                else:
                    logger.warning(f"  Warning: Parsed scale factor {parsed_sf} from filename is invalid.")
            else: # Fallback for names like "..._scaled_0_25_mask.tif" (if mask name is passed)
                match_mask = re.search(r"_scaled_([0-9]+(?:_[0-9]+)?).*_mask\.[^.]+$", os.path.basename(target_processing_unit_name))
                if match_mask:
                    scale_str = match_mask.group(1).replace('_', '.')
                    parsed_sf = float(scale_str)
                    if 0 < parsed_sf <= 1.0:
                        applied_scale_factor = parsed_sf
                        logger.info(f"  Inferred applied_scale_factor {applied_scale_factor} from target_processing_unit_name (mask pattern) '{target_processing_unit_name}'")
                    else:
                        logger.warning(f"  Warning: Parsed scale factor {parsed_sf} from mask filename is invalid.")
        except ValueError:
            logger.error(f"  Could not parse scale factor from filename: {target_processing_unit_name}")


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

def get_base_experiment_id(image_id: str, param_set_id: str) -> str:
    """Constructs the base part of an experiment ID."""
    return f"{image_id}_{param_set_id}"

def format_scale_factor_for_path(scale_factor: float | None) -> str:
    """Formats a scale factor for use in paths/IDs (e.g., _scaled_0_5). Returns empty string if scale is 1.0 or None."""
    if scale_factor is not None and scale_factor != 1.0:
        return f"_scaled_{str(scale_factor).replace('.', '_')}"
    return ""

def construct_full_experiment_id(image_id: str, param_set_id: str, scale_factor: float | None = None, processing_unit_name_for_tile: str | None = None, is_tile: bool = False) -> str:
    """
    Constructs the full experiment ID used for result folder names.
    For tiles, it incorporates a sanitized version of the tile name.
    For non-tiles, it incorporates a scale factor string if applicable.
    """
    base_id = get_base_experiment_id(image_id, param_set_id)
    if is_tile and processing_unit_name_for_tile:
        # Use clean_filename_for_dir to clean the tile name part for the ID
        # clean_filename_for_dir removes the extension, which is appropriate here.
        sanitized_tile_name = clean_filename_for_dir(processing_unit_name_for_tile)
        return f"{base_id}_{sanitized_tile_name}"
    else:
        scale_str = format_scale_factor_for_path(scale_factor)
        return f"{base_id}{scale_str}"

def construct_mask_path(results_dir: str, experiment_id: str, processing_unit_name: str) -> str:
    """
    Constructs the full path to a mask file within the results directory.
    Ensures the mask filename ends with _mask.tif.
    Args:
        results_dir (str): Base directory for all results.
        experiment_id (str): The specific experiment subfolder ID.
        processing_unit_name (str): The name of the image/tile that was processed.
    Returns:
        str: The full path to the mask file.
    """
    mask_basename = os.path.splitext(processing_unit_name)[0]
    if not mask_basename.endswith("_mask"):
        mask_filename = f"{mask_basename}_mask.tif"
    else: # Already contains _mask, just ensure .tif extension
        mask_filename = f"{mask_basename}.tif"
    
    return os.path.join(results_dir, experiment_id, mask_filename)

def normalize_to_8bit_for_display(img_array):
    if img_array is None or img_array.size == 0:
        logger.warning("normalize_to_8bit received None or empty input.")
        # Determine placeholder shape based on whether original img_array hinted at being 3-channel
        placeholder_shape = (100, 100, 3) if (img_array is not None and hasattr(img_array, 'ndim') and img_array.ndim == 3 and hasattr(img_array, 'shape') and len(img_array.shape) > 2 and img_array.shape[-1] == 3) else (100, 100)
        return np.zeros(placeholder_shape, dtype=np.uint8)

    try:
        # If already uint8, return as is
        if img_array.dtype == np.uint8:
            logger.info("Input image is already uint8. Returning as is.")
            return img_array

        # Perform robust intensity scaling using percentiles
        min_val = np.percentile(img_array, 1)
        max_val = np.percentile(img_array, 99)

        # Handle cases where the image has no dynamic range after percentile clipping
        if max_val <= min_val:
            logger.warning(f"Image has no dynamic range after percentile clipping (min_val: {min_val}, max_val: {max_val}). Original dtype: {img_array.dtype}. Returning a zero image.")
            return np.zeros_like(img_array, dtype=np.uint8) # Return black image of same shape

        # Clip values to the determined range then scale to 0-255
        img_clipped = np.clip(img_array, min_val, max_val)
        
        # Perform normalization
        img_normalized = (img_clipped - min_val) / (max_val - min_val) * 255.0
        
        # Convert to uint8
        img_8bit = img_normalized.astype(np.uint8)
        
        logger.info(f"Successfully normalized image from dtype {img_array.dtype} (original min: {img_array.min()}, original max: {img_array.max()}; "
                    f"clipped & scaled using min: {min_val}, max: {max_val}). Output shape: {img_8bit.shape}")
        return img_8bit

    except Exception as e:
        logger.error(f"Error during 8-bit normalization for input of dtype {img_array.dtype} and shape {img_array.shape}: {e}")
        logger.error(traceback.format_exc())
        # Fallback: return a black image
        try:
            placeholder_shape = img_array.shape
        except: # Handle if img_array itself has no shape (shouldn't happen if initial check passed)
            placeholder_shape = (100,100,3) if (hasattr(img_array, 'ndim') and img_array.ndim == 3 and hasattr(img_array, 'shape') and len(img_array.shape) > 2 and img_array.shape[-1] == 3) else (100,100)
        return np.zeros(placeholder_shape, dtype=np.uint8)

def get_image_mpp_and_path_from_config(all_image_configs, target_image_id, target_processing_unit_name=None):
    """
    Finds the image configuration for a given image_id and returns the config,
    the full path to the original image, and its MPP values.

    Args:
        all_image_configs (list): A list of image configuration dictionaries.
        target_image_id (str): The image_id to search for.
        target_processing_unit_name (str, optional): The specific processing unit name.
            Currently primarily used for logging context if image_id is not found,
            but could be used for more complex logic in the future.

    Returns:
        tuple: (image_config, original_image_full_path, mpp_x, mpp_y)
               Returns (None, None, None, None) if the image_id is not found
               or if essential information is missing.
    """
    if not all_image_configs:
        logger.warning("get_image_mpp_and_path_from_config: all_image_configs list is empty or None.")
        return None, None, None, None

    for img_cfg in all_image_configs:
        if img_cfg.get("image_id") == target_image_id:
            original_filename = img_cfg.get("original_image_filename")
            mpp_x = img_cfg.get("mpp_x")
            mpp_y = img_cfg.get("mpp_y")

            if not original_filename:
                logger.error(f"Image config for '{target_image_id}' found, but 'original_image_filename' is missing.")
                return img_cfg, None, mpp_x, mpp_y # Return what we have, path will be None

            if mpp_x is None or mpp_y is None:
                logger.warning(f"Image config for '{target_image_id}' found, but 'mpp_x' or 'mpp_y' is missing.")
                # Continue, but mpp values might be None

            original_image_full_path = os.path.join(PROJECT_ROOT, original_filename)
            
            # Log if the constructed path doesn't exist, but still return it.
            # The caller can decide how to handle a non-existent path.
            if not os.path.exists(original_image_full_path):
                logger.warning(f"Original image for '{target_image_id}' not found at derived path: {original_image_full_path}")

            return img_cfg, original_image_full_path, mpp_x, mpp_y

    processing_unit_context = f" (for processing unit '{target_processing_unit_name}')" if target_processing_unit_name else ""
    logger.error(f"Image ID '{target_image_id}' not found in any image_configurations{processing_unit_context}.")
    return None, None, None, None

__all__ = [
    "clean_filename_for_dir",
    "get_cv2_interpolation_method",
    "rescale_image_and_save",
    "determine_image_unit_for_segmentation_and_mask_path",
    "get_base_experiment_id",
    "format_scale_factor_for_path",
    "construct_full_experiment_id",
    "construct_mask_path",
    "normalize_to_8bit_for_display",
    "get_image_mpp_and_path_from_config"
]
