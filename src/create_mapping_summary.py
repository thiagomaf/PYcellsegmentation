import os
import json
import numpy as np
import pandas as pd
import cv2 # For loading images for re-plotting
import re
import tifffile # <<< ADD IMPORT

import matplotlib # Import matplotlib first
matplotlib.use('Agg') # Set backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import matplotlib.colors
import math
import argparse
from .file_paths import (
    PROJECT_ROOT,
    RESULTS_DIR_RELATIVE_TO_PROJECT,
    VISUALIZATION_CONFIG_FILENAME,
    PROCESSING_CONFIG_FILENAME, # For finding original image paths
    RESCALED_IMAGE_CACHE_DIR # <<< ADD IMPORT HERE
)
from .pipeline_utils import (
    clean_filename_for_dir, 
    normalize_to_8bit_for_display, # For background image display
    get_image_mpp_and_path_from_config, # To get background image path
    construct_full_experiment_id # For constructing full experiment ID
)
from cellpose import io as cellpose_io # For loading masks

# Default configuration
IMAGES_PER_ROW_DEFAULT = 4
DEFAULT_SUMMARY_SUBDIR = "summary"

logger = None # Placeholder, can be replaced with proper logging

def _parse_gene_visualization_list(genes_spec):
    """
    Parses the gene specification (list of IDs or dict of Alias:ID)
    into a list of (display_name, actual_gene_id) tuples.
    (Identical to the one in visualize_gene_expression.py for consistency)
    """
    parsed_genes = []
    if isinstance(genes_spec, dict):
        for display_name, actual_id in genes_spec.items():
            if isinstance(actual_id, str) and isinstance(display_name, str):
                parsed_genes.append((display_name, actual_id))
            else:
                logger.warning(f"Invalid entry in gene dictionary: key '{display_name}' or value '{actual_id}' is not a string. Skipping.")
    elif isinstance(genes_spec, list):
        for gene_entry in genes_spec:
            if isinstance(gene_entry, str):
                parsed_genes.append((gene_entry, gene_entry)) # Display name is the same as actual ID
            elif isinstance(gene_entry, dict) and len(gene_entry) == 1:
                display_name, actual_id = list(gene_entry.items())[0]
                if isinstance(actual_id, str) and isinstance(display_name, str):
                    parsed_genes.append((display_name, actual_id))
                else:
                    logger.warning(f"Invalid entry in gene list of dicts: {gene_entry}. Key or value not strings. Skipping.")
            else:
                logger.warning(f"Unsupported gene entry type in list: {gene_entry}. Skipping.")
    elif genes_spec is not None:
        logger.warning(f"genes_to_visualize is neither a list nor a dict, but: {type(genes_spec)}. No genes will be processed.")
    
    if not parsed_genes and genes_spec:
        logger.warning(f"Could not parse any valid genes from specification: {genes_spec}")
    elif not parsed_genes and not genes_spec:
        logger.info("Gene specification is empty. No genes to parse.")
    return parsed_genes

def setup_logging():
    # Basic logger for now, can be expanded
    global logger
    import logging
    import traceback # <<< ADD IMPORT FOR TRACEBACK IN EXCEPTION LOGGING
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

def load_json_config(config_path, config_name="JSON"):
    resolved_path = os.path.abspath(config_path)
    logger.info(f"Attempting to load {config_name} config from: {resolved_path}")
    if not os.path.exists(resolved_path):
        logger.error(f"{config_name} config file not found at resolved path: {resolved_path}")
        return None
    try:
        with open(resolved_path, 'r') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded {config_name}. Top keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            return data
    except Exception as e:
        logger.error(f"Error reading/parsing {config_name} config {resolved_path}: {e}")
        return None

def find_task_in_config(config_data, task_id):
    if config_data and "visualization_tasks" in config_data and isinstance(config_data["visualization_tasks"], dict):
        logger.info(f"Keys within config_data['visualization_tasks']: {list(config_data['visualization_tasks'].keys())}")
    else:
        logger.info("config_data or config_data['visualization_tasks'] is not as expected before checking for 'tasks' key.")

    if not config_data or "visualization_tasks" not in config_data or "tasks" not in config_data["visualization_tasks"]:
        logger.error("Invalid visualization config structure. Problem is with 'tasks' key presence or parent structure.")
        return None, None

    all_tasks = config_data["visualization_tasks"]["tasks"]
    for task in all_tasks:
        if task.get("task_id") == task_id:
            return task, config_data["visualization_tasks"].get("default_genes_to_visualize", [])
    logger.error(f"Task ID '{task_id}' not found in visualization config.")
    return None, None

# Function to be adapted from visualize_gene_expression.py for re-plotting a single gene overlay
# This will be a simplified version focused on generating the overlay with a given global normalization
def generate_single_gene_overlay_globally_scaled(
    background_img_8bit, 
    mask_data,
    gene_cell_counts_df, # DataFrame with 'cell_id' and 'transcript_count' for the current gene
    colormap_func, # Result of plt.get_cmap()
    global_norm, # Result of matplotlib.colors.Normalize(vmin=global_min, vmax=global_max)
    log_scale_current_gene_counts # Boolean, indicates if current gene's counts need log1p (already factored into global_norm range)
    ):
    
    overlay_img = background_img_8bit.copy()
    # If background is grayscale, convert to color for overlaying colored cells
    if overlay_img.ndim == 2:
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2RGB)
    elif overlay_img.ndim == 3 and overlay_img.shape[-1] == 1: # Grayscale with channel dim
        overlay_img = cv2.cvtColor(overlay_img[:,:,0], cv2.COLOR_GRAY2RGB)
    
    # Ensure it's RGB for consistent color application
    if overlay_img.shape[-1] == 4: # RGBA
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2RGB)

    # Create a color map for cells based on their transcript count for THIS gene, using the GLOBAL scale
    for index, row in gene_cell_counts_df.iterrows():
        cell_id_val = row['cell_id']
        # Ensure cell_id is not None or NaN before converting to int
        if pd.isna(cell_id_val):
            logger.warning(f"Skipping row with null/NaN cell_id in gene_cell_counts_df. Row: {row}")
            continue
        try:
            cell_id = int(cell_id_val)
        except ValueError:
            logger.warning(f"Skipping row with non-integer cell_id '{cell_id_val}'. Row: {row}")
            continue
            
        count = row['transcript_count']
        
        if cell_id == 0: continue # Skip background pseudo-cell if present

        current_value_for_norm = np.log1p(count) if log_scale_current_gene_counts else count
        normalized_value = global_norm(current_value_for_norm) # Apply global normalization
        rgba_color = colormap_func(normalized_value) # Get color from colormap
        color_to_apply = (np.array(rgba_color[:3]) * 255).astype(np.uint8)

        # --- Debugging logs ---
        logger.debug(f"Processing cell_id: {cell_id} (type: {type(cell_id)})")
        if mask_data is not None:
            logger.debug(f"mask_data shape: {mask_data.shape}, dtype: {mask_data.dtype}")
            condition_mask = (mask_data == cell_id)
            logger.debug(f"Condition mask (mask_data == cell_id) sum: {condition_mask.sum()}, shape: {condition_mask.shape}")
            # If no pixels match this cell_id in the mask, skip trying to color it.
            if condition_mask.sum() == 0:
                logger.debug(f"No pixels found for cell_id {cell_id} in mask_data. Skipping addWeighted for this cell.")
                continue # Skip to the next cell_id
            try:
                selected_overlay_pixels = overlay_img[condition_mask]
                logger.debug(f"overlay_img[condition_mask] shape: {selected_overlay_pixels.shape}")
            except Exception as e_debug_shape:
                logger.debug(f"Error getting shape of overlay_img[condition_mask]: {e_debug_shape}")
        else:
            logger.debug("mask_data is None.")
        logger.debug(f"overlay_img shape: {overlay_img.shape}, dtype: {overlay_img.dtype}")
        logger.debug(f"color_to_apply: {color_to_apply}, type: {type(color_to_apply)}, dtype: {color_to_apply.dtype}")
        # --- End Debugging logs ---

        # Apply color to the mask region for this cell_id
        try:
            overlay_img[mask_data == cell_id] = cv2.addWeighted(
                overlay_img[mask_data == cell_id],
                0.5, # Alpha for original background under cell
                np.full_like(overlay_img[mask_data == cell_id], color_to_apply, dtype=np.uint8),
                0.5, # Alpha for cell color
                0
            )
        except TypeError as te:
            logger.error(f"TypeError during cv2.addWeighted for cell_id {cell_id}: {te}")
            logger.error(f"  Details - cell_id: {cell_id}, type(cell_id): {type(cell_id)}")
            if mask_data is not None:
                logger.error(f"  mask_data.shape: {mask_data.shape}, mask_data.dtype: {mask_data.dtype}")
                condition_mask_err = (mask_data == cell_id)
                logger.error(f"  condition_mask_err.sum(): {condition_mask_err.sum()}")
                selected_pixels_err_shape = overlay_img[condition_mask_err].shape
                logger.error(f"  overlay_img[condition_mask_err].shape: {selected_pixels_err_shape}")
            else:
                logger.error("  mask_data is None at point of error.")
            logger.error(f"  overlay_img.shape: {overlay_img.shape}, overlay_img.dtype: {overlay_img.dtype}")
            logger.error(f"  color_to_apply: {color_to_apply}")
            # Optionally, re-raise the error or handle more gracefully depending on desired behavior
            raise # Re-raise to see full traceback if debugging locally
        # More simply, just paint the cell (can be adjusted with alpha later if needed via mask_overlay type func)
        # overlay_img[mask_data == cell_id] = color_to_apply

    return overlay_img

def create_mapping_summary(args):
    setup_logging()
    logger.info(f"Starting mapping summary creation for task_id: {args.task_id}")

    viz_config_data = load_json_config(args.viz_config, "Visualization Config")
    proc_config_data = load_json_config(args.proc_config, "Processing Config") # Load processing config

    if not viz_config_data or not proc_config_data:
        logger.error("Failed to load necessary configuration files.")
        return

    task_info, default_genes = find_task_in_config(viz_config_data, args.task_id)
    if not task_info:
        return

    # --- Get image_config for the source_image_id --- 
    # This is needed early for several path derivations.
    source_image_id = task_info.get("source_image_id") # Ensure source_image_id is defined before this call
    source_processing_unit_name_from_task = task_info.get("source_processing_unit_display_name") # Get this too, might be needed by func
    image_configurations_from_proc = proc_config_data.get("image_configurations", [])
    image_config, _, _, _ = get_image_mpp_and_path_from_config(
        image_configurations_from_proc, 
        source_image_id, 
        source_processing_unit_name_from_task # Pass this in case the function uses it
    )
    if not image_config: # Check if image_config was successfully retrieved
        logger.error(f"Could not find image configuration for source_image_id '{source_image_id}' in processing_config. Cannot proceed with task '{args.task_id}'.")
        return

    # --- Extract other task parameters (some were fetched above for image_config) ---
    task_output_subfolder = task_info.get("output_subfolder_name")
    source_param_set_id = task_info.get("source_param_set_id")
    source_processing_unit_name_from_task = task_info.get("source_processing_unit_display_name")

    if not all([task_output_subfolder, source_image_id, source_param_set_id]):
        logger.error(f"Task '{args.task_id}' is missing critical fields: output_subfolder_name, source_image_id, or source_param_set_id.")
        return

    # Get and parse genes for this task
    genes_spec_from_task = task_info.get("genes_to_visualize", default_genes)
    genes_to_process = _parse_gene_visualization_list(genes_spec_from_task)

    if not genes_to_process:
        logger.error(f"No valid genes specified or parsed for task '{args.task_id}'.")
        return
    
    logger.info(f"Parsed genes for task '{args.task_id}': {genes_to_process}")
    
    viz_params = task_info.get("visualization_params", {})
    log_scale_all_counts_for_global_norm = viz_params.get("log_scale_counts_for_colormap", False)
    colormap_name = viz_params.get("colormap", "viridis") 
    min_percentile = viz_params.get("colormap_min_percentile", 0.0)
    max_percentile = viz_params.get("colormap_max_percentile", 100.0)
    # background_override_path is fetched later, when actual_background_image_path is determined

    # --- Determine paths --- (Moved results_base_dir determination here)
    results_base_dir = args.results_base_dir or os.path.join(PROJECT_ROOT, RESULTS_DIR_RELATIVE_TO_PROJECT)
    task_gene_outputs_dir = os.path.join(results_base_dir, "visualizations", task_output_subfolder)

    if not os.path.isdir(task_gene_outputs_dir):
        logger.error(f"Task gene outputs directory not found: {task_gene_outputs_dir}")
        logger.error(f"This directory is expected to contain '_cell_counts.csv' files generated by 'visualize_gene_expression.py' for task '{args.task_id}'.")
        logger.error("Please ensure that 'visualize_gene_expression.py' has been run successfully for this task and its outputs are in the correct location.")
        return
    else:
        # Check if the directory, though existing, contains relevant CSV files
        found_csvs = False
        for display_name, actual_id in genes_to_process: # genes_to_process should be defined before this
            short_actual_id_name = actual_id.split('.')[-1]
            safe_short_actual_id_name = clean_filename_for_dir(short_actual_id_name)
            counts_csv_filename = f"{safe_short_actual_id_name}_expression_overlay_cell_counts.csv"
            counts_csv_path = os.path.join(task_gene_outputs_dir, counts_csv_filename)
            if os.path.exists(counts_csv_path):
                found_csvs = True
                break # Found at least one, that's enough for this check
        if not found_csvs and genes_to_process: # only error if genes were expected
            logger.error(f"Task gene outputs directory found: {task_gene_outputs_dir}, but it does not contain the expected '_cell_counts.csv' files for the specified genes.")
            logger.error(f"Please ensure that 'visualize_gene_expression.py' has generated these files for task '{args.task_id}'.")
            return 
        elif not genes_to_process:
            logger.info(f"Task gene outputs directory is {task_gene_outputs_dir}. No genes specified for summary, so not checking for CSVs.")
        else:
            logger.info(f"Task gene outputs directory found at {task_gene_outputs_dir} and contains relevant CSVs (checked for at least one). Proceeding.")

    # --- Determine name_of_unit_that_was_segmented (used for both mask and potentially background image) ---
    # This logic was previously part of background image path determination but is also crucial for mask path.
    name_of_unit_that_was_segmented = source_processing_unit_name_from_task # This is from viz_config task
    img_options_for_paths = image_config.get("segmentation_options", {}) 
    rescale_conf_for_paths = img_options_for_paths.get("rescaling_config", {})
    scale_factor_for_paths = rescale_conf_for_paths.get("scale_factor", 1.0)
    
    original_image_filename_from_proc_config_for_paths = image_config.get("original_image_filename")
    if not original_image_filename_from_proc_config_for_paths:
        logger.error(f"Task {args.task_id}: 'original_image_filename' is missing in image_config '{source_image_id}'. Cannot derive unit name.")
        return
    original_img_basename_for_paths = os.path.basename(original_image_filename_from_proc_config_for_paths)
    original_img_base_for_paths, original_img_ext_for_paths = os.path.splitext(original_img_basename_for_paths)

    if not name_of_unit_that_was_segmented: # If not provided in task, derive it
        tiling_params_for_paths = img_options_for_paths.get("tiling_parameters", {})
        is_tiled_for_paths = tiling_params_for_paths.get("apply_tiling", False)

        if is_tiled_for_paths:
            logger.error(f"Task {args.task_id} seems to be for a tiled image config '{source_image_id}', but 'source_processing_unit_display_name' (tile name) is missing in the task. Cannot derive unit name.")
            return
        elif scale_factor_for_paths != 1.0:
            scale_factor_str_file = str(scale_factor_for_paths).replace('.', '_')
            name_of_unit_that_was_segmented = f"{original_img_base_for_paths}_scaled_{scale_factor_str_file}{original_img_ext_for_paths}"
            logger.info(f"Derived name_of_unit_that_was_segmented (scaled): {name_of_unit_that_was_segmented}")
        else:
            name_of_unit_that_was_segmented = original_img_basename_for_paths
            logger.info(f"Derived name_of_unit_that_was_segmented (original): {name_of_unit_that_was_segmented}")
    
    if not name_of_unit_that_was_segmented: # Should be set by now or returned
        logger.error("Critical error: name_of_unit_that_was_segmented could not be determined.")
        return

    # --- Construct Mask Path --- 
    # This uses name_of_unit_that_was_segmented which has been determined above.
    tiling_params_for_mask = image_config.get("segmentation_options", {}).get("tiling_parameters", {})
    is_source_tiled_for_mask = tiling_params_for_mask.get("apply_tiling", False)
    scale_factor_for_mask_exp_id = scale_factor_for_paths # Consistent scale factor from proc_config

    experiment_id_for_mask = construct_full_experiment_id(
        image_id=source_image_id,
        param_set_id=source_param_set_id,
        scale_factor=scale_factor_for_mask_exp_id,
        processing_unit_name_for_tile=name_of_unit_that_was_segmented if is_source_tiled_for_mask else None,
        is_tile=is_source_tiled_for_mask
    )
    mask_filename_stem = os.path.splitext(name_of_unit_that_was_segmented)[0]
    mask_path = os.path.join(results_base_dir, experiment_id_for_mask, f"{mask_filename_stem}_mask.tif")

    # --- Load Mask Data (now that mask_path is defined) ---
    if not os.path.exists(mask_path):
        logger.error(f"Segmentation mask not found at: {mask_path}. This path was derived for task '{args.task_id}'.")
        logger.info(f"  Used experiment_id: {experiment_id_for_mask}, mask_stem: {mask_filename_stem}")
        return
    mask_data = cellpose_io.imread(mask_path)
    if mask_data is None:
        logger.error(f"Failed to load mask data from {mask_path}")
        return
    logger.info(f"Successfully loaded mask '{mask_path}', shape: {mask_data.shape}")

    # --- Get Background Image (find, or create if scaled version missing) ---
    # This section now has mask_data available if on-the-fly rescaling needs its dimensions.
    actual_background_image_path = None
    user_background_override = viz_params.get("background_image_path_override")

    if user_background_override:
        logger.info(f"Task '{args.task_id}' has 'background_image_path_override': {user_background_override}")
        resolved_override = os.path.join(PROJECT_ROOT, user_background_override) if not os.path.isabs(user_background_override) else user_background_override
        if os.path.exists(resolved_override):
            actual_background_image_path = resolved_override
            logger.info(f"Using overridden background image: {actual_background_image_path}")
        else:
            logger.warning(f"Warning: 'background_image_path_override' specified but not found: {resolved_override}. Will attempt to derive background path.")

    if not actual_background_image_path:
        # Attempt to find the image unit that corresponds to name_of_unit_that_was_segmented
        # This could be in RESCALED_IMAGE_CACHE_DIR or IMAGE_DIR_BASE (for originals or un-cached scaled)
        
        potential_paths_to_check = []
        path_in_rescaled_cache = None
        if "_scaled_" in name_of_unit_that_was_segmented: # Only check cache if it's supposed to be a scaled name
            path_in_rescaled_cache = os.path.join(RESCALED_IMAGE_CACHE_DIR, source_image_id, name_of_unit_that_was_segmented)
            potential_paths_to_check.append(path_in_rescaled_cache)
        
        path_in_original_dir = None
        if original_image_filename_from_proc_config_for_paths:
             original_image_full_raw_path = os.path.join(PROJECT_ROOT, original_image_filename_from_proc_config_for_paths)
             base_dir_of_original_raw = os.path.dirname(original_image_full_raw_path)
             path_in_original_dir = os.path.join(base_dir_of_original_raw, name_of_unit_that_was_segmented)
             potential_paths_to_check.append(path_in_original_dir)
        
        # Check if name_of_unit_that_was_segmented itself is a full path (less likely if derived)
        if os.path.isabs(name_of_unit_that_was_segmented) and os.path.exists(name_of_unit_that_was_segmented):
             potential_paths_to_check.append(name_of_unit_that_was_segmented)

        found_preexisting_scaled_and_matching = False # Initialize before loop
        for p_path in potential_paths_to_check:
            if os.path.exists(p_path):
                actual_background_image_path = p_path
                logger.info(f"Found candidate pre-existing background image: {actual_background_image_path}")
                # Try to load it and check dimensions against mask_data
                try:
                    candidate_img = tifffile.imread(actual_background_image_path)
                    # Attempt to get a 2D/3D plane from candidate_img for dimension check
                    plane_to_check_dims = None
                    if candidate_img.ndim == 2: plane_to_check_dims = candidate_img
                    elif candidate_img.ndim == 3:
                        if candidate_img.shape[0] <= 4 and candidate_img.shape[0] < candidate_img.shape[1] and candidate_img.shape[0] < candidate_img.shape[2]: # CxHxW
                            plane_to_check_dims = candidate_img[0] # Check one channel's HxW
                        else: # HxWxC or ZxHxW (take first slice HxW)
                            plane_to_check_dims = candidate_img if candidate_img.shape[-1] <=4 else candidate_img[0]
                    elif candidate_img.ndim > 3:
                        squeezed_cand = candidate_img.copy()
                        while squeezed_cand.ndim > 3 and squeezed_cand.shape[0] == 1: squeezed_cand = squeezed_cand[0]
                        if squeezed_cand.ndim == 3: 
                            if squeezed_cand.shape[0] <= 4 and squeezed_cand.shape[0] < squeezed_cand.shape[1] and squeezed_cand.shape[0] < squeezed_cand.shape[2]:
                                 plane_to_check_dims = squeezed_cand[0]
                            elif squeezed_cand.shape[-1] <=4:
                                 plane_to_check_dims = squeezed_cand
                            else: 
                                 plane_to_check_dims = squeezed_cand[squeezed_cand.shape[0] // 2, :, :]
                        elif squeezed_cand.ndim == 2: plane_to_check_dims = squeezed_cand

                    if plane_to_check_dims is not None and plane_to_check_dims.shape[:2] == mask_data.shape[:2]:
                        logger.info(f"Pre-existing background image '{actual_background_image_path}' matches mask dimensions. Using it.")
                        found_preexisting_scaled_and_matching = True
                        break # Found a good one
                    else:
                        logger.warning(f"Pre-existing background '{actual_background_image_path}' (shape {plane_to_check_dims.shape[:2] if plane_to_check_dims is not None else 'unknown'}) does not match mask dimensions ({mask_data.shape[:2]}). Will attempt to recreate it.")
                        actual_background_image_path = None # Discard this candidate
                except Exception as e_load_candidate:
                    logger.warning(f"Could not load or process candidate pre-existing background '{actual_background_image_path}': {e_load_candidate}. Will attempt to recreate.")
                    actual_background_image_path = None # Discard this candidate
        
        if not found_preexisting_scaled_and_matching:
            if not found_preexisting_scaled_and_matching: # Renamed variable for clarity
                 logger.warning(f"Matching pre-existing background image unit '{name_of_unit_that_was_segmented}' was either not found or did not match mask dimensions.")
            # Attempt to create it if it was a scaled name and we have the original raw image
            if "_scaled_" in name_of_unit_that_was_segmented and scale_factor_for_paths != 1.0 and original_image_filename_from_proc_config_for_paths:
                original_raw_img_full_path = os.path.join(PROJECT_ROOT, original_image_filename_from_proc_config_for_paths)
                if os.path.exists(original_raw_img_full_path):
                    logger.info(f"Attempting to rescale original image '{original_raw_img_full_path}' by factor {scale_factor_for_paths} to create '{name_of_unit_that_was_segmented}'.")
                    try:
                        # Load original raw image (using tifffile for robustness as it might be OME-TIFF)
                        raw_img_for_rescale = tifffile.imread(original_raw_img_full_path)
                        
                        # Handle multi-dimensional images from tifffile to get a 2D/3D plane for rescaling
                        temp_raw_plane = None
                        if raw_img_for_rescale.ndim == 2: temp_raw_plane = raw_img_for_rescale
                        elif raw_img_for_rescale.ndim == 3:
                            if raw_img_for_rescale.shape[0] <= 4 and raw_img_for_rescale.shape[0] < raw_img_for_rescale.shape[1] and raw_img_for_rescale.shape[0] < raw_img_for_rescale.shape[2]: # CxHxW
                                temp_raw_plane = np.transpose(raw_img_for_rescale, (1,2,0)) # to HxWxC
                            else: # HxWxC or ZxHxW (take first slice)
                                temp_raw_plane = raw_img_for_rescale if raw_img_for_rescale.shape[-1] <=4 else raw_img_for_rescale[0] 
                        elif raw_img_for_rescale.ndim > 3: # Squeeze and take middle Z if needed
                            squeezed = raw_img_for_rescale.copy()
                            while squeezed.ndim > 3 and squeezed.shape[0] == 1: squeezed = squeezed[0]
                            if squeezed.ndim == 3: # ZHW or CHW or HWC after squeeze
                                if squeezed.shape[0] <= 4 and squeezed.shape[0] < squeezed.shape[1] and squeezed.shape[0] < squeezed.shape[2]: # CxHxW
                                     temp_raw_plane = np.transpose(squeezed, (1, 2, 0))
                                elif squeezed.shape[-1] <=4 and squeezed.shape[-1] < squeezed.shape[0] and squeezed.shape[-1] < squeezed.shape[1]: # HWC
                                     temp_raw_plane = squeezed
                                else: # Probably ZHW, take middle slice
                                     temp_raw_plane = squeezed[squeezed.shape[0] // 2, :, :]
                            elif squeezed.ndim == 2: temp_raw_plane = squeezed
                        
                        if temp_raw_plane is None:
                            raise ValueError(f"Could not extract a suitable 2D/3D plane from original image {original_raw_img_full_path} for rescaling.")

                        original_height, original_width = temp_raw_plane.shape[:2]
                        new_width = int(original_width * scale_factor_for_paths)
                        new_height = int(original_height * scale_factor_for_paths)
                        
                        if new_width == 0 or new_height == 0:
                             raise ValueError(f"Calculated new dimensions are zero ({new_width}x{new_height}). Original: {original_width}x{original_height}, Scale: {scale_factor_for_paths}")

                        # Choose interpolation method (INTER_AREA is good for downscaling)
                        interpolation = cv2.INTER_AREA if scale_factor_for_paths < 1.0 else cv2.INTER_LINEAR
                        rescaled_image = cv2.resize(temp_raw_plane, (new_width, new_height), interpolation=interpolation)
                        
                        # Determine save path (prefer RESCALED_IMAGE_CACHE_DIR)
                        save_path_for_rescaled = path_in_rescaled_cache # This was defined earlier
                        if not save_path_for_rescaled: # Fallback if not constructed (e.g. if RESCALED_IMAGE_CACHE_DIR was None or issues)
                            save_path_for_rescaled = path_in_original_dir # Save alongside original if cache path is problematic

                        if save_path_for_rescaled:
                            # Ensure mask_data is loaded to get target dimensions for resize
                            if mask_data is None: # mask_data should be loaded before this point normally
                                logger.error("mask_data not loaded before attempting to rescale background. This is a logic error.")
                                raise ReferenceError("mask_data required for rescaling but not loaded.")
                            
                            target_h, target_w = mask_data.shape[:2]
                            logger.info(f"Target dimensions for rescaling from mask: {target_w}x{target_h}")

                            # temp_raw_plane is the plane extracted from the original full-res image
                            if temp_raw_plane.shape[:2] == (target_h, target_w):
                                logger.info("Original image plane already matches mask dimensions. No resize needed, but will save as scaled name if it was not found under that name.")
                                rescaled_image = temp_raw_plane # Use as is
                            else:
                                interpolation = cv2.INTER_AREA if temp_raw_plane.shape[0] > target_h else cv2.INTER_LINEAR
                                rescaled_image = cv2.resize(temp_raw_plane, (target_w, target_h), interpolation=interpolation)

                            os.makedirs(os.path.dirname(save_path_for_rescaled), exist_ok=True)
                            tifffile.imwrite(save_path_for_rescaled, rescaled_image) # Use tifffile to save, might preserve dtype better
                            logger.info(f"Successfully rescaled and saved image to: {save_path_for_rescaled}")
                            actual_background_image_path = save_path_for_rescaled
                        else:
                            logger.warning(f"Could not determine a valid save path for the rescaled image. Using in-memory rescaled image for now but it won't be cached.")
                            # To use in-memory, we need to handle it slightly differently downstream,
                            # actual_background_image_path would remain None, and display_background_raw would be set directly
                            # For now, let's assume save_path_for_rescaled will be valid if RESCALED_IMAGE_CACHE_DIR logic is okay
                            # If saving fails, this path won't be set, and it will fall through.

                    except Exception as e_rescale:
                        logger.error(f"Error during on-the-fly rescaling of '{original_raw_img_full_path}': {e_rescale}")
                        logger.error(traceback.format_exc())
            
            # Fallback to the raw original image path if specific unit not found AND rescaling failed/not attempted
            if not actual_background_image_path:
                raw_original_full_path = os.path.join(PROJECT_ROOT, original_image_filename_from_proc_config_for_paths) if original_image_filename_from_proc_config_for_paths else None
                if raw_original_full_path and os.path.exists(raw_original_full_path):
                    actual_background_image_path = raw_original_full_path
                    logger.warning(f"Using original raw image: {actual_background_image_path}, as scaled version not found and on-the-fly rescaling either failed or was not applicable. This might lead to dimension mismatch with scaled masks.")
                else:
                    logger.error(f"Could not find or derive a suitable background image. Attempted for '{name_of_unit_that_was_segmented}'. Original raw path also missing or invalid: {raw_original_full_path}")
                    return
    
    if not actual_background_image_path or not os.path.exists(actual_background_image_path): 
        logger.error(f"Background image ultimately not found or created: {actual_background_image_path}")
        return

    # Load the chosen/created background image 
    display_background_raw = None
    try:
        img_raw_tifffile = tifffile.imread(actual_background_image_path)
        logger.info(f"Successfully loaded background image {actual_background_image_path} with tifffile. Shape: {img_raw_tifffile.shape}, dtype: {img_raw_tifffile.dtype}")

        if img_raw_tifffile.ndim == 2: # Grayscale HxW
            display_background_raw = img_raw_tifffile
        elif img_raw_tifffile.ndim == 3:
            if img_raw_tifffile.shape[0] <= 4 and img_raw_tifffile.shape[0] < img_raw_tifffile.shape[1] and img_raw_tifffile.shape[0] < img_raw_tifffile.shape[2]: # Likely C x H x W
                logger.info(f"  Background is {img_raw_tifffile.shape}, assuming CxHxW, transposing to HxWxC.")
                display_background_raw = np.transpose(img_raw_tifffile, (1, 2, 0))
            else: # Assume HxWxC or a single slice of ZxHxW
                 logger.info(f"  Background is {img_raw_tifffile.shape}, assuming HxWxC or will take first slice if it's ZxHxW like.")
                 display_background_raw = img_raw_tifffile
        elif img_raw_tifffile.ndim > 3:
            squeezed_img = img_raw_tifffile.copy()
            while squeezed_img.ndim > 3 and squeezed_img.shape[0] == 1:
                squeezed_img = squeezed_img[0]
            
            if squeezed_img.ndim == 3: # ZHW or CHW or HWC
                if squeezed_img.shape[0] <= 4 and squeezed_img.shape[0] < squeezed_img.shape[1] and squeezed_img.shape[0] < squeezed_img.shape[2]: # CxHxW
                     display_background_raw = np.transpose(squeezed_img, (1, 2, 0))
                elif squeezed_img.shape[-1] <=4 and squeezed_img.shape[-1] < squeezed_img.shape[0] and squeezed_img.shape[-1] < squeezed_img.shape[1]: # HWC
                     display_background_raw = squeezed_img
                else: # Probably ZHW, take middle slice
                     display_background_raw = squeezed_img[squeezed_img.shape[0] // 2, :, :]
                logger.info(f"  Background was >3D, selected/transposed slice has shape: {display_background_raw.shape}")
            elif squeezed_img.ndim == 2: # HW
                display_background_raw = squeezed_img
                logger.info(f"  Background was >3D, selected slice has shape: {display_background_raw.shape}")
            else: 
                logger.warning(f"  Complex background image structure ({img_raw_tifffile.shape}). Trying to take middle Z plane assuming (..., Z, H, W).")
                if img_raw_tifffile.ndim >= 3:
                    display_background_raw = img_raw_tifffile[..., img_raw_tifffile.shape[-3] // 2, :, :]
                elif img_raw_tifffile.ndim == 2:
                     display_background_raw = img_raw_tifffile
                else:
                    logger.error(f"  Failed to extract a suitable 2D/3D plane from >3D background image ({img_raw_tifffile.shape}).")
                    display_background_raw = None

        if display_background_raw is None:
             raise ValueError(f"Could not extract a usable 2D/3-channel image from raw background {img_raw_tifffile.shape}")
        
        logger.info(f"  Successfully processed raw background. Final shape for display: {display_background_raw.shape}, dtype: {display_background_raw.dtype}")

    except Exception as e_load_bg:
        logger.error(f"Error loading background image {actual_background_image_path} with tifffile: {e_load_bg}")
        logger.error(traceback.format_exc()) # Make sure traceback is imported if not already
        return # Exit if background loading fails

    if display_background_raw is None: # Should be caught by the return above, but as a safeguard
        logger.error(f"Failed to load background image from: {actual_background_image_path} (display_background_raw is None after tifffile attempt).")
        return

    background_img_8bit = normalize_to_8bit_for_display(display_background_raw)
    if background_img_8bit is None:
        logger.error(f"Failed to normalize background image (loaded by tifffile) to 8-bit for display. Source: {actual_background_image_path}")
        return

    # --- Collect counts for global normalization (already started, continue) ---
    all_counts_for_global_norm = []
    gene_data_for_plotting = [] # Store dicts with display_name, actual_id, and its cell_counts_df

    logger.info(f"Collecting counts for genes (display name, actual ID): {genes_to_process}")
    for display_name, actual_id in genes_to_process:
        # CSV filename is based on the *actual_id*
        short_actual_id_name = actual_id.split('.')[-1]
        safe_short_actual_id_name = clean_filename_for_dir(short_actual_id_name)
        counts_csv_filename = f"{safe_short_actual_id_name}_expression_overlay_cell_counts.csv"
        counts_csv_path = os.path.join(task_gene_outputs_dir, counts_csv_filename)

        if os.path.exists(counts_csv_path):
            try:
                counts_df = pd.read_csv(counts_csv_path)
                if 'transcript_count' in counts_df.columns and 'cell_id' in counts_df.columns:
                    all_counts_for_global_norm.extend(counts_df['transcript_count'].tolist())
                    gene_data_for_plotting.append({"display_name": display_name, "actual_id": actual_id, "counts_df": counts_df})
                else:
                    logger.warning(f"Required columns not found in {counts_csv_path}. Skipping gene (display: {display_name}, actual: {actual_id}).")
            except Exception as e:
                logger.warning(f"Error reading counts CSV {counts_csv_path} for gene (display: {display_name}, actual: {actual_id}): {e}. Skipping.")
        else:
            logger.warning(f"Counts CSV not found for gene (display: {display_name}, actual: {actual_id}) at {counts_csv_path}. Skipping this gene.")

    if not gene_data_for_plotting:
        logger.error("No valid gene count data found to create a summary.")
        return
    
    logger.info(f"Collected count data for {len(gene_data_for_plotting)} genes. Aggregated total {len(all_counts_for_global_norm)} cell counts for global normalization.")

    # --- Global normalization for colorbar (same logic as before) ---
    norm_min_val_actual = 0
    norm_max_val_actual = 1 
    if all_counts_for_global_norm:
        values_for_norm = np.array(all_counts_for_global_norm)
        if log_scale_all_counts_for_global_norm:
            values_for_norm = np.log1p(values_for_norm)
        if np.any(values_for_norm > 0):
            positive_values = values_for_norm[values_for_norm > 0]
            if positive_values.size > 0:
                norm_min_val_actual = np.percentile(positive_values, min_percentile)
                norm_max_val_actual = np.percentile(positive_values, max_percentile)
        if norm_max_val_actual <= norm_min_val_actual: 
            norm_max_val_actual = norm_min_val_actual + 1e-6 
    logger.info(f"Global colormap scale: Min={norm_min_val_actual:.2f}, Max={norm_max_val_actual:.2f} (log_scaled={log_scale_all_counts_for_global_norm})")
    
    global_normalize_transform = matplotlib.colors.Normalize(vmin=norm_min_val_actual, vmax=norm_max_val_actual)
    try:
        cmap_function = plt.get_cmap(colormap_name)
    except ValueError:
        logger.warning(f"Colormap '{colormap_name}' not found, defaulting to 'viridis'.")
        cmap_function = plt.get_cmap("viridis")

    # --- Create Montage by re-plotting each gene ---    
    num_images = len(gene_data_for_plotting)
    cols = args.images_per_row
    rows = math.ceil(num_images / cols)
    fig_width = cols * 5 
    fig_height = rows * 5 + 1.5 
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    for i, gene_plot_data in enumerate(gene_data_for_plotting):
        ax = axes[i]
        # Use display_name for the title
        logger.info(f"Re-plotting overlay for gene: {gene_plot_data['display_name']} (Actual ID: {gene_plot_data['actual_id']})")
        re_rendered_img = generate_single_gene_overlay_globally_scaled(
            background_img_8bit, 
            mask_data,
            gene_plot_data["counts_df"],
            cmap_function, 
            global_normalize_transform,
            log_scale_all_counts_for_global_norm 
        )
        ax.imshow(re_rendered_img)
        ax.set_title(gene_plot_data['display_name'], fontsize=10) # Use display_name for title
        ax.axis('off')

    for j in range(num_images, len(axes)): 
        fig.delaxes(axes[j])

    # --- Add Global Colorbar (same logic as before) ---
    cbar_ax_rect = [0.15, 0.05, 0.7, 0.03]
    if rows == 1 and num_images < cols:
        cbar_ax_rect = [0.15, 0.08, 0.7 * (num_images/cols), 0.03]
    cbar_ax = fig.add_axes(cbar_ax_rect)
    cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap_function, norm=global_normalize_transform, orientation='horizontal')
    colorbar_label = f"log(1 + Tx Count)" if log_scale_all_counts_for_global_norm else "Transcript Count"
    cb.set_label(colorbar_label, fontsize=10)
    plt.suptitle(f"Gene Expression Summary: Task '{args.task_id}' (Globally Scaled)", fontsize=14, y=0.98 if rows > 1 else 1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])

    # --- Save Output (same logic as before) ---
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(task_gene_outputs_dir, DEFAULT_SUMMARY_SUBDIR)
    if not os.path.exists(output_dir):
        try: os.makedirs(output_dir); logger.info(f"Created output directory: {output_dir}")
        except Exception as e: logger.error(f"Could not create output directory {output_dir}: {e}"); plt.close(fig); return
    output_filename = args.output_filename or f"{args.task_id}_expression_summary_global_scale.png"
    final_output_path = os.path.join(output_dir, output_filename)
    try:
        plt.savefig(final_output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Mapping summary image saved to: {final_output_path}")
    except Exception as e: logger.error(f"Error saving summary image: {e}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a summary image (montage) of gene expression visualizations for a given task, with a global colorbar by re-plotting.")
    
    parser.add_argument("--viz_config", default=os.path.join(PROJECT_ROOT, VISUALIZATION_CONFIG_FILENAME),
                        help=f"Path to the visualization JSON configuration file (default: {VISUALIZATION_CONFIG_FILENAME} in project root).")
    # Add proc_config argument, as it's needed to find original image details for background
    parser.add_argument("--proc_config", default=os.path.join(PROJECT_ROOT, PROCESSING_CONFIG_FILENAME),
                        help=f"Path to the processing JSON configuration file (default: {PROCESSING_CONFIG_FILENAME} in project root).")
    parser.add_argument("--task_id", required=True,
                        help="The specific 'task_id' from the visualization_config.json to summarize.")
    parser.add_argument("--results_base_dir", default=None, 
                        help=f"Base directory where visualization AND segmentation results are stored (default derived from '{RESULTS_DIR_RELATIVE_TO_PROJECT}').")
    parser.add_argument("--output_dir", default=None,
                        help="Specific directory to save the summary image. If None, saves to a 'summary' subfolder within the task's gene output directory.")
    parser.add_argument("--output_filename", default=None,
                        help="Filename for the output summary image (default: <task_id>_expression_summary_global_scale.png).")
    parser.add_argument("--images_per_row", type=int, default=IMAGES_PER_ROW_DEFAULT,
                        help=f"Number of images per row in the montage (default: {IMAGES_PER_ROW_DEFAULT}).")

    parsed_args = parser.parse_args()
    create_mapping_summary(parsed_args) 