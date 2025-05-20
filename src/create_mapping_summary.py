import os
import json
import numpy as np
import pandas as pd
import cv2 # For loading images for re-plotting
import re

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
    PROCESSING_CONFIG_FILENAME # For finding original image paths
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

    # --- Extract task parameters ---
    task_output_subfolder = task_info.get("output_subfolder_name")
    source_image_id = task_info.get("source_image_id")
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
    background_override_path = viz_params.get("background_image_path_override")

    # --- Determine paths --- (Moved results_base_dir determination here)
    results_base_dir = args.results_base_dir or os.path.join(PROJECT_ROOT, RESULTS_DIR_RELATIVE_TO_PROJECT)
    task_gene_outputs_dir = os.path.join(results_base_dir, "visualizations", task_output_subfolder)

    if not os.path.isdir(task_gene_outputs_dir):
        logger.error(f"Task gene outputs directory not found: {task_gene_outputs_dir}")
        return

    # --- Get Background Image and Mask Path (once for the task) ---
    # Use get_image_mpp_and_path_from_config to find original/background image path
    image_config, _, _, _ = get_image_mpp_and_path_from_config(proc_config_data.get("image_configurations", []), source_image_id, source_processing_unit_name_from_task)
    if not image_config:
        logger.error(f"Could not find image configuration for source_image_id '{source_image_id}' in processing_config.")
        return

    actual_background_image_path = background_override_path
    if not actual_background_image_path:
        # Derive from image_config if no override
        original_image_rel_path = image_config.get("original_image_filename")
        if not original_image_rel_path:
            logger.error(f"original_image_filename not found in image_config for {source_image_id}")
            return
        actual_background_image_path = os.path.join(PROJECT_ROOT, original_image_rel_path)
    
    if not os.path.exists(actual_background_image_path):
        logger.error(f"Background image not found at: {actual_background_image_path}")
        return
    
    background_img_raw = cv2.imread(actual_background_image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if background_img_raw is None:
        logger.error(f"Failed to load background image from: {actual_background_image_path}")
        return
    background_img_8bit = normalize_to_8bit_for_display(background_img_raw)

    # Determine derived_unit_name_for_mask (the name of the actual unit that was segmented)
    derived_unit_name_for_mask = source_processing_unit_name_from_task
    img_options = image_config.get("segmentation_options", {}) # Get options once

    if not derived_unit_name_for_mask:
        # If task didn't specify, try to derive for non-tiled/non-rescaled case
        # This logic needs to be robust and consistent with how segmentation_worker names outputs.
        tiling_params_local = img_options.get("tiling_parameters", {})
        is_tiled_local = tiling_params_local.get("apply_tiling", False)
        rescale_conf_local = img_options.get("rescaling_config", {})
        is_rescaled_local = rescale_conf_local.get("apply_rescaling", False) and rescale_conf_local.get("scale_factor") != 1.0

        if is_tiled_local:
            logger.error(f"Task {args.task_id} refers to a tiled image {source_image_id}, but 'source_processing_unit_display_name' is missing in the task. Cannot derive mask path without the specific tile name.")
            return
        elif is_rescaled_local:
            # If rescaled, the name would typically include the scale factor.
            # Constructing this name perfectly without knowing the exact output format of segmentation_worker is hard.
            # For now, assume it might be the original filename if no specific name given for a rescaled-but-not-tiled case.
            # This part is a common source of path mismatch if naming conventions aren't strictly followed.
            original_img_basename = os.path.basename(image_config.get("original_image_filename", "unknown_original.tif"))
            scale_factor = rescale_conf_local.get("scale_factor", 1.0)
            scale_factor_str_file = str(scale_factor).replace('.', '_')
            derived_unit_name_for_mask = f"{os.path.splitext(original_img_basename)[0]}_scaled_{scale_factor_str_file}.tif"
            logger.info(f"Derived unit name for mask (rescaled, not in task): {derived_unit_name_for_mask}")

        else: # Not tiled, not rescaled, and not specified in task
            derived_unit_name_for_mask = os.path.basename(image_config.get("original_image_filename", "unknown_original.tif"))
            logger.info(f"Derived unit name for mask (original, not in task): {derived_unit_name_for_mask}")
    
    if not derived_unit_name_for_mask: # Should not happen if logic above is correct
        logger.error("Could not determine derived_unit_name_for_mask for mask path construction.")
        return

    # Construct mask path
    tiling_params = img_options.get("tiling_parameters", {})
    is_source_tiled = tiling_params.get("apply_tiling", False)
    
    rescale_conf = img_options.get("rescaling_config", {})
    applied_scale_factor_for_mask = rescale_conf.get("scale_factor", 1.0)

    if is_source_tiled:
        # If the source image config indicates tiling, then 'derived_unit_name_for_mask' MUST be the tile name.
        # The scale factor is that of the image *before* tiling.
        logger.info(f"Source image {source_image_id} is configured for tiling. Mask unit: {derived_unit_name_for_mask}, Pre-tiling scale: {applied_scale_factor_for_mask}")
    elif "_scaled_" in derived_unit_name_for_mask: # Infer from name if not tiled by config, and name has scale
        match = re.search(r"_scaled_([0-9]+(?:_[0-9]+)?)(?:[._]|$)", os.path.basename(derived_unit_name_for_mask))
        if match:
            try:
                scale_str_from_name = match.group(1).replace('_', '.')
                parsed_sf_from_name = float(scale_str_from_name)
                if 0 < parsed_sf_from_name <= 1.0:
                    applied_scale_factor_for_mask = parsed_sf_from_name
                    logger.info(f"Inferred scale factor {applied_scale_factor_for_mask} for mask from derived_unit_name: {derived_unit_name_for_mask}.")
            except ValueError:
                logger.warning(f"Could not parse scale factor from derived_unit_name: {derived_unit_name_for_mask}")
    # If not tiled by config and name doesn't have _scaled_, applied_scale_factor_for_mask remains from rescale_conf (default 1.0)

    experiment_id_for_mask = construct_full_experiment_id(
        image_id=source_image_id,
        param_set_id=source_param_set_id,
        scale_factor=applied_scale_factor_for_mask,
        processing_unit_name_for_tile=derived_unit_name_for_mask if is_source_tiled else None,
        is_tile=is_source_tiled
    )
    
    # The mask_filename_stem is derived from the unit name that was segmented.
    # If it's a tile, derived_unit_name_for_mask is the tile name (e.g., tile_r0_c0.tif)
    # If not tiled, it's the (possibly rescaled) original image name.
    mask_filename_stem = os.path.splitext(derived_unit_name_for_mask)[0]

    mask_path = os.path.join(results_base_dir, experiment_id_for_mask, f"{mask_filename_stem}_mask.tif")
    
    if not os.path.exists(mask_path):
        logger.error(f"Segmentation mask not found at: {mask_path}. This path was derived for task '{args.task_id}'.")
        logger.info(f"  Used experiment_id: {experiment_id_for_mask}, mask_stem: {mask_filename_stem}")
        return
    mask_data = cellpose_io.imread(mask_path)
    if mask_data is None:
        logger.error(f"Failed to load mask data from {mask_path}")
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