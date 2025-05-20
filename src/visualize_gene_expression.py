import os
import argparse
import pandas as pd
import numpy as np
import cv2
from cellpose import io as cellpose_io
from cellpose import plot as cellpose_plot

import matplotlib
matplotlib.use('Agg') # <--- ADD THIS to set a non-interactive backend BEFORE pyplot import
import matplotlib.pyplot as plt
import matplotlib.colors
import json
import re
import traceback
import logging

from .pipeline_utils import (
    normalize_to_8bit_for_display, 
    clean_filename_for_dir, # Changed from sanitize_string_for_filesystem
    construct_full_experiment_id, # New
    construct_mask_path          # New
)
from .file_paths import (
    PROJECT_ROOT, # Import for resolving config path if needed when run as script
    IMAGE_DIR_BASE,
    TILED_IMAGE_OUTPUT_BASE,
    RESCALED_IMAGE_CACHE_DIR,
    RESULTS_DIR_BASE
) # Import from new file_paths

logger = logging.getLogger(__name__)

UNASSIGNED_CELL_ID = -1
DEFAULT_BACKGROUND_COLOR_FOR_CELLS_NO_EXPRESSION = np.array([200, 200, 200], dtype=np.uint8)
TRANSCRIPT_DOT_COLOR = (255, 0, 255) 
TRANSCRIPT_DOT_SIZE = 1

# PROJECT_ROOT, IMAGE_DIR_BASE etc. are now imported from file_paths

def load_mapped_transcripts(mapped_transcripts_csv_path):
    logger.info(f"Loading mapped transcripts from: {mapped_transcripts_csv_path} ...")
    try:
        df = pd.read_csv(mapped_transcripts_csv_path)
        required_cols = ['transcript_id', 'x_location', 'y_location', 'feature_name', 'assigned_cell_id']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in mapped transcripts file.")
        logger.info(f"Loaded {len(df)} mapped transcript records.")
        return df
    except Exception as e:
        logger.error(f"Error loading mapped transcripts CSV: {e}")
        return None

def get_colors_for_gene_expression(instance_mask, cell_expression_counts, colormap_func, default_color):
    num_mask_labels = instance_mask.max()
    cell_colors_rgb = np.zeros((int(num_mask_labels) + 1, 3), dtype=np.uint8)
    cell_colors_rgb[:] = default_color 

    if cell_expression_counts.empty:
        return cell_colors_rgb

    min_count = 0 
    max_count = cell_expression_counts.max()
    if max_count == 0 : 
        return cell_colors_rgb 

    for cell_id, count in cell_expression_counts.items():
        cell_id_int = int(cell_id)
        if 0 < cell_id_int <= num_mask_labels:
            norm_count = 0.0
            if max_count > min_count: 
                norm_count = (count - min_count) / (max_count - min_count)
            elif max_count > 0: 
                norm_count = 1.0
            
            rgba_color = colormap_func(norm_count)
            cell_colors_rgb[cell_id_int] = (np.array(rgba_color[:3]) * 255).astype(np.uint8)
    return cell_colors_rgb

def visualize_gene_expression_for_job(
    mapped_transcripts_df, 
    segmentation_mask_array, 
    background_image_for_overlay_rgb, 
    gene_of_interest, 
    output_png_path, 
    effective_mpp_x, effective_mpp_y, 
    task_visualization_params,
    x_offset_microns=0, y_offset_microns=0, 
    plot_transcript_dots=False
    ):

    logger.info(f"--- Visualizing expression for gene: {gene_of_interest} ---")
    gene_transcripts_df = mapped_transcripts_df[mapped_transcripts_df['feature_name'] == gene_of_interest]
    total_gene_transcripts = len(gene_transcripts_df)
    if total_gene_transcripts == 0:
        logger.info(f"No transcripts found for gene '{gene_of_interest}'. Skipping visualization."); return

    assigned_gene_transcripts_df = gene_transcripts_df[gene_transcripts_df['assigned_cell_id'] != UNASSIGNED_CELL_ID]
    num_assigned = len(assigned_gene_transcripts_df)
    num_unassigned = total_gene_transcripts - num_assigned
    percent_unassigned = (num_unassigned / total_gene_transcripts * 100) if total_gene_transcripts > 0 else 0
    logger.info(f"  Total '{gene_of_interest}' transcripts: {total_gene_transcripts} (Assigned: {num_assigned}, Unassigned: {num_unassigned} [{percent_unassigned:.2f}%])")

    cell_expression_counts = assigned_gene_transcripts_df.groupby('assigned_cell_id').size()
    max_cell_id_in_mask = int(segmentation_mask_array.max())
    cell_colors_rgb = np.zeros((max_cell_id_in_mask + 1, 3), dtype=np.uint8)
    cell_colors_rgb[1:] = DEFAULT_BACKGROUND_COLOR_FOR_CELLS_NO_EXPRESSION 

    # Get visualization parameters from the task_visualization_params dictionary
    colormap_name = task_visualization_params.get("colormap", "viridis")
    log_scale_counts = task_visualization_params.get("log_scale_counts_for_colormap", False)
    min_percentile = task_visualization_params.get("colormap_min_percentile", 0.0)
    max_percentile = task_visualization_params.get("colormap_max_percentile", 100.0)

    logger.info(f"  Colormap: {colormap_name}, Log-scale: {log_scale_counts}, Percentiles: [{min_percentile}, {max_percentile}]")

    # These will store the actual values used for colormap normalization limits
    norm_min_val_actual = 0.0
    norm_max_val_actual = 0.0
    colormap_label = "Transcript Count"
    
    if not cell_expression_counts.empty:
        values_for_normalization = cell_expression_counts.copy()
        if log_scale_counts:
            values_for_normalization = np.log1p(values_for_normalization)
            logger.info(f"  Applied log1p transformation to cell counts for colormap scaling.")
            colormap_label = "log(1 + Count)"

        # Determine min/max for normalization based on percentiles of positive values
        positive_values = values_for_normalization[values_for_normalization > 0]
        if positive_values.empty:
            logger.info(f"  No cells with positive (or log-positive) expression for '{gene_of_interest}'. Using default 0-1 range for colormap if any cells exist.")
            norm_min_val_actual = 0.0
            norm_max_val_actual = 1.0 # Avoid division by zero if positive_values is empty
        else:
            norm_min_val_actual = np.percentile(positive_values, min_percentile)
            norm_max_val_actual = np.percentile(positive_values, max_percentile)
            if norm_max_val_actual <= norm_min_val_actual: # Ensure max > min
                logger.warning(f"  norm_max_val_actual ({norm_max_val_actual:.2f}) <= norm_min_val_actual ({norm_min_val_actual:.2f}) after percentile clipping. Using full range of positive values or defaults.")
                # Fallback to min/max of positive values if percentiles resulted in invalid range
                if positive_values.min() < positive_values.max():
                    norm_min_val_actual = positive_values.min()
                    norm_max_val_actual = positive_values.max()
                else: # Still no range, use 0-1 to avoid errors
                    norm_min_val_actual = 0.0
                    norm_max_val_actual = 1.0
        
        logger.info(f"  Colormap normalization range for '{gene_of_interest}': Min={norm_min_val_actual:.2f}, Max={norm_max_val_actual:.2f} (based on {'log-scaled' if log_scale_counts else 'raw'} counts)")

        try: colormap_func = matplotlib.colormaps[colormap_name]
        except AttributeError: colormap_func = plt.cm.get_cmap(colormap_name) # older matplotlib

        for cell_id, count_val in cell_expression_counts.items(): # Iterate original counts
            value_to_normalize = count_val
            if log_scale_counts:
                value_to_normalize = np.log1p(count_val)
            
            cell_id_int = int(cell_id)
            if 0 < cell_id_int <= max_cell_id_in_mask:
                if norm_max_val_actual > norm_min_val_actual:
                    norm_intensity = (value_to_normalize - norm_min_val_actual) / (norm_max_val_actual - norm_min_val_actual)
                elif value_to_normalize > norm_min_val_actual : # if max == min, and value is above, it's max intensity
                    norm_intensity = 1.0
                else: # if max == min, and value is at or below, it's min intensity
                    norm_intensity = 0.0
                
                norm_intensity = np.clip(norm_intensity, 0.0, 1.0) # Ensure it's strictly within [0,1]
                rgba_color = colormap_func(norm_intensity)
                cell_colors_rgb[cell_id_int] = (np.array(rgba_color[:3]) * 255).astype(np.uint8)
    else:
        logger.info(f"  No cells found with expression of '{gene_of_interest}'. Cells will have default color.")
        # For colorbar, if no expression, we might use a dummy range or not show it
        norm_min_val_actual = 0.0 
        norm_max_val_actual = 1.0 # Default range for a placeholder colorbar if needed
        if log_scale_counts: colormap_label = "log(1 + Count)" # Keep label consistent if log scaling was requested

    overlay_img_rgb = cellpose_plot.mask_overlay(background_image_for_overlay_rgb, segmentation_mask_array, colors=cell_colors_rgb)

    if plot_transcript_dots and not gene_transcripts_df.empty:
        logger.info(f"  Plotting {total_gene_transcripts} transcript dots for '{gene_of_interest}'...")
        dot_color_rgb = TRANSCRIPT_DOT_COLOR 
        
        t_x_pixels = ((gene_transcripts_df['x_location'] - x_offset_microns) / effective_mpp_x).astype(int)
        t_y_pixels = ((gene_transcripts_df['y_location'] - y_offset_microns) / effective_mpp_y).astype(int)

        h, w = overlay_img_rgb.shape[:2]
        for tx, ty in zip(t_x_pixels, t_y_pixels):
            if 0 <= tx < w and 0 <= ty < h:
                cv2.circle(overlay_img_rgb, (tx, ty), TRANSCRIPT_DOT_SIZE, dot_color_rgb, -1)
    try:
        # cv2.imwrite(output_png_path, cv2.cvtColor(overlay_img_rgb, cv2.COLOR_RGB2BGR))
        # Replace cv2.imwrite with matplotlib save to include colorbar
        fig, ax = plt.subplots(figsize=(10,10)) # Adjust figsize as needed
        ax.imshow(overlay_img_rgb)
        ax.axis('off') # Turn off axis numbers and ticks

        # Add colorbar if there was any expression to scale by
        # (even if all cells ended up with 0 after log/percentile, the range might be non-zero)
        if not cell_expression_counts.empty or (total_gene_transcripts > 0 and (norm_min_val_actual != 0.0 or norm_max_val_actual != 1.0)):
            # Create a normalizer for the colorbar
            norm = matplotlib.colors.Normalize(vmin=norm_min_val_actual, vmax=norm_max_val_actual)
            # Create a scalar mappable for the colorbar
            sm = plt.cm.ScalarMappable(cmap=colormap_func, norm=norm)
            sm.set_array([]) # You need to set_array for the ScalarMappable to work, even an empty one

            # Add colorbar to the figure - adjust fraction, pad, and orientation as needed
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label(colormap_label, rotation=270, labelpad=20)
        
        fig.savefig(output_png_path, bbox_inches='tight', dpi=150)
        plt.close(fig) # Close the figure to free memory
        logger.info(f"  Expression overlay for '{gene_of_interest}' saved to: {output_png_path}")

        # Save cell expression counts to CSV
        if not cell_expression_counts.empty:
            # Derive CSV path from PNG path
            csv_filename_base = os.path.splitext(output_png_path)[0]
            csv_output_path = f"{csv_filename_base}_cell_counts.csv"
            try:
                counts_df_to_save = cell_expression_counts.reset_index()
                counts_df_to_save.columns = ['cell_id', 'transcript_count']
                counts_df_to_save.to_csv(csv_output_path, index=False)
                logger.info(f"  Cell expression counts for '{gene_of_interest}' saved to: {csv_output_path}")
            except Exception as e_csv:
                logger.error(f"  Error saving cell expression counts CSV for '{gene_of_interest}': {e_csv}")
        elif total_gene_transcripts > 0: # Transcripts exist but none assigned to cells in this mask
            csv_filename_base = os.path.splitext(output_png_path)[0]
            csv_output_path = f"{csv_filename_base}_cell_counts.csv"
            try:
                # Create and save an empty DataFrame with correct headers
                empty_counts_df = pd.DataFrame(columns=['cell_id', 'transcript_count'])
                empty_counts_df.to_csv(csv_output_path, index=False)
                logger.info(f"  No cell-assigned transcripts for '{gene_of_interest}'. Empty cell counts CSV saved to: {csv_output_path}")
            except Exception as e_csv_empty:
                logger.error(f"  Error saving empty cell counts CSV for '{gene_of_interest}': {e_csv_empty}")

    except Exception as e:
        logger.error(f"  Error saving expression overlay for '{gene_of_interest}': {e} (Path: {output_png_path})") 

def _get_image_config_from_params(param_sets_path, target_image_id):
    if not os.path.exists(param_sets_path):
        logger.error(f"Error: Config file '{param_sets_path}' not found."); return None
    try:
        with open(param_sets_path, 'r') as f: config_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading/parsing {param_sets_path}: {e}"); return None

    for img_cfg in config_data.get("image_configurations", []):
        if img_cfg.get("image_id") == target_image_id:
            return img_cfg
    logger.error(f"Error: Image ID '{target_image_id}' not found in image_configurations."); return None

def _determine_applied_scale_factor(image_config, target_processing_unit_name):
    """Determines applied scale factor, from config first, then by inferring from filename."""
    applied_scale_factor = 1.0
    if image_config:
        rescaling_cfg = image_config.get("rescaling_config")
        if rescaling_cfg and "scale_factor" in rescaling_cfg:
            sf_from_config = rescaling_cfg["scale_factor"]
            if isinstance(sf_from_config, (int, float)) and 0 < sf_from_config <= 1.0:
                applied_scale_factor = sf_from_config
                logger.info(f"  Scale factor from config: {applied_scale_factor}")
                # If config specifies a scale factor (even 1.0), we primarily trust it.
                # We only try to infer from filename if config did NOT specify a scale_factor or it was 1.0 initially.
                if applied_scale_factor != 1.0:
                    return applied_scale_factor 
            else:
                logger.warning(f"  Warning: Invalid scale_factor {sf_from_config} in config. Ignoring.")

    # If scale_factor is still 1.0 (either from config or default), try to infer from filename
    if applied_scale_factor == 1.0 and "_scaled_" in target_processing_unit_name:
        # Regex to find "_scaled_X_Y" or "_scaled_X" where X, Y are digits
        # It can be part of a filename like "..._scaled_0_5.tif" or "..._scaled_0_5_mask.tif"
        match = re.search(r"_scaled_([0-9]+(?:_[0-9]+)?)(?:[._]|$)", os.path.basename(target_processing_unit_name))
        if match:
            try:
                scale_str_from_name = match.group(1).replace('_', '.')
                parsed_sf_from_name = float(scale_str_from_name)
                if 0 < parsed_sf_from_name < 1.0: # Strict < 1.0 as 1.0 is default
                    applied_scale_factor = parsed_sf_from_name
                    logger.info(f"  Inferred applied_scale_factor {applied_scale_factor} from target_processing_unit_name '{target_processing_unit_name}'")
                elif parsed_sf_from_name == 1.0:
                     logger.info(f"  Inferred scale factor 1.0 from filename, no change from default/config.")
                else:
                    logger.warning(f"  Parsed scale factor {parsed_sf_from_name} from filename is invalid (not between 0-1).")
            except ValueError:
                logger.warning(f"  Could not parse scale factor from filename part: {match.group(1)}")
    return applied_scale_factor

def get_job_info_and_paths(param_sets_path, target_image_id, target_param_set_id, target_processing_unit_name):
    default_return = None, None, 1.0, None, None
    
    image_config = _get_image_config_from_params(param_sets_path, target_image_id)
    if not image_config:
        return default_return

    original_image_filename = image_config.get("original_image_filename")
    if not original_image_filename: 
        logger.error(f"Error: original_image_filename missing for Image ID '{target_image_id}'."); return default_return
    
    original_source_image_full_path = os.path.join(PROJECT_ROOT, original_image_filename) # Changed IMAGE_DIR_BASE to PROJECT_ROOT
    if not os.path.exists(original_source_image_full_path):
        logger.warning(f"Warning: Original source image (path constructed from PROJECT_ROOT + original_image_filename from config) not found: {original_source_image_full_path}")
        logger.info(f"  Attempted path was: PROJECT_ROOT ('{PROJECT_ROOT}') + original_image_filename ('{original_image_filename}')")
        # Do not return here, allow continuation if background_image_path_override will be used later or if only metadata is needed from original_image_filename

    path_of_image_unit_for_segmentation = original_source_image_full_path # Default background, might not exist
    applied_scale_factor = _determine_applied_scale_factor(image_config, target_processing_unit_name)
    
    is_tiled_job = False
    tiling_cfg = image_config.get("tiling_config")
    
    # Determine the base name of the image unit (original or rescaled if scale < 1.0) before tiling
    # This helps identify if target_processing_unit_name is a tile or the main (potentially rescaled) image
    base_name_after_potential_rescale = original_image_filename
    if applied_scale_factor != 1.0:
        scale_factor_str_file = str(applied_scale_factor).replace('.', '_')
        base_name_after_potential_rescale = f"{os.path.splitext(original_image_filename)[0]}_scaled_{scale_factor_str_file}.tif"
        
        expected_rescaled_path = os.path.join(RESCALED_IMAGE_CACHE_DIR, target_image_id, base_name_after_potential_rescale) # Uses imported RESCALED_IMAGE_CACHE_DIR
        if os.path.exists(expected_rescaled_path):
            path_of_image_unit_for_segmentation = expected_rescaled_path
            logger.info(f"  Using explicitly rescaled image from cache: {path_of_image_unit_for_segmentation}")
        else:
            # Fallback if target_processing_unit_name itself is the scaled image path (e.g. if not using standard cache structure)
            if target_processing_unit_name == base_name_after_potential_rescale and os.path.exists(os.path.join(IMAGE_DIR_BASE, target_processing_unit_name)): # Check in root image dir
                 path_of_image_unit_for_segmentation = os.path.join(IMAGE_DIR_BASE, target_processing_unit_name)
                 logger.info(f"  Using rescaled image (matches target_processing_unit_name, found in image base): {path_of_image_unit_for_segmentation}")
            elif os.path.exists(target_processing_unit_name): # If full path was given and exists
                 path_of_image_unit_for_segmentation = target_processing_unit_name
                 logger.info(f"  Using rescaled image (target_processing_unit_name as path): {path_of_image_unit_for_segmentation}")
            else:
                 logger.warning(f"  Expected rescaled image {expected_rescaled_path} not found. And target_processing_unit_name {target_processing_unit_name} not found as alternative. Check paths.")
                 # Keep original_source_image_full_path as default, but this state is ambiguous for scaling.

    # Check if target_processing_unit_name indicates a tile
    # A simple check: if tiling is configured and target_processing_unit_name is not the main image (even if rescaled)
    if tiling_cfg and tiling_cfg.get("tile_size") and target_processing_unit_name != os.path.basename(path_of_image_unit_for_segmentation): # if target is not the (scaled) source
        is_tiled_job = True
        tile_storage_parent_dir_name = target_image_id
        if applied_scale_factor != 1.0: # If the image was scaled *before* tiling
            scale_factor_str_for_tile_parent = str(applied_scale_factor).replace('.', '_')
            tile_storage_parent_dir_name += f"_scaled{scale_factor_str_for_tile_parent}"
        
        # Path to the specific tile
        path_of_image_unit_for_segmentation = os.path.join(TILED_IMAGE_OUTPUT_BASE, tile_storage_parent_dir_name, target_processing_unit_name) # Uses imported TILED_IMAGE_OUTPUT_BASE
        logger.info(f"  This is a tiled job. Segmentation unit (tile) path: {path_of_image_unit_for_segmentation}")
        if not os.path.exists(path_of_image_unit_for_segmentation):
             logger.warning(f"  WARNING: Tile image file not found: {path_of_image_unit_for_segmentation}.")
    elif applied_scale_factor == 1.0 and target_processing_unit_name != original_image_filename :
        # If not tiled, not scaled by config, but target name is different from original, assume target_processing_unit_name is the direct path if it exists
        potential_direct_path = os.path.join(IMAGE_DIR_BASE, target_processing_unit_name) # Check relative to IMAGE_DIR_BASE
        if os.path.exists(target_processing_unit_name): # Check if absolute or already correct relative path
             path_of_image_unit_for_segmentation = target_processing_unit_name
        elif os.path.exists(potential_direct_path):
            path_of_image_unit_for_segmentation = potential_direct_path


    # Construct experiment ID and mask path using new utilities
    experiment_id_for_results_folder = construct_full_experiment_id(
        image_id=target_image_id,
        param_set_id=target_param_set_id,
        scale_factor=applied_scale_factor, # Use the determined scale factor
        processing_unit_name_for_tile=target_processing_unit_name if is_tiled_job else None,
        is_tile=is_tiled_job
    )
    
    mask_path = construct_mask_path(
        results_dir=RESULTS_DIR_BASE, # Uses imported RESULTS_DIR_BASE
        experiment_id=experiment_id_for_results_folder,
        processing_unit_name=target_processing_unit_name # The mask is named after the unit that was processed
    )
    
    logger.info(f"  Final derived path for image unit to use for display/finding mask: {path_of_image_unit_for_segmentation}")
    logger.info(f"  Final derived mask path: {mask_path}")
    logger.info(f"  Final derived experiment folder ID for results: {experiment_id_for_results_folder}")

    return path_of_image_unit_for_segmentation, mask_path, applied_scale_factor, original_source_image_full_path, experiment_id_for_results_folder


if __name__ == "__main__":
    # Setup basic logging for direct script running
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Visualize gene expression on segmented cells, using separate processing and visualization JSON configuration files.")
    parser.add_argument("--proc_config", default="processing_config.json",
                        help="Path to the processing JSON configuration file (e.g., processing_config.json). Contains image_configurations. Default: processing_config.json")
    parser.add_argument("--viz_config", default="visualization_config.json",
                        help="Path to the visualization tasks JSON configuration file (e.g., visualization_config.json). Default: visualization_config.json")
    parser.add_argument("--task_id", default=None,
                        help="Optional ID of a specific visualization task to run from the visualization_config.json. If not provided, all active tasks are run.")
    
    args = parser.parse_args()

    # Resolve paths for config files
    proc_config_full_path = args.proc_config
    if not os.path.isabs(proc_config_full_path):
        proc_config_full_path = os.path.join(PROJECT_ROOT, proc_config_full_path)

    viz_config_full_path = args.viz_config
    if not os.path.isabs(viz_config_full_path):
        viz_config_full_path = os.path.join(PROJECT_ROOT, viz_config_full_path)

    # Load Processing Config
    if not os.path.exists(proc_config_full_path):
        logger.error(f"Error: Processing configuration file not found: {proc_config_full_path}"); exit(1)
    try:
        with open(proc_config_full_path, 'r') as f:
            processing_config_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading/parsing processing configuration file {proc_config_full_path}: {e}"); exit(1)
    
    image_configurations = {img_cfg["image_id"]: img_cfg for img_cfg in processing_config_data.get("image_configurations", [])}
    if not image_configurations:
        logger.error(f"No 'image_configurations' found in {proc_config_full_path}. This is required for visualization backgrounds and metadata."); exit(1)

    # Load Visualization Config
    if not os.path.exists(viz_config_full_path):
        logger.error(f"Error: Visualization configuration file not found: {viz_config_full_path}"); exit(1)
    try:
        with open(viz_config_full_path, 'r') as f:
            viz_config_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading/parsing visualization configuration file {viz_config_full_path}: {e}"); exit(1)

    visualization_tasks_object = viz_config_data.get("visualization_tasks", {})
    default_genes_global = visualization_tasks_object.get("default_genes_to_visualize", [])
    if default_genes_global:
        logger.info(f"Found default_genes_to_visualize: {default_genes_global}")

    individual_tasks_list = visualization_tasks_object.get("tasks", [])
    if not individual_tasks_list:
        logger.info("No 'tasks' found within the 'visualization_tasks' object in {viz_config_full_path}."); exit(0)

    # Filter tasks to run
    tasks_to_run = []
    if args.task_id:
        task_found = False
        for task in individual_tasks_list:
            if task.get("task_id") == args.task_id:
                tasks_to_run.append(task)
                task_found = True
                break
        if not task_found:
            logger.error(f"Error: Visualization task with ID '{args.task_id}' not found in {viz_config_full_path}."); exit(1)
    else:
        for task in individual_tasks_list:
            if task.get("is_active", True):
                tasks_to_run.append(task)

    if not tasks_to_run:
        logger.info("No active visualization tasks to run."); exit(0)

    # --- The rest of the loop is largely the same, ensuring param_sets_full_path for get_job_info_and_paths refers to proc_config_full_path ---
    base_visualization_output_dir = os.path.join(RESULTS_DIR_BASE, "visualizations")
    if not os.path.exists(base_visualization_output_dir):
        try: 
            os.makedirs(base_visualization_output_dir)
            logger.info(f"Created base visualization directory: {base_visualization_output_dir}")
        except OSError as e: 
            logger.error(f"Error creating base visualization directory {base_visualization_output_dir}: {e}"); exit(1)

    for task_idx, task in enumerate(tasks_to_run):
        task_id_str = task.get("task_id", f"unnamed_task_{task_idx+1}")
        logger.info(f"\n--- Processing Visualization Task: {task_id_str} ---")

        genes_for_this_task = task.get("genes_to_visualize") 
        if genes_for_this_task is None: 
            genes_for_this_task = default_genes_global 
            if genes_for_this_task: 
                 logger.info(f"  Task '{task_id_str}': Using default_genes_to_visualize: {genes_for_this_task}")
        else:
            logger.info(f"  Task '{task_id_str}': Using task-specific genes_to_visualize: {genes_for_this_task}")

        if not genes_for_this_task: 
            logger.warning(f"  Task '{task_id_str}': No genes specified for visualization. Plotting may be skipped or empty.")
            genes_for_this_task = [] 
        
        genes_to_visualize = genes_for_this_task

        source_image_id = task.get("source_image_id")
        source_param_set_id = task.get("source_param_set_id")
        mapped_transcripts_csv_rel_path = task.get("mapped_transcripts_csv_path")
        output_subfolder_name = task.get("output_subfolder_name", task_id_str) 

        source_processing_unit_name = task.get("source_processing_unit_name")
        # image_config_for_task is now sourced from the loaded image_configurations dict from processing_config.json
        image_config_for_task = image_configurations.get(source_image_id)

        if not image_config_for_task:
            logger.warning(f"Skipping task '{task_id_str}': Source image_id '{source_image_id}' not found in image_configurations (from {proc_config_full_path}).")
            continue
        
        segmentation_opts = image_config_for_task.get("segmentation_options", {})
        tiling_params = segmentation_opts.get("tiling_parameters", {})
        image_was_configured_for_tiling = tiling_params.get("apply_tiling", False)

        if not source_processing_unit_name:
            if image_was_configured_for_tiling:
                logger.warning(f"Skipping task '{task_id_str}': 'source_processing_unit_display_name' is mandatory when the linked image_configuration (id: {source_image_id}) has apply_tiling=true, but was not provided.")
                continue
            else:
                original_fname = image_config_for_task.get("original_image_filename")
                if not original_fname:
                    logger.warning(f"Skipping task '{task_id_str}': Cannot derive source_processing_unit_name because 'original_image_filename' is missing in image_config '{source_image_id}'.")
                    continue
                
                rescaling_cfg = segmentation_opts.get("rescaling_config")
                scale_factor = 1.0
                if rescaling_cfg and "scale_factor" in rescaling_cfg:
                    sf_from_config = rescaling_cfg["scale_factor"]
                    if isinstance(sf_from_config, (int, float)) and 0 < sf_from_config <= 1.0:
                        scale_factor = sf_from_config
                
                if scale_factor != 1.0:
                    base, ext = os.path.splitext(os.path.basename(original_fname))
                    scale_factor_str_file = str(scale_factor).replace('.', '_')
                    source_processing_unit_name = f"{base}_scaled_{scale_factor_str_file}{ext}" 
                    logger.info(f"  Derived source_processing_unit_name for non-tiled task '{task_id_str}': {source_processing_unit_name} (scale: {scale_factor})")
                else:
                    source_processing_unit_name = os.path.basename(original_fname)
                    logger.info(f"  Derived source_processing_unit_name for non-tiled task '{task_id_str}': {source_processing_unit_name} (no rescaling)")
        
        if not all([source_image_id, source_param_set_id, source_processing_unit_name, mapped_transcripts_csv_rel_path, genes_to_visualize]):
            logger.warning(f"Skipping task '{task_id_str}': Missing one or more required fields OR no genes to visualize were resolved. Check: source_image_id, source_param_set_id, source_processing_unit_name, mapped_transcripts_csv_path, genes_to_visualize."); continue

        mpp_x_original = image_config_for_task.get("mpp_x")
        mpp_y_original = image_config_for_task.get("mpp_y")
        if mpp_x_original is None or mpp_y_original is None:
            logger.warning(f"Skipping task '{task_id_str}': mpp_x or mpp_y not found for image_id '{source_image_id}' in image_configurations."); continue
        
        mapped_transcripts_full_path = mapped_transcripts_csv_rel_path
        if not os.path.isabs(mapped_transcripts_full_path):
            mapped_transcripts_full_path = os.path.join(PROJECT_ROOT, mapped_transcripts_full_path)

        if not os.path.exists(mapped_transcripts_full_path):
            logger.warning(f"Skipping task '{task_id_str}': Mapped transcripts CSV not found: {mapped_transcripts_full_path}"); continue

        task_output_dir = os.path.join(base_visualization_output_dir, clean_filename_for_dir(output_subfolder_name))
        if not os.path.exists(task_output_dir):
            try: 
                os.makedirs(task_output_dir)
                logger.info(f"Created output directory for task '{task_id_str}': {task_output_dir}")
            except OSError as e: 
                logger.error(f"Error creating output dir {task_output_dir} for task '{task_id_str}': {e}"); continue
        
        # get_job_info_and_paths now needs the path to the processing config for its _get_image_config_from_params call
        background_image_to_display_path, segmentation_mask_path, scale_factor_applied, _, experiment_folder_id = get_job_info_and_paths(
            proc_config_full_path, # Pass the processing config path here
            source_image_id, 
            source_param_set_id, 
            source_processing_unit_name
        )

        # Determine the actual background image path to use
        actual_background_image_to_load = None
        task_viz_params = task.get("visualization_params", {})
        background_override_rel_path = task_viz_params.get("background_image_path_override")

        if background_override_rel_path:
            logger.info(f"  Task '{task_id_str}' has 'background_image_path_override': {background_override_rel_path}")
            # Assume override path is relative to PROJECT_ROOT if not absolute
            if not os.path.isabs(background_override_rel_path):
                resolved_override_path = os.path.join(PROJECT_ROOT, background_override_rel_path)
            else:
                resolved_override_path = background_override_rel_path
            
            if os.path.exists(resolved_override_path):
                actual_background_image_to_load = resolved_override_path
                logger.info(f"  Using overridden background image: {actual_background_image_to_load}")
            else:
                logger.warning(f"  Warning: 'background_image_path_override' specified but not found: {resolved_override_path}. Will attempt fallback.")
        
        # If override was not provided or was invalid, fall back to the path derived by get_job_info_and_paths
        if not actual_background_image_to_load:
            if background_image_to_display_path: # This is the path from get_job_info_and_paths
                if os.path.exists(background_image_to_display_path):
                    actual_background_image_to_load = background_image_to_display_path
                    logger.info(f"  Using derived background image (from original_image_filename in image_config): {actual_background_image_to_load}")
                else:
                    logger.warning(f"  Warning: Derived background image (from original_image_filename) not found: {background_image_to_display_path}")
            else: # background_image_to_display_path itself might be None if get_job_info_and_paths had issues earlier
                 logger.warning(f"  Warning: No valid background image path derived by get_job_info_and_paths.")


        if not segmentation_mask_path or not experiment_folder_id: # These are critical from get_job_info_and_paths
            logger.warning(f"Skipping task '{task_id_str}': Could not determine mask_path or experiment_folder_id from configuration."); continue
        
        if not actual_background_image_to_load: # If still no valid background after override and fallback
            logger.warning(f"Skipping task '{task_id_str}': No valid background image could be found or determined."); continue

        # Check existence of mask (background is now checked above)
        if not os.path.exists(segmentation_mask_path):
            logger.warning(f"Skipping task '{task_id_str}': Segmentation mask does not exist: {segmentation_mask_path}"); continue

        logger.info(f"  Task '{task_id_str}' - Results experiment folder ID: {experiment_folder_id}")
        logger.info(f"  Task '{task_id_str}' - Using mask: {segmentation_mask_path}")
        logger.info(f"  Task '{task_id_str}' - Confirmed background image for overlay: {actual_background_image_to_load} (Derived scale factor for MPP: {scale_factor_applied})")

        effective_mpp_x = mpp_x_original / scale_factor_applied if scale_factor_applied != 0 else mpp_x_original
        effective_mpp_y = mpp_y_original / scale_factor_applied if scale_factor_applied != 0 else mpp_y_original
        logger.info(f"  Task '{task_id_str}' - Original MPP (x,y): ({mpp_x_original}, {mpp_y_original})")
        logger.info(f"  Task '{task_id_str}' - Applied Scale Factor: {scale_factor_applied}")
        logger.info(f"  Task '{task_id_str}' - Effective MPP (x,y): ({effective_mpp_x:.4f}, {effective_mpp_y:.4f})")

        plot_dots_flag = task.get("plot_transcript_dots", False)
        if plot_dots_flag and (effective_mpp_x == 0 or effective_mpp_y == 0) :
             logger.warning(f"Warning for task '{task_id_str}': Effective MPP is zero, cannot plot dots. Check scale factor and original MPP."); plot_dots_flag = False

        mapped_df = load_mapped_transcripts(mapped_transcripts_full_path)
        seg_mask = cellpose_io.imread(segmentation_mask_path)
        display_background_raw = cv2.imread(actual_background_image_to_load, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) # Use the confirmed path

        if mapped_df is None or seg_mask is None or display_background_raw is None:
            logger.warning(f"Skipping task '{task_id_str}': Error loading one or more critical input files for visualization (MappedDF: {mapped_df is not None}, Mask: {seg_mask is not None}, BackgroundRaw: {display_background_raw is not None} from {actual_background_image_to_load})."); continue
            
        display_background_8bit = normalize_to_8bit_for_display(display_background_raw)
        if display_background_8bit is None:
            logger.error(f"Skipping task '{task_id_str}': Failed to normalize background image to 8-bit (normalize_to_8bit_for_display returned None)."); continue

        display_background_8bit_rgb = None
        if display_background_8bit.ndim == 2:
            display_background_8bit_rgb = cv2.cvtColor(display_background_8bit, cv2.COLOR_GRAY2RGB)
        elif display_background_8bit.ndim == 3 and display_background_8bit.shape[-1] == 4: 
            display_background_8bit_rgb = cv2.cvtColor(display_background_8bit, cv2.COLOR_BGRA2RGB)
        elif display_background_8bit.ndim == 3 and display_background_8bit.shape[-1] == 3:
            display_background_8bit_rgb = cv2.cvtColor(display_background_8bit, cv2.COLOR_BGR2RGB)
        else: 
            logger.warning(f"Warning for task '{task_id_str}': Background image for overlay has unexpected shape {display_background_8bit.shape}. Attempting grayscale.")
            if display_background_8bit.ndim > 2 and display_background_8bit.shape[-1] > 0 : display_background_8bit_gray = normalize_to_8bit_for_display(display_background_8bit[:,:,0]) 
            else: display_background_8bit_gray = display_background_8bit
            display_background_8bit_rgb = cv2.cvtColor(normalize_to_8bit_for_display(display_background_8bit_gray), cv2.COLOR_GRAY2RGB)
        
        if display_background_8bit_rgb.shape[:2] != seg_mask.shape[:2]:
            logger.error(f"FATAL Error for task '{task_id_str}': Background image shape {display_background_8bit_rgb.shape[:2]} and mask shape {seg_mask.shape[:2]} DO NOT MATCH!")
            logger.error(f"  Background image path used: {actual_background_image_to_load}"); continue

        file_suffix = "_expression_overlay.png"
        x_offset = task.get("x_offset_microns", 0.0)
        y_offset = task.get("y_offset_microns", 0.0)
        colormap = task.get("colormap", "viridis")

        for gene in genes_to_visualize: 
            # Extract the last part of the gene name after splitting by '.'
            gene_parts = gene.split('.')
            short_gene_name = gene_parts[-1] if gene_parts else gene
            safe_gene_name = clean_filename_for_dir(short_gene_name) # Sanitize the shortened name
            
            output_filename = f"{safe_gene_name}{file_suffix}"
            output_png_path = os.path.join(task_output_dir, output_filename)
            
            visualize_gene_expression_for_job(
                mapped_df, seg_mask, display_background_8bit_rgb, 
                gene, output_png_path,
                effective_mpp_x=effective_mpp_x, 
                effective_mpp_y=effective_mpp_y,
                task_visualization_params=task_viz_params,
                x_offset_microns=x_offset, 
                y_offset_microns=y_offset,
                plot_transcript_dots=plot_dots_flag
            )
        logger.info(f"--- Finished Visualization Task: {task_id_str} ---")
    
    logger.info("\nAll specified visualization tasks processed.")
