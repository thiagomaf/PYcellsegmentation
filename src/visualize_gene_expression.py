import os
import argparse
import pandas as pd
import numpy as np
import cv2
from cellpose import io as cellpose_io
from cellpose import plot as cellpose_plot
import matplotlib.pyplot as plt
import matplotlib.colors
import json
import re
import traceback

# Global settings
UNASSIGNED_CELL_ID = -1
DEFAULT_BACKGROUND_COLOR_FOR_CELLS_NO_EXPRESSION = np.array([200, 200, 200], dtype=np.uint8)
TRANSCRIPT_DOT_COLOR = (255, 0, 255) # Magenta (RGB)
TRANSCRIPT_DOT_SIZE = 1

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR_BASE = os.path.join(PROJECT_ROOT, "images")
TILED_IMAGE_OUTPUT_BASE = os.path.join(IMAGE_DIR_BASE, "tiled_outputs")
RESCALED_IMAGE_CACHE_DIR = os.path.join(IMAGE_DIR_BASE, "rescaled_cache")
RESULTS_DIR_BASE = os.path.join(PROJECT_ROOT, "results")

def clean_filename_for_dir(filename):
    name_without_ext = os.path.splitext(filename)[0]
    cleaned_name = re.sub(r'[^\w\.-]', '_', name_without_ext)
    return cleaned_name

def load_mapped_transcripts(mapped_transcripts_csv_path):
    print(f"Loading mapped transcripts from: {mapped_transcripts_csv_path} ...")
    try:
        df = pd.read_csv(mapped_transcripts_csv_path)
        required_cols = ['transcript_id', 'x_location', 'y_location', 'feature_name', 'assigned_cell_id']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in mapped transcripts file.")
        print(f"Loaded {len(df)} mapped transcript records.")
        return df
    except Exception as e:
        print(f"Error loading mapped transcripts CSV: {e}")
        return None

def normalize_to_8bit_for_display(img_array):
    if img_array is None or img_array.size == 0:
        print("Warning: normalize_to_8bit received None or empty input.")
        return np.zeros((100,100,3) if (img_array is not None and hasattr(img_array, 'ndim') and img_array.ndim == 3) else (100,100), dtype=np.uint8)
    
    if img_array.dtype == np.uint8: return img_array
    
    img_out = None
    if np.issubdtype(img_array.dtype, np.floating):
        img_min, img_max = np.min(img_array), np.max(img_array)
        if img_max > img_min: img_out = ((img_array - img_min) / (img_max - img_min) * 255.0)
        else: img_out = np.zeros_like(img_array) + 128 
    elif np.issubdtype(img_array.dtype, np.integer):
        img_min_val, img_max_val = np.min(img_array), np.max(img_array)
        if img_max_val > img_min_val: img_out = ((img_array.astype(np.float32) - img_min_val) / (img_max_val - img_min_val) * 255.0)
        elif img_max_val == img_min_val: img_out = np.zeros_like(img_array, dtype=np.float32) + 128.0
        else: img_out = np.zeros_like(img_array, dtype=np.float32)
    else:
        print(f"Warning: Unhandled dtype {img_array.dtype} in normalize_to_8bit. Attempting direct cast.")
        try:
            img_out = img_array.astype(np.float32) 
            img_min, img_max = np.min(img_out), np.max(img_out)
            if img_max > img_min: img_out = ((img_out - img_min) / (img_max - img_min) * 255.0)
            else: img_out = np.zeros_like(img_out) + 128
        except Exception as e:
            print(f"Could not convert {img_array.dtype} to float and scale: {e}")
            return np.zeros_like(img_array, dtype=np.uint8) 

    if img_out is None: return np.zeros(img_array.shape if hasattr(img_array, 'shape') else (100,100), dtype=np.uint8)
    return img_out.astype(np.uint8)

def get_colors_for_gene_expression(instance_mask, cell_expression_counts, colormap_func, default_color):
    num_mask_labels = instance_mask.max()
    cell_colors_rgb = np.zeros((int(num_mask_labels) + 1, 3), dtype=np.uint8)
    cell_colors_rgb[:] = default_color # Fill with default (e.g., for cells with no expression)

    if cell_expression_counts.empty:
        return cell_colors_rgb

    min_count = 0 
    max_count = cell_expression_counts.max()
    if max_count == 0 : # All expressing cells have 0 count (or no expressing cells)
        return cell_colors_rgb # All remain default color

    for cell_id, count in cell_expression_counts.items():
        cell_id_int = int(cell_id)
        if 0 < cell_id_int <= num_mask_labels:
            norm_count = 0.0
            if max_count > min_count: # Should always be true if max_count > 0
                norm_count = (count - min_count) / (max_count - min_count)
            elif max_count > 0: # Handles if all counts are the same and > min_count (e.g. all 1)
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
    x_offset_microns=0, y_offset_microns=0, 
    plot_transcript_dots=False,
    colormap_name='viridis'
    ):

    print(f"--- Visualizing expression for gene: {gene_of_interest} ---")
    gene_transcripts_df = mapped_transcripts_df[mapped_transcripts_df['feature_name'] == gene_of_interest]
    total_gene_transcripts = len(gene_transcripts_df)
    if total_gene_transcripts == 0:
        print(f"No transcripts found for gene '{gene_of_interest}'. Skipping visualization."); return

    assigned_gene_transcripts_df = gene_transcripts_df[gene_transcripts_df['assigned_cell_id'] != UNASSIGNED_CELL_ID]
    num_assigned = len(assigned_gene_transcripts_df)
    num_unassigned = total_gene_transcripts - num_assigned
    percent_unassigned = (num_unassigned / total_gene_transcripts * 100) if total_gene_transcripts > 0 else 0
    print(f"  Total '{gene_of_interest}' transcripts: {total_gene_transcripts} (Assigned: {num_assigned}, Unassigned: {num_unassigned} [{percent_unassigned:.2f}%])")

    cell_expression_counts = assigned_gene_transcripts_df.groupby('assigned_cell_id').size()
    
    try: colormap_func = matplotlib.colormaps[colormap_name]
    except AttributeError: colormap_func = plt.cm.get_cmap(colormap_name)

    cell_specific_colors = get_colors_for_gene_expression(
        segmentation_mask_array, cell_expression_counts, colormap_func, 
        DEFAULT_BACKGROUND_COLOR_FOR_CELLS_NO_EXPRESSION
    )
    if not cell_expression_counts.empty():
         print(f"  Coloring {len(cell_expression_counts)} cells based on '{gene_of_interest}' expression.")
    else:
        print(f"  No cells expressed '{gene_of_interest}'. Using default background color for all cells.")

    overlay_img_rgb = cellpose_plot.mask_overlay(background_image_for_overlay_rgb, segmentation_mask_array, colors=cell_specific_colors)

    if plot_transcript_dots and not gene_transcripts_df.empty:
        print(f"  Plotting {total_gene_transcripts} transcript dots for '{gene_of_interest}'...")
        dot_color_rgb = TRANSCRIPT_DOT_COLOR 
        
        t_x_pixels = ((gene_transcripts_df['x_location'] - x_offset_microns) / effective_mpp_x).astype(int)
        t_y_pixels = ((gene_transcripts_df['y_location'] - y_offset_microns) / effective_mpp_y).astype(int)

        h, w = overlay_img_rgb.shape[:2]
        for tx, ty in zip(t_x_pixels, t_y_pixels):
            if 0 <= tx < w and 0 <= ty < h:
                cv2.circle(overlay_img_rgb, (tx, ty), TRANSCRIPT_DOT_SIZE, dot_color_rgb, -1)
    try:
        cv2.imwrite(output_png_path, cv2.cvtColor(overlay_img_rgb, cv2.COLOR_RGB2BGR))
        print(f"  Expression overlay for '{gene_of_interest}' saved to: {output_png_path}")
    except Exception as e:
        print(f"  Error saving expression overlay for '{gene_of_interest}': {e}")

def get_job_paths_and_scale(param_sets_path, target_image_id, target_param_set_id, target_processing_unit_name):
    if not os.path.exists(param_sets_path):
        print(f"Error: Config file '{param_sets_path}' not found."); return None, None, 1.0, None

    try:
        with open(param_sets_path, 'r') as f: config_data = json.load(f)
    except Exception as e:
        print(f"Error reading/parsing {param_sets_path}: {e}"); return None, None, 1.0, None

    image_config = None
    for img_cfg in config_data.get("image_configurations", []):
        if img_cfg.get("image_id") == target_image_id:
            image_config = img_cfg; break
    if not image_config:
        print(f"Error: Image ID '{target_image_id}' not found."); return None, None, 1.0, None

    original_image_filename = image_config.get("original_image_filename")
    if not original_image_filename: print("Error: original_image_filename missing."); return None,None,1.0, None
    
    original_image_full_path = os.path.join(IMAGE_DIR_BASE, original_image_filename)
    path_of_image_unit_for_segmentation = original_image_full_path 
    applied_scale_factor = 1.0
    is_tiled_job = False

    rescaling_cfg = image_config.get("rescaling_config")
    if rescaling_cfg and "scale_factor" in rescaling_cfg and rescaling_cfg["scale_factor"] != 1.0:
        applied_scale_factor = rescaling_cfg["scale_factor"]
        scale_factor_str = str(applied_scale_factor).replace('.', '_')
        rescaled_fn = f"{os.path.splitext(original_image_filename)[0]}_scaled_{scale_factor_str}.tif"
        path_of_image_unit_for_segmentation = os.path.join(RESCALED_IMAGE_CACHE_DIR, target_image_id, rescaled_fn)

    tiling_cfg = image_config.get("tiling_config")
    if tiling_cfg and tiling_cfg.get("tile_size"):
        is_tiled_job = True
        tile_storage_dir_suffix = target_image_id
        if applied_scale_factor != 1.0:
            scale_factor_str = str(applied_scale_factor).replace('.', '_')
            tile_storage_dir_suffix += f"_scaled{scale_factor_str}"
        tile_dir = os.path.join(TILED_IMAGE_OUTPUT_BASE, tile_storage_dir_suffix)
        path_of_image_unit_for_segmentation = os.path.join(tile_dir, target_processing_unit_name)

    cleaned_proc_unit_name = clean_filename_for_dir(target_processing_unit_name)
    experiment_id_final_for_mask_folder = ""
    if is_tiled_job:
        experiment_id_final_for_mask_folder = f"{target_image_id}_{target_param_set_id}_{cleaned_proc_unit_name}"
    else: 
        experiment_id_final_for_mask_folder = f"{target_image_id}_{target_param_set_id}"
        if applied_scale_factor != 1.0: 
            scale_factor_str = str(applied_scale_factor).replace('.', '_')
            experiment_id_final_for_mask_folder += f"_scaled{scale_factor_str}"
            
    mask_filename_part = os.path.splitext(target_processing_unit_name)[0] + "_mask.tif"
    mask_path = os.path.join(RESULTS_DIR_BASE, experiment_id_final_for_mask_folder, mask_filename_part)

    if not os.path.exists(path_of_image_unit_for_segmentation): print(f"Warning: Constructed background image path non-existent: {path_of_image_unit_for_segmentation}")
    if not os.path.exists(mask_path): print(f"Warning: Constructed mask path non-existent: {mask_path}")
        
    return path_of_image_unit_for_segmentation, mask_path, applied_scale_factor, original_image_full_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize gene expression on segmented cells, using parameter_sets.json for paths.")
    parser.add_argument("parameter_sets_json", help="Path to the parameter_sets.json file (relative to project root).")
    parser.add_argument("image_id", help="The 'image_id' from image_configurations in JSON.")
    parser.add_argument("param_set_id", help="The 'param_set_id' from cellpose_parameter_configurations in JSON.")
    parser.add_argument("processing_unit_name", help="Filename of the specific image/tile that was segmented (e.g., 'tile_r0_c0.tif', 'original_image_scaled_0_5.tif', or 'original_image.tif').")
    parser.add_argument("mapped_transcripts_csv", help="Path to the CSV with mapped transcripts.")
    parser.add_argument("output_dir", help="Directory to save visualizations.")
    parser.add_argument("--genes", nargs='+', required=True, help="List of gene names to visualize.")
    parser.add_argument("--mpp_x_original", type=float, required=True, help="Microns per pixel in X of the *original, unscaled* source image.")
    parser.add_argument("--mpp_y_original", type=float, required=True, help="Microns per pixel in Y of the *original, unscaled* source image.")
    parser.add_argument("--x_offset_microns", type=float, default=0.0)
    parser.add_argument("--y_offset_microns", type=float, default=0.0)
    parser.add_argument("--plot_dots", action='store_true')
    parser.add_argument("--colormap", default='viridis')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        try: os.makedirs(args.output_dir); print(f"Created output directory: {args.output_dir}")
        except OSError as e: print(f"Error creating output dir {args.output_dir}: {e}"); exit(1)

    # parameter_sets_json is relative to project root where script is run from with python -m src.visualize_gene_expression
    # So, directly use args.parameter_sets_json, or construct full path if needed.
    # Assuming it's run from project root, args.parameter_sets_json is fine.

    background_image_to_segment_path, segmentation_mask_path, scale_factor_applied, original_source_image_path = get_job_paths_and_scale(
        args.parameter_sets_json, args.image_id, args.param_set_id, args.processing_unit_name
    )

    if not background_image_to_segment_path or not segmentation_mask_path:
        print("Could not determine necessary file paths from configuration. Exiting."); exit(1)
    if not os.path.exists(background_image_to_segment_path):
        print(f"Determined background image for segmentation path does not exist: {background_image_to_segment_path}"); exit(1)
    if not os.path.exists(segmentation_mask_path):
        print(f"Determined segmentation mask path does not exist: {segmentation_mask_path}"); exit(1)
    if not os.path.exists(original_source_image_path):
        print(f"Original source image path for display does not exist: {original_source_image_path}"); exit(1)

    print(f"--- Visualizing for Image ID: {args.image_id}, Param Set: {args.param_set_id}, Unit: {args.processing_unit_name} ---")
    print(f"  Using mask: {segmentation_mask_path}")
    print(f"  Original source image for display: {original_source_image_path}")
    print(f"  Image unit that was segmented: {background_image_to_segment_path} (Scale factor from original: {scale_factor_applied})")

    effective_mpp_x = args.mpp_x_original / scale_factor_applied if scale_factor_applied != 0 else args.mpp_x_original
    effective_mpp_y = args.mpp_y_original / scale_factor_applied if scale_factor_applied != 0 else args.mpp_y_original
    print(f"  Original MPP (x,y): ({args.mpp_x_original}, {args.mpp_y_original})")
    print(f"  Applied Scale Factor to segmented unit: {scale_factor_applied}")
    print(f"  Effective MPP for Mask & transcript mapping (x,y): ({effective_mpp_x:.4f}, {effective_mpp_y:.4f})")

    if args.plot_dots and (effective_mpp_x == 0 or effective_mpp_y == 0) :
         parser.error("Effective MPP is zero, cannot plot dots. Check scale factor and original MPP.")

    mapped_df = load_mapped_transcripts(args.mapped_transcripts_csv)
    seg_mask = cellpose_io.imread(segmentation_mask_path)
    
    # The background for display should be the original image, rescaled to match the mask if necessary.
    # The mask corresponds to background_image_to_segment_path. We need to load THAT for display.
    display_background_raw = cv2.imread(background_image_to_segment_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    if mapped_df is None or seg_mask is None or display_background_raw is None:
        print("Error loading one or more critical input files for visualization. Exiting."); exit(1)
        
    display_background_8bit = normalize_to_8bit_for_display(display_background_raw)
    display_background_8bit_rgb = None
    if display_background_8bit.ndim == 2:
        display_background_8bit_rgb = cv2.cvtColor(display_background_8bit, cv2.COLOR_GRAY2RGB)
    elif display_background_8bit.ndim == 3 and display_background_8bit.shape[-1] == 4: 
        display_background_8bit_rgb = cv2.cvtColor(display_background_8bit, cv2.COLOR_BGRA2RGB)
    elif display_background_8bit.ndim == 3 and display_background_8bit.shape[-1] == 3:
        display_background_8bit_rgb = cv2.cvtColor(display_background_8bit, cv2.COLOR_BGR2RGB)
    else:
        print(f"Warning: Background image for overlay has unexpected shape {display_background_8bit.shape}. Using as is.")
        display_background_8bit_rgb = display_background_8bit 

    if display_background_8bit_rgb.shape[:2] != seg_mask.shape[:2]:
        print(f"FATAL Error: Background image shape {display_background_8bit_rgb.shape[:2]} and mask shape {seg_mask.shape[:2]} DO NOT MATCH!")
        print(f"  Background image path used: {background_image_to_segment_path}")
        exit(1)

    for gene in args.genes:
        cleaned_gene_name = clean_filename_for_dir(gene)
        output_png_path = os.path.join(args.output_dir, f"{cleaned_gene_name}_expression_overlay.png")
        
        visualize_gene_expression_for_job(
            mapped_df, seg_mask, display_background_8bit_rgb, 
            gene, output_png_path,
            effective_mpp_x=effective_mpp_x, 
            effective_mpp_y=effective_mpp_y,
            x_offset_microns=args.x_offset_microns, 
            y_offset_microns=args.y_offset_microns,
            plot_transcript_dots=args.plot_dots,
            colormap_name=args.colormap
        )
    
    print("Gene expression visualization process finished.")

