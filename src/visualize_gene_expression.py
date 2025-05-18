import os
import argparse
import pandas as pd
import numpy as np
import cv2
from cellpose import io as cellpose_io
from cellpose import plot as cellpose_plot # For mask_overlay
import matplotlib.pyplot as plt
import matplotlib.colors
import json 
import re

# Global settings
UNASSIGNED_CELL_ID = -1 
DEFAULT_BACKGROUND_COLOR_FOR_CELLS_NO_EXPRESSION = np.array([200, 200, 200], dtype=np.uint8) # Light gray
TRANSCRIPT_DOT_COLOR = (255, 0, 255) # Magenta for individual transcript dots (RGB)
TRANSCRIPT_DOT_SIZE = 1 # Pixels

# Base directories (assuming script is in src/, so project root is one level up)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR_BASE = os.path.join(PROJECT_ROOT, "images")
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
        # Return a small black image of appropriate dimensions if possible
        return np.zeros((100,100,3) if (img_array is not None and img_array.ndim == 3) else (100,100), dtype=np.uint8)
    
    # print(f"Normalizing. Input dtype: {img_array.dtype}, shape: {img_array.shape}, min: {np.min(img_array) if img_array.size > 0 else 'N/A'}, max: {np.max(img_array) if img_array.size > 0 else 'N/A'}")

    if img_array.dtype == np.uint8:
        # print("Image is already uint8.")
        return img_array
    
    img_out = None
    if np.issubdtype(img_array.dtype, np.floating):
        img_min, img_max = np.min(img_array), np.max(img_array)
        if img_max > img_min:
            img_out = ((img_array - img_min) / (img_max - img_min) * 255.0)
        else: 
            img_out = np.zeros_like(img_array) + 128 
    elif np.issubdtype(img_array.dtype, np.integer):
        img_min_val = np.min(img_array)
        img_max_val = np.max(img_array)
        if img_max_val > img_min_val:
            img_out = ((img_array.astype(np.float32) - img_min_val) / (img_max_val - img_min_val) * 255.0)
        elif img_max_val == img_min_val: 
            img_out = np.zeros_like(img_array, dtype=np.float32) + 128.0
        else: 
            img_out = np.zeros_like(img_array, dtype=np.float32)
    else:
        print(f"Warning: Unhandled dtype {img_array.dtype} in normalize_to_8bit. Attempting direct cast.")
        try:
            img_out = img_array.astype(np.float32) 
            img_min, img_max = np.min(img_out), np.max(img_out)
            if img_max > img_min:
                img_out = ((img_out - img_min) / (img_max - img_min) * 255.0)
            else:
                img_out = np.zeros_like(img_out) + 128
        except Exception as e:
            print(f"Could not convert {img_array.dtype} to float and scale: {e}")
            return np.zeros_like(img_array, dtype=np.uint8) 

    if img_out is None: 
        return np.zeros(img_array.shape if hasattr(img_array, 'shape') else (100,100), dtype=np.uint8)

    return img_out.astype(np.uint8)


def get_consistency_colors_for_mask(instance_mask, consensus_prob_map, colormap_func): # Changed colormap to colormap_func
    num_cells = instance_mask.max()
    cell_colors_rgb = np.zeros((int(num_cells) + 1, 3), dtype=np.uint8) 

    if num_cells == 0:
        return cell_colors_rgb 

    for i in range(1, int(num_cells) + 1):
        cell_pixels = (instance_mask == i)
        if np.any(cell_pixels):
            avg_consistency = np.mean(consensus_prob_map[cell_pixels]) 
            rgba_color = colormap_func(avg_consistency) 
            cell_colors_rgb[i] = (np.array(rgba_color[:3]) * 255).astype(np.uint8)
    return cell_colors_rgb


def visualize_gene_expression_on_cells(
    mapped_transcripts_df, 
    segmentation_mask, 
    original_image_8bit_rgb, 
    gene_of_interest, 
    output_path,
    mpp_x=None, mpp_y=None, 
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
    num_assigned_gene_transcripts = len(assigned_gene_transcripts_df)
    num_unassigned_gene_transcripts = total_gene_transcripts - num_assigned_gene_transcripts
    percent_unassigned = (num_unassigned_gene_transcripts / total_gene_transcripts * 100) if total_gene_transcripts > 0 else 0

    print(f"  Total transcripts for '{gene_of_interest}': {total_gene_transcripts}")
    print(f"  Assigned to cells: {num_assigned_gene_transcripts}")
    print(f"  Unassigned: {num_unassigned_gene_transcripts} ({percent_unassigned:.2f}%)")

    cell_expression_counts = assigned_gene_transcripts_df.groupby('assigned_cell_id').size()
    
    max_cell_id_in_mask = int(segmentation_mask.max())
    cell_colors_rgb = np.zeros((max_cell_id_in_mask + 1, 3), dtype=np.uint8)
    cell_colors_rgb[1:] = DEFAULT_BACKGROUND_COLOR_FOR_CELLS_NO_EXPRESSION 

    if not cell_expression_counts.empty:
        min_count = 0 
        max_count = cell_expression_counts.max()
        
        try: colormap_func = matplotlib.colormaps[colormap_name]
        except AttributeError: colormap_func = plt.cm.get_cmap(colormap_name)

        for cell_id, count in cell_expression_counts.items():
            cell_id_int = int(cell_id) # Ensure cell_id is int for indexing
            if cell_id_int > max_cell_id_in_mask or cell_id_int <=0 : continue
            
            normalized_count = 0.0
            if max_count > min_count:
                normalized_count = (count - min_count) / (max_count - min_count)
            elif max_count > 0: 
                normalized_count = 1.0 
            
            rgba_color = colormap_func(normalized_count)
            cell_colors_rgb[cell_id_int] = (np.array(rgba_color[:3]) * 255).astype(np.uint8)
    else:
        print(f"  No cells found with expression of '{gene_of_interest}'. Cells will have default color.")

    overlay_img_rgb_colored = cellpose_plot.mask_overlay(original_image_8bit_rgb, segmentation_mask, colors=cell_colors_rgb)

    if plot_transcript_dots and not gene_transcripts_df.empty and mpp_x is not None and mpp_y is not None:
        print(f"  Plotting {total_gene_transcripts} transcript dots for '{gene_of_interest}'...")
        # Ensure TRANSCRIPT_DOT_COLOR is in a format cv2.circle expects for an RGB image (tuple of BGR uint8)
        # plot.mask_overlay returns RGB, so if we draw on it directly, color should be RGB.
        # cv2.circle takes BGR by default if image is BGR. Let's assume overlay_img_rgb_colored is RGB.
        dot_color_rgb = TRANSCRIPT_DOT_COLOR # Assuming (R,G,B)
        
        t_x_pixels = ((gene_transcripts_df['x_location'] - x_offset_microns) / mpp_x).astype(int)
        t_y_pixels = ((gene_transcripts_df['y_location'] - y_offset_microns) / mpp_y).astype(int)

        height, width = overlay_img_rgb_colored.shape[:2]
        for tx, ty in zip(t_x_pixels, t_y_pixels):
            if 0 <= tx < width and 0 <= ty < height:
                cv2.circle(overlay_img_rgb_colored, (tx, ty), TRANSCRIPT_DOT_SIZE, dot_color_rgb, -1)
    
    try:
        cv2.imwrite(output_path, cv2.cvtColor(overlay_img_rgb_colored, cv2.COLOR_RGB2BGR))
        print(f"  Expression overlay for '{gene_of_interest}' saved to: {output_path}")
    except Exception as e:
        print(f"  Error saving expression overlay for '{gene_of_interest}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize gene expression on segmented cells.")
    parser.add_argument("mapped_transcripts_csv", help="Path to CSV with mapped transcripts.")
    parser.add_argument("segmentation_mask_tif", help="Path to segmentation mask TIFF.")
    parser.add_argument("original_image_tif", help="Path to original microscopy image for background.")
    parser.add_argument("output_dir", help="Directory to save visualizations.")
    parser.add_argument("--genes", nargs='+', required=True, help="List of gene names to visualize.")
    parser.add_argument("--mpp_x", type=float, help="Microns per pixel in X. Required if --plot_dots.")
    parser.add_argument("--mpp_y", type=float, help="Microns per pixel in Y. Required if --plot_dots.")
    parser.add_argument("--x_offset_microns", type=float, default=0.0)
    parser.add_argument("--y_offset_microns", type=float, default=0.0)
    parser.add_argument("--plot_dots", action='store_true', help="Plot individual transcript locations.")
    parser.add_argument("--colormap", default='viridis', help="Matplotlib colormap (default: 'viridis').")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        try: os.makedirs(args.output_dir); print(f"Created output directory: {args.output_dir}")
        except OSError as e: print(f"Error creating output directory {args.output_dir}: {e}"); exit(1)
            
    if args.plot_dots and (args.mpp_x is None or args.mpp_y is None):
        parser.error("--mpp_x and --mpp_y are required when --plot_dots is enabled.")

    mapped_df = load_mapped_transcripts(args.mapped_transcripts_csv)
    seg_mask = cellpose_io.imread(args.segmentation_mask_tif) # Use cellpose_io for mask consistently
    orig_img_raw = cv2.imread(args.original_image_tif, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    if mapped_df is None or seg_mask is None or orig_img_raw is None:
        print("Error loading one or more input files. Exiting."); exit(1)
        
    original_img_8bit = normalize_to_8bit_for_display(orig_img_raw)
    if original_img_8bit.ndim == 2:
        original_image_8bit_rgb = cv2.cvtColor(original_img_8bit, cv2.COLOR_GRAY2RGB)
    elif original_img_8bit.ndim == 3 and original_img_8bit.shape[-1] == 4: 
        original_image_8bit_rgb = cv2.cvtColor(original_img_8bit, cv2.COLOR_BGRA2RGB)
    elif original_img_8bit.ndim == 3 and original_img_8bit.shape[-1] == 3:
        original_image_8bit_rgb = cv2.cvtColor(original_img_8bit, cv2.COLOR_BGR2RGB)
    else:
        print(f"Warning: Original image has unexpected shape {original_img_8bit.shape}. Attempting to use as is.")
        original_image_8bit_rgb = original_img_8bit 

    for gene in args.genes:
        # Clean gene name for use in filename
        cleaned_gene_name = clean_filename_for_dir(gene)
        output_png_path = os.path.join(args.output_dir, f"{cleaned_gene_name}_expression_overlay.png")
        visualize_gene_expression_on_cells(
            mapped_df, seg_mask, original_image_8bit_rgb, 
            gene, output_png_path,
            mpp_x=args.mpp_x, mpp_y=args.mpp_y,
            x_offset_microns=args.x_offset_microns, y_offset_microns=args.y_offset_microns,
            plot_transcript_dots=args.plot_dots,
            colormap_name=args.colormap
        )
    
    print("Gene expression visualization process finished.")

