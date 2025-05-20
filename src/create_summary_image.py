import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import math
from cellpose import io, plot
import argparse
from .pipeline_utils import construct_full_experiment_id

# --- Configuration ---
DEFAULT_CONFIG_FILE = "processing_config.json"
IMAGE_DIR_BASE = "images" 
RESULTS_DIR_BASE = "results"
SUMMARY_IMAGE_FILENAME_BASE = "segmentation_summary_consistency"
IMAGES_PER_ROW = 3
CONSENSUS_THRESHOLD_FOR_DICE = 0.5 
COLORMAP_NAME = 'RdYlGn' 

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR_PATH = os.path.join(PROJECT_ROOT, IMAGE_DIR_BASE) 
RESULTS_DIR_PATH = os.path.join(PROJECT_ROOT, RESULTS_DIR_BASE)
SUMMARY_IMAGE_SAVE_PATH = os.path.join(RESULTS_DIR_PATH, SUMMARY_IMAGE_FILENAME_BASE + ".png")


def load_config_data(json_file_path):
    if not os.path.exists(json_file_path):
        print(f"Error: Configuration file '{json_file_path}' not found.")
        return None
    try:
        with open(json_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading/parsing {json_file_path}: {e}")
        return None

def calculate_dice_coefficient(mask1_binary, mask2_binary):
    intersection = np.sum(mask1_binary & mask2_binary)
    sum_masks = np.sum(mask1_binary) + np.sum(mask2_binary)
    if sum_masks == 0: return 1.0 if intersection == 0 else 0.0
    return (2.0 * intersection) / sum_masks

def normalize_to_8bit_simple(img_array): 
    if img_array is None or img_array.size == 0: return np.zeros((100,100), dtype=np.uint8)
    if img_array.dtype == np.uint8: return img_array
    
    img_min, img_max = np.min(img_array), np.max(img_array)
    if img_max > img_min:
        img_out = ((img_array.astype(np.float32) - img_min) / (img_max - img_min) * 255.0)
    else:
        img_out = np.zeros_like(img_array, dtype=np.float32) + (128.0 if img_array.size > 0 else 0)
    return img_out.astype(np.uint8)


def get_consistency_colors_for_mask(instance_mask, consensus_prob_map, colormap_func):
    num_cells = instance_mask.max()
    cell_colors_rgb = np.zeros((int(num_cells) + 1, 3), dtype=np.uint8)
    if num_cells == 0: return cell_colors_rgb 

    for i in range(1, int(num_cells) + 1):
        cell_pixels = (instance_mask == i)
        if np.any(cell_pixels):
            avg_consistency = np.mean(consensus_prob_map[cell_pixels]) 
            rgba_color = colormap_func(avg_consistency) 
            cell_colors_rgb[i] = (np.array(rgba_color[:3]) * 255).astype(np.uint8)
    return cell_colors_rgb


def create_summary_image_with_consistency_coloring(
    all_image_configs, 
    all_cellpose_params, 
    base_results_dir,
    base_image_dir,
    output_summary_image_path 
    ):
    if not all_image_configs or not all_cellpose_params:
        print("No image configurations or cellpose parameters provided for summary.")
        return

    print("Gathering masks and original images for summary...")
    all_masks_data = []
    original_images_cache = {} 
    mask_shape = None
    
    active_image_configs = [ic for ic in all_image_configs if ic.get("is_active", True)]
    active_cellpose_params = [cp for cp in all_cellpose_params if cp.get("is_active", True)]

    if not active_image_configs or not active_cellpose_params:
        print("No active image configurations or cellpose parameters to generate summary for.")
        return

    for img_config in active_image_configs:
        image_id = img_config.get("image_id")
        original_image_relative_path = img_config.get("original_image_filename")
        if not image_id or not original_image_relative_path:
            print(f"Skipping image config due to missing 'image_id' or 'original_image_filename': {img_config}")
            continue

        original_image_full_path = os.path.normpath(os.path.join(PROJECT_ROOT, original_image_relative_path))

        if original_image_relative_path not in original_images_cache:
            if os.path.exists(original_image_full_path):
                try:
                    orig_img_cv = cv2.imread(original_image_full_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    if orig_img_cv is None: raise ValueError(f"cv2.imread returned None for {original_image_full_path}")
                    original_images_cache[original_image_relative_path] = normalize_to_8bit_simple(orig_img_cv)
                except Exception as e:
                    print(f"Error loading original image {original_image_full_path}: {e}. Skipping this image for summary.")
                    continue
            else:
                print(f"Original image {original_image_full_path} not found. Skipping this image for summary.")
                continue
        
        for cp_config in active_cellpose_params:
            param_set_id = cp_config.get("param_set_id")
            if not param_set_id:
                print(f"Skipping cellpose config due to missing 'param_set_id': {cp_config}")
                continue

            derived_experiment_id = f"{image_id}_{param_set_id}" 
            
            original_image_basename = os.path.splitext(os.path.basename(original_image_relative_path))[0]
            mask_filename = f"{original_image_basename}_mask.tif"
            
            mask_filepath = os.path.join(base_results_dir, derived_experiment_id, mask_filename)

            if os.path.exists(mask_filepath):
                try:
                    mask = io.imread(mask_filepath) 
                    if mask_shape is None: mask_shape = mask.shape
                    elif mask.shape != mask_shape:
                        print(f"Shape mismatch for mask {mask_filepath}. Expected {mask_shape}, got {mask.shape}. Skipping.")
                        continue
                    all_masks_data.append({
                        "id": derived_experiment_id, 
                        "mask": mask, 
                        "original_image_relative_path": original_image_relative_path, 
                        "image_id": image_id,
                        "param_set_id": param_set_id
                    })
                except Exception as e:
                    print(f"Error loading mask {mask_filepath} for {derived_experiment_id}: {e}. Skipping.")
            else:
                print(f"Mask file not found for {derived_experiment_id} at {mask_filepath}. Skipping.")

    if not all_masks_data or mask_shape is None:
        print("No valid segmentation masks found (from active configurations) or mask_shape not determined for summary.")
        return
    print(f"Loaded {len(all_masks_data)} segmentation masks for summary processing.")

    print("Generating consensus probability map (from loaded masks)...")
    consensus_accumulator = np.zeros(mask_shape, dtype=np.float32)
    for data in all_masks_data:
        consensus_accumulator += (data["mask"] > 0).astype(np.float32)
    
    if len(all_masks_data) == 0: print("Cannot divide by zero: no masks for consensus."); return
        
    consensus_probability_map = consensus_accumulator / len(all_masks_data)
    binary_consensus_for_dice = (consensus_probability_map > CONSENSUS_THRESHOLD_FOR_DICE).astype(np.uint8)
    print("Consensus probability map generated.")

    plot_data = []
    for data in all_masks_data: 
        exp_id = data["id"]
        individual_binary_mask = (data["mask"] > 0).astype(np.uint8)
        dice_score = calculate_dice_coefficient(individual_binary_mask, binary_consensus_for_dice)
        plot_data.append({"id": exp_id, "mask": data["mask"], "original_image_relative_path": data["original_image_relative_path"], "dice": dice_score})
        print(f"Experiment {exp_id}: Dice against consensus = {dice_score:.4f}")
    
    num_plots = len(plot_data)
    cols = IMAGES_PER_ROW
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5.5)) 
    if num_plots == 0 : print("No data to plot for summary."); plt.close(fig); return
    if num_plots == 1 and not isinstance(axes, np.ndarray): axes = np.array([axes])
    axes = axes.flatten()

    try: colormap_func = matplotlib.colormaps[COLORMAP_NAME]
    except AttributeError: colormap_func = plt.cm.get_cmap(COLORMAP_NAME)

    for i, data in enumerate(plot_data): 
        ax = axes[i]
        original_img_8bit_for_display = original_images_cache[data["original_image_relative_path"]]
        
        img_to_show_under_mask = original_img_8bit_for_display.copy()
        if img_to_show_under_mask.ndim == 3 and img_to_show_under_mask.shape[-1] == 3: 
            img_to_show_under_mask = cv2.cvtColor(img_to_show_under_mask, cv2.COLOR_BGR2RGB)
        elif img_to_show_under_mask.ndim == 2: pass 
        elif img_to_show_under_mask.ndim == 3 and img_to_show_under_mask.shape[-1] == 1: 
            img_to_show_under_mask = img_to_show_under_mask[:,:,0]
        else: 
            if img_to_show_under_mask.ndim > 2: img_to_show_under_mask = img_to_show_under_mask[:,:,0]
            if img_to_show_under_mask.dtype != np.uint8 : img_to_show_under_mask = normalize_to_8bit_simple(img_to_show_under_mask)

        cell_specific_colors = get_consistency_colors_for_mask(data["mask"], consensus_probability_map, colormap_func)
        overlay_img_rgb = plot.mask_overlay(img_to_show_under_mask, data["mask"], colors=cell_specific_colors)
        
        ax.imshow(overlay_img_rgb)
        ax.set_title(f"{data['id']} Dice: {data['dice']:.3f}", fontsize=9)
        ax.axis('off')

    for j in range(num_plots, len(axes)): fig.delaxes(axes[j])
    plt.tight_layout(pad=2.5, h_pad=2.5, w_pad=1.5) 
    
    summary_output_dir = os.path.dirname(output_summary_image_path)
    if not os.path.exists(summary_output_dir):
        try: os.makedirs(summary_output_dir); print(f"Created directory for summary image: {summary_output_dir}")
        except OSError as e: print(f"Error creating dir {summary_output_dir}: {e}"); return

    try:
        plt.savefig(output_summary_image_path, dpi=120)
        print(f"Summary image with consistency colors saved to: {output_summary_image_path}")
    except Exception as e:
        print(f"Error saving summary image: {e}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a summary image with consistency-colored overlays and Dice scores from segmentation results.")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, DEFAULT_CONFIG_FILE),
                        help=f"Path to the processing JSON configuration file (default: {DEFAULT_CONFIG_FILE} in project root).")
    parser.add_argument("--results_dir", default=os.path.join(PROJECT_ROOT, "results"),
                        help="Base directory where segmentation results (experiment folders with masks) are stored (default: 'results' in project root).")
    parser.add_argument("--output_filename", default=SUMMARY_IMAGE_FILENAME_BASE + ".png",
                        help=f"Filename for the output summary image (default: {SUMMARY_IMAGE_FILENAME_BASE}.png). Will be saved in the --results_dir.")

    args = parser.parse_args()

    print(f"Creating summary image. Using config: {args.config}")
    
    config_data = load_config_data(args.config)

    if config_data:
        image_configurations = config_data.get("image_configurations", [])
        cellpose_param_configs = config_data.get("cellpose_parameter_configurations", [])
        
        if not image_configurations:
            print("No 'image_configurations' found in the config file.")
        elif not cellpose_param_configs:
            print("No 'cellpose_parameter_configurations' found in the config file.")
        else:
            print(f"Found {len(image_configurations)} image configurations and {len(cellpose_param_configs)} Cellpose parameter sets.")
            
            create_summary_image_with_consistency_coloring(
                image_configurations, 
                cellpose_param_configs,
                args.results_dir,
                PROJECT_ROOT,
                os.path.join(args.results_dir, args.output_filename)
            )
    else:
        print(f"Could not load configuration from {args.config}. Exiting summary creation.")

