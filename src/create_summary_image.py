import os
import json
import cv2 # For image loading and color conversion if needed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import math
from cellpose import io, plot # <--- CORRECTED IMPORT

# --- Configuration ---
PARAMETER_SETS_FILE = "parameter_sets.json"
IMAGE_DIR_BASE = "images" # Base directory for original images
RESULTS_DIR_BASE = "results"
SUMMARY_IMAGE_FILENAME = "segmentation_summary_consistency.png"
IMAGES_PER_ROW = 3
CONSENSUS_THRESHOLD_FOR_DICE = 0.5 # Threshold for binarizing consensus map for Dice score calculation
COLORMAP_NAME = 'RdYlGn' # Red (low consistency) to Yellow (mid) to Green (high consistency)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # This script is in src/
PARAMETER_SETS_FILE_PATH = os.path.join(PROJECT_ROOT, PARAMETER_SETS_FILE)
IMAGE_DIR_PATH = os.path.join(PROJECT_ROOT, IMAGE_DIR_BASE)
RESULTS_DIR_PATH = os.path.join(PROJECT_ROOT, RESULTS_DIR_BASE)
SUMMARY_IMAGE_SAVE_PATH = os.path.join(RESULTS_DIR_PATH, SUMMARY_IMAGE_FILENAME)


def load_parameter_sets(json_file_path):
    if not os.path.exists(json_file_path):
        print(f"Error: Parameter configuration file '{json_file_path}' not found.")
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

def normalize_to_8bit(img_array):
    """Normalizes an image array to 8-bit (0-255) for display."""
    if img_array.dtype == np.uint8:
        return img_array
    
    if np.issubdtype(img_array.dtype, np.floating):
        img_min, img_max = np.min(img_array), np.max(img_array)
        if img_max > img_min:
            img_array = ((img_array - img_min) / (img_max - img_min) * 255.0)
        else: 
            img_array = np.zeros_like(img_array) + 128 
    elif img_array.dtype == np.uint16:
        img_actual_max = np.max(img_array)
        if img_actual_max > 0:
            img_array = (img_array / img_actual_max * 255.0)
        else: 
            img_array = np.zeros_like(img_array)
    else: 
        img_actual_max = np.max(img_array)
        if img_actual_max > 0:
            img_array = (img_array / img_actual_max * 255.0)
        else: 
            img_array = np.zeros_like(img_array)
            
    return img_array.astype(np.uint8)


def get_consistency_colors_for_mask(instance_mask, consensus_prob_map, colormap):
    num_cells = instance_mask.max()
    cell_colors_rgb = np.zeros((num_cells + 1, 3), dtype=np.uint8) 

    if num_cells == 0:
        return cell_colors_rgb 

    for i in range(1, num_cells + 1):
        cell_pixels = (instance_mask == i)
        if np.any(cell_pixels):
            avg_consistency = np.mean(consensus_prob_map[cell_pixels]) 
            rgba_color = colormap(avg_consistency) 
            cell_colors_rgb[i] = (np.array(rgba_color[:3]) * 255).astype(np.uint8)
    return cell_colors_rgb


def create_summary_image_with_consistency_coloring(parameter_sets):
    if not parameter_sets:
        print("No parameter sets to process.")
        return

    print("Loading all segmentation masks and original images...")
    all_masks_data = []
    original_images_cache = {} 
    mask_shape = None
    base_image_filename_for_consensus_display = None 

    for i, params in enumerate(parameter_sets):
        experiment_id = params.get("experiment_id", f"unknown_exp_{i}")
        image_filename = params.get("image_filename", "unknown_image.tif")
        
        if base_image_filename_for_consensus_display is None:
            base_image_filename_for_consensus_display = image_filename

        mask_filepath = os.path.join(RESULTS_DIR_PATH, experiment_id, os.path.splitext(image_filename)[0] + "_mask.tif")
        original_image_path = os.path.join(IMAGE_DIR_PATH, image_filename)

        if image_filename not in original_images_cache:
            if os.path.exists(original_image_path):
                try:
                    orig_img_cv = cv2.imread(original_image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    if orig_img_cv is None: raise ValueError("cv2.imread returned None")
                    original_images_cache[image_filename] = normalize_to_8bit(orig_img_cv)
                    print(f"Loaded and normalized original image: {image_filename}")
                except Exception as e:
                    print(f"Error loading original image {original_image_path}: {e}. Skipping experiment {experiment_id}.")
                    continue
            else:
                print(f"Original image {original_image_path} not found. Skipping experiment {experiment_id}.")
                continue
        
        if os.path.exists(mask_filepath):
            try:
                mask = io.imread(mask_filepath) 
                if mask_shape is None: mask_shape = mask.shape
                elif mask.shape != mask_shape:
                    print(f"Shape mismatch for mask {experiment_id}. Skipping.")
                    continue
                all_masks_data.append({"id": experiment_id, "mask": mask, "image_filename": image_filename, "params": params})
            except Exception as e:
                print(f"Error loading mask {mask_filepath} for {experiment_id}: {e}")
        else:
            print(f"Mask file not found for {experiment_id} at {mask_filepath}. Skipping.")

    if not all_masks_data:
        print("No valid segmentation masks found.")
        return
    print(f"Loaded {len(all_masks_data)} segmentation masks for processing.")

    print("Generating consensus probability map...")
    consensus_accumulator = np.zeros(mask_shape, dtype=np.float32)
    for data in all_masks_data:
        consensus_accumulator += (data["mask"] > 0).astype(np.float32)
    consensus_probability_map = consensus_accumulator / len(all_masks_data)
    binary_consensus_for_dice = (consensus_probability_map > CONSENSUS_THRESHOLD_FOR_DICE).astype(np.uint8)
    print("Consensus probability map generated.")

    plot_data = []
    for data in all_masks_data:
        exp_id = data["id"]
        individual_binary_mask = (data["mask"] > 0).astype(np.uint8)
        dice_score = calculate_dice_coefficient(individual_binary_mask, binary_consensus_for_dice)
        plot_data.append({
            "id": exp_id,
            "mask": data["mask"],
            "image_filename": data["image_filename"],
            "dice": dice_score
        })
        print(f"Experiment {exp_id}: Dice against consensus = {dice_score:.4f}")
    
    num_plots = len(plot_data)
    cols = IMAGES_PER_ROW
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5.5)) 
    if num_plots == 0 : print("No data to plot."); plt.close(fig); return
    if num_plots == 1 and not isinstance(axes, np.ndarray): axes = np.array([axes])
    axes = axes.flatten()

    # Corrected: Use matplotlib.colormaps instead of plt.cm.get_cmap
    try:
        colormap = matplotlib.colormaps[COLORMAP_NAME]
    except AttributeError: # Fallback for older matplotlib
        colormap = plt.cm.get_cmap(COLORMAP_NAME)


    for i, data in enumerate(plot_data):
        ax = axes[i]
        original_img_8bit = original_images_cache[data["image_filename"]]
        
        img_display_for_overlay = original_img_8bit.copy()
        if img_display_for_overlay.ndim == 3 and img_display_for_overlay.shape[-1] == 3: 
            img_display_for_overlay = cv2.cvtColor(img_display_for_overlay, cv2.COLOR_BGR2RGB)
        elif img_display_for_overlay.ndim == 2: 
            pass 
        else: 
            print(f"Warning: Unexpected image format for overlay for {data['id']}. Shape: {img_display_for_overlay.shape}. Attempting grayscale.")
            if img_display_for_overlay.ndim > 2: img_display_for_overlay = img_display_for_overlay[:,:,0]
            if img_display_for_overlay.dtype != np.uint8 : img_display_for_overlay = normalize_to_8bit(img_display_for_overlay)

        cell_specific_colors = get_consistency_colors_for_mask(data["mask"], consensus_probability_map, colormap)
        
        overlay_img_rgb = plot.mask_overlay(img_display_for_overlay, data["mask"], colors=cell_specific_colors) # 'plot' is now defined
        
        ax.imshow(overlay_img_rgb)
        ax.set_title(f"{data['id']} Dice: {data['dice']:.3f}", fontsize=9)
        ax.axis('off')

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=2.5, h_pad=2.5, w_pad=1.5) 
    
    if not os.path.exists(RESULTS_DIR_PATH):
        try: os.makedirs(RESULTS_DIR_PATH)
        except OSError as e: print(f"Error creating base results dir for summary: {RESULTS_DIR_PATH}: {e}"); return

    try:
        plt.savefig(SUMMARY_IMAGE_SAVE_PATH, dpi=120) 
        print(f"Summary image with consistency colors saved to: {SUMMARY_IMAGE_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving summary image: {e}")
    
    plt.close(fig)


if __name__ == "__main__":
    print("Creating summary image with consistency-colored overlays and Dice scores...")
    print(f"Looking for parameter sets file at: {PARAMETER_SETS_FILE_PATH}")
    print(f"Base results directory is: {RESULTS_DIR_PATH}")

    loaded_params = load_parameter_sets(PARAMETER_SETS_FILE_PATH)
    if loaded_params:
        create_summary_image_with_consistency_coloring(loaded_params)
    else:
        print("Could not load parameter sets. Exiting summary creation.")

