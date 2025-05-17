import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Configuration ---
PARAMETER_SETS_FILE = "parameter_sets.json"
RESULTS_DIR_BASE = "results"
SUMMARY_IMAGE_FILENAME = "segmentation_summary_with_scores.png" # New filename
IMAGES_PER_ROW = 3
CONSENSUS_THRESHOLD = 0.5 # Pixels included in >50% of segmentations form the consensus

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMETER_SETS_FILE_PATH = os.path.join(PROJECT_ROOT, PARAMETER_SETS_FILE)
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
    """Calculates the Dice coefficient between two binary masks."""
    intersection = np.sum(mask1_binary & mask2_binary)
    sum_masks = np.sum(mask1_binary) + np.sum(mask2_binary)
    if sum_masks == 0: # Both masks are empty
        return 1.0 if intersection == 0 else 0.0 # Perfect match if both empty, else 0
    return (2.0 * intersection) / sum_masks

def create_summary_image(parameter_sets):
    if not parameter_sets:
        print("No parameter sets to process.")
        return

    print("Loading all segmentation masks for consensus calculation...")
    all_masks_data = []
    mask_shape = None

    for i, params in enumerate(parameter_sets):
        experiment_id = params.get("experiment_id", f"unknown_exp_{i}")
        image_filename_base = params.get("image_filename", "unknown_image.tif")
        mask_filename = os.path.splitext(image_filename_base)[0] + "_mask.tif"
        mask_path = os.path.join(RESULTS_DIR_PATH, experiment_id, mask_filename)

        if os.path.exists(mask_path):
            try:
                # Use cv2.imread for TIFF masks, ensuring it's loaded as is
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) 
                if mask is None:
                    print(f"Warning: cv2.imread failed to load mask {mask_path}. Skipping.")
                    continue
                if mask_shape is None:
                    mask_shape = mask.shape
                elif mask.shape != mask_shape:
                    print(f"Warning: Mask shape mismatch for {experiment_id} ({mask.shape}) vs expected ({mask_shape}). Skipping.")
                    continue
                all_masks_data.append({"id": experiment_id, "mask": mask, "params": params})
            except Exception as e:
                print(f"Error loading mask {mask_path}: {e}")
        else:
            print(f"Warning: Mask file not found for {experiment_id} at {mask_path}. Skipping.")

    if not all_masks_data:
        print("No valid segmentation masks found to create a consensus or summary.")
        return
    
    print(f"Loaded {len(all_masks_data)} segmentation masks.")

    # Create Consensus Mask
    print("Generating consensus mask...")
    consensus_accumulator = np.zeros(mask_shape, dtype=np.float32)
    valid_masks_for_consensus = 0
    for mask_data in all_masks_data:
        binary_mask = (mask_data["mask"] > 0).astype(np.float32)
        consensus_accumulator += binary_mask
        valid_masks_for_consensus += 1
    
    if valid_masks_for_consensus == 0:
        print("No masks were valid for consensus generation.")
        return

    consensus_probability_map = consensus_accumulator / valid_masks_for_consensus
    binary_consensus_mask = (consensus_probability_map > CONSENSUS_THRESHOLD).astype(np.uint8)
    print(f"Consensus mask generated using threshold > {CONSENSUS_THRESHOLD}.")

    # Calculate Dice scores for each mask against the consensus
    experiment_scores = {}
    for mask_data in all_masks_data:
        individual_binary_mask = (mask_data["mask"] > 0).astype(np.uint8)
        dice_score = calculate_dice_coefficient(individual_binary_mask, binary_consensus_mask)
        experiment_scores[mask_data["id"]] = dice_score
        print(f"Experiment {mask_data['id']}: Dice score against consensus = {dice_score:.4f}")

    # Plotting
    num_experiments_to_plot = len(all_masks_data)
    cols = IMAGES_PER_ROW
    rows = math.ceil(num_experiments_to_plot / cols) 

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    if num_experiments_to_plot == 0 : # Should be caught by earlier checks but defensive
        print("No experiments to plot.")
        if isinstance(axes, plt.Axes) : fig.delaxes(axes) # Delete single axis if it exists
        plt.close(fig)
        return
        
    if num_experiments_to_plot == 1 and not isinstance(axes, np.ndarray): # Single subplot case
        axes = np.array([axes])
    axes = axes.flatten()

    for i, mask_data in enumerate(all_masks_data):
        experiment_id = mask_data["id"]
        params = mask_data["params"]
        image_filename_base = params.get("image_filename", "unknown_image.tif")
        
        overlay_filename = os.path.splitext(image_filename_base)[0] + "_overlay.png"
        overlay_path = os.path.join(RESULTS_DIR_PATH, experiment_id, overlay_filename)
        
        score = experiment_scores.get(experiment_id, "N/A")
        title = f"{experiment_id}
Dice: {score:.3f}" if isinstance(score, float) else f"{experiment_id}
Dice: N/A"

        ax = axes[i]
        ax.set_title(title, fontsize=9)
        ax.axis('off')

        if os.path.exists(overlay_path):
            try:
                img_bgr = cv2.imread(overlay_path)
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                else:
                    ax.text(0.5, 0.5, 'Overlay Not Loaded', ha='center', va='center', transform=ax.transAxes, fontsize=8)
            except Exception as e:
                ax.text(0.5, 0.5, 'Error Loading Overlay', ha='center', va='center', transform=ax.transAxes, fontsize=8)
                print(f"Error loading or displaying overlay {overlay_path} for {experiment_id}: {e}")
        else:
            ax.text(0.5, 0.5, 'Overlay Not Found', ha='center', va='center', transform=ax.transAxes, fontsize=8)
            print(f"Warning: Overlay image not found for {experiment_id} at {overlay_path}")

    for j in range(num_experiments_to_plot, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
    
    if not os.path.exists(RESULTS_DIR_PATH):
        try: os.makedirs(RESULTS_DIR_PATH)
        except OSError as e: print(f"Error creating base results dir for summary: {RESULTS_DIR_PATH}: {e}"); return

    try:
        plt.savefig(SUMMARY_IMAGE_SAVE_PATH, dpi=150)
        print(f"Summary image with scores saved to: {SUMMARY_IMAGE_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving summary image: {e}")
    
    plt.close(fig)

if __name__ == "__main__":
    print("Creating summary image with consensus scores...")
    print(f"Looking for parameter sets file at: {PARAMETER_SETS_FILE_PATH}")
    print(f"Base results directory is: {RESULTS_DIR_PATH}")

    loaded_params = load_parameter_sets(PARAMETER_SETS_FILE_PATH)
    if loaded_params:
        create_summary_image(loaded_params)
    else:
        print("Could not load parameter sets. Exiting summary creation.")

