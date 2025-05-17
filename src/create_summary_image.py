import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Configuration ---
PARAMETER_SETS_FILE = "parameter_sets.json"
RESULTS_DIR_BASE = "results"
SUMMARY_IMAGE_FILENAME = "segmentation_summary.png"
IMAGES_PER_ROW = 3 # Adjust how many images you want per row in the summary

# Ensure this script is in the 'src' directory, or adjust paths accordingly
# Assuming script is in src, so parameter_sets.json and results are in parent dir
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMETER_SETS_FILE_PATH = os.path.join(PROJECT_ROOT, PARAMETER_SETS_FILE)
RESULTS_DIR_PATH = os.path.join(PROJECT_ROOT, RESULTS_DIR_BASE)
SUMMARY_IMAGE_SAVE_PATH = os.path.join(RESULTS_DIR_PATH, SUMMARY_IMAGE_FILENAME)


def load_parameter_sets(json_file_path):
    """Loads the parameter sets from the JSON configuration file."""
    if not os.path.exists(json_file_path):
        print(f"Error: Parameter configuration file '{json_file_path}' not found.")
        return None
    try:
        with open(json_file_path, 'r') as f:
            parameter_sets = json.load(f)
        return parameter_sets
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error reading {json_file_path}: {e}")
        return None

def create_summary_image(parameter_sets):
    """
    Creates a summary image by combining overlay images from different experiments.
    """
    if not parameter_sets:
        print("No parameter sets to process for summary image.")
        return

    num_experiments = len(parameter_sets)
    if num_experiments == 0:
        print("No experiments found in parameter sets.")
        return

    cols = IMAGES_PER_ROW
    rows = math.ceil(num_experiments / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5)) # Adjust figsize as needed
    axes = axes.flatten() # Flatten to 1D array for easy iteration, even if 1 row/col

    for i, params in enumerate(parameter_sets):
        experiment_id = params.get("experiment_id", f"unknown_exp_{i}")
        image_filename_base = params.get("image_filename", "unknown_image.tif")
        
        overlay_filename = os.path.splitext(image_filename_base)[0] + "_overlay.png"
        overlay_path = os.path.join(RESULTS_DIR_PATH, experiment_id, overlay_filename)

        ax = axes[i]
        ax.set_title(experiment_id, fontsize=10)
        ax.axis('off') # Turn off axis numbers and ticks

        if os.path.exists(overlay_path):
            try:
                # Load image using OpenCV (matplotlib loads as RGB, OpenCV as BGR)
                img_bgr = cv2.imread(overlay_path)
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                else:
                    ax.text(0.5, 0.5, 'Overlay Not Loaded', ha='center', va='center', transform=ax.transAxes)
                    print(f"Warning: Could not load overlay image: {overlay_path}")
            except Exception as e:
                ax.text(0.5, 0.5, 'Error Loading Overlay', ha='center', va='center', transform=ax.transAxes)
                print(f"Error loading or displaying overlay {overlay_path}: {e}")
        else:
            ax.text(0.5, 0.5, 'Overlay Not Found', ha='center', va='center', transform=ax.transAxes)
            print(f"Warning: Overlay image not found at {overlay_path}")

    # Hide any unused subplots
    for j in range(num_experiments, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=2.0) # Add some padding between title and plots, and between plots
    
    # Ensure the base results directory exists for the summary image
    if not os.path.exists(RESULTS_DIR_PATH):
        try:
            os.makedirs(RESULTS_DIR_PATH)
        except OSError as e:
            print(f"Error creating base results directory for summary: {RESULTS_DIR_PATH}: {e}")
            return

    try:
        plt.savefig(SUMMARY_IMAGE_SAVE_PATH)
        print(f"Summary image saved to: {SUMMARY_IMAGE_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving summary image: {e}")
    
    plt.close(fig) # Close the figure to free memory

if __name__ == "__main__":
    print("Creating summary image...")
    # Correct path construction assuming this script is in src/
    
    print(f"Looking for parameter sets file at: {PARAMETER_SETS_FILE_PATH}")
    print(f"Base results directory is: {RESULTS_DIR_PATH}")

    loaded_params = load_parameter_sets(PARAMETER_SETS_FILE_PATH)
    if loaded_params:
        create_summary_image(loaded_params)
    else:
        print("Could not load parameter sets. Exiting summary creation.")

