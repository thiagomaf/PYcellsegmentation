import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Must be before other imports that might use OpenMP
import cv2
import numpy as np
from cellpose import models, io

# --- Configuration ---
USE_GPU = False
FORCE_GRAYSCALE = True
CELLPROB_THRESHOLD = 0.0 # <--- ADDED: Adjust this threshold
# Try values like -1.0, -2.0, or even lower if cells are faint or not detected.
# For very confident segmentations, you might use positive values like 0.5 or 1.0.

IMAGE_DIR = "images"
RESULTS_DIR = "results"
DEFAULT_IMAGE_NAME = "test_image.tif"

def segment_image(image_path, output_dir, use_gpu_flag, force_grayscale_flag, cellprob_thresh):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None

    print(f"Loading image: {image_path}")
    try:
        img = io.imread(image_path)
        print(f"Image loaded. Shape: {img.shape}, Data type: {img.dtype}")
    except Exception as e:
        print(f"Error loading image with cellpose.io.imread: {e}")
        return None, None

    print(f"Initializing Cellpose model...")
    model = None # Initialize model to None
    try:
        model = models.CellposeModel(gpu=use_gpu_flag)
    except AttributeError:
        try:
            print("AttributeError with CellposeModel, trying models.Cellpose...")
            model = models.Cellpose(gpu=use_gpu_flag, model_type='cyto')
        except Exception as e_fallback:
            print(f"Error initializing Cellpose model (Cellpose or CellposeModel): {e_fallback}")
            return None, None
    except Exception as e:
        print(f"Error initializing Cellpose model: {e}")
        return None, None

    if model is None: # Should not happen if try-except is structured well, but as a safeguard
        print("Model initialization failed.")
        return None, None

    print("Running Cellpose segmentation...")
    try:
        eval_args = {
            "diameter": None, # Let Cellpose estimate diameter initially
            "flow_threshold": None, # Default is 0.4, can be tuned
            "cellprob_threshold": cellprob_thresh # Use the configured threshold
        }
        if force_grayscale_flag:
            print(f"Forcing grayscale processing (channels=[0,0]), cellprob_threshold={cellprob_thresh}.")
            eval_args["channels"] = [0,0]
        else:
            print(f"Using Cellpose default channel processing, cellprob_threshold={cellprob_thresh}.")
            # Do not pass 'channels' argument to use Cellpose 4.x defaults

        masks, flows, styles = model.eval(img, **eval_args)
        
        print(f"Segmentation complete. Masks shape: {masks.shape}, Unique mask values: {np.unique(masks)}")
        # In Cellpose 3.0+, model.diam_labels gives the diameter of the ROIs *after* segmentation
        # For CellposeModel, the diameter might be stored differently or as part of styles.
        # For now, we'll focus on getting non-empty masks.
        # print(f"Estimated diameter (model.diam_labels): {model.diam_labels if hasattr(model, 'diam_labels') else 'N/A'}")


    except Exception as e:
        print(f"Error during Cellpose model evaluation: {e}")
        return None, None

    mask_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_mask.tif")
    try:
        io.imsave(mask_filename, masks)
        print(f"Segmentation mask saved to: {mask_filename}")
    except Exception as e:
        print(f"Error saving mask: {e}")

    num_cells = masks.max()
    print(f"Found {num_cells} cells.")
    coordinates = []
    if num_cells > 0:
        for i in range(1, num_cells + 1):
            cell_mask_pixels = (masks == i)
            if np.any(cell_mask_pixels):
                y, x = np.where(cell_mask_pixels)
                centroid_y, centroid_x = np.mean(y), np.mean(x)
                coordinates.append((centroid_x, centroid_y))
    
    coord_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_coords.txt")
    try:
        with open(coord_filename, 'w') as f:
            for i, (x_coord, y_coord) in enumerate(coordinates):
                f.write(f"Cell_{i+1},{x_coord:.2f},{y_coord:.2f}") # *** Corrected f-string with newline ***
        print(f"Cell coordinates saved to: {coord_filename}")
    except Exception as e:
        print(f"Error saving coordinates: {e}")

    return masks, coordinates

if __name__ == "__main__":
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    image_file_path = os.path.join(IMAGE_DIR, DEFAULT_IMAGE_NAME)

    if not os.path.exists(image_file_path):
        print(f"Default image '{DEFAULT_IMAGE_NAME}' not found in '{IMAGE_DIR}'.")
    else:
        print(f"Starting segmentation process... (GPU: {USE_GPU}, Grayscale: {FORCE_GRAYSCALE}, Cellprob: {CELLPROB_THRESHOLD})")
        masks, coords = segment_image(image_file_path, RESULTS_DIR, USE_GPU, FORCE_GRAYSCALE, CELLPROB_THRESHOLD)
        if masks is not None and coords is not None:
            if masks.max() == 0:
                print("Processing complete, but NO CELLS WERE SEGMENTED (mask is empty).")
                print("Consider the following:")
                print("1. Adjust CELLPROB_THRESHOLD in the script (e.g., to -1.0, -2.0 or lower).")
                print("2. If FORCE_GRAYSCALE is True, ensure cells are in the first channel of your image.")
                print("3. Try setting FORCE_GRAYSCALE to False to let Cellpose handle multi-channel images differently.")
                print("4. Check your image for contrast and visibility of cells.")
                print("5. Manually specify 'diameter' in eval_args if default estimation seems off.")
            else:
                print("Processing complete.")
        else:
            print("Processing encountered errors.")
