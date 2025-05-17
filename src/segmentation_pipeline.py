import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Must be before other imports that might use OpenMP
import cv2
import numpy as np
from cellpose import models, io

# --- Configuration ---
USE_GPU = False
FORCE_GRAYSCALE = True # <--- ADDED: True to force [0,0] channels, False to let Cellpose decide
IMAGE_DIR = "images"
RESULTS_DIR = "results"
DEFAULT_IMAGE_NAME = "test_image.tif"

def segment_image(image_path, output_dir, use_gpu_flag, force_grayscale_flag):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None

    print(f"Loading image: {image_path}")
    try:
        img = io.imread(image_path)
    except Exception as e:
        print(f"Error loading image with cellpose.io.imread: {e}")
        return None, None

    print(f"Initializing Cellpose model...")
    try:
        model = models.CellposeModel(gpu=use_gpu_flag)
    except AttributeError:
        try:
            print("AttributeError with CellposeModel, trying models.Cellpose...")
            model = models.Cellpose(gpu=use_gpu_flag, model_type='cyto') # model_type for older fallback
        except Exception as e_fallback:
            print(f"Error initializing Cellpose model (Cellpose or CellposeModel): {e_fallback}")
            return None, None
    except Exception as e:
        print(f"Error initializing Cellpose model: {e}")
        return None, None

    print("Running Cellpose segmentation...")
    try:
        eval_args = {
            "diameter": None,
            "flow_threshold": None,
            "cellprob_threshold": 0.0
        }
        if force_grayscale_flag:
            print("Forcing grayscale processing (channels=[0,0]).")
            eval_args["channels"] = [0,0]
        else:
            print("Using Cellpose default channel processing.")
            # Do not pass 'channels' argument to use Cellpose 4.x defaults

        masks, flows, styles = model.eval(img, **eval_args)
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
                f.write(f"Cell_{i+1},{x_coord:.2f},{y_coord:.2f}") # Ensured newline
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
        print(f"Starting segmentation process... (GPU Enabled: {USE_GPU}, Force Grayscale: {FORCE_GRAYSCALE})")
        masks, coords = segment_image(image_file_path, RESULTS_DIR, USE_GPU, FORCE_GRAYSCALE)
        if masks is not None and coords is not None:
            print("Processing complete.")
        else:
            print("Processing encountered errors.")
