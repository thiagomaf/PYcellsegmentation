import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Must be before other imports that might use OpenMP
import cv2
import numpy as np
from cellpose import models, io, plot
import json # Added for JSON output

# --- Configuration ---
USE_GPU = False
FORCE_GRAYSCALE = True
CELLPROB_THRESHOLD = 0.0

IMAGE_DIR = "images"
RESULTS_DIR = "results"
DEFAULT_IMAGE_NAME = "test_image.tif"

def segment_image(image_path, output_dir, use_gpu_flag, force_grayscale_flag, cellprob_thresh):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None, None

    print(f"Loading image: {image_path}")
    original_img_for_overlay = None
    try:
        img = io.imread(image_path) # This will be used by Cellpose
        original_img_for_overlay = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # Load with OpenCV for overlay
        print(f"Image loaded. Shape: {img.shape}, Data type: {img.dtype}")
        if original_img_for_overlay is None:
            print(f"Warning: OpenCV could not load {image_path} for overlay. Overlay might use Cellpose's internal image representation.")
            original_img_for_overlay = img # Fallback
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None

    print(f"Initializing Cellpose model...")
    model = None
    try:
        model = models.CellposeModel(gpu=use_gpu_flag)
    except AttributeError:
        try:
            print("AttributeError with CellposeModel, trying models.Cellpose...")
            model = models.Cellpose(gpu=use_gpu_flag, model_type='cyto')
        except Exception as e_fallback:
            print(f"Error initializing Cellpose model (Cellpose or CellposeModel): {e_fallback}")
            return None, None, None
    except Exception as e:
        print(f"Error initializing Cellpose model: {e}")
        return None, None, None

    if model is None:
        print("Model initialization failed.")
        return None, None, None

    print("Running Cellpose segmentation...")
    try:
        eval_args = {
            "diameter": None,
            "flow_threshold": None,
            "cellprob_threshold": cellprob_thresh
        }
        if force_grayscale_flag:
            print(f"Forcing grayscale processing (channels=[0,0]), cellprob_threshold={cellprob_thresh}.")
            eval_args["channels"] = [0,0]
        else:
            print(f"Using Cellpose default channel processing, cellprob_threshold={cellprob_thresh}.")

        masks, flows, styles = model.eval(img, **eval_args)
        print(f"Segmentation complete. Masks shape: {masks.shape}, Unique mask values: {np.unique(masks)[:20]} (showing first 20 unique values)")
    except Exception as e:
        print(f"Error during Cellpose model evaluation: {e}")
        return None, None, None

    # Save segmentation mask (integer labels)
    mask_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_mask.tif")
    try:
        io.imsave(mask_filename, masks)
        print(f"Segmentation mask saved to: {mask_filename} (Note: May appear black in basic viewers, open with ImageJ/Fiji or similar)")
    except Exception as e:
        print(f"Error saving mask: {e}")

    # Create and save overlay image
    overlay_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_overlay.png")
    try:
        # Ensure image for overlay is suitable (e.g., 2D grayscale or 3D RGB)
        # Cellpose plot.mask_overlay expects img to be 2D or 3D (RGB)
        img_display = img
        if img.ndim > 2 and img.shape[-1] > 3 and force_grayscale_flag : # If original image was multi-channel > 3 and we forced grayscale
             img_display = img[:,:,0] # Use the first channel for display
        elif img.ndim == 2: # Grayscale
            pass # Already 2D
        elif img.ndim > 2 and img.shape[-1] == 3: # RGB
            pass # Already RGB
        # Add other conditions if needed based on typical image formats

        overlay_img = plot.mask_overlay(img_display, masks, colors=None) # colors=None uses default random colors
        cv2.imwrite(overlay_filename, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)) # OpenCV expects BGR
        print(f"Overlay image saved to: {overlay_filename}")
    except Exception as e:
        print(f"Error creating or saving overlay image: {e}")


    num_cells = masks.max()
    print(f"Found {num_cells} cells.")
    coordinates_list = []
    if num_cells > 0:
        for i in range(1, num_cells + 1):
            cell_mask_pixels = (masks == i)
            if np.any(cell_mask_pixels):
                y, x = np.where(cell_mask_pixels)
                centroid_y, centroid_x = np.mean(y), np.mean(x)
                coordinates_list.append({
                    "cell_id": int(i),
                    "centroid_x": float(f"{centroid_x:.2f}"),
                    "centroid_y": float(f"{centroid_y:.2f}")
                })
    
    # Save coordinates to a JSON file
    coord_json_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_coords.json")
    try:
        with open(coord_json_filename, 'w') as f_json:
            json.dump(coordinates_list, f_json, indent=4)
        print(f"Cell coordinates saved to: {coord_json_filename}")
    except Exception as e:
        print(f"Error saving coordinates JSON: {e}")

    return masks, coordinates_list, overlay_filename # Return more info

if __name__ == "__main__":
    if not os.path.exists(IMAGE_DIR): os.makedirs(IMAGE_DIR)
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

    image_file_path = os.path.join(IMAGE_DIR, DEFAULT_IMAGE_NAME)

    if not os.path.exists(image_file_path):
        print(f"Default image '{DEFAULT_IMAGE_NAME}' not found in '{IMAGE_DIR}'.")
    else:
        print(f"Starting segmentation process... (GPU: {USE_GPU}, Grayscale: {FORCE_GRAYSCALE}, Cellprob: {CELLPROB_THRESHOLD})")
        masks, coords, overlay_file = segment_image(image_file_path, RESULTS_DIR, USE_GPU, FORCE_GRAYSCALE, CELLPROB_THRESHOLD)
        
        if masks is not None and coords is not None:
            if masks.max() == 0:
                print("Processing complete, but NO CELLS WERE SEGMENTED (mask is empty).")
                # ... (previous suggestions remain valid) ...
            else:
                print(f"Processing complete. {masks.max()} cells found.")
                if overlay_file:
                    print(f"Visual feedback saved to {overlay_file}")
        else:
            print("Processing encountered errors.")
