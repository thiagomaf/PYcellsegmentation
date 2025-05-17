import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Must be before other imports that might use OpenMP
import cv2
import numpy as np
from cellpose import models, io, plot
import json
import traceback # For detailed error printing

# --- Configuration ---
USE_GPU = False
FORCE_GRAYSCALE = True # True to force [0,0] channels, False to let Cellpose 4.x decide
CELLPROB_THRESHOLD = 0.0 # Adjust this threshold as needed

IMAGE_DIR = "images"
RESULTS_DIR = "results"
DEFAULT_IMAGE_NAME = "test_image.tif" # Make sure you have an image with this name

def segment_image(image_path, output_dir, use_gpu_flag, force_grayscale_flag, cellprob_thresh):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None, None

    print(f"Loading image for Cellpose: {image_path}")
    try:
        img_for_cellpose = io.imread(image_path) # Used by Cellpose for segmentation
        print(f"Image for Cellpose loaded. Shape: {img_for_cellpose.shape}, Data type: {img_for_cellpose.dtype}")
    except Exception as e:
        print(f"Error loading image with cellpose.io.imread: {e}")
        traceback.print_exc()
        return None, None, None

    print(f"Initializing Cellpose model...")
    model = None
    try:
        model = models.CellposeModel(gpu=use_gpu_flag)
    except AttributeError:
        try:
            print("AttributeError with CellposeModel, trying models.Cellpose...")
            model = models.Cellpose(gpu=use_gpu_flag, model_type='cyto') # model_type for older fallback
        except Exception as e_fallback:
            print(f"Error initializing Cellpose model (fallback): {e_fallback}")
            traceback.print_exc()
            return None, None, None
    except Exception as e:
        print(f"Error initializing Cellpose model: {e}")
        traceback.print_exc()
        return None, None, None

    if model is None:
        print("Model initialization failed.")
        return None, None, None

    print("Running Cellpose segmentation...")
    try:
        eval_args = {
            "diameter": None, # Let Cellpose estimate
            "flow_threshold": None, # Default is 0.4, can be tuned
            "cellprob_threshold": cellprob_thresh
        }
        if force_grayscale_flag:
            print(f"Forcing grayscale processing (channels=[0,0]), cellprob_threshold={cellprob_thresh}.")
            eval_args["channels"] = [0,0]
        else:
            print(f"Using Cellpose default channel processing, cellprob_threshold={cellprob_thresh}.")
        
        masks, flows, styles = model.eval(img_for_cellpose, **eval_args)
        print(f"Segmentation complete. Masks shape: {masks.shape}, Unique mask values (first 20): {np.unique(masks)[:20]}")
    except Exception as e:
        print(f"Error during Cellpose model evaluation: {e}")
        traceback.print_exc()
        return None, None, None

    # Save segmentation mask (integer labels)
    mask_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_mask.tif")
    try:
        io.imsave(mask_filename, masks.astype(np.uint16)) # Save masks as uint16
        print(f"Segmentation mask saved to: {mask_filename} (View with ImageJ/Fiji for best results)")
    except Exception as e:
        print(f"Error saving mask: {e}")
        traceback.print_exc()

    # Create and save overlay image
    overlay_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_overlay.png")
    try:
        print(f"Attempting to create overlay for: {image_path}")
        # Load image with OpenCV for display. IMREAD_ANYCOLOR | IMREAD_ANYDEPTH ensures we get data.
        img_for_display_cv = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        if img_for_display_cv is None:
            print(f"Warning: cv2.imread could not load {image_path} for overlay. Using image loaded by Cellpose (may differ).")
            img_for_display_normalized = img_for_cellpose.copy() # Fallback to image Cellpose loaded
        else:
            img_for_display_normalized = img_for_display_cv.copy()
        
        print(f"Image for overlay initial shape: {img_for_display_normalized.shape}, dtype: {img_for_display_normalized.dtype}")

        # Normalize to 8-bit (0-255) for consistent display
        if img_for_display_normalized.dtype == np.uint16:
            print("Overlay: Converting 16-bit to 8-bit.")
            img_max_val = np.iinfo(np.uint16).max # Use max possible value for uint16 for scaling
            # img_max_val = np.max(img_for_display_normalized) # Alternative: scale by actual max in image
            if img_max_val > 0: # Avoid division by zero if image is all black
                 img_for_display_normalized = (img_for_display_normalized / img_max_val * 255.0).astype(np.uint8)
            else:
                 img_for_display_normalized = img_for_display_normalized.astype(np.uint8) # All zero
        elif img_for_display_normalized.dtype not in [np.uint8, np.float32, np.float64]: # Catch other int types like int32
             print(f"Overlay: Converting {img_for_display_normalized.dtype} to 8-bit by scaling.")
             img_max_val = np.max(img_for_display_normalized)
             if img_max_val > 0:
                img_for_display_normalized = (img_for_display_normalized / img_max_val * 255.0).astype(np.uint8)
             else: # All zero or negative
                img_for_display_normalized = np.zeros_like(img_for_display_normalized, dtype=np.uint8)
        elif np.issubdtype(img_for_display_normalized.dtype, np.floating): # Handle float images (e.g., 0.0-1.0 or arbitrary range)
            print("Overlay: Converting float to 8-bit.")
            img_min, img_max = np.min(img_for_display_normalized), np.max(img_for_display_normalized)
            if img_max > img_min: # Avoid division by zero
                img_for_display_normalized = ((img_for_display_normalized - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
            else: # Flat float image
                # Create a mid-gray image if min and max are the same
                img_for_display_normalized = (np.zeros_like(img_for_display_normalized) + 128).astype(np.uint8)
        
        # img_for_display_normalized is now ideally uint8
        print(f"Image for overlay normalized. Shape: {img_for_display_normalized.shape}, dtype: {img_for_display_normalized.dtype}")

        # Prepare image for plot.mask_overlay (expects 2D grayscale or 3D RGB)
        img_to_plot_for_overlay = None
        if force_grayscale_flag:
            if img_for_display_normalized.ndim == 3 and img_for_display_normalized.shape[-1] >= 3: # BGR or BGRA
                img_to_plot_for_overlay = cv2.cvtColor(img_for_display_normalized, cv2.COLOR_BGR2GRAY)
            else: # Already grayscale or 1-channel from multi-channel
                img_to_plot_for_overlay = img_for_display_normalized
            print(f"Overlay: Forced grayscale for plot. Shape: {img_to_plot_for_overlay.shape}")
        else: # Not forcing grayscale, prefer color if available
            if img_for_display_normalized.ndim == 2: # Is grayscale
                img_to_plot_for_overlay = cv2.cvtColor(img_for_display_normalized, cv2.COLOR_GRAY2BGR)
            elif img_for_display_normalized.ndim == 3 and img_for_display_normalized.shape[-1] == 4: # BGRA
                 img_to_plot_for_overlay = cv2.cvtColor(img_for_display_normalized, cv2.COLOR_BGRA2BGR)
            elif img_for_display_normalized.ndim == 3 and img_for_display_normalized.shape[-1] == 3: # BGR
                img_to_plot_for_overlay = img_for_display_normalized
            else: # Fallback for unknown multi-channel that is not BGR/BGRA
                print(f"Overlay: Unknown color format (shape {img_for_display_normalized.shape}). Defaulting to grayscale.")
                if img_for_display_normalized.ndim > 2 : img_to_plot_for_overlay = img_for_display_normalized[:,:,0] # First channel
                else : img_to_plot_for_overlay = img_for_display_normalized # Already 2D
                if img_to_plot_for_overlay.ndim == 3: # If first channel was still 3D (unlikely but safeguard)
                     img_to_plot_for_overlay = cv2.cvtColor(img_to_plot_for_overlay, cv2.COLOR_BGR2GRAY)

        if img_to_plot_for_overlay.ndim == 3 and img_to_plot_for_overlay.shape[-1] == 3:
             # plot.mask_overlay expects RGB, but OpenCV loads as BGR
             img_to_plot_for_overlay_rgb = cv2.cvtColor(img_to_plot_for_overlay, cv2.COLOR_BGR2RGB)
             overlay_rgb = plot.mask_overlay(img_to_plot_for_overlay_rgb, masks)
        else: # Grayscale
            overlay_rgb = plot.mask_overlay(img_to_plot_for_overlay, masks)

        # Convert final overlay (which is RGB) to BGR for OpenCV's imwrite.
        overlay_bgr_to_save = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(overlay_filename, overlay_bgr_to_save)
        print(f"Overlay image saved to: {overlay_filename}")

    except Exception as e:
        print(f"Error creating or saving overlay image: {e}")
        traceback.print_exc()

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
    
    coord_json_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_coords.json")
    try:
        with open(coord_json_filename, 'w') as f_json:
            json.dump(coordinates_list, f_json, indent=4)
        print(f"Cell coordinates saved to: {coord_json_filename}")
    except Exception as e:
        print(f"Error saving coordinates JSON: {e}")
        traceback.print_exc()

    return masks, coordinates_list, overlay_filename

if __name__ == "__main__":
    if not os.path.exists(IMAGE_DIR): os.makedirs(IMAGE_DIR)
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

    image_file_path = os.path.join(IMAGE_DIR, DEFAULT_IMAGE_NAME)

    if not os.path.exists(image_file_path):
        print(f"Default image '{DEFAULT_IMAGE_NAME}' not found in '{IMAGE_DIR}'.")
        print(f"Please add a TIFF image (e.g., {DEFAULT_IMAGE_NAME}) to the '{IMAGE_DIR}' folder,")
        print("or update the DEFAULT_IMAGE_NAME variable in the script.")
    else:
        print(f"Starting segmentation process... (GPU: {USE_GPU}, Grayscale: {FORCE_GRAYSCALE}, Cellprob: {CELLPROB_THRESHOLD})")
        masks, coords, overlay_file = segment_image(image_file_path, RESULTS_DIR, USE_GPU, FORCE_GRAYSCALE, CELLPROB_THRESHOLD)
        
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
                print(f"Processing complete. {masks.max()} cells found.")
                if overlay_file: # Check if overlay_file was successfully created
                    print(f"Visual feedback saved to {overlay_file}")
        else:
            print("Processing encountered errors.")
