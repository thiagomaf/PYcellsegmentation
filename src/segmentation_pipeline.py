import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Must be before other imports that might use OpenMP
import cv2 # Will be imported if opencv-python is installed
import numpy as np
from cellpose import models, io

# Define input and output directories
IMAGE_DIR = "images"
RESULTS_DIR = "results"
# Define a default image name for now, we can make this dynamic later
DEFAULT_IMAGE_NAME = "test_image.tif" # Make sure you have an image with this name in the 'images' folder

def segment_image(image_path, output_dir):
    """
    Segments cells in an image using Cellpose and saves the mask.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the segmentation mask.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None

    print(f"Loading image: {image_path}")
    try:
        img = io.imread(image_path)
    except Exception as e:
        print(f"Error loading image with cellpose.io.imread: {e}")
        return None, None

    # Initialize Cellpose model
    # model_type='cyto' or model_type='nuclei'
    # 'cyto' is for cytoplasm segmentation, 'nuclei' for nuclei
    # You can also specify a pre-trained model path
    print("Initializing Cellpose model (cyto)...")
    try:
        # *** Changed models.Cellpose to models.CellposeModel ***
        model = models.CellposeModel(gpu=False, model_type='cyto') # Set gpu=True if you have a compatible GPU
    except AttributeError:
        # Fallback for older versions or different naming conventions if CellposeModel is not found
        try:
            print("AttributeError with CellposeModel, trying models.Cellpose...")
            model = models.Cellpose(gpu=False, model_type='cyto')
        except Exception as e_fallback:
            print(f"Error initializing Cellpose model (Cellpose or CellposeModel): {e_fallback}")
            return None, None
    except Exception as e:
        print(f"Error initializing Cellpose model: {e}")
        return None, None


    # Run segmentation
    # channels: [0,0] for grayscale, [1,2] for R=cytoplasm, G=nucleus, [2,1] for R=nucleus, G=cytoplasm
    # You might need to adjust channels based on your image type (e.g., single channel grayscale or multi-channel)
    # For a typical grayscale TIFF, channels=[0,0] should work.
    # If your TIFF has a specific channel for cell signal, adjust accordingly.
    print("Running Cellpose segmentation...")
    try:
        masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0,0], flow_threshold=None, cellprob_threshold=0.0)
        # diameter=None lets Cellpose estimate the diameter.
        # flow_threshold and cellprob_threshold can be tuned for sensitivity.
    except Exception as e:
        print(f"Error during Cellpose model evaluation: {e}")
        return None, None

    # Save segmentation mask
    mask_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_mask.tif")
    try:
        io.imsave(mask_filename, masks)
        print(f"Segmentation mask saved to: {mask_filename}")
    except Exception as e:
        print(f"Error saving mask: {e}")

    # Extract and print coordinates (centroids) of segmented cells
    num_cells = masks.max()
    print(f"Found {num_cells} cells.")
    coordinates = []
    if num_cells > 0:
        for i in range(1, num_cells + 1):
            cell_mask_pixels = (masks == i)
            if np.any(cell_mask_pixels):
                # Calculate centroid
                y, x = np.where(cell_mask_pixels)
                centroid_y, centroid_x = np.mean(y), np.mean(x)
                coordinates.append((centroid_x, centroid_y))
                # print(f"Cell {i}: Centroid (X, Y) = ({centroid_x:.2f}, {centroid_y:.2f})")
    
    # Save coordinates to a file
    coord_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_coords.txt")
    try:
        with open(coord_filename, 'w') as f:
            for i, (x_coord, y_coord) in enumerate(coordinates):
                # *** Corrected f-string with newline ***
                f.write(f"Cell_{i+1},{x_coord:.2f},{y_coord:.2f}")
        print(f"Cell coordinates saved to: {coord_filename}")
    except Exception as e:
        print(f"Error saving coordinates: {e}")

    return masks, coordinates

if __name__ == "__main__":
    # Create directories if they don't exist
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"Created directory: {IMAGE_DIR}. Please add your TIFF images here.")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created directory: {RESULTS_DIR}")

    # Construct the full path to the image
    image_file_path = os.path.join(IMAGE_DIR, DEFAULT_IMAGE_NAME)

    # Check if the default image exists
    if not os.path.exists(image_file_path):
        print(f"Default image '{DEFAULT_IMAGE_NAME}' not found in '{IMAGE_DIR}'.")
        print(f"Please add a TIFF image (e.g., {DEFAULT_IMAGE_NAME}) to the '{IMAGE_DIR}' folder,")
        print("or update the DEFAULT_IMAGE_NAME variable in the script.")
    else:
        print("Starting segmentation process...")
        masks, coords = segment_image(image_file_path, RESULTS_DIR)
        if masks is not None and coords is not None:
            print("Processing complete.")
        else:
            print("Processing encountered errors.")
