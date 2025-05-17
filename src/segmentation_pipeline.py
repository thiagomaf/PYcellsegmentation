import os
import cv2
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
        return

    print(f"Loading image: {image_path}")
    # Load image using OpenCV, then Cellpose's io.imread
    # OpenCV is good for general image reading, Cellpose's io might have specific optimizations
    img = io.imread(image_path)

    # Initialize Cellpose model
    # model_type='cyto' or model_type='nuclei'
    # 'cyto' is for cytoplasm segmentation, 'nuclei' for nuclei
    # You can also specify a pre-trained model path
    print("Initializing Cellpose model (cyto)...")
    model = models.Cellpose(gpu=False, model_type='cyto') # Set gpu=True if you have a compatible GPU

    # Run segmentation
    # channels: [0,0] for grayscale, [1,2] for R=cytoplasm, G=nucleus, [2,1] for R=nucleus, G=cytoplasm
    # You might need to adjust channels based on your image type (e.g., single channel grayscale or multi-channel)
    # For a typical grayscale TIFF, channels=[0,0] should work.
    # If your TIFF has a specific channel for cell signal, adjust accordingly.
    print("Running Cellpose segmentation...")
    masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0,0], flow_threshold=None, cellprob_threshold=0.0)
    # diameter=None lets Cellpose estimate the diameter.
    # flow_threshold and cellprob_threshold can be tuned for sensitivity.

    # Save segmentation mask
    mask_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_mask.tif")
    io.imsave(mask_filename, masks)
    print(f"Segmentation mask saved to: {mask_filename}")

    # Extract and print coordinates (centroids) of segmented cells
    num_cells = masks.max()
    print(f"Found {num_cells} cells.")
    coordinates = []
    for i in range(1, num_cells + 1):
        cell_mask = (masks == i)
        if np.any(cell_mask):
            # Calculate centroid
            y, x = np.where(cell_mask)
            centroid_y, centroid_x = np.mean(y), np.mean(x)
            coordinates.append((centroid_x, centroid_y))
            # print(f"Cell {i}: Centroid (X, Y) = ({centroid_x:.2f}, {centroid_y:.2f})")
    
    # Save coordinates to a file (optional)
    coord_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_coords.txt")
    with open(coord_filename, 'w') as f:
        for i, (x, y) in enumerate(coordinates):
            f.write(f"Cell_{i+1},{x:.2f},{y:.2f}
") # Corrected line
    print(f"Cell coordinates saved to: {coord_filename}")

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
        print("Please add a TIFF image to the 'images' folder and update DEFAULT_IMAGE_NAME if needed, or provide an image path.")
    else:
        segment_image(image_file_path, RESULTS_DIR)
        print("Processing complete.")
