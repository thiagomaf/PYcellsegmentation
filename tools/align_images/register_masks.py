import argparse
import zipfile
import sys
import numpy as np
import pandas as pd
from skimage import io as skimage_io
from skimage import transform as skimage_transform

def transform_mask_robust(zip_filepath, mask_filepath, output_filepath):
    """
    Applies an affine transformation to a TIFF mask, automatically calculating
    the output image size with a safety margin to prevent clipping.

    Args:
        zip_filepath (str): Path to the ZIP file containing 'matrix.csv'.
        mask_filepath (str): Path to the input TIFF segmentation mask.
        output_filepath (str): Path to save the transformed TIFF mask.
    """
    try:
        # --- 1. Read the transformation matrix from the ZIP file ---
        with zipfile.ZipFile(zip_filepath, 'r') as zf:
            if 'matrix.csv' not in zf.namelist():
                print(f"Error: 'matrix.csv' not found inside {zip_filepath}")
                sys.exit(1)
            with zf.open('matrix.csv') as f:
                transform_matrix = pd.read_csv(f, header=None).values
        
        print(f"Successfully loaded transformation matrix from '{zip_filepath}'")

        # --- 2. Read the segmentation mask TIFF file ---
        mask_image = skimage_io.imread(mask_filepath)
        h, w = mask_image.shape[:2]
        print(f"Loaded mask image '{mask_filepath}' with dimensions {h}x{w}")

        # --- 3. Calculate the new position of the image corners ---
        corners = np.array([
            [0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]
        ])
        transformed_corners = (transform_matrix @ corners.T).T
        
        min_coords = np.min(transformed_corners[:, :2], axis=0)
        max_coords = np.max(transformed_corners[:, :2], axis=0)

        # --- 4. Calculate the new canvas size WITH a safety margin ---
        # We use floor on min and ceil on max to get the full range.
        # Coordinates are (x, y), corresponding to (width, height).
        output_w = int(np.ceil(max_coords[0]) - np.floor(min_coords[0]))
        output_h = int(np.ceil(max_coords[1]) - np.floor(min_coords[1]))
        print(f"Calculated robust canvas size: {output_h}x{output_w}")

        # --- 5. Create the full transformation for warping ---
        # This translation moves the top-left corner of the transformed
        # bounding box to the coordinate (0,0) in our new canvas.
        translation = -np.floor(min_coords)
        shift_matrix = np.array([
            [1, 0, translation[0]],
            [0, 1, translation[1]],
            [0, 0, 1]
        ])
        full_transform_matrix = shift_matrix @ transform_matrix
        
        inverse_full_transform = np.linalg.inv(full_transform_matrix)
        affine_transform = skimage_transform.AffineTransform(matrix=inverse_full_transform)

        # --- 6. Apply the transformation using 'warp' ---
        warped_mask = skimage_transform.warp(
            mask_image,
            affine_transform,
            output_shape=(output_h, output_w),
            order=0,  # Use nearest-neighbor interpolation for masks
            preserve_range=True
        )
        warped_mask = warped_mask.astype(mask_image.dtype)

        # --- 7. Save the new transformed TIFF ---
        skimage_io.imsave(output_filepath, warped_mask, check_contrast=False, compression='zlib')
        print(f"Transformation complete. Saved final mask to '{output_filepath}'")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robustly apply an affine transformation to a TIFF mask with auto-sizing.",
        epilog="Example: python transform_mask_robust.py alignment.zip my_mask.tiff aligned_mask_final.tiff"
    )
    parser.add_argument("zip_filepath", help="Path to the ZIP file with 'matrix.csv'.")
    parser.add_argument("mask_filepath", help="Path to the input TIFF segmentation mask.")
    parser.add_argument("output_filepath", help="Path for the output transformed TIFF file.")
    args = parser.parse_args()
    transform_mask_robust(args.zip_filepath, args.mask_filepath, args.output_filepath)