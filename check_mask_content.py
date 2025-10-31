import tifffile
import numpy as np
import argparse
import os

def check_mask_file(mask_path):
    """
    Loads a TIFF mask file and prints diagnostic information.
    """
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found at '{mask_path}'")
        return

    try:
        print(f"Checking mask: {mask_path}")
        
        # Load the mask image
        mask = tifffile.imread(mask_path)
        
        # Print diagnostic information
        print(f"  - Shape: {mask.shape}")
        print(f"  - Data Type (dtype): {mask.dtype}")
        print(f"  - Minimum Value: {np.min(mask)}")
        print(f"  - Maximum Value: {np.max(mask)}")
        
        unique_values = np.unique(mask)
        num_segments = len(unique_values) - 1 if 0 in unique_values else len(unique_values)
        
        print(f"  - Number of unique values: {len(unique_values)}")
        print(f"  - Number of segmented objects (excluding background): {num_segments}")

        if num_segments == 0:
            print("\n---")
            print("Warning: This mask file contains NO segmented objects.")
            print("This is why the overlays appear empty.")
            print("---\n")
        else:
            print("\n---")
            print("Success: This mask file contains segmented objects.")
            print("---\n")

    except Exception as e:
        print(f"An error occurred while reading the mask file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Check a Cellpose mask TIFF file for issues.")
    parser.add_argument("mask_file", help="Path to the mask.tif file to check.")
    args = parser.parse_args()
    
    check_mask_file(args.mask_file)

if __name__ == "__main__":
    main()