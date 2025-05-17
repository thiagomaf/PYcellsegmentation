import os
import argparse
import tifffile
import numpy as np
import json
import math
import traceback

def tile_image(input_image_path, output_dir, tile_size=1024, overlap=100, output_prefix="tile"):
    """
    Tiles a large 2D image into smaller, overlapping TIFF files and saves a JSON manifest.
    Returns the manifest data (dictionary) on success, None on failure.
    """
    if not os.path.exists(input_image_path):
        print(f"Error: Input image file not found: {input_image_path}")
        return None

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return None

    try:
        print(f"Loading large image for tiling: {input_image_path} ...")
        large_image = tifffile.imread(input_image_path)
        
        if large_image.ndim != 2:
            if large_image.ndim == 3: # Try to take first plane for ZYX or CYX etc.
                print(f"Warning: Input image is 3D (shape: {large_image.shape}). Using the first 2D plane (index 0 of the first axis).")
                large_image = large_image[0, :, :]
            elif large_image.ndim > 3: # Try to take first plane for ZCYX etc.
                 print(f"Warning: Input image is >3D (shape: {large_image.shape}). Using the first 2D plane from the first two axes (e.g. Z=0, C=0).")
                 large_image = large_image[0, 0, :, :] 
            else: 
                print(f"Error: Input image is not 2D (shape: {large_image.shape}). Tiling requires a 2D image.")
                return None
        
        print(f"Image loaded for tiling. Shape: {large_image.shape}, dtype: {large_image.dtype}")

        img_height, img_width = large_image.shape
        tile_manifest = {
            "original_image_path": os.path.abspath(input_image_path),
            "original_height": img_height,
            "original_width": img_width,
            "tile_size": tile_size,
            "overlap": overlap,
            "output_tile_directory": os.path.abspath(output_dir),
            "tiles": []
        }

        step_size = tile_size - overlap

        if step_size <= 0:
            print("Error: Overlap must be less than tile_size.")
            return None

        print(f"Tiling image into {tile_size}x{tile_size} tiles with {overlap}px overlap (step size: {step_size})...")

        tile_count = 0
        for r_idx, y in enumerate(range(0, img_height, step_size)):
            for c_idx, x in enumerate(range(0, img_width, step_size)):
                y_start = y
                y_end = y + tile_size
                x_start = x
                x_end = x + tile_size

                actual_y_end = min(y_end, img_height)
                actual_x_end = min(x_end, img_width)

                tile_data = large_image[y_start:actual_y_end, x_start:actual_x_end]

                if tile_data.size == 0: continue

                tile_filename = f"{output_prefix}_r{r_idx:03d}_c{c_idx:03d}.tif"
                tile_filepath = os.path.join(output_dir, tile_filename)
                
                try:
                    tifffile.imwrite(tile_filepath, tile_data)
                    tile_info = {
                        "filename": tile_filename,
                        "path": os.path.abspath(tile_filepath),
                        "x_start_in_original": x_start,
                        "y_start_in_original": y_start,
                        "width_tile": tile_data.shape[1],
                        "height_tile": tile_data.shape[0]
                    }
                    tile_manifest["tiles"].append(tile_info)
                    tile_count += 1
                except Exception as e_write:
                    print(f"Error writing tile {tile_filepath}: {e_write}")

        print(f"Generated {tile_count} tiles.")
        if tile_count == 0 and (img_width > 0 and img_height > 0) : # If image had size but no tiles (e.g. smaller than tile_size)
             print("Warning: No tiles were generated. Image might be smaller than tile_size or overlap is too large.")
             # Consider if a single tile should be made if image < tile_size

        manifest_filename = f"{output_prefix}_manifest.json"
        manifest_filepath = os.path.join(output_dir, manifest_filename)
        try:
            with open(manifest_filepath, 'w') as f_manifest:
                json.dump(tile_manifest, f_manifest, indent=4)
            print(f"Tile manifest saved to: {manifest_filepath}")
            return tile_manifest # Return the manifest data
        except Exception as e_json:
            print(f"Error saving tile manifest {manifest_filepath}: {e_json}")
            return None # Indicate failure

    except Exception as e:
        print(f"An error occurred during tiling: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile a large 2D TIFF image into smaller, overlapping tiles.")
    parser.add_argument("input_image", help="Path to the large input 2D TIFF image.")
    parser.add_argument("output_dir", help="Directory to save the tiled images and manifest file.")
    parser.add_argument("--tile_size", type=int, default=1024, 
                        help="Desired size (width and height) of each square tile (default: 1024).")
    parser.add_argument("--overlap", type=int, default=100, 
                        help="Number of pixels to overlap between adjacent tiles (default: 100).")
    parser.add_argument("--prefix", type=str, default="tile", 
                        help="Prefix for the output tiled image filenames and manifest (default: 'tile').")

    args = parser.parse_args()

    print(f"Tiling image: {args.input_image}")
    print(f"Output directory: {args.output_dir}")
    
    tile_manifest_data = tile_image(args.input_image, args.output_dir, 
                                   tile_size=args.tile_size, 
                                   overlap=args.overlap, 
                                   output_prefix=args.prefix)
    
    if tile_manifest_data:
        print("Tiling process finished successfully.")
    else:
        print("Tiling process encountered errors or produced no manifest.")

