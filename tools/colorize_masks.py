#!/usr/bin/env python3
"""
Colorize Segmentation Masks
===========================

This script takes one or more segmentation mask images (e.g., from Cellpose)
and generates colored versions with a transparent background. The output is a
PNG file where only the segmented regions are visible, colored according to the
specified mode.

This is useful for creating figures or overlays where the original image is
not needed, and only the colored cell shapes are desired.

How to Use
----------
1.  **Provide Input Masks**:
    -   Specify one or more mask files (TIFF format is common) as input.

2.  **Choose a Color Mode**:
    -   `--color multi`: (Default) Each individual cell/segment is assigned a
      unique, random, bright color.
    -   `--color "#RRGGBB"`: All segments are colored with the specified hex
      color (e.g., `#FF0000` for red, `#00FF00` for green).

3.  **Set Transparency (Optional)**:
    -   Use the `--alpha` flag to set the transparency of the segments, from
      0.0 (fully transparent) to 1.0 (fully opaque).

4.  **Specify Output Directory**:
    -   Use `--output_dir` to define where the colored masks will be saved.

Example Usage
-------------

**Multi-color mode:**
```bash
python tools/colorize_masks.py path/to/mask1.tif path/to/mask2.tif --output_dir colored_masks/
```

**Single-color mode (red):**
```bash
python tools/colorize_masks.py path/to/mask1.tif --output_dir colored_masks/ --color "#FF0000" --alpha 0.8
```
"""

import os
import argparse
import numpy as np
from pathlib import Path

# --- Dependency Handling ---
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False
# --- End Dependency Handling ---


def load_image(image_path):
    """Loads a mask image, preferring tifffile."""
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return None
    
    if HAS_TIFFFILE and image_path.suffix.lower() in ['.tif', '.tiff']:
        try:
            return tifffile.imread(image_path)
        except Exception as e:
            print(f"Warning: Failed to load {image_path} with tifffile, trying OpenCV. Error: {e}")

    if HAS_OPENCV:
        try:
            # IMREAD_UNCHANGED is important for masks
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img
        except Exception as e:
            print(f"Warning: Failed to load {image_path} with OpenCV: {e}")

    print(f"Error: Could not load image {image_path} with any available library.")
    return None

def hex_to_rgb(hex_color):
    """Converts a hex color string (e.g., "#FF0000") to an RGB tuple (255, 0, 0)."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Invalid hex color format. Must be #RRGGBB.")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def colorize_mask(mask, color_mode="multi", alpha=1.0):
    """
    Creates an RGBA image from a mask using efficient, vectorized operations.

    Args:
        mask (np.ndarray): The 2D integer segmentation mask.
        color_mode (str): Coloring mode. Can be "multi" for random colors or a
                          hex string (e.g., "#FF0000") for a single color.
        alpha (float): The transparency of the colored segments (0.0 to 1.0).

    Returns:
        np.ndarray: An RGBA image (height, width, 4) of type uint8.
    """
    if mask is None:
        return None
    
    # Get the maximum label value to create an appropriately sized color map
    max_label = np.max(mask)
    if max_label == 0:
        print("Warning: Mask contains no segments (all background).")
        return np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

    alpha_value = int(alpha * 255)

    if color_mode == "multi":
        # 1. Create a color map (lookup table) for all possible labels
        # Index 0 is for the background, which will remain transparent black
        color_map = np.zeros((max_label + 1, 4), dtype=np.uint8)
        
        # 2. Generate random colors for all labels from 1 to max_label
        # This is done in a single batch for speed
        colors = np.random.randint(60, 256, size=(max_label, 3), dtype=np.uint8)
        color_map[1:, :3] = colors
        
        # 3. Set the alpha for all segmented regions
        color_map[1:, 3] = alpha_value

        # 4. Apply the color map to the mask in a single, fast operation
        rgba_image = color_map[mask]

    else:
        # Single color mode (already reasonably fast)
        try:
            rgb_color = hex_to_rgb(color_mode)
            rgba_image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            
            # Create a mask for all segmented pixels
            all_segments_mask = (mask > 0)
            
            # Apply the single color and alpha to all segments
            rgba_image[all_segments_mask, 0] = rgb_color[0]
            rgba_image[all_segments_mask, 1] = rgb_color[1]
            rgba_image[all_segments_mask, 2] = rgb_color[2]
            rgba_image[all_segments_mask, 3] = alpha_value
        except ValueError as e:
            print(f"Error: Invalid color format '{color_mode}'. {e}")
            print("Defaulting to multi-color mode.")
            return colorize_mask(mask, color_mode="multi", alpha=alpha)

    return rgba_image

def save_rgba_image(filepath, image_array):
    """
    Saves an RGBA NumPy array to a file, trying faster libraries first.
    """
    filepath = Path(filepath)
    # Ensure the image is in the expected uint8 format
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)

    # 1. Try OpenCV (cv2) - often the fastest
    if HAS_OPENCV:
        try:
            # OpenCV expects BGRA for saving, so we convert RGBA to BGRA
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(str(filepath), bgr_image)
            return True, "OpenCV"
        except Exception as e:
            print(f"Warning: Failed to save with OpenCV, trying Pillow. Error: {e}")

    # 2. Try Pillow (PIL) - also very fast
    if HAS_PILLOW:
        try:
            Image.fromarray(image_array, 'RGBA').save(filepath)
            return True, "Pillow"
        except Exception as e:
            print(f"Warning: Failed to save with Pillow, trying Matplotlib. Error: {e}")
    
    # 3. Fallback to Matplotlib
    if HAS_MATPLOTLIB:
        try:
            plt.imsave(filepath, image_array)
            return True, "Matplotlib"
        except Exception as e:
            print(f"Error: Failed to save image with any available library. Reason: {e}")
            return False, "None"
            
    print("Error: No suitable image saving library (OpenCV, Pillow, Matplotlib) found.")
    return False, "None"

def main():
    parser = argparse.ArgumentParser(
        description="Colorize segmentation masks with a transparent background.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example Usage:
  # Multi-color mode for two masks
  python %(prog)s masks/mask1.tif masks/mask2.tif --output_dir colored_masks/

  # Single-color mode (red) with 80% opacity
  python %(prog)s masks/mask1.tif --output_dir colored_masks/ --color "#FF0000" --alpha 0.8
"""
    )
    parser.add_argument(
        "input_masks", 
        nargs='+', 
        help="One or more paths to input mask files (e.g., .tif)."
    )
    parser.add_argument(
        "--output_dir", 
        required=True, 
        help="Directory to save the colored mask PNG files."
    )
    parser.add_argument(
        "--color", 
        type=str, 
        default="multi", 
        help='Color mode for segments.\n- "multi" (default): Assign a unique random color to each segment.\n- Hex RGB string (e.g., "#FF0000"): Use a single color for all segments.'
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=1.0, 
        help="Transparency of the segmentation mask overlay (0.0 to 1.0). Default is 1.0 (opaque)."
    )
    args = parser.parse_args()

    if not any([HAS_OPENCV, HAS_PILLOW, HAS_MATPLOTLIB]):
        print("Error: At least one of OpenCV, Pillow, or Matplotlib is required to save images.")
        return

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving colored masks to: {output_path.resolve()}")

    for mask_path_str in args.input_masks:
        mask_path = Path(mask_path_str)
        print(f"\nProcessing: {mask_path.name}")

        # 1. Load mask
        mask_image = load_image(mask_path)
        if mask_image is None:
            print(f"  - Skipping: Failed to load mask.")
            continue

        # 2. Colorize mask
        colored_rgba_mask = colorize_mask(mask_image, args.color, args.alpha)
        if colored_rgba_mask is None:
            print(f"  - Skipping: Failed to colorize mask.")
            continue
            
        # 3. Save the result
        output_filename = f"{mask_path.stem}_colorized.png"
        save_path = output_path / output_filename
        
        success, library = save_rgba_image(save_path, colored_rgba_mask)

        if success:
            print(f"  + Successfully saved colored mask to: {save_path.name} (using {library})")
        else:
            print(f"  - Error: Failed to save image '{save_path}'.")

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
