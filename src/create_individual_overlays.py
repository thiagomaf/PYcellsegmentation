#!/usr/bin/env python3
"""
Generates individual overlay images for each image-model combination defined
in a processing configuration file.

Each output image shows the original image with the corresponding segmentation
mask overlaid. Each segmented cell is filled with a unique, semi-transparent
random color, and a scale bar is added for physical reference.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path

from src.file_paths import RESCALED_IMAGE_CACHE_DIR

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

def load_image(image_path):
    """Loads an image, preferring tifffile."""
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found: {image_path}")
        return None
    
    if HAS_TIFFFILE:
        try:
            return tifffile.imread(image_path)
        except Exception:
            pass
    
    if HAS_OPENCV:
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        except Exception as e:
            print(f"Warning: Failed to load {image_path} with OpenCV: {e}")

    print(f"Error: Could not load image {image_path} with available libraries.")
    return None

def add_scale_bar(ax, image_width_pixels, mpp, position='bottom-right', length_microns=50):
    """
    Adds a scale bar to a matplotlib axes object.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to add the scale bar to.
        image_width_pixels (int): The width of the image in pixels.
        mpp (float): Microns per pixel.
        position (str): Position for the scale bar ('bottom-right' or 'bottom-left').
        length_microns (int): The desired length of the scale bar in microns.
    """
    if mpp is None or mpp == 0:
        print("Warning: Cannot add scale bar. Microns-per-pixel (mpp) is not defined.")
        return

    length_pixels = length_microns / mpp
    
    # Padding from the edge
    pad = 0.05 * image_width_pixels
    
    if position == 'bottom-right':
        x0 = image_width_pixels - pad - length_pixels
        x1 = image_width_pixels - pad
    else: # bottom-left
        x0 = pad
        x1 = pad + length_pixels
        
    y = image_width_pixels * 0.95 # Positioned at 95% of the height
    
    ax.plot([x0, x1], [y, y], color='white', linewidth=4)
    ax.text((x0 + x1) / 2, y - image_width_pixels * 0.01, f'{length_microns} µm', 
            color='white', ha='center', va='bottom', fontsize=10, fontweight='bold')

def find_background_image_path(image_config, image_id):
    """
    Determines the correct background image to use, prioritizing a cached,
    rescaled image if one was created during segmentation.
    """
    original_path = Path(image_config['original_image_filename'])
    rescaling_config = image_config.get('segmentation_options', {}).get('rescaling_config')

    if rescaling_config and rescaling_config['scale_factor'] != 1.0:
        scale_factor = rescaling_config['scale_factor']
        scale_str = str(scale_factor).replace('.', '_')
        
        # Construct the expected cached filename
        cache_dir = Path(RESCALED_IMAGE_CACHE_DIR) / image_id
        cached_filename = f"{original_path.stem}_scaled_{scale_str}{original_path.suffix}"
        cached_image_path = cache_dir / cached_filename
        
        if cached_image_path.exists():
            print(f"  - Found cached rescaled image: {cached_image_path.resolve()}")
            return cached_image_path
        else:
            print(f"  - Warning: Rescaling was configured, but cached image not found at '{cached_image_path}'. Falling back to original.")
            return original_path
    
    return original_path

def create_colored_overlay(image, mask, color_mode="multi", alpha=0.4, force_8bit_conversion=False, brightness_factor=1.0):
    """
    Creates a single blended overlay image with randomly colored, transparent mask segments.
    This function manually blends the images for robust display.
    """
    if image is None or mask is None:
        return None
    
    # 1. Convert base image to 8-bit for consistent visualization and contrast
    if force_8bit_conversion:
        # Robust conversion for better contrast, especially for 16-bit images
        if image.dtype == np.uint16:
            p_high = np.percentile(image, 99.9)
            image_scaled = np.clip(image, 0, p_high)
            image_8bit = ((image_scaled / (p_high + 1e-6)) * 255).astype(np.uint8)
        elif image.dtype == np.uint8:
            image_8bit = image
        else: # float, etc.
            min_val, max_val = np.min(image), np.max(image)
            if max_val > min_val:
                image_8bit = (((image - min_val) / (max_val - min_val)) * 255).astype(np.uint8)
            else:
                image_8bit = image.astype(np.uint8)
    else:
        # Simple min-max scaling for all types to get to 8-bit for blending
        min_val, max_val = np.min(image), np.max(image)
        if max_val > min_val:
            image_8bit = (((image - min_val) / (max_val - min_val)) * 255).astype(np.uint8)
        else:
            image_8bit = image.astype(np.uint8)

    # Adjust brightness if requested
    if brightness_factor != 1.0:
        print(f"  - Adjusting brightness by factor: {brightness_factor}")
        # Convert to float for safe multiplication, apply factor, and clip back to [0, 255]
        bright_image_float = image_8bit.astype(np.float32) * brightness_factor
        image_8bit = np.clip(bright_image_float, 0, 255).astype(np.uint8)

    # 2. Create an RGB version of the grayscale image
    if HAS_OPENCV:
        image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = np.stack([image_8bit]*3, axis=-1)

    # Ensure mask and image have same dimensions by cropping to the smallest
    min_h = min(image_rgb.shape[0], mask.shape[0])
    min_w = min(image_rgb.shape[1], mask.shape[1])
    image_rgb = image_rgb[:min_h, :min_w]
    mask = mask[:min_h, :min_w]

    # 3. Create a colored version of the mask based on the color_mode
    colored_mask = np.zeros_like(image_rgb)
    unique_labels = np.unique(mask)[1:] # Get all non-background labels

    if color_mode == "multi":
        for label in unique_labels:
            # Generate a random color, ensuring it's reasonably bright
            color = np.random.randint(60, 256, size=3, dtype=np.uint8)
            colored_mask[mask == label] = color
    else:
        # Assume single color mode (hex string)
        try:
            # Convert hex to RGB tuple
            hex_color = color_mode.lstrip('#')
            rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            # Apply the single color to all segmented regions
            all_segments_mask = (mask > 0)
            colored_mask[all_segments_mask] = rgb_color
        except Exception as e:
            print(f"Warning: Could not parse color '{color_mode}'. Defaulting to multi-color mode. Error: {e}")
            # Fallback to multi-color if parsing fails
            for label in unique_labels:
                color = np.random.randint(60, 256, size=3, dtype=np.uint8)
                colored_mask[mask == label] = color
    
    # 4. Blend the images using cv2.addWeighted for efficiency
    # The formula is: dst = src1*alpha + src2*(1-alpha) + gamma
    # Here, src1 is the colored mask, and src2 is the original image.
    blended_image = cv2.addWeighted(colored_mask, alpha, image_rgb, 1 - alpha, 0)
    
    return blended_image


def main():
    parser = argparse.ArgumentParser(description="Create individual segmentation overlay images.")
    parser.add_argument("--config", required=True, help="Path to processing config JSON file.")
    parser.add_argument("--results_dir", default="results", help="Directory containing segmentation results.")
    parser.add_argument("--output_dir", default="results/individual_overlays", help="Directory to save overlay images.")
    parser.add_argument("--scale_bar_microns", type=int, default=50, help="Length of the scale bar in microns.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Transparency of the segmentation mask overlay (0.0 to 1.0).")
    parser.add_argument("--force_8bit", action="store_true", help="If TRUE, use robust 8-bit conversion for better contrast. Defaults to FALSE.")
    parser.add_argument("--color", type=str, default="multi", help='Color mode for segments. "multi" (default) or a hex RGB string (e.g., "#FF0000").')
    parser.add_argument("--brightness_factor", type=float, default=1.0, help="Factor to adjust background image brightness (e.g., 1.5 for brighter).")
    parser.add_argument("--zoom_factor", type=float, default=1.0, help="Factor to zoom in on a region (e.g., 2.0 for 2x zoom). Requires --zoom_x and --zoom_y.")
    parser.add_argument("--zoom_x", type=int, default=None, help="X-coordinate (in pixels) of the center of the zoom.")
    parser.add_argument("--zoom_y", type=int, default=None, help="Y-coordinate (in pixels) of the center of the zoom.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for the output image.")
    
    args = parser.parse_args()

    if not HAS_MATPLOTLIB or not HAS_OPENCV:
        print("Error: Matplotlib and OpenCV are required to create visualizations.")
        return

    # Load config file
    with open(args.config, 'r') as f:
        config = json.load(f)

    image_configs = {img['image_id']: img for img in config.get('image_configurations', []) if img.get('is_active', False)}
    param_configs = {param['param_set_id']: param for param in config.get('cellpose_parameter_configurations', []) if param.get('is_active', False)}

    if not image_configs or not param_configs:
        print("\nWarning: No active 'image_configurations' or 'cellpose_parameter_configurations' found in the config file.")
        print("Please check your config file to ensure at least one of each has 'is_active': true.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving overlays to: {args.output_dir}")

    # Process each image-model combination
    for image_id, image_config in image_configs.items():
        for param_set_id, param_config in param_configs.items():
            print(f"\nProcessing: Image '{image_id}' with Model '{param_set_id}'")

            # 1. Find the correct background image (original or rescaled)
            background_image_path = find_background_image_path(image_config, image_id)
            if not background_image_path.exists():
                print(f"  - Skipping: Background image not found at '{background_image_path}'")
                continue
            
            print(f"  - Using Background Image: {background_image_path.resolve()}")

            # 2. Construct experiment ID and find mask
            scale_factor = 1.0
            rescaling_config = image_config.get('segmentation_options', {}).get('rescaling_config')
            if rescaling_config:
                scale_factor = rescaling_config.get('scale_factor', 1.0)
            
            scale_str = f"_scaled_{str(scale_factor).replace('.', '_')}" if scale_factor != 1.0 else ""
            experiment_id = f"{image_id}_{param_set_id}{scale_str}"
            
            mask_dir = Path(args.results_dir) / experiment_id
            mask_files = list(mask_dir.glob("*_mask.tif"))
            if not mask_files:
                print(f"  - Skipping: No mask file found in '{mask_dir}'")
                continue
            
            mask_path = mask_files[0]
            print(f"  - Using Mask File:      {mask_path.resolve()}")

            # 3. Load data
            background_image = load_image(background_image_path)
            mask = load_image(mask_path)
            
            if background_image is None or mask is None:
                print("  - Skipping due to loading errors.")
                continue

            # 4. Create overlay
            blended_image = create_colored_overlay(
                background_image, mask, 
                color_mode=args.color,
                alpha=args.alpha, 
                force_8bit_conversion=args.force_8bit,
                brightness_factor=args.brightness_factor
            )
            
            if blended_image is None:
                print("  - Skipping: Failed to create blended image.")
                continue

            # 5. Apply zoom if specified
            final_image = blended_image
            zoom_active = args.zoom_factor > 1.0 and args.zoom_x is not None and args.zoom_y is not None
            
            if zoom_active:
                print(f"  - Applying {args.zoom_factor}x zoom centered at ({args.zoom_x}, {args.zoom_y})")
                h, w = blended_image.shape[:2]
                
                # Calculate the size of the crop window
                crop_h = int(h / args.zoom_factor)
                crop_w = int(w / args.zoom_factor)

                # Calculate top-left corner of the crop window, ensuring it's within bounds
                y1 = max(0, args.zoom_y - crop_h // 2)
                x1 = max(0, args.zoom_x - crop_w // 2)

                # Ensure the crop window does not exceed image dimensions
                y2 = min(h, y1 + crop_h)
                x2 = min(w, x1 + crop_w)
                
                # Adjust if we hit an edge
                if y2 == h: y1 = h - crop_h
                if x2 == w: x1 = w - crop_w

                cropped_region = blended_image[y1:y2, x1:x2]
                
                # Resize the cropped region back to the original dimensions
                final_image = cv2.resize(cropped_region, (w, h), interpolation=cv2.INTER_NEAREST)

            # 6. Generate plot
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(final_image)
            ax.axis('off')

            # 7. Add scale bar
            # MPP needs to be adjusted by the scale factor for the scale bar to be accurate
            original_mpp = image_config.get('mpp_x') 
            effective_mpp = original_mpp / scale_factor if original_mpp and scale_factor != 0 else None
            
            # Further adjust MPP for zoom factor
            final_mpp = effective_mpp / args.zoom_factor if effective_mpp and zoom_active else effective_mpp
            add_scale_bar(ax, final_image.shape[1], final_mpp, length_microns=args.scale_bar_microns)

            # 8. Save figure
            output_filename = f"{image_id}_{param_set_id}_overlay.png"
            output_path = Path(args.output_dir) / output_filename
            
            plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight', pad_inches=0, facecolor='black')
            plt.close(fig)
            print(f"  + Successfully created overlay: {output_path.name}")

    print("\nOverlay generation complete.")

if __name__ == "__main__":
    main() 