#!/usr/bin/env python3
"""
Create Model Comparison Overlays
================================

This script generates a 2x2 visual comparison of segmentation results from different
Cellpose models for a given set of images. The output is a single PNG and PDF figure
for each input image, showing the original image alongside overlays of masks from
up to three different models (e.g., 'nuclei', 'cyto2', 'cyto3').

The script is configured through a JSON file that specifies the images to process
and the parameters for each model. It automatically finds the corresponding mask
files in the results directory based on a naming convention.

How to Use
----------
1.  **Configuration**:
    -   Create a JSON configuration file (e.g., `config/processing_config_comparison.json`).
    -   In the `image_configurations` section, define each image with its `image_id`
        and `original_image_filename`. Set `is_active` to `true` for images you
        want to process.
    -   In the `cellpose_parameter_configurations` section, define parameter sets
        for each model. The `MODEL_CHOICE` should be one of 'nuclei', 'cyto2', etc.

2.  **Run Segmentation First**:
    -   Ensure that you have already run the main segmentation pipeline to generate the mask files.
    -   Masks should be located in a directory structure like:
        `<results_dir>/<image_id>_<param_set_id>_scaled_<scale>/..._mask.tif`

3.  **Execute the Script**:
    -   Run the script from the command line.

    Example:
    ```bash
    python src/create_model_comparison_overlay.py \\
        --config config/processing_config_comparison5.json \\
        --results_dir results \\
        --output_dir results/model_comparisons
    ```

4.  **Output**:
    -   The script will generate `model_comparison_<image_id>.png` and `.pdf` files
        in the specified output directory.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Optional imports with fallbacks
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Cannot create visualizations.")

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    from skimage import measure, segmentation
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

def load_image(image_path):
    """Load an image using available libraries."""
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found: {image_path}")
        return None
    
    if HAS_TIFFFILE:
        try:
            return tifffile.imread(image_path)
        except Exception as e:
            print(f"Warning: Could not load {image_path} with tifffile: {e}")
    
    if HAS_OPENCV:
        try:
            return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f"Warning: Could not load {image_path} with OpenCV: {e}")
    
    try:
        from PIL import Image
        return np.array(Image.open(image_path))
    except ImportError:
        print(f"Error: No image loading library available for {image_path}")
        return None

def create_mask_overlay(image, mask, alpha=0.3, colormap='viridis'):
    """Create an overlay of segmentation mask on the original image."""
    if image is None or mask is None:
        return None
    
    # Ensure image is 2D and normalize to 0-1
    if len(image.shape) > 2:
        image = image[:, :, 0] if image.shape[2] > 1 else image.squeeze()
    
    # Normalize image to 0-1 range
    image_norm = (image - image.min()) / (image.max() - image.min()) if image.max() > image.min() else image
    
    # Create RGB version of grayscale image
    image_rgb = np.stack([image_norm, image_norm, image_norm], axis=2)
    
    # Create mask overlay
    if HAS_SKIMAGE:
        # Use skimage to create nice boundaries
        boundaries = segmentation.find_boundaries(mask, mode='outer')
        
        # Ensure boundaries and image_rgb have matching dimensions
        min_height = min(boundaries.shape[0], image_rgb.shape[0])
        min_width = min(boundaries.shape[1], image_rgb.shape[1])
        
        # Crop both arrays to matching dimensions
        boundaries = boundaries[:min_height, :min_width]
        image_rgb = image_rgb[:min_height, :min_width]
        
        overlay = np.zeros_like(image_rgb)
        overlay[boundaries] = [1, 0, 0]  # Red boundaries
        
        # Blend with original image
        result = image_rgb.copy()
        mask_pixels = boundaries > 0
        result[mask_pixels] = alpha * overlay[mask_pixels] + (1 - alpha) * image_rgb[mask_pixels]
        
    else:
        # Simple overlay without skimage - also handle dimension mismatches
        min_height = min(mask.shape[0], image_rgb.shape[0])
        min_width = min(mask.shape[1], image_rgb.shape[1])
        
        # Crop both arrays to matching dimensions
        mask = mask[:min_height, :min_width]
        image_rgb = image_rgb[:min_height, :min_width]
        
        result = image_rgb.copy()
        mask_pixels = mask > 0
        result[mask_pixels, 0] = alpha + (1 - alpha) * image_rgb[mask_pixels, 0]  # Add red channel
    
    return result

def create_comparison_figure(image_path, mask_paths, model_names, output_path, image_id, scale_factor=1.0):
    """
    Create a 2x2 comparison figure showing original image and 3 model overlays.
    
    Args:
        image_path (str): Path to original image
        mask_paths (dict): Dictionary mapping model names to mask file paths
        model_names (list): List of model names to compare
        output_path (str): Output file path for the figure
        image_id (str): Image identifier for the title
        scale_factor (float): Scale factor used for segmentation
    """
    if not HAS_MATPLOTLIB:
        print("Cannot create visualizations without matplotlib")
        return False
    
    # Load original image
    original_image = load_image(image_path)
    if original_image is None:
        print(f"Could not load original image: {image_path}")
        return False
    
    # Scale the original image to match the mask dimensions if needed
    if scale_factor != 1.0:
        if HAS_OPENCV:
            new_height = int(original_image.shape[0] * scale_factor)
            new_width = int(original_image.shape[1] * scale_factor)
            original_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            print(f"Warning: Cannot scale image without OpenCV. Scale factor {scale_factor} ignored.")
            print("Mask and image dimensions may not match.")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Cellpose Model Comparison - Image {image_id}', fontsize=16, fontweight='bold')
    
    # Top-left: Original image
    ax = axes[0, 0]
    ax.imshow(original_image, cmap='gray')
    ax.set_title('Original Image', fontweight='bold')
    ax.axis('off')
    
    # Load masks and create overlays for each model
    positions = [(0, 1), (1, 0), (1, 1)]  # Top-right, bottom-left, bottom-right
    colors = ['red', 'blue', 'green']
    
    for i, model_name in enumerate(model_names):
        if i >= len(positions):
            break
            
        row, col = positions[i]
        ax = axes[row, col]
        
        # Load mask
        mask_path = mask_paths.get(model_name)
        if mask_path and os.path.exists(mask_path):
            mask = load_image(mask_path)
            if mask is not None:
                # Create overlay
                overlay = create_mask_overlay(original_image, mask, alpha=0.4)
                if overlay is not None:
                    ax.imshow(overlay)
                    
                    # Count cells
                    num_cells = len(np.unique(mask)) - 1  # Exclude background
                    ax.set_title(f'{model_name.capitalize()} Model\n({num_cells} cells)', fontweight='bold', color=colors[i])
                else:
                    ax.imshow(original_image, cmap='gray')
                    ax.set_title(f'{model_name.capitalize()} Model\n(Overlay failed)', fontweight='bold', color='red')
            else:
                ax.imshow(original_image, cmap='gray')
                ax.set_title(f'{model_name.capitalize()} Model\n(Mask load failed)', fontweight='bold', color='red')
        else:
            ax.imshow(original_image, cmap='gray') 
            ax.set_title(f'{model_name.capitalize()} Model\n(No mask found)', fontweight='bold', color='gray')
        
        ax.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Comparison figure saved: {output_path}")
    return True

def find_mask_file(results_dir, experiment_id):
    """Find the mask file in a result directory."""
    result_dir = os.path.join(results_dir, experiment_id)
    if not os.path.exists(result_dir):
        return None
    
    # Look for mask files
    mask_files = [f for f in os.listdir(result_dir) if f.endswith('_mask.tif')]
    if mask_files:
        return os.path.join(result_dir, mask_files[0])
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Create visual comparisons of Cellpose models")
    parser.add_argument("--config", default="config/processing_config_comparison5.json",
                       help="Path to processing config JSON file")
    parser.add_argument("--results_dir", default="results", 
                       help="Directory containing segmentation results")
    parser.add_argument("--output_dir", default="results/model_comparisons",
                       help="Directory to save comparison figures")
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for creating visualizations")
        return 1
    
    print("Creating Cellpose model comparison visualizations...")
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Extract configurations
    image_configs = {img['image_id']: img for img in config.get('image_configurations', [])}
    param_configs = {param['param_set_id']: param for param in config.get('cellpose_parameter_configurations', [])}
    
    print(f"Found {len(image_configs)} image configs: {list(image_configs.keys())}")
    print(f"Found {len(param_configs)} parameter configs: {list(param_configs.keys())}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Expected models based on common configurations
    expected_models = ['nuclei', 'cyto2', 'cyto3']
    
    # Process each image
    for image_id, image_config in image_configs.items():
        if not image_config.get('is_active', True):
            continue
        
        print(f"\nProcessing image: {image_id}")
        
        # Get original image path
        original_image_path = image_config['original_image_filename']
        if not os.path.exists(original_image_path):
            print(f"  Warning: Original image not found: {original_image_path}")
            continue
        
        # Get scale factor for experiment ID construction
        scale_factor = 1.0
        seg_options = image_config.get('segmentation_options', {})
        rescaling_config = seg_options.get('rescaling_config')
        if rescaling_config and 'scale_factor' in rescaling_config:
            scale_factor = rescaling_config['scale_factor']
        
        def format_scale_factor_for_path(scale_factor):
            if scale_factor is not None and scale_factor != 1.0:
                return f"_scaled_{str(scale_factor).replace('.', '_')}"
            return ""
        
        # Find mask files for each model
        mask_paths = {}
        model_names = []
        
        for param_set_id, param_config in param_configs.items():
            if not param_config.get('is_active', True):
                continue
            
            model = param_config['cellpose_parameters']['MODEL_CHOICE']
            if model in expected_models:
                # Construct experiment ID
                experiment_id = f"{image_id}_{param_set_id}{format_scale_factor_for_path(scale_factor)}"
                
                # Find mask file
                mask_path = find_mask_file(args.results_dir, experiment_id)
                if mask_path:
                    mask_paths[model] = mask_path
                    model_names.append(model)
                    print(f"  Found {model} mask: {mask_path}")
                else:
                    print(f"  Warning: No mask found for {model} (experiment_id: {experiment_id})")
        
                 # Create comparison figure if we have at least one mask
        if mask_paths:
            output_filename = f"model_comparison_{image_id}.png"
            output_path = os.path.join(args.output_dir, output_filename)
            
            success = create_comparison_figure(
                original_image_path,
                mask_paths,
                expected_models,  # Use consistent order
                output_path,
                image_id,
                scale_factor  # Pass the scale factor
            )
            
            if success:
                print(f"  Created comparison figure: {output_path}")
            else:
                print(f"  Failed to create comparison figure for {image_id}")
        else:
            print(f"  No masks found for image {image_id}, skipping...")
    
    print("\nModel comparison visualization complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 