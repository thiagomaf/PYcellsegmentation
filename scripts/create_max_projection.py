#!/usr/bin/env python3
"""
Create a maximum projection of an OME-TIFF image and save it in the same folder.
"""

import os
import numpy as np
import tifffile
from pathlib import Path

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not available. Scaling will be disabled.")

def create_max_projection(input_path, output_path=None, scale_factor=1.0):
    """
    Load an OME-TIFF image and create a maximum projection, optionally scaling it.
    
    Args:
        input_path: Path to the input OME-TIFF file
        output_path: Path for the output file (if None, auto-generates in same folder)
        scale_factor: Scale factor to apply after projection (default: 1.0, use 0.5 for 50% size)
    """
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return False
    
    print(f"Loading image: {input_path}")
    
    try:
        # Load the image
        image = tifffile.imread(input_path)
        print(f"  Original shape: {image.shape}, dtype: {image.dtype}")
        
        # Handle different dimensionalities
        if image.ndim == 2:
            print("  Image is already 2D, no projection needed.")
            projected = image
        elif image.ndim == 3:
            dim0, dim1, dim2 = image.shape
            # Determine if it's (Z, H, W), (H, W, C), or (C, H, W)
            if dim0 < dim1 and dim0 < dim2:
                # Likely (Z, H, W) or (C, H, W) - project along first axis
                print(f"  Detected 3D image (likely Z-stack or multi-channel), applying max projection along axis 0")
                projected = np.max(image, axis=0)
            else:
                # Likely (H, W, C) - project along channel axis
                print(f"  Detected 3D image (likely RGB), applying max projection along channel axis")
                projected = np.max(image, axis=2)
        elif image.ndim == 4:
            # Could be (Z, C, H, W), (T, Z, H, W), or (T, C, H, W)
            print(f"  Detected 4D image, applying max projection")
            # Project along first two dimensions (time/Z and channels)
            projected = np.max(np.max(image, axis=0), axis=0)
        else:
            print(f"  Warning: Unusual dimensionality ({image.ndim}D), attempting max projection along all but last two axes")
            # Project along all axes except the last two (height and width)
            for axis in range(image.ndim - 2):
                image = np.max(image, axis=0)
            projected = image
        
        print(f"  Projected shape: {projected.shape}, dtype: {projected.dtype}")
        
        # Apply scaling if requested
        if scale_factor != 1.0:
            if not HAS_OPENCV:
                print(f"  Warning: Cannot apply scale factor {scale_factor} without OpenCV. Skipping scaling.")
            else:
                original_h, original_w = projected.shape[:2]
                new_w = int(round(original_w * scale_factor))
                new_h = int(round(original_h * scale_factor))
                print(f"  Applying scale factor {scale_factor}: {original_w}x{original_h} -> {new_w}x{new_h}")
                projected = cv2.resize(projected, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"  Scaled shape: {projected.shape}, dtype: {projected.dtype}")
        
        # Generate output path if not provided
        if output_path is None:
            input_path_obj = Path(input_path)
            if scale_factor != 1.0:
                scale_str = str(scale_factor).replace('.', '_')
                output_path = input_path_obj.parent / f"{input_path_obj.stem}_max_projection_scaled_{scale_str}.tif"
            else:
                output_path = input_path_obj.parent / f"{input_path_obj.stem}_max_projection.tif"
        
        # Save the projected image
        print(f"Saving max projection to: {output_path}")
        tifffile.imwrite(str(output_path), projected)
        print(f"  Successfully saved max projection!")
        print(f"  Output dimensions: {projected.shape}, dtype: {projected.dtype}")
        
        return True
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Path to the input image
    input_image = "data/raw/images/xenium_HE/xenium/Xenium_28DPI_5A_taugat_QuPath.ome.tif"
    
    # Create max projection with 0.5 scale factor
    success = create_max_projection(input_image, scale_factor=0.5)
    
    if success:
        print("\nMax projection created successfully!")
    else:
        print("\nFailed to create max projection.")

