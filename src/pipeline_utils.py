# src/pipeline_utils.py
import os
import re
import cv2
import tifffile
import numpy as np
import traceback

RESCALED_IMAGE_CACHE_DIR = os.path.join("images", "rescaled_cache") # Relative to project root

def clean_filename_for_dir(filename):
    name_without_ext = os.path.splitext(filename)[0]
    cleaned_name = re.sub(r'[^\w\.-]', '_', name_without_ext)
    return cleaned_name

def get_cv2_interpolation_method(method_str="INTER_AREA"):
    methods = {
        "INTER_NEAREST": cv2.INTER_NEAREST,
        "INTER_LINEAR": cv2.INTER_LINEAR,
        "INTER_AREA": cv2.INTER_AREA,
        "INTER_CUBIC": cv2.INTER_CUBIC,
        "INTER_LANCZOS4": cv2.INTER_LANCZOS4
    }
    return methods.get(method_str.upper(), cv2.INTER_AREA)

def rescale_image_and_save(original_image_path, image_id_for_cache, rescaling_config):
    """
    Loads an image, rescales it, and saves it to a cache directory.
    Returns the path to the rescaled image and the actual scale factor used.
    Assumes RESCALED_IMAGE_CACHE_DIR is accessible from project root.
    """
    if not os.path.exists(original_image_path):
        print(f"  Error: Original image for rescaling not found: {original_image_path}")
        return original_image_path, 1.0

    if not rescaling_config or "scale_factor" not in rescaling_config:
        return original_image_path, 1.0

    scale_factor = rescaling_config["scale_factor"]
    if not (isinstance(scale_factor, (int, float)) and 0 < scale_factor <= 1.0):
        print(f"  Warning: Invalid scale_factor {scale_factor} for {original_image_path}. Must be float between 0 (excl) and 1.0. Skipping rescale.")
        return original_image_path, 1.0
    if scale_factor == 1.0: # No actual rescaling
        return original_image_path, 1.0

    interpolation_str = rescaling_config.get("interpolation", "INTER_AREA")
    interpolation_method = get_cv2_interpolation_method(interpolation_str)
    
    if not os.path.exists(RESCALED_IMAGE_CACHE_DIR):
        try: os.makedirs(RESCALED_IMAGE_CACHE_DIR)
        except OSError as e: print(f"Could not create base rescaled cache dir {RESCALED_IMAGE_CACHE_DIR}: {e}")
    
    rescaled_image_specific_dir = os.path.join(RESCALED_IMAGE_CACHE_DIR, image_id_for_cache)
    if not os.path.exists(rescaled_image_specific_dir):
        try: os.makedirs(rescaled_image_specific_dir)
        except OSError as e:
            print(f"  Error creating cache subdir {rescaled_image_specific_dir}: {e}. Cannot cache rescaled image.")
            return original_image_path, 1.0

    original_filename = os.path.basename(original_image_path)
    scale_factor_str = str(scale_factor).replace('.', '_')
    rescaled_image_filename = f"{os.path.splitext(original_filename)[0]}_scaled_{scale_factor_str}.tif"
    rescaled_image_path = os.path.join(rescaled_image_specific_dir, rescaled_image_filename)

    if os.path.exists(rescaled_image_path):
        print(f"  Found cached rescaled image: {rescaled_image_path}")
        return rescaled_image_path, scale_factor

    try:
        print(f"  Rescaling {original_image_path} by factor {scale_factor} using {interpolation_str}...")
        img = cv2.imread(original_image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if img is None:
            print(f"  Error: Could not load original image {original_image_path} for rescaling.")
            return original_image_path, 1.0

        new_width = int(round(img.shape[1] * scale_factor))
        new_height = int(round(img.shape[0] * scale_factor))
        
        if new_width == 0 or new_height == 0:
            print(f"  Error: Rescaled dimensions are zero or near-zero ({new_width}x{new_height}). Scale factor {scale_factor} too small for image size {img.shape[:2]}.")
            return original_image_path, 1.0

        rescaled_img = cv2.resize(img, (new_width, new_height), interpolation=interpolation_method)
        
        tifffile.imwrite(rescaled_image_path, rescaled_img)
        print(f"  Saved rescaled image to: {rescaled_image_path} (Shape: {rescaled_img.shape})")
        return rescaled_image_path, scale_factor
    except Exception as e:
        print(f"  Error during rescaling or saving {original_image_path}: {e}")
        traceback.print_exc()
        return original_image_path, 1.0
