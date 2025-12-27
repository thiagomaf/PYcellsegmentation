#!/usr/bin/env python3
"""
Visualization Asset Generator
=============================

This script generates screen-optimized assets for the Napari visualization workflow.
It processes segmentation results to create:
1. Resized image and label mask (e.g. max dimension 4096px) for fast loading.
2. A statistics JSON file containing per-cell metrics (Area, Circularity, etc.).

Usage:
    python tools/create_vis_assets.py --config config/processing_config.json --image_id RNAlater_S3
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import tifffile
import cv2
import pandas as pd
from pathlib import Path
from skimage import measure

# Ensure src modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline_utils import resolve_image_path, clean_filename_for_dir, setup_logging

logger = logging.getLogger(__name__)

class AssetGenerator:
    def __init__(self, config_path, max_dim=4096):
        self.config_path = config_path
        self.max_dim = max_dim
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config = self._load_config()

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _resolve_path(self, path_str):
        """Resolves path string to absolute local path, handling /content/drive etc."""
        if not path_str:
            return None
            
        # 1. Try generic pipeline resolution
        resolved = resolve_image_path(path_str, str(self.project_root))
        if resolved and os.path.exists(resolved):
            return resolved

        # 2. Heuristic for Colab/Remote paths:
        # Look for the project name "PYcellsegmentation" in the path and take everything after
        # Or look for known folders like "results/", "data/"
        parts = path_str.replace('\\', '/').split('/')
        
        # Try finding known anchors
        anchors = ["results", "data", "images"]
        for anchor in anchors:
            if anchor in parts:
                idx = parts.index(anchor)
                # Reconstruct relative path from anchor
                rel_path = os.path.join(*parts[idx:])
                local_candidate = self.project_root / rel_path
                if local_candidate.exists():
                    return str(local_candidate)
        
        return resolved

    def _get_output_path(self, result_dir, filename):
        return Path(result_dir) / filename

    def _resize_image(self, image, is_mask=False):
        """Resizes image to fit within max_dim while maintaining aspect ratio."""
        h, w = image.shape[:2]
        scale = min(self.max_dim / h, self.max_dim / w)
        
        if scale >= 1.0:
            return image, 1.0  # No downsampling needed

        new_w, new_h = int(w * scale), int(h * scale)
        
        if is_mask:
            # Nearest neighbor for masks to preserve integer labels
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            # Area interpolation for images (good for downsampling)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        return resized, scale

    def analyze_mask(self, mask, pixel_size_um=1.0):
        """
        Calculates per-cell statistics from the full-resolution mask.
        """
        logger.info("  Calculating cell statistics...")
        props = measure.regionprops(mask)
        
        stats = []
        for prop in props:
            # Basic stats
            area_px = prop.area
            perimeter = prop.perimeter
            
            # Shape descriptors
            circularity = (4 * np.pi * area_px) / (perimeter ** 2) if perimeter > 0 else 0
            
            # Convert to physical units
            area_um2 = area_px * (pixel_size_um ** 2)
            # Use equivalent_diameter_area to avoid deprecation warning (scikit-image >= 0.26)
            try:
                equiv_diam_um = prop.equivalent_diameter_area * pixel_size_um
            except AttributeError:
                # Fallback for older scikit-image versions
                equiv_diam_um = prop.equivalent_diameter * pixel_size_um
            
            stats.append({
                'label': prop.label,
                'area_um2': round(area_um2, 2),
                'diameter_um': round(equiv_diam_um, 2),
                'circularity': round(circularity, 3),
                'solidity': round(prop.solidity, 3),
                'eccentricity': round(prop.eccentricity, 3)
            })
            
        return stats

    def process_experiment(self, image_id):
        """
        Main processing loop for a given image ID.
        """
        logger.info(f"Processing image ID: {image_id}")
        
        # 1. Find Image Config
        image_config = next((cfg for cfg in self.config.get('image_configurations', []) 
                             if cfg.get('image_id') == image_id), None)
        if not image_config:
            logger.error(f"Image ID {image_id} not found in config.")
            return

        # Find all relevant result directories first
        results_base = self.project_root / "results"
        # We look for directories starting with image_id
        # Pattern: {image_id}_diam_{d}_scaled_{s} or similar
        potential_dirs = [d for d in results_base.iterdir() if d.is_dir() and d.name.startswith(image_id)]
        
        if not potential_dirs:
            logger.warning(f"No result directories found for image ID: {image_id}")
            return
        
        # Load the processed image from the first summary (all results for same image_id should use same processed image)
        first_summary_file = next(potential_dirs[0].glob("*segmentation_summary.json"), None)
        if not first_summary_file:
            logger.error(f"No summary file found in {potential_dirs[0]}")
            return
        
        with open(first_summary_file, 'r') as f:
            first_summary = json.load(f)
        
        # Load the processed image (rescaled version that was actually segmented)
        processed_image_path = self._resolve_path(first_summary.get('image_path_processed'))
        if not processed_image_path or not os.path.exists(processed_image_path):
            logger.warning(f"Processed image not found: {first_summary.get('image_path_processed')}. Falling back to original.")
            # Fallback to original
            processed_image_path = self._resolve_path(image_config['original_image_filename'])
            if not os.path.exists(processed_image_path):
                logger.error(f"Source image not found: {processed_image_path}")
                return

        logger.info(f"Loading processed image: {processed_image_path}")
        image = tifffile.imread(processed_image_path)
        
        # Handle dimensionality (ensure 2D/3D is handled correctly)
        if image.ndim == 3 and image.shape[0] < 10: # Assume channels first if small dim 0
             image = image[0] # Take first channel for viz
        
        # Resize Source Image
        resized_image, img_scale = self._resize_image(image, is_mask=False)
        
        processed_count = 0
        
        for res_dir in potential_dirs:
            summary_file = next(res_dir.glob("*segmentation_summary.json"), None)
            if not summary_file:
                continue

            logger.info(f"  Found result directory: {res_dir.name}")
            
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                # Resolve Mask Path
                mask_path_raw = summary.get('mask_output_path')
                mask_path = self._resolve_path(mask_path_raw)
                
                if not mask_path or not os.path.exists(mask_path):
                    logger.warning(f"    Mask not found: {mask_path_raw}")
                    continue
                
                # Load Mask
                logger.info(f"    Loading mask: {mask_path}")
                mask = tifffile.imread(mask_path)
                
                # Get Pixel Size (MPP) for stats
                # Note: Analysis should use original MPP if mask is on original scale,
                # or scaled MPP if mask is on scaled image.
                # Usually pipeline outputs masks matching the input to segmentation (which might be scaled).
                
                mpp_x = image_config.get('mpp_x', 1.0)
                # Adjust MPP if the segmentation was done on a scaled image
                scale_applied = summary.get('job_params_received', {}).get('scale_factor_applied_for_log', 1.0)
                if scale_applied:
                     mpp_x = mpp_x / float(scale_applied)

                # Calculate Stats (on full res mask)
                stats = self.analyze_mask(mask, pixel_size_um=mpp_x)
                
                # Resize Mask (to match resized image)
                # Note: We resize based on the *viz* scale factor calculated from the source image.
                # Ideally source image and mask match in dimensions.
                if mask.shape != image.shape:
                    logger.warning(f"    Shape mismatch! Image: {image.shape}, Mask: {mask.shape}. Resize might be misaligned.")
                
                resized_mask, _ = self._resize_image(mask, is_mask=True)
                
                # Save Assets
                viz_image_path = self._get_output_path(res_dir, f"{image_id}_viz_image.tif")
                viz_labels_path = self._get_output_path(res_dir, f"{image_id}_viz_labels.tif")
                viz_stats_path = self._get_output_path(res_dir, f"{image_id}_viz_stats.json")
                
                # We save the image once per result dir (redundant but robust if results move)
                # Or we could just save it once centrally. For now, following plan (assets in result folder).
                tifffile.imwrite(viz_image_path, resized_image)
                tifffile.imwrite(viz_labels_path, resized_mask, compression='zlib')
                
                with open(viz_stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                logger.info(f"    Saved viz assets to {res_dir.name}")
                processed_count += 1
                
            except Exception as e:
                logger.error(f"    Error processing {res_dir.name}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        if processed_count == 0:
            logger.warning("No valid segmentation results found for this image.")

def main():
    parser = argparse.ArgumentParser(description="Generate Napari visualization assets.")
    parser.add_argument("--config", required=True, help="Path to processing config JSON.")
    parser.add_argument("--image_id", required=True, help="ID of the image to process.")
    parser.add_argument("--max_dim", type=int, default=4096, help="Maximum dimension for resized assets.")
    
    args = parser.parse_args()
    
    setup_logging()
    
    generator = AssetGenerator(args.config, args.max_dim)
    generator.process_experiment(args.image_id)

if __name__ == "__main__":
    main()

