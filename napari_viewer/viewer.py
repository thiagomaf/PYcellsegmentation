#!/usr/bin/env python3
"""
Napari Comparison App
=====================

A Napari-based viewer for comparing cell segmentation results.
Loads pre-generated "Viz Assets" (resized images and masks with stats).

Usage:
    python napari/napari.py --config config/processing_config.json --image_id RNAlater_S3
"""

import argparse
import json
import logging
import sys
import os
import glob
from pathlib import Path
import napari
import numpy as np
import tifffile
import pandas as pd

# Add src to path if needed (though this script is in napari/ folder)
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(project_root))

from src.pipeline_utils import setup_logging

# Import color control widget
sys.path.insert(0, str(Path(__file__).parent))
from color_control_widget import LayerColorControl

logger = logging.getLogger(__name__)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Napari Segmentation Comparison Viewer")
    parser.add_argument("--config", required=True, help="Path to processing config JSON.")
    parser.add_argument("--image_id", required=True, help="ID of the image to visualize.")
    parser.add_argument("--dry-run", action="store_true", help="Run setup but skip starting the viewer loop.")
    
    args = parser.parse_args()
    setup_logging()
    
    logger.info(f"Starting Napari Viewer for {args.image_id}...")
    
    # Initialize Viewer
    viewer = napari.Viewer(title=f"Comparison: {args.image_id}")
    
    # 1. Find and Load Base Image
    # We look for ANY result directory that has the viz_image
    results_base = project_root / "results"
    
    # Find all result folders for this image
    # Pattern matching: {image_id}*
    potential_dirs = [d for d in results_base.iterdir() if d.is_dir() and d.name.startswith(args.image_id)]
    
    if not potential_dirs:
        logger.error(f"No results found for image ID: {args.image_id}")
        return

    # Try to load the base image from the first available result
    base_image_loaded = False
    
    for res_dir in potential_dirs:
        viz_image_path = res_dir / f"{args.image_id}_viz_image.tif"
        if viz_image_path.exists():
            logger.info(f"Loading base image from: {viz_image_path}")
            image_data = tifffile.imread(viz_image_path)
            viewer.add_image(image_data, name=f"Image: {args.image_id}", blending='additive')
            base_image_loaded = True
            break
            
    if not base_image_loaded:
        logger.warning("Could not find a _viz_image.tif in any result folder. Have you run 'tools/create_vis_assets.py'?")
        # Fallback: Try to load original from config? (Not implemented for simplicity/performance)
        return

    # 2. Load Label Layers for each Result
    for res_dir in potential_dirs:
        viz_labels_path = res_dir / f"{args.image_id}_viz_labels.tif"
        viz_stats_path = res_dir / f"{args.image_id}_viz_stats.json"
        
        if not viz_labels_path.exists():
            continue
            
        logger.info(f"Loading result: {res_dir.name}")
        
        # Load Mask
        labels_data = tifffile.imread(viz_labels_path)
        
        # Determine Layer Name (parse useful info from dir name)
        # Directory format is often: {image_id}_{param_set_id}_{scale_info}
        # We want to show the param_set_id (e.g. diam_15, diam_30)
        layer_name = res_dir.name.replace(args.image_id, "").strip("_")
        if not layer_name: layer_name = "Result"
        
        # Load Stats/Features
        features = {}
        if viz_stats_path.exists():
            try:
                stats_list = load_json(viz_stats_path)
                # Convert list of dicts to dict of lists (DataFrame-style) for Napari
                if stats_list:
                    df = pd.DataFrame(stats_list)
                    features = df.to_dict(orient='list')
            except Exception as e:
                logger.error(f"Failed to load stats from {viz_stats_path}: {e}")
        
        # Add Labels Layer
        # visibility=False by default to avoid clutter
        # Colors can be changed via the GUI widget
        layer = viewer.add_labels(
            labels_data, 
            name=layer_name, 
            visible=False,
            features=features
        )
    
    # 3. Add Color Control Widget
    color_widget = LayerColorControl(viewer)
    viewer.window.add_dock_widget(color_widget, name="Layer Colors", area='right')
    logger.info("Color control widget added to viewer.")
    
    logger.info("Viewer launched. ready.")
    
    if args.dry_run:
        logger.info("Dry run complete. Exiting before napari.run()")
        return

    napari.run()

if __name__ == "__main__":
    main()

