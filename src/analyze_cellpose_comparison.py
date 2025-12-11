#!/usr/bin/env python3
"""
Script to analyze Cellpose model comparison results.
Extracts metrics from segmentation summary files and mask files,
calculates statistics, and generates a comparison table.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import tifffile
from skimage import measure

def process_result(image_config, param_config, mask_path, summary_path):
    """
    Processes a single result: loads mask and summary, computes metrics.
    """
    try:
        # Load mask file
        mask = tifffile.imread(mask_path)
        
        # Load summary file
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)

        # Get pixel size (microns per pixel)
        mpp = image_config.get('mpp_x', 1.0)  # Assume square pixels
        pixel_to_um2 = mpp**2

        # Basic cell count
        unique_labels = np.unique(mask)
        cell_count = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)

        if cell_count == 0:
            print(f"    - No cells found in mask: {mask_path.parent}")
            return {
                'Sample': image_config['image_id'],
                'Diameter (px)': param_config['cellpose_parameters']['DIAMETER'],
                'Cell Count': 0,
                'Mean_Area (µm²)': 0,
                'Max_Area (µm²)': 0,
                'Total_Segmented_Area (µm²)': 0
            }

        # Calculate area for each cell
        props = measure.regionprops(mask)
        areas_pixels = [prop.area for prop in props]
        
        # Calculate metrics
        mean_area_pixels = np.mean(areas_pixels)
        max_area_pixels = np.max(areas_pixels)
        total_segmented_area_pixels = np.sum(areas_pixels)

        # Convert to µm²
        mean_area_um2 = mean_area_pixels * pixel_to_um2
        max_area_um2 = max_area_pixels * pixel_to_um2
        total_segmented_area_um2 = total_segmented_area_pixels * pixel_to_um2

        return {
            'Sample': image_config['image_id'],
            'Diameter (px)': param_config['cellpose_parameters']['DIAMETER'],
            'Cell Count': cell_count,
            'Mean_Area (µm²)': round(mean_area_um2, 2),
            'Max_Area (µm²)': round(max_area_um2, 2),
            'Total_Segmented_Area (µm²)': round(total_segmented_area_um2, 2)
        }

    except FileNotFoundError:
        print(f"    - Error: Mask or summary file not found for {mask_path.parent}")
        return None
    except Exception as e:
        print(f"    - Error processing result {mask_path.parent}: {e}")
        return None

def analyze_and_generate_table(config_file, results_dir, output_dir):
    """
    Main analysis function.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_file}")
        return

    image_configs = {img['image_id']: img for img in config.get('image_configurations', []) if img.get('is_active', True)}
    param_configs = {param['param_set_id']: param for param in config.get('cellpose_parameter_configurations', []) if param.get('is_active', True)}

    print(f"Found {len(image_configs)} active image configs: {list(image_configs.keys())}")
    print(f"Found {len(param_configs)} active parameter configs: {list(param_configs.keys())}")

    results_data = []
    all_subdirs_in_results = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

    for image_id, image_config in image_configs.items():
        print(f"\nProcessing image: {image_id}")
        
        # Find result directories that belong to this image
        image_result_dirs = [d for d in all_subdirs_in_results if d.startswith(image_id)]
        
        if not image_result_dirs:
            print(f"  - Warning: No result directories found for image_id '{image_id}'")
            continue

        for dir_name in image_result_dirs:
            # Reconstruct param_set_id from directory name
            # e.g., "5A_rev2_v1_comparison_30" -> "comparison_30"
            parts = dir_name.replace(f"{image_id}_", "").split('_scaled_')
            param_set_id = parts[0]
            
            if param_set_id in param_configs:
                print(f"  + Analyzing result: {dir_name}")
                param_config = param_configs[param_set_id]
                full_dir_path = Path(results_dir) / dir_name
                
                mask_files = list(full_dir_path.glob("*_mask.tif"))
                summary_files = list(full_dir_path.glob("*_segmentation_summary.json"))

                if not mask_files or not summary_files:
                    print(f"    - Warning: Missing mask or summary file in {full_dir_path}. Skipping.")
                    continue
                
                stats = process_result(image_config, param_config, mask_files[0], summary_files[0])
                if stats:
                    results_data.append(stats)
            else:
                print(f"  - Skipping directory {dir_name} (param_set_id '{param_set_id}' not active in config).")

    if not results_data:
        print("\nNo valid results found to generate a table. Please check your results directory and config file.")
        return

    # Create and save the table
    df = pd.DataFrame(results_data)
    df_sorted = df.sort_values(by=['Sample', 'Diameter (px)'])

    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = Path(output_dir) / 'segmentation_comparison_by_image.csv'
    md_path = Path(output_dir) / 'segmentation_comparison_by_image.md'
    
    df_sorted.to_csv(csv_path, index=False)
    df_sorted.to_markdown(md_path, index=False)

    print("\n" + "="*50)
    print("Analysis Complete!")
    print(f"Comparison table saved to:")
    print(f"  - CSV: {csv_path}")
    print(f"  - MD:  {md_path}")
    print("="*50)
    print("\nTable Preview:")
    print(df_sorted.to_string(index=False))


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze and compare Cellpose segmentation results.")
    parser.add_argument("--config", required=True,
                        help="Path to the processing configuration JSON file.")
    parser.add_argument("--results_dir", default="results",
                        help="Base directory where segmentation results are stored.")
    parser.add_argument("--output_dir", default="results/comparison_tables",
                        help="Directory to save the output tables.")
    
    args = parser.parse_args()
    
    print("Analyzing Cellpose model comparison results...")
    analyze_and_generate_table(args.config, args.results_dir, args.output_dir)


if __name__ == "__main__":
    main() 