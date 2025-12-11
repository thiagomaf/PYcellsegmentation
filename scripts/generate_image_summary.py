"""
Generate a CSV summary of which input images have corresponding output folders.

This script scans input images from visium, HE, and xenium directories and checks
for corresponding output folders in results/results_paper/.
"""

import os
import csv
from pathlib import Path


def derive_output_base_name(input_filename, source_dir):
    """
    Derive the expected output base name from an input filename.
    
    Args:
        input_filename: Name of the input image file
        source_dir: Source directory type ('visium', 'HE', or 'xenium')
    
    Returns:
        Expected base name for output folders
    """
    # Remove .ome.tif extension
    base = input_filename.replace('.ome.tif', '')
    
    if source_dir == 'visium':
        # Convert dashes to underscores
        base = base.replace('-', '_')
    elif source_dir == 'HE':
        # Convert dash to underscore (e.g., H&E_XEN-115A -> H&E_XEN_115A)
        base = base.replace('-', '_')
    elif source_dir == 'xenium':
        # Remove _merged or _taugat_QuPath suffix
        if base.endswith('_merged'):
            base = base[:-7]  # Remove '_merged'
        elif base.endswith('_taugat_QuPath'):
            base = base[:-14]  # Remove '_taugat_QuPath'
    
    return base


def check_output_folders(output_base, results_dir):
    """
    Check if output folders exist for a given base name.
    
    Args:
        output_base: Base name for output folders
        results_dir: Path to results/results_paper directory
    
    Returns:
        Tuple of (has_60, has_90) booleans
    """
    has_60 = False
    has_90 = False
    
    # Check for comparison_60 (regular or scaled)
    folder_60 = os.path.join(results_dir, f"{output_base}_comparison_60")
    folder_60_scaled = os.path.join(results_dir, f"{output_base}_comparison_60_scaled_0_5")
    
    if os.path.isdir(folder_60) or os.path.isdir(folder_60_scaled):
        has_60 = True
    
    # Check for comparison_90 (regular or scaled)
    folder_90 = os.path.join(results_dir, f"{output_base}_comparison_90")
    folder_90_scaled = os.path.join(results_dir, f"{output_base}_comparison_90_scaled_0_5")
    
    if os.path.isdir(folder_90) or os.path.isdir(folder_90_scaled):
        has_90 = True
    
    return has_60, has_90


def main():
    """Main function to generate the image processing summary CSV."""
    
    # Define paths
    project_root = Path(__file__).parent.parent
    visium_dir = project_root / "data" / "raw" / "images" / "visium"
    he_dir = project_root / "data" / "raw" / "images" / "xenium_HE" / "HE"
    xenium_dir = project_root / "data" / "raw" / "images" / "xenium_HE" / "xenium"
    results_dir = project_root / "results" / "results_paper"
    output_csv = results_dir / "image_processing_summary.csv"
    
    # Collect all input images
    image_data = []
    
    # Scan visium directory
    if visium_dir.exists():
        for file in visium_dir.glob("*.ome.tif"):
            image_data.append({
                'input_image': file.name,
                'source_directory': 'visium'
            })
    
    # Scan HE directory
    if he_dir.exists():
        for file in he_dir.glob("*.ome.tif"):
            image_data.append({
                'input_image': file.name,
                'source_directory': 'HE'
            })
    
    # Scan xenium directory
    if xenium_dir.exists():
        for file in xenium_dir.glob("*.ome.tif"):
            image_data.append({
                'input_image': file.name,
                'source_directory': 'xenium'
            })
    
    # Check output folders for each image
    results = []
    for img_info in image_data:
        output_base = derive_output_base_name(
            img_info['input_image'],
            img_info['source_directory']
        )
        
        has_60, has_90 = check_output_folders(output_base, results_dir)
        
        results.append({
            'input_image': img_info['input_image'],
            'source_directory': img_info['source_directory'],
            'comparison_60': has_60,
            'comparison_90': has_90
        })
    
    # Write CSV
    if results:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['input_image', 'source_directory', 'comparison_60', 'comparison_90']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Summary CSV generated: {output_csv}")
        print(f"Total images processed: {len(results)}")
        print(f"Images with comparison_60: {sum(1 for r in results if r['comparison_60'])}")
        print(f"Images with comparison_90: {sum(1 for r in results if r['comparison_90'])}")
    else:
        print("No input images found.")


if __name__ == "__main__":
    main()





