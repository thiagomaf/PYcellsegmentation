#!/usr/bin/env python3
"""
Script to create JSON configuration files for cell segmentation pipeline.
Extracts DPI from image metadata, calculates MPP, determines scale factors,
and groups images by DPI for xenium images.
"""

import os
import json
import re
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
import tifffile
from PIL import Image

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
XENIUM_DIR = PROJECT_ROOT / "data" / "raw" / "images" / "xenium_HE" / "xenium"
HE_DIR = PROJECT_ROOT / "data" / "raw" / "images" / "xenium_HE" / "HE"
VISIUM_DIR = PROJECT_ROOT / "data" / "raw" / "images" / "visium"
CONFIG_DIR = PROJECT_ROOT / "config" / "xenium_HE"

# Target scaled image size (MB)
TARGET_SCALED_SIZE_MB = 275  # Target around 250-300 MB

def extract_dpi_from_tiff(image_path):
    """
    Extract DPI value from TIFF/OME-TIFF metadata.
    Returns DPI value or None if not found.
    """
    try:
        # Try using tifffile for OME-TIFF metadata
        with tifffile.TiffFile(image_path) as tif:
            # Check OME metadata first
            if tif.ome_metadata:
                try:
                    root = ET.fromstring(tif.ome_metadata)
                    # Look for DPI in various locations
                    # Check ImageDescription or other metadata fields
                    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                    pixels = root.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')
                    if pixels is not None:
                        # Check for PhysicalSizeX/PhysicalSizeY in microns, convert back to DPI
                        # But we need actual DPI, not physical size
                        pass
                except Exception as e:
                    print(f"    Warning: Could not parse OME XML: {e}")
            
            # Check TIFF tags for resolution
            for page in tif.pages:
                tags = page.tags
                # Look for XResolution and YResolution
                if 282 in tags:  # XResolution tag
                    x_res = tags[282].value
                    if isinstance(x_res, tuple):
                        x_res = x_res[0] / x_res[1] if x_res[1] != 0 else x_res[0]
                    
                    # Check ResolutionUnit (296 = inches, 3 = centimeters)
                    unit = tags.get(296)  # ResolutionUnit
                    unit_value = unit.value if unit else 2  # Default to inches (2)
                    
                    if unit_value == 2:  # Inches
                        dpi = x_res
                        return float(dpi)
                    elif unit_value == 3:  # Centimeters
                        dpi = x_res * 2.54
                        return float(dpi)
        
        # Fallback: Try PIL
        try:
            with Image.open(image_path) as img:
                if hasattr(img, 'tag') and 282 in img.tag:  # XResolution
                    x_res = img.tag[282]
                    if isinstance(x_res, tuple):
                        x_res = x_res[0] / x_res[1] if x_res[1] != 0 else x_res[0]
                    
                    unit = img.tag.get(296, (2,))  # ResolutionUnit, default to inches
                    if unit[0] == 2:  # Inches
                        return float(x_res)
                    elif unit[0] == 3:  # Centimeters
                        return float(x_res * 2.54)
        except Exception as e:
            print(f"    Warning: PIL failed: {e}")
    
    except Exception as e:
        print(f"    Error reading {image_path}: {e}")
    
    return None

def dpi_to_mpp(dpi):
    """Convert DPI to micrometers per pixel (MPP)."""
    if dpi is None or dpi <= 0:
        return 0.0
    mpp = 25400.0 / dpi
    return round(mpp, 4)

def get_file_size_mb(file_path):
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)

def calculate_scale_factor(file_size_mb):
    """
    Calculate appropriate scale factor based on file size.
    Target: Keep scaled images around 250-300 MB.
    """
    if file_size_mb < 250:
        return 1.0
    elif file_size_mb <= 300:
        return 0.5
    elif file_size_mb <= 600:
        # Use sqrt to get 2D scaling that results in ~275 MB
        target_ratio = TARGET_SCALED_SIZE_MB / file_size_mb
        scale_factor = target_ratio ** 0.5  # Square root for 2D
        # Round to reasonable values
        if scale_factor > 0.5:
            return 0.5
        elif scale_factor > 0.3:
            return round(scale_factor, 2)
        else:
            return 0.25
    else:
        # For very large files, calculate scale to target size
        target_ratio = TARGET_SCALED_SIZE_MB / file_size_mb
        scale_factor = target_ratio ** 0.5
        # Round to reasonable increments
        if scale_factor > 0.5:
            return 0.5
        elif scale_factor > 0.3:
            return round(scale_factor, 2)
        elif scale_factor > 0.2:
            return 0.25
        else:
            return 0.2

def extract_dpi_from_filename(filename):
    """
    Extract DPI value from filename pattern like '14DPI', '28DPI', etc.
    Returns DPI value as integer or None.
    """
    # Pattern to match DPI values in filename
    patterns = [
        r'(\d+)DPI',  # Matches "14DPI", "28DPI", etc.
        r'_(\d+)DPI',  # Matches "_14DPI", "_28DPI", etc.
        r'DPI[_-]?(\d+)',  # Alternative format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    
    return None

def sanitize_image_id(filename):
    """Convert filename to image_id format."""
    # Remove extension
    base = os.path.splitext(filename)[0]
    # Remove .ome if present
    base = base.replace('.ome', '')
    # Replace hyphens with underscores
    base = base.replace('-', '_')
    # Remove common suffixes
    base = re.sub(r'_merged$', '', base)
    base = re.sub(r'_taugat_QuPath$', '', base)
    return base

def get_image_info(image_path, image_type='xenium'):
    """
    Extract image information including DPI, MPP, file size, and scale factor.
    Returns dict with all image metadata.
    """
    filename = os.path.basename(image_path)
    file_size_mb = get_file_size_mb(image_path)
    
    # Try to extract DPI from metadata
    dpi = extract_dpi_from_tiff(image_path)
    
    # Fallback: try to extract from filename
    if dpi is None:
        dpi = extract_dpi_from_filename(filename)
        if dpi:
            print(f"  Using DPI from filename for {filename}: {dpi}")
    
    # Calculate MPP
    mpp = dpi_to_mpp(dpi)
    
    # Determine scale factor
    if image_type == 'visium':
        scale_factor = 1.0
    else:
        scale_factor = calculate_scale_factor(file_size_mb)
    
    # Convert path to use forward slashes (JSON standard)
    relative_path = str(image_path.relative_to(PROJECT_ROOT)).replace('\\', '/')
    
    return {
        'filename': filename,
        'full_path': str(image_path),
        'relative_path': relative_path,
        'image_id': sanitize_image_id(filename),
        'dpi': dpi,
        'mpp_x': mpp,
        'mpp_y': mpp,
        'file_size_mb': round(file_size_mb, 2),
        'scale_factor': scale_factor
    }

def group_xenium_images(image_infos):
    """Group xenium images by DPI extracted from filename."""
    groups = defaultdict(list)
    
    for info in image_infos:
        # Prioritize DPI from filename
        dpi = extract_dpi_from_filename(info['filename'])
        if dpi is None:
            # Fallback: use DPI from metadata if available, but round to integer for grouping
            if info['dpi'] is not None:
                # For grouping, round metadata DPI to nearest reasonable value
                # If DPI is very high (>1000), it might be in different units
                metadata_dpi = info['dpi']
                if metadata_dpi > 10000:
                    # Might be wrong unit, try to infer from filename pattern or skip grouping
                    print(f"  Warning: Unusual DPI value {metadata_dpi} for {info['filename']}, using filename-based grouping")
                    continue
                dpi = int(round(metadata_dpi))
            else:
                print(f"  Warning: Could not extract DPI for {info['filename']}, skipping")
                continue
        
        groups[dpi].append(info)
    
    # Sort groups by DPI
    return dict(sorted(groups.items()))

def create_config_json(image_configs, image_type='xenium', dpi=None, batch_num=None):
    """Create a JSON configuration file."""
    config = {
        "global_segmentation_settings": {
            "default_log_level": "INFO",
            "max_processes": 1,
            "FORCE_GRAYSCALE": True,
            "USE_GPU_IF_AVAILABLE": True
        },
        "image_configurations": [],
        "cellpose_parameter_configurations": [
            {
                "param_set_id": "comparison_60",
                "is_active": True,
                "cellpose_parameters": {
                    "MODEL_CHOICE": "cyto3",
                    "DIAMETER": 60,
                    "MIN_SIZE": 15,
                    "CELLPROB_THRESHOLD": 0.0,
                    "FORCE_GRAYSCALE": True,
                    "Z_PROJECTION_METHOD": "max",
                    "CHANNEL_INDEX": 0,
                    "ENABLE_3D_SEGMENTATION": False
                }
            },
            {
                "param_set_id": "comparison_90",
                "is_active": True,
                "cellpose_parameters": {
                    "MODEL_CHOICE": "cyto3",
                    "DIAMETER": 90,
                    "MIN_SIZE": 15,
                    "CELLPROB_THRESHOLD": 0.0,
                    "FORCE_GRAYSCALE": True,
                    "Z_PROJECTION_METHOD": "max",
                    "CHANNEL_INDEX": 0,
                    "ENABLE_3D_SEGMENTATION": False
                }
            }
        ],
        "mapping_tasks": []
    }
    
    # Add image configurations
    for info in image_configs:
        image_config = {
            "image_id": info['image_id'],
            "original_image_filename": info['relative_path'],
            "is_active": True,
            "mpp_x": info['mpp_x'],
            "mpp_y": info['mpp_y'],
            "segmentation_options": {
                "apply_segmentation": True,
                "rescaling_config": {
                    "scale_factor": info['scale_factor'],
                    "interpolation": "INTER_LINEAR"
                },
                "tiling_parameters": {
                    "apply_tiling": False
                }
            }
        }
        config["image_configurations"].append(image_config)
    
    # Determine filename
    if image_type == 'xenium' and dpi is not None:
        filename = f"processing_config_xenium_{dpi}DPI_batch_{batch_num}.json"
    elif image_type == 'he':
        filename = f"processing_config_HE_batch_{batch_num}.json"
    elif image_type == 'visium':
        filename = f"processing_config_visium_batch_{batch_num}.json"
    else:
        filename = f"processing_config_batch_{batch_num}.json"
    
    output_path = CONFIG_DIR / filename
    
    # Write JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return filename, len(image_configs)

def main():
    """Main function to process all images and create config files."""
    print("Creating JSON configuration files for cell segmentation pipeline...")
    print(f"Project root: {PROJECT_ROOT}")
    
    # Ensure config directory exists
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    all_configs = []
    
    # Process Xenium images
    print("\n=== Processing Xenium Images ===")
    # Use set to avoid duplicates (as *.tif pattern also matches *.ome.tif)
    xenium_files = set(XENIUM_DIR.glob("*.tif")) | set(XENIUM_DIR.glob("*.ome.tif"))
    xenium_files = sorted([f for f in xenium_files if f.is_file()])
    
    print(f"Found {len(xenium_files)} xenium image files")
    
    xenium_infos = []
    for file_path in sorted(xenium_files):
        print(f"\nProcessing: {file_path.name}")
        info = get_image_info(file_path, image_type='xenium')
        xenium_infos.append(info)
        print(f"  DPI: {info['dpi']}, MPP: {info['mpp_x']}, Size: {info['file_size_mb']} MB, Scale: {info['scale_factor']}")
    
    # Group xenium images by DPI
    xenium_groups = group_xenium_images(xenium_infos)
    print(f"\nGrouped into {len(xenium_groups)} DPI groups: {sorted(xenium_groups.keys())}")
    
    # Create config files for xenium images
    for dpi, images in sorted(xenium_groups.items()):
        # Split into batches of max 5 images
        for batch_idx in range(0, len(images), 5):
            batch = images[batch_idx:batch_idx + 5]
            batch_num = (batch_idx // 5) + 1
            filename, count = create_config_json(batch, image_type='xenium', dpi=dpi, batch_num=batch_num)
            all_configs.append({
                'filename': filename,
                'type': 'xenium',
                'dpi': dpi,
                'batch': batch_num,
                'count': count,
                'images': [img['filename'] for img in batch]
            })
            print(f"  Created: {filename} ({count} images)")
    
    # Process H&E images
    print("\n=== Processing H&E Images ===")
    # Use set to avoid duplicates
    he_files = set(HE_DIR.glob("*.tif")) | set(HE_DIR.glob("*.ome.tif"))
    he_files = sorted([f for f in he_files if f.is_file()])
    
    print(f"Found {len(he_files)} H&E image files")
    
    he_infos = []
    for file_path in sorted(he_files):
        print(f"\nProcessing: {file_path.name}")
        info = get_image_info(file_path, image_type='he')
        he_infos.append(info)
        print(f"  DPI: {info['dpi']}, MPP: {info['mpp_x']}, Size: {info['file_size_mb']} MB, Scale: {info['scale_factor']}")
    
    # Create config files for H&E images (batches of 5)
    for batch_idx in range(0, len(he_infos), 5):
        batch = he_infos[batch_idx:batch_idx + 5]
        batch_num = (batch_idx // 5) + 1
        filename, count = create_config_json(batch, image_type='he', batch_num=batch_num)
        all_configs.append({
            'filename': filename,
            'type': 'he',
            'dpi': None,
            'batch': batch_num,
            'count': count,
            'images': [img['filename'] for img in batch]
        })
        print(f"  Created: {filename} ({count} images)")
    
    # Process Visium images (if any)
    print("\n=== Processing Visium Images ===")
    # Use set to avoid duplicates
    visium_files = set(VISIUM_DIR.glob("*.tif")) | set(VISIUM_DIR.glob("*.ome.tif"))
    visium_files = sorted([f for f in visium_files if f.is_file()])
    
    if visium_files:
        print(f"Found {len(visium_files)} visium image files")
        
        visium_infos = []
        for file_path in sorted(visium_files):
            print(f"\nProcessing: {file_path.name}")
            info = get_image_info(file_path, image_type='visium')
            visium_infos.append(info)
            print(f"  DPI: {info['dpi']}, MPP: {info['mpp_x']}, Size: {info['file_size_mb']} MB, Scale: {info['scale_factor']}")
        
        # Create config files for visium images (batches of 5)
        for batch_idx in range(0, len(visium_infos), 5):
            batch = visium_infos[batch_idx:batch_idx + 5]
            batch_num = (batch_idx // 5) + 1
            filename, count = create_config_json(batch, image_type='visium', batch_num=batch_num)
            all_configs.append({
                'filename': filename,
                'type': 'visium',
                'dpi': None,
                'batch': batch_num,
                'count': count,
                'images': [img['filename'] for img in batch]
            })
            print(f"  Created: {filename} ({count} images)")
    
    # Create tracking file
    print("\n=== Creating Tracking File ===")
    tracking_path = CONFIG_DIR / "PROCESSING_STATUS.md"
    
    with open(tracking_path, 'w', encoding='utf-8') as f:
        f.write("# Processing Status\n\n")
        f.write("## Configuration Files\n\n")
        f.write("| Config File | Type | DPI | Batch | Images | Status | Notes |\n")
        f.write("|-------------|------|-----|-------|--------|--------|-------|\n")
        
        for config in all_configs:
            dpi_str = str(config['dpi']) if config['dpi'] else "-"
            f.write(f"| {config['filename']} | {config['type']} | {dpi_str} | {config['batch']} | {config['count']} | Pending | - |\n")
        
        f.write("\n## Status Legend\n\n")
        f.write("- **Pending**: Config file created, ready for processing\n")
        f.write("- **Processing**: Currently being processed\n")
        f.write("- **Completed**: Processing finished successfully\n")
        f.write("- **Failed**: Processing encountered errors\n")
    
    print(f"Created tracking file: {tracking_path}")
    print(f"\nTotal config files created: {len(all_configs)}")
    
    # Print summary
    print("\n=== Summary ===")
    xenium_count = sum(1 for c in all_configs if c['type'] == 'xenium')
    he_count = sum(1 for c in all_configs if c['type'] == 'he')
    visium_count = sum(1 for c in all_configs if c['type'] == 'visium')
    print(f"Xenium configs: {xenium_count}")
    print(f"H&E configs: {he_count}")
    print(f"Visium configs: {visium_count}")

if __name__ == "__main__":
    main()

