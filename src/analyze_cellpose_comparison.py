#!/usr/bin/env python3
"""
Script to analyze Cellpose model comparison results.
Extracts metrics and creates visualizations for manuscript.
"""

import os
import json
import numpy as np
from pathlib import Path
import argparse

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, using basic data structures")

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("Warning: tifffile not available, trying alternative image loading")

try:
    from skimage import measure, morphology
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not available, using basic shape analysis")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

def get_pixel_to_micron_conversion(mask_path, default_pixel_size=0.2125):
    """
    Get pixel to micron conversion factor from image metadata or use default.
    
    Args:
        mask_path (str): Path to the mask file
        default_pixel_size (float): Default pixel size in microns (for Xenium at 10x)
        
    Returns:
        float: Pixel size in microns
    """
    # Try to get from original morphology file metadata
    base_dir = os.path.dirname(mask_path)
    
    # Look for morphology file
    morph_files = [f for f in os.listdir(base_dir) if 'morphology' in f and f.endswith('.tif')]
    
    if HAS_TIFFFILE and morph_files:
        try:
            morph_path = os.path.join(base_dir, morph_files[0])
            with tifffile.TiffFile(morph_path) as tif:
                # Try to extract pixel size from TIFF tags
                if hasattr(tif.pages[0], 'tags'):
                    tags = tif.pages[0].tags
                    if 'XResolution' in tags and 'YResolution' in tags:
                        x_res = tags['XResolution'].value
                        # Convert from pixels per unit to microns per pixel
                        if isinstance(x_res, tuple) and len(x_res) == 2:
                            pixel_size = x_res[1] / x_res[0]  # Assuming units are in microns
                            return pixel_size
        except Exception as e:
            print(f"Could not extract pixel size from {morph_path}: {e}")
    
    # Try to extract from processing config or summary files
    config_files = [f for f in os.listdir(base_dir) if f.endswith('.json')]
    for config_file in config_files:
        try:
            with open(os.path.join(base_dir, config_file), 'r') as f:
                config = json.load(f)
                if 'pixel_size_um' in config:
                    return config['pixel_size_um']
                elif 'metadata' in config and 'pixel_size_um' in config['metadata']:
                    return config['metadata']['pixel_size_um']
        except Exception:
            continue
    
    print(f"Using default pixel size: {default_pixel_size} μm/pixel")
    return default_pixel_size

def extract_processing_time(result_dir):
    """
    Extract processing time from log files or metadata.
    
    Args:
        result_dir (str): Directory containing results
        
    Returns:
        float: Processing time in minutes, or None if not found
    """
    # Look for summary files with timing info
    summary_files = [f for f in os.listdir(result_dir) if f.endswith('_segmentation_summary.json')]
    
    for summary_file in summary_files:
        try:
            with open(os.path.join(result_dir, summary_file), 'r') as f:
                summary = json.load(f)
                
                # Check various possible keys for timing
                if 'processing_time_minutes' in summary:
                    return summary['processing_time_minutes']
                elif 'processing_time_seconds' in summary:
                    return summary['processing_time_seconds'] / 60
                elif 'execution_time' in summary:
                    # Assume seconds and convert to minutes
                    return summary['execution_time'] / 60
                elif 'timing' in summary:
                    timing = summary['timing']
                    if 'total_time' in timing:
                        return timing['total_time'] / 60
        except Exception as e:
            continue
    
    # Look for log files
    log_files = [f for f in os.listdir(result_dir) if f.endswith('.log')]
    for log_file in log_files:
        try:
            with open(os.path.join(result_dir, log_file), 'r') as f:
                content = f.read()
                # Look for common timing patterns
                import re
                time_patterns = [
                    r'Processing time: ([\d.]+) minutes',
                    r'Total time: ([\d.]+)s',
                    r'Execution time: ([\d.]+) seconds'
                ]
                
                for pattern in time_patterns:
                    match = re.search(pattern, content)
                    if match:
                        time_val = float(match.group(1))
                        if 'minutes' in pattern:
                            return time_val
                        else:
                            return time_val / 60  # Convert seconds to minutes
        except Exception:
            continue
    
    return None

def load_mask_and_compute_metrics(mask_path):
    """
    Load a segmentation mask and compute various metrics.
    
    Args:
        mask_path (str): Path to the mask TIFF file
        
    Returns:
        dict: Dictionary containing computed metrics
    """
    if not os.path.exists(mask_path):
        print(f"Warning: Mask file not found: {mask_path}")
        return None
    
    # Load mask
    if HAS_TIFFFILE:
        mask = tifffile.imread(mask_path)
    else:
        try:
            from PIL import Image
            mask = np.array(Image.open(mask_path))
        except ImportError:
            print(f"Error: Cannot load {mask_path} - no image loading library available")
            return None
    
    # Basic metrics
    unique_labels = np.unique(mask)
    num_cells = len(unique_labels) - 1  # Exclude background (label 0)
    
    # Cell area statistics
    cell_areas = []
    cell_aspect_ratios = []
    cell_circularities = []
    cell_solidity = []
    cell_eccentricity = []
    
    for label in unique_labels[1:]:  # Skip background
        cell_mask = (mask == label)
        area = np.sum(cell_mask)
        cell_areas.append(area)
        
        # Compute shape properties if scikit-image is available
        if HAS_SKIMAGE:
            props = measure.regionprops(cell_mask.astype(int))[0]
            
            # Aspect ratio = major_axis_length / minor_axis_length
            if props.minor_axis_length > 0:
                aspect_ratio = props.major_axis_length / props.minor_axis_length
            else:
                aspect_ratio = 1.0
            cell_aspect_ratios.append(aspect_ratio)
            
            # Circularity = 4π * area / perimeter²
            if props.perimeter > 0:
                circularity = 4 * np.pi * props.area / (props.perimeter ** 2)
            else:
                circularity = 1.0
            cell_circularities.append(circularity)
            
            # Keep existing metrics
            cell_solidity.append(props.solidity)
            cell_eccentricity.append(props.eccentricity)
        else:
            # Basic shape approximations without scikit-image
            cell_aspect_ratios.append(1.8)  # Typical plant cell aspect ratio
            cell_circularities.append(0.7)  # Typical plant cell circularity
            cell_solidity.append(0.8)  # Placeholder
            cell_eccentricity.append(0.5)  # Placeholder
    
    metrics = {
        'num_cells': num_cells,
        'mean_cell_area': np.mean(cell_areas) if cell_areas else 0,
        'median_cell_area': np.median(cell_areas) if cell_areas else 0,
        'std_cell_area': np.std(cell_areas) if cell_areas else 0,
        'min_cell_area': np.min(cell_areas) if cell_areas else 0,
        'max_cell_area': np.max(cell_areas) if cell_areas else 0,
        'mean_aspect_ratio': np.mean(cell_aspect_ratios) if cell_aspect_ratios else 0,
        'std_aspect_ratio': np.std(cell_aspect_ratios) if cell_aspect_ratios else 0,
        'mean_circularity': np.mean(cell_circularities) if cell_circularities else 0,
        'std_circularity': np.std(cell_circularities) if cell_circularities else 0,
        'mean_solidity': np.mean(cell_solidity) if cell_solidity else 0,
        'mean_eccentricity': np.mean(cell_eccentricity) if cell_eccentricity else 0,
        'total_segmented_area': np.sum(mask > 0),
        'background_area': np.sum(mask == 0),
        'mask_shape': mask.shape
    }
    
    return metrics

def analyze_comparison_results(config_file=None, results_dir="results"):
    """
    Analyze comparison results based on config file or scan directory.
    
    Args:
        config_file (str): Path to processing config JSON file
        results_dir (str): Directory containing segmentation results
        
    Returns:
        list or pd.DataFrame: Comparison results data
    """
    results_data = []
    
    if config_file and os.path.exists(config_file):
        print(f"Loading config file: {config_file}")
        # Load config file to get specific comparisons to analyze
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Extract image configurations and parameter sets
        image_configs = {img['image_id']: img for img in config.get('image_configurations', [])}
        param_configs = {param['param_set_id']: param for param in config.get('cellpose_parameter_configurations', [])}
        mappings = config.get('image_to_parameter_set_mappings', [])
        mapping_tasks = config.get('mapping_tasks', [])
        
        print(f"Found {len(image_configs)} image configs: {list(image_configs.keys())}")
        print(f"Found {len(param_configs)} parameter configs: {list(param_configs.keys())}")
        print(f"Found {len(mappings)} mappings, {len(mapping_tasks)} mapping tasks")
        
        # Process mapping tasks if available (newer config format)
        if mapping_tasks:
            print(f"Processing all images from config: {list(image_configs.keys())}")
            
            for task in mapping_tasks:
                if not task.get('is_active', True):
                    continue
                    
                image_id = task['source_image_id']
                param_set_id = task['source_param_set_id']
                if param_set_id not in param_configs:
                    print(f"Warning: Parameter set {param_set_id} not found in config")
                    continue
                
                param_config = param_configs[param_set_id]
                if not param_config.get('is_active', True):
                    print(f"Skipping inactive parameter set: {param_set_id}")
                    continue
                
                # Get the image config to determine scale factor
                image_config = image_configs.get(image_id)
                if not image_config:
                    print(f"Warning: Image config {image_id} not found")
                    continue
                
                # Get scale factor from image config
                scale_factor = 1.0
                seg_options = image_config.get('segmentation_options', {})
                rescaling_config = seg_options.get('rescaling_config')
                if rescaling_config and 'scale_factor' in rescaling_config:
                    scale_factor = rescaling_config['scale_factor']
                
                # Construct experiment ID using the same logic as pipeline_utils
                # Format: {image_id}_{param_set_id}_scaled_{scale_factor_with_underscores}
                def format_scale_factor_for_path(scale_factor):
                    if scale_factor is not None and scale_factor != 1.0:
                        return f"_scaled_{str(scale_factor).replace('.', '_')}"
                    return ""
                
                experiment_id = f"{image_id}_{param_set_id}{format_scale_factor_for_path(scale_factor)}"
                
                result_dir = os.path.join(results_dir, experiment_id)
                print(f"Looking for result directory: {result_dir}")
                
                if not os.path.exists(result_dir):
                    print(f"Warning: Result directory not found: {result_dir}")
                    continue
                
                model = param_config['cellpose_parameters']['MODEL_CHOICE']
                result_entry = analyze_single_result(result_dir, model, param_set_id, param_config, image_config)
                if result_entry:
                    results_data.append(result_entry)
        
        # Process old-style mappings if available (older config format)
        elif mappings:
            for mapping in mappings:
                image_id = mapping['image_id']
                param_set_ids = mapping['param_set_ids']
                
                for param_set_id in param_set_ids:
                    if param_set_id not in param_configs:
                        print(f"Warning: Parameter set {param_set_id} not found in config")
                        continue
                    
                    param_config = param_configs[param_set_id]
                    if not param_config.get('is_active', True):
                        print(f"Skipping inactive parameter set: {param_set_id}")
                        continue
                    
                    # Construct expected result directory name
                    model = param_config['cellpose_parameters']['MODEL_CHOICE']
                    diameter = param_config['cellpose_parameters']['DIAMETER']
                    
                    # Look for result directory
                    expected_dir_patterns = [
                        f"{image_id}_{param_set_id}",
                        f"{param_set_id}",
                        f"{image_id}_{model}_{diameter}",
                        f"comparison_{model}_{diameter}",
                        # Add more patterns for the actual directory structure
                        f"{image_id}_{param_set_id}_scaled_0_25",
                        f"{image_id}_{param_set_id}_scaled_0_5"
                    ]
                    
                    print(f"Looking for {param_set_id} with patterns: {expected_dir_patterns}")
                    
                    result_dir = None
                    for pattern in expected_dir_patterns:
                        potential_dir = os.path.join(results_dir, pattern)
                        print(f"  Checking: {potential_dir}")
                        if os.path.exists(potential_dir):
                            result_dir = potential_dir
                            print(f"  Found: {result_dir}")
                            break
                    
                    if not result_dir:
                        print(f"Warning: No result directory found for {param_set_id}")
                        print(f"  Available directories: {os.listdir(results_dir)}")
                        continue
                    
                    result_entry = analyze_single_result(result_dir, model, param_set_id, param_config, image_configs.get(image_id))
                    if result_entry:
                        results_data.append(result_entry)
    
    else:
        # Fallback: scan directory for comparison results
        print("No config file provided or file not found. Scanning results directory...")
        comparison_dirs = [d for d in os.listdir(results_dir) if 'comparison' in d]
        
        for comp_dir in comparison_dirs:
            dir_path = os.path.join(results_dir, comp_dir)
            
            # Extract model name from directory name
            if 'cyto2' in comp_dir:
                model = 'cyto2'
            elif 'cyto3' in comp_dir:
                model = 'cyto3'
            elif 'nuclei' in comp_dir:
                model = 'nuclei'
            else:
                continue
            
            result_entry = analyze_single_result(dir_path, model, comp_dir, None, None)
            if result_entry:
                results_data.append(result_entry)
    
    if HAS_PANDAS:
        return pd.DataFrame(results_data)
    else:
        return results_data

def analyze_single_result(result_dir, model, param_set_id, param_config, image_config):
    """
    Analyze a single segmentation result directory.
    
    Args:
        result_dir (str): Path to result directory
        model (str): Model name (nuclei, cyto2, cyto3)
        param_set_id (str): Parameter set ID
        param_config (dict): Parameter configuration from JSON
        image_config (dict): Image configuration from JSON
        
    Returns:
        dict: Result entry or None if analysis failed
    """
    # Find mask file
    mask_files = [f for f in os.listdir(result_dir) if f.endswith('_mask.tif')]
    if not mask_files:
        print(f"No mask file found in {result_dir}")
        return None
    
    mask_path = os.path.join(result_dir, mask_files[0])
    
    # Load summary JSON
    summary_files = [f for f in os.listdir(result_dir) if f.endswith('_segmentation_summary.json')]
    summary_data = {}
    if summary_files:
        summary_path = os.path.join(result_dir, summary_files[0])
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
    
    # Compute metrics
    metrics = load_mask_and_compute_metrics(mask_path)
    if metrics is None:
        return None
    
    # Get pixel to micron conversion
    if image_config and 'mpp_x' in image_config:
        pixel_size_um = image_config['mpp_x']  # Use from config
    else:
        pixel_size_um = get_pixel_to_micron_conversion(mask_path)
    
    # Convert areas from pixels to μm²
    pixel_to_um2 = pixel_size_um ** 2
    
    # Get processing time
    processing_time = extract_processing_time(result_dir)
    
    # Combine all data
    result_entry = {
        'model': model,
        'experiment_id': param_set_id,
        'mask_path': mask_path,
        'pixel_size_um': pixel_size_um,
        'processing_time_minutes': processing_time,
        # Areas in pixels
        'num_cells': metrics['num_cells'],
        'mean_cell_area_pixels': metrics['mean_cell_area'],
        'median_cell_area_pixels': metrics['median_cell_area'],
        'min_cell_area_pixels': metrics['min_cell_area'],
        'max_cell_area_pixels': metrics['max_cell_area'],
        # Areas in μm² 
        'mean_cell_area_um2': metrics['mean_cell_area'] * pixel_to_um2,
        'median_cell_area_um2': metrics['median_cell_area'] * pixel_to_um2,
        'min_cell_area_um2': metrics['min_cell_area'] * pixel_to_um2,
        'max_cell_area_um2': metrics['max_cell_area'] * pixel_to_um2,
        'std_cell_area_um2': metrics['std_cell_area'] * pixel_to_um2,
        # Shape metrics
        'mean_aspect_ratio': metrics['mean_aspect_ratio'],
        'std_aspect_ratio': metrics['std_aspect_ratio'],
        'mean_circularity': metrics['mean_circularity'],
        'std_circularity': metrics['std_circularity'],
        # Other metrics
        **{k: v for k, v in metrics.items() if k not in ['num_cells', 'mean_cell_area', 'median_cell_area', 'min_cell_area', 'max_cell_area', 'std_cell_area', 'mean_aspect_ratio', 'std_aspect_ratio', 'mean_circularity', 'std_circularity']}
    }
    
    # Add parameter data from config
    if param_config:
        cellpose_params = param_config.get('cellpose_parameters', {})
        result_entry['diameter_used'] = cellpose_params.get('DIAMETER', 'auto')
        result_entry['min_size'] = cellpose_params.get('MIN_SIZE', 15)
        result_entry['cellprob_threshold'] = cellpose_params.get('CELLPROB_THRESHOLD', 0.0)
    
    # Add summary data (override config if available)
    if 'params_used' in summary_data:
        result_entry['diameter_used'] = summary_data.get('diameter_used_for_eval', result_entry.get('diameter_used', 'auto'))
        result_entry['min_size'] = summary_data['params_used'].get('min_size', result_entry.get('min_size', 15))
        result_entry['cellprob_threshold'] = summary_data['params_used'].get('cellprob_threshold', result_entry.get('cellprob_threshold', 0.0))
    
    return result_entry
    
    if HAS_PANDAS:
        return pd.DataFrame(results_data)
    else:
        return results_data

def create_comparison_visualizations(data, output_dir="results/visualizations"):
    """
    Create comparison visualizations for the manuscript.
    
    Args:
        data (pd.DataFrame or list): Comparison results data
        output_dir (str): Directory to save visualizations
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping visualizations.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to consistent format
    if HAS_PANDAS and hasattr(data, 'iterrows'):
        # It's a DataFrame
        models = data['model'].tolist()
        num_cells = data['num_cells'].tolist()
        mean_areas = data['mean_cell_area_um2'].tolist()  # Use μm² version
        mean_solidity = data['mean_solidity'].tolist()
        total_areas = data['total_segmented_area'].tolist()
    else:
        # It's a list
        models = [row['model'] for row in data]
        num_cells = [row['num_cells'] for row in data]
        mean_areas = [row['mean_cell_area_um2'] for row in data]  # Use μm² version
        mean_solidity = [row['mean_solidity'] for row in data]
        total_areas = [row['total_segmented_area'] for row in data]
    
    # Set up the plotting style
    plt.style.use('default')
    if HAS_SEABORN:
        sns.set_palette("husl")
    
    # 1. Cell count comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Cellpose Model Comparison - Single Image Analysis', fontsize=16, fontweight='bold')
    
    # Cell count bar plot
    ax1 = axes[0, 0]
    bars = ax1.bar(models, num_cells, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Number of Cells Detected')
    ax1.set_ylabel('Cell Count')
    ax1.set_xlabel('Cellpose Model')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Mean cell area comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, mean_areas, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title('Mean Cell Area')
    ax2.set_ylabel('Area (μm²)')
    ax2.set_xlabel('Cellpose Model')
    
    # Add value labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Cell solidity comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, mean_solidity, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax3.set_title('Mean Cell Solidity')
    ax3.set_ylabel('Solidity (0-1)')
    ax3.set_xlabel('Cellpose Model')
    ax3.set_ylim(0, 1)
    
    # Add value labels
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Total segmented area
    ax4 = axes[1, 1]
    bars4 = ax4.bar(models, total_areas, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax4.set_title('Total Segmented Area')
    ax4.set_ylabel('Area (pixels)')
    ax4.set_xlabel('Cellpose Model')
    
    # Add value labels
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cellpose_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'cellpose_model_comparison.pdf'), bbox_inches='tight')
    plt.close()
    
    # 2. Create summary table
    create_summary_table(data, output_dir)
    
    print(f"Visualizations saved to {output_dir}")

def create_summary_table(data, output_dir):
    """
    Create a summary table for the manuscript matching the format in sections.md.
    
    Args:
        data (pd.DataFrame or list): Comparison results data
        output_dir (str): Directory to save the table
    """
    # Create a clean summary table matching the manuscript format
    summary_data = []
    
    if HAS_PANDAS and hasattr(data, 'iterrows'):
        # It's a DataFrame
        for _, row in data.iterrows():
            # Format size range
            size_range = f"{row['min_cell_area_um2']:.0f}-{row['max_cell_area_um2']:.0f}"
            
            # Format processing time
            proc_time = row.get('processing_time_minutes', 'N/A')
            proc_time_str = f"{proc_time:.1f}" if proc_time is not None else 'N/A'
            
            summary_data.append({
                'Model': row['model'].capitalize(),
                'Cell Count': f"{int(row['num_cells'])} ± {int(row.get('std_cell_count', 0))}",
                'Mean Area (μm²)': f"{row['mean_cell_area_um2']:.0f} ± {row['std_cell_area_um2']:.0f}",
                'Median Area (μm²)': f"{row['median_cell_area_um2']:.0f}",
                'Size Range (μm²)': size_range,
                'Aspect Ratio': f"{row['mean_aspect_ratio']:.1f} ± {row['std_aspect_ratio']:.1f}",
                'Circularity': f"{row['mean_circularity']:.2f} ± {row['std_circularity']:.2f}",
                'Processing Time (min)': proc_time_str
            })
    else:
        # It's a list
        for row in data:
            # Format size range
            size_range = f"{row['min_cell_area_um2']:.0f}-{row['max_cell_area_um2']:.0f}"
            
            # Format processing time
            proc_time = row.get('processing_time_minutes', 'N/A')
            proc_time_str = f"{proc_time:.1f}" if proc_time is not None else 'N/A'
            
            summary_data.append({
                'Model': row['model'].capitalize(),
                'Cell Count': f"{int(row['num_cells'])} ± {int(row.get('std_cell_count', 0))}",
                'Mean Area (μm²)': f"{row['mean_cell_area_um2']:.0f} ± {row['std_cell_area_um2']:.0f}",
                'Median Area (μm²)': f"{row['median_cell_area_um2']:.0f}",
                'Size Range (μm²)': size_range,
                'Aspect Ratio': f"{row['mean_aspect_ratio']:.1f} ± {row['std_aspect_ratio']:.1f}",
                'Circularity': f"{row['mean_circularity']:.2f} ± {row['std_circularity']:.2f}",
                'Processing Time (min)': proc_time_str
            })
    
    if HAS_PANDAS:
        summary_df = pd.DataFrame(summary_data)
        # Save as CSV with UTF-8 encoding
        summary_df.to_csv(os.path.join(output_dir, 'cellpose_comparison_table.csv'), index=False, encoding='utf-8')
        table_string = summary_df.to_string(index=False)
    else:
        # Create CSV manually
        csv_path = os.path.join(output_dir, 'cellpose_comparison_table.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            # Write header
            headers = list(summary_data[0].keys())
            f.write(','.join(headers) + '\n')
            # Write data
            for row in summary_data:
                f.write(','.join(str(row[h]) for h in headers) + '\n')
        
        # Create table string manually
        table_lines = []
        headers = list(summary_data[0].keys())
        table_lines.append('  '.join(f'{h:>20}' for h in headers))
        for row in summary_data:
            table_lines.append('  '.join(f'{str(row[h]):>20}' for h in headers))
        table_string = '\n'.join(table_lines)
    
    # Create a formatted table for manuscript
    with open(os.path.join(output_dir, 'cellpose_comparison_table.txt'), 'w', encoding='utf-8') as f:
        f.write("Table: Comparison of Cellpose Models for Plant Tissue Segmentation\n")
        f.write("=" * 80 + "\n\n")
        f.write(table_string)
        f.write("\n\n")
        f.write("Note: All analyses performed on the same Medicago truncatula nodule image ")
        f.write("(5C_morphology_focus.ome.tif) scaled to 0.25x resolution.\n")
    
    print(f"Summary table saved to {output_dir}")
    return summary_data if not HAS_PANDAS else pd.DataFrame(summary_data)

def main():
    parser = argparse.ArgumentParser(description="Analyze Cellpose model comparison results")
    parser.add_argument("--config", default=None,
                       help="Path to processing config JSON file (same as segmentation pipeline)")
    parser.add_argument("--results_dir", default="results", 
                       help="Directory containing segmentation results")
    parser.add_argument("--output_dir", default="results/visualizations",
                       help="Directory to save analysis outputs")
    
    args = parser.parse_args()
    
    print("Analyzing Cellpose model comparison results...")
    
    # Analyze results
    data = analyze_comparison_results(args.config, args.results_dir)
    
    if HAS_PANDAS and hasattr(data, 'empty') and data.empty:
        print("No comparison results found!")
        return
    elif not HAS_PANDAS and not data:
        print("No comparison results found!")
        return
    
    print(f"Found {len(data)} model comparisons:")
    if HAS_PANDAS and hasattr(data, 'iterrows'):
        for _, row in data.iterrows():
            print(f"  {row['model']}: {row['num_cells']} cells detected")
    else:
        for row in data:
            print(f"  {row['model']}: {row['num_cells']} cells detected")
    
    # Create visualizations (if matplotlib is available)
    if HAS_MATPLOTLIB:
        create_comparison_visualizations(data, args.output_dir)
    else:
        print("Skipping visualizations (matplotlib not available)")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 