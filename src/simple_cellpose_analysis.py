#!/usr/bin/env python3
"""
Simplified script to analyze Cellpose model comparison results.
Extracts basic metrics without plotting to avoid GUI dependencies.
"""

import os
import json
import numpy as np

def load_config_and_get_param_sets(config_path):
    """
    Load config file and extract parameter set IDs to determine what results to analyze.
    
    Args:
        config_path (str): Path to the JSON config file
        
    Returns:
        tuple: (param_set_ids, config_name)
    """
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return [], ""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract parameter set IDs
    param_sets = config.get("cellpose_parameter_configurations", [])
    param_set_ids = [ps["param_set_id"] for ps in param_sets if ps.get("is_active", True)]
    
    # Extract config name from filename
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    
    return param_set_ids, config_name

def load_mask_basic_metrics(mask_path, mpp=None):
    """
    Load a segmentation mask and compute basic metrics without external dependencies.
    Converts area to µm² if mpp is provided.
    """
    if not os.path.exists(mask_path):
        print(f"Warning: Mask file not found: {mask_path}")
        return None
    
    try:
        # Try to load with tifffile first
        try:
            import tifffile
            mask = tifffile.imread(mask_path)
        except ImportError:
            # Fallback to PIL
            from PIL import Image
            mask = np.array(Image.open(mask_path))
    except Exception as e:
        print(f"Error loading {mask_path}: {e}")
        return None
    
    # Basic metrics
    unique_labels = np.unique(mask)
    num_cells = len(unique_labels) - 1  # Exclude background (label 0)
    
    # Cell area statistics
    cell_areas = []
    for label in unique_labels[1:]:  # Skip background
        cell_mask = (mask == label)
        area = np.sum(cell_mask)
        cell_areas.append(area)
    
    # Convert areas to µm² if mpp is available
    units = "pixels"
    if mpp and mpp > 0 and cell_areas:
        px_to_um2 = mpp * mpp
        cell_areas = [area * px_to_um2 for area in cell_areas]
        units = "µm²"

    metrics = {
        'num_cells': num_cells,
        'mean_cell_area': np.mean(cell_areas) if cell_areas else 0,
        'median_cell_area': np.median(cell_areas) if cell_areas else 0,
        'std_cell_area': np.std(cell_areas) if cell_areas else 0,
        'min_cell_area': np.min(cell_areas) if cell_areas else 0,
        'max_cell_area': np.max(cell_areas) if cell_areas else 0,
        'total_segmented_area': np.sum(cell_areas) if cell_areas else 0,
        'mask_shape': mask.shape,
        'units': units
    }
    
    return metrics

def analyze_comparison_results(results_dir="results", config=None, filter_pattern="comparison"):
    """
    Analyze comparison results and print summary.
    If a config is provided, it will be used to find results and convert units to µm².
    
    Args:
        results_dir (str): Directory containing segmentation results
        config (dict): Loaded processing config JSON.
        filter_pattern (str): Pattern to filter result directories (used if config is None)
    """
    results_data = []

    if config:
        # --- Config-driven approach (more robust and enables unit conversion) ---
        image_configs = {img['image_id']: img for img in config.get('image_configurations', []) if img.get('is_active', True)}
        param_configs = {param['param_set_id']: param for param in config.get('cellpose_parameter_configurations', []) if param.get('is_active', True)}

        for image_id, image_config in image_configs.items():
            for param_set_id, param_config in param_configs.items():
                
                # Determine scale factor to find correct directory and effective MPP
                scale_factor = 1.0
                rescaling_config = image_config.get('segmentation_options', {}).get('rescaling_config')
                if rescaling_config:
                    scale_factor = rescaling_config.get('scale_factor', 1.0)
                
                scale_str = f"_scaled_{str(scale_factor).replace('.', '_')}" if scale_factor != 1.0 else ""
                experiment_id = f"{image_id}_{param_set_id}{scale_str}"
                result_dir = os.path.join(results_dir, experiment_id)

                if not os.path.exists(result_dir):
                    continue
                
                # Calculate effective MPP for unit conversion
                original_mpp = image_config.get('mpp_x')
                effective_mpp = original_mpp / scale_factor if original_mpp and scale_factor != 0 else None

                # Find mask file
                mask_files = [f for f in os.listdir(result_dir) if f.endswith('_mask.tif')]
                if not mask_files:
                    print(f"No mask file found in {result_dir}")
                    continue
                
                mask_path = os.path.join(result_dir, mask_files[0])
                
                # Compute metrics (will be in µm² if effective_mpp is valid)
                metrics = load_mask_basic_metrics(mask_path, mpp=effective_mpp)
                if metrics is None:
                    continue
                
                model_name = param_config['cellpose_parameters']['MODEL_CHOICE']
                result_entry = {
                    'model': model_name,
                    'experiment_id': experiment_id,
                    'mask_path': mask_path,
                    **metrics
                }
                results_data.append(result_entry)
        
    else:
        # --- Fallback: scan directory with filter pattern (pixel units only) ---
        comparison_dirs = [d for d in os.listdir(results_dir) if filter_pattern in d]
        
        for comp_dir in comparison_dirs:
            dir_path = os.path.join(results_dir, comp_dir)
            
            # Extract model name
            model = comp_dir # Simplified model name extraction
            for m in ['cyto2', 'cyto3', 'nuclei']:
                if m in comp_dir:
                    model = m
                    break
            
            mask_files = [f for f in os.listdir(dir_path) if f.endswith('_mask.tif')]
            if not mask_files:
                continue
            
            mask_path = os.path.join(dir_path, mask_files[0])
            metrics = load_mask_basic_metrics(mask_path, mpp=None) # No MPP available
            if metrics is None:
                continue
            
            summary_files = [f for f in os.listdir(dir_path) if f.endswith('_segmentation_summary.json')]
            summary_data = {}
            if summary_files:
                with open(os.path.join(dir_path, summary_files[0]), 'r') as f:
                    summary_data = json.load(f)

            result_entry = { 'model': model, 'experiment_id': comp_dir, 'mask_path': mask_path, **metrics }
            if 'params_used' in summary_data:
                result_entry['diameter_used'] = summary_data.get('diameter_used_for_eval', 'auto')
                result_entry['min_size'] = summary_data['params_used'].get('min_size', 15)
            results_data.append(result_entry)

    return results_data

def print_comparison_table(data):
    """
    Print a formatted comparison table.
    """
    if not data:
        return

    print("\n" + "="*80)
    print("CELLPOSE MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Determine units from the first data entry
    units = data[0].get('units', 'pixels')
    header_unit_str = f"({units})"
    
    # Print header
    print(f"{'Model':<20} {'Cells':<8} {'Mean Area':<18} {'Std Area':<18} {'Total Area':<18} {'Diameter':<10}")
    print(f"{'':<20} {'':<8} {header_unit_str:<18} {header_unit_str:<18} {header_unit_str:<18} {'(px)':<10}")
    print("-" * 95)
    
    # Print data rows
    for row in data:
        print(f"{row['experiment_id']:<20} "
              f"{row['num_cells']:<8} "
              f"{row['mean_cell_area']:<18.1f} "
              f"{row['std_cell_area']:<18.1f} "
              f"{row['total_segmented_area']:<18.1f} "
              f"{str(row.get('diameter_used', 'auto')):<10}")
    
    print("\n" + "="*95)
    print("DETAILED ANALYSIS:")
    print("="*95)
    
    for row in data:
        units = row.get('units', 'pixels')
        print(f"\n{row['experiment_id'].upper()}:")
        print(f"  - Cells detected: {row['num_cells']}")
        print(f"  - Mean cell area: {row['mean_cell_area']:.1f} {units}")
        print(f"  - Cell area range: {row['min_cell_area']:.1f} - {row['max_cell_area']:.1f} {units}")
        print(f"  - Total segmented area: {row['total_segmented_area']:.1f} {units}")
        print(f"  - Image dimensions: {row['mask_shape']}")
        print(f"  - Diameter used: {row.get('diameter_used', 'auto')} px")
        print(f"  - Min size threshold: {row.get('min_size', 'N/A')} px")

def save_results_csv(data, output_path="results/cellpose_comparison_results.csv"):
    """
    Save results to CSV file.
    """
    if not data:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    units = data[0].get('units', 'pixels')
    
    with open(output_path, 'w') as f:
        # Write header
        headers = ['Experiment_ID', 'Model', 'Cells_Detected', f'Mean_Cell_Area_({units})', f'Std_Cell_Area_({units})', 
                  f'Min_Cell_Area_({units})', f'Max_Cell_Area_({units})', f'Total_Segmented_Area_({units})', 
                  'Diameter_Used_(px)', 'Min_Size_Threshold_(px)']
        f.write(','.join(headers) + '\n')
        
        # Write data
        for row in data:
            values = [
                row['experiment_id'],
                row['model'],
                str(row['num_cells']),
                f"{row['mean_cell_area']:.2f}",
                f"{row['std_cell_area']:.2f}",
                f"{row['min_cell_area']:.2f}",
                f"{row['max_cell_area']:.2f}",
                f"{row['total_segmented_area']:.2f}",
                str(row.get('diameter_used', 'auto')),
                str(row.get('min_size', 'N/A'))
            ]
            f.write(','.join(values) + '\n')
    
    print(f"\nResults saved to: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Cellpose model comparison results")
    parser.add_argument("--config", default=None,
                       help="Path to JSON config file to analyze results for (e.g., config/processing_config_comparison.json). Enables unit conversion to µm².")
    parser.add_argument("--results_dir", default="results", 
                       help="Directory containing segmentation results (default: results)")
    parser.add_argument("--filter_pattern", default="comparison",
                       help="Pattern to filter result directories (used if --config not provided)")
    parser.add_argument("--output_csv", default=None,
                       help="Output CSV file path (default: auto-generated based on config)")
    
    args = parser.parse_args()
    
    print(f"Analyzing Cellpose model comparison results...")
    print(f"Results directory: {args.results_dir}")
    
    config = None
    output_path = args.output_csv

    # Determine what to analyze
    if args.config:
        # Load config file and get parameter set IDs
        print(f"Loading config file: {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Analyze results based on parameter set IDs
        data = analyze_comparison_results(args.results_dir, config=config)
        
        # Generate output filename based on config
        if not output_path:
            config_name = os.path.splitext(os.path.basename(args.config))[0]
            output_path = f"results/{config_name}_analysis_results.csv"
    else:
        # Use filter pattern approach
        print(f"Filter pattern: {args.filter_pattern}. Units will be in pixels.")
        data = analyze_comparison_results(args.results_dir, config=None, filter_pattern=args.filter_pattern)
        
        if not output_path:
            output_path = "results/cellpose_comparison_results.csv"
    
    if not data:
        print("No comparison results found!")
        if args.config:
            print(f"Searched for result directories based on active configurations in {args.config}")
        else:
            print(f"Searched for directories containing '{args.filter_pattern}' in {args.results_dir}")
        return
    
    print(f"Found {len(data)} model comparisons")
    
    # Print comparison table
    print_comparison_table(data)
    
    # Save results
    save_results_csv(data, output_path)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 