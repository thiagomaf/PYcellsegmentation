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

def load_mask_basic_metrics(mask_path):
    """
    Load a segmentation mask and compute basic metrics without external dependencies.
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
    
    metrics = {
        'num_cells': num_cells,
        'mean_cell_area': np.mean(cell_areas) if cell_areas else 0,
        'median_cell_area': np.median(cell_areas) if cell_areas else 0,
        'std_cell_area': np.std(cell_areas) if cell_areas else 0,
        'min_cell_area': np.min(cell_areas) if cell_areas else 0,
        'max_cell_area': np.max(cell_areas) if cell_areas else 0,
        'total_segmented_area': np.sum(mask > 0),
        'mask_shape': mask.shape
    }
    
    return metrics

def analyze_comparison_results(results_dir="results", filter_pattern="comparison", param_set_ids=None):
    """
    Analyze comparison results and print summary.
    
    Args:
        results_dir (str): Directory containing segmentation results
        filter_pattern (str): Pattern to filter result directories (used if param_set_ids is None)
        param_set_ids (list): Specific parameter set IDs to look for in directory names
    """
    if param_set_ids:
        # Look for directories containing any of the parameter set IDs
        comparison_dirs = []
        for d in os.listdir(results_dir):
            if any(param_id in d for param_id in param_set_ids):
                comparison_dirs.append(d)
    else:
        # Use the filter pattern
        comparison_dirs = [d for d in os.listdir(results_dir) if filter_pattern in d]
    
    results_data = []
    
    for comp_dir in comparison_dirs:
        dir_path = os.path.join(results_dir, comp_dir)
        
        # Extract model name and parameter info from directory name
        if 'cyto2' in comp_dir:
            model = 'cyto2'
        elif 'cyto3' in comp_dir:
            model = 'cyto3'
        elif 'nuclei' in comp_dir:
            # For nuclei comparisons, extract the specific parameter set
            if 'nuclei_1' in comp_dir:
                model = 'nuclei_diam_40'
            elif 'nuclei_2' in comp_dir:
                model = 'nuclei_diam_30'
            elif 'nuclei_3' in comp_dir:
                model = 'nuclei_diam_20'
            else:
                model = 'nuclei'
        else:
            # Try to extract model from parameter set ID in directory name
            for model_type in ['cyto', 'nuclei', 'livecell']:
                if model_type in comp_dir.lower():
                    model = model_type
                    break
            else:
                # If no recognizable model type, use a cleaned directory name
                model = comp_dir.replace('5A_', '').replace('_scaled_0_5', '')
                if not model:
                    continue
        
        # Find mask file
        mask_files = [f for f in os.listdir(dir_path) if f.endswith('_mask.tif')]
        if not mask_files:
            print(f"No mask file found in {dir_path}")
            continue
        
        mask_path = os.path.join(dir_path, mask_files[0])
        
        # Load summary JSON
        summary_files = [f for f in os.listdir(dir_path) if f.endswith('_segmentation_summary.json')]
        summary_data = {}
        if summary_files:
            summary_path = os.path.join(dir_path, summary_files[0])
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
        
        # Compute metrics
        metrics = load_mask_basic_metrics(mask_path)
        if metrics is None:
            continue
        
        # Combine all data
        result_entry = {
            'model': model,
            'experiment_id': comp_dir,
            'mask_path': mask_path,
            **metrics
        }
        
        # Add summary data
        if 'params_used' in summary_data:
            result_entry['diameter_used'] = summary_data.get('diameter_used_for_eval', 'auto')
            result_entry['min_size'] = summary_data['params_used'].get('min_size', 15)
            result_entry['cellprob_threshold'] = summary_data['params_used'].get('cellprob_threshold', 0.0)
        
        results_data.append(result_entry)
    
    return results_data

def print_comparison_table(data):
    """
    Print a formatted comparison table.
    """
    print("\n" + "="*80)
    print("CELLPOSE MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Print header
    print(f"{'Model':<10} {'Cells':<8} {'Mean Area':<12} {'Std Area':<12} {'Total Area':<12} {'Diameter':<10}")
    print("-" * 80)
    
    # Print data rows
    for row in data:
        print(f"{row['model']:<10} "
              f"{row['num_cells']:<8} "
              f"{row['mean_cell_area']:<12.1f} "
              f"{row['std_cell_area']:<12.1f} "
              f"{row['total_segmented_area']:<12} "
              f"{str(row.get('diameter_used', 'auto')):<10}")
    
    print("\n" + "="*80)
    print("DETAILED ANALYSIS:")
    print("="*80)
    
    for row in data:
        print(f"\n{row['model'].upper()} MODEL:")
        print(f"  - Cells detected: {row['num_cells']}")
        print(f"  - Mean cell area: {row['mean_cell_area']:.1f} pixels")
        print(f"  - Cell area range: {row['min_cell_area']:.0f} - {row['max_cell_area']:.0f} pixels")
        print(f"  - Total segmented area: {row['total_segmented_area']} pixels")
        print(f"  - Image dimensions: {row['mask_shape']}")
        print(f"  - Diameter used: {row.get('diameter_used', 'auto')}")
        print(f"  - Min size threshold: {row.get('min_size', 'N/A')}")

def save_results_csv(data, output_path="results/cellpose_comparison_results.csv"):
    """
    Save results to CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write header
        headers = ['Model', 'Cells_Detected', 'Mean_Cell_Area', 'Std_Cell_Area', 
                  'Min_Cell_Area', 'Max_Cell_Area', 'Total_Segmented_Area', 
                  'Diameter_Used', 'Min_Size_Threshold']
        f.write(','.join(headers) + '\n')
        
        # Write data
        for row in data:
            values = [
                row['model'],
                str(row['num_cells']),
                f"{row['mean_cell_area']:.1f}",
                f"{row['std_cell_area']:.1f}",
                str(row['min_cell_area']),
                str(row['max_cell_area']),
                str(row['total_segmented_area']),
                str(row.get('diameter_used', 'auto')),
                str(row.get('min_size', 'N/A'))
            ]
            f.write(','.join(values) + '\n')
    
    print(f"\nResults saved to: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Cellpose model comparison results")
    parser.add_argument("--config", default=None,
                       help="Path to JSON config file to analyze results for (e.g., config/processing_config_comparison.json)")
    parser.add_argument("--results_dir", default="results", 
                       help="Directory containing segmentation results (default: results)")
    parser.add_argument("--filter_pattern", default="comparison",
                       help="Pattern to filter result directories (used if --config not provided)")
    parser.add_argument("--output_csv", default=None,
                       help="Output CSV file path (default: auto-generated based on config)")
    
    args = parser.parse_args()
    
    print(f"Analyzing Cellpose model comparison results...")
    print(f"Results directory: {args.results_dir}")
    
    # Determine what to analyze
    if args.config:
        # Load config file and get parameter set IDs
        param_set_ids, config_name = load_config_and_get_param_sets(args.config)
        if not param_set_ids:
            print(f"No active parameter sets found in config: {args.config}")
            return
        
        print(f"Config file: {args.config}")
        print(f"Parameter sets to analyze: {param_set_ids}")
        
        # Analyze results based on parameter set IDs
        data = analyze_comparison_results(args.results_dir, param_set_ids=param_set_ids)
        
        # Generate output filename based on config
        if args.output_csv:
            output_path = args.output_csv
        else:
            output_path = f"results/{config_name}_analysis_results.csv"
    else:
        # Use filter pattern approach
        print(f"Filter pattern: {args.filter_pattern}")
        data = analyze_comparison_results(args.results_dir, args.filter_pattern)
        
        if args.output_csv:
            output_path = args.output_csv
        else:
            output_path = "results/cellpose_comparison_results.csv"
    
    if not data:
        print("No comparison results found!")
        if args.config:
            print(f"Searched for directories containing parameter set IDs: {param_set_ids}")
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