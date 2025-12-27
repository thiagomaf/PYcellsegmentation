#!/usr/bin/env python3
"""
Create a single overlay for a specific mask from a config file.
Usage: 
    python create_single_mask_overlay.py --config processing_config_grid_2.json --image_id IMAGE_ID --param_set_id PARAM_SET_ID
    python create_single_mask_overlay.py --config processing_config_grid_2.json --image_id IMAGE_ID --param_set_id PARAM_SET_ID --output overlay.png
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import cv2
    import numpy as np
    import tifffile
    from src.create_individual_overlays import (
        create_colored_overlay, 
        load_image, 
        find_background_image_path
    )
    from src.file_paths import RESCALED_IMAGE_CACHE_DIR, RESULTS_DIR_BASE, PROJECT_ROOT
    from src.pipeline_utils import resolve_image_path
    HAS_DEPS = True
except ImportError as e:
    print(f"Error: Missing dependencies: {e}")
    HAS_DEPS = False

def find_mask_from_config(config_path, image_id, param_set_id, results_dir=None):
    """Find mask file for a specific image_id and param_set_id from config."""
    if results_dir is None:
        results_dir = Path(RESULTS_DIR_BASE) if isinstance(RESULTS_DIR_BASE, (str, Path)) else Path(RESULTS_DIR_BASE)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Find image config
    image_configs = config.get('image_configurations', [])
    image_config = next((img for img in image_configs if img.get('image_id') == image_id), None)
    
    if not image_config:
        raise ValueError(f"Image ID '{image_id}' not found in config")
    
    # Get scale factor
    scale_factor = 1.0
    rescaling_config = image_config.get('segmentation_options', {}).get('rescaling_config')
    if rescaling_config:
        scale_factor = rescaling_config.get('scale_factor', 1.0)
    
    # Construct experiment ID
    scale_str = f"_scaled_{str(scale_factor).replace('.', '_')}" if scale_factor != 1.0 else ""
    experiment_id = f"{image_id}_{param_set_id}{scale_str}"
    
    # Find mask directory
    mask_dir = Path(results_dir) / experiment_id
    if not mask_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {mask_dir}")
    
    # Find mask file
    mask_files = list(mask_dir.glob("*_mask.tif"))
    if not mask_files:
        # Try alternative patterns
        mask_files = list(mask_dir.glob("*.tif"))
        mask_files = [m for m in mask_files if 'mask' in m.name.lower()]
    
    if not mask_files:
        raise FileNotFoundError(f"No mask file found in {mask_dir}")
    
    return mask_files[0], image_config, scale_factor

def main():
    parser = argparse.ArgumentParser(
        description="Create overlay for a specific mask from config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create overlay for specific image and param set
  python create_single_mask_overlay.py --config processing_config_grid_2.json --image_id IMAGE_ID --param_set_id PARAM_SET_ID
  
  # With custom output and color
  python create_single_mask_overlay.py --config processing_config_grid_2.json --image_id IMAGE_ID --param_set_id PARAM_SET_ID --output my_overlay.png --color "#FF0000" --alpha 0.5
  
  # List available image IDs and param sets
  python create_single_mask_overlay.py --config processing_config_grid_2.json --list
        """
    )
    
    parser.add_argument("--config", required=True, help="Path to processing config JSON file")
    parser.add_argument("--image_id", help="Image ID from config (required unless --list)")
    parser.add_argument("--param_set_id", help="Parameter set ID from config (required unless --list)")
    parser.add_argument("--output", help="Output overlay image path (default: {image_id}_{param_set_id}_overlay.png)")
    parser.add_argument("--results_dir", help="Results directory (default: from config)")
    parser.add_argument("--color", default="multi", help='Color mode: "multi" or hex like "#FF0000"')
    parser.add_argument("--alpha", type=float, default=0.4, help="Transparency (0.0-1.0)")
    parser.add_argument("--force_8bit", action="store_true", help="Use robust 8-bit conversion")
    parser.add_argument("--list", action="store_true", help="List available image IDs and param sets")
    
    args = parser.parse_args()
    
    if not HAS_DEPS:
        return 1
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # List mode
    if args.list:
        print("\nAvailable Image IDs:")
        print("-" * 50)
        image_configs = config.get('image_configurations', [])
        for img in image_configs:
            img_id = img.get('image_id', 'N/A')
            is_active = img.get('is_active', True)
            status = "✓" if is_active else "✗"
            print(f"  {status} {img_id}")
        
        print("\nAvailable Parameter Set IDs:")
        print("-" * 50)
        param_configs = config.get('cellpose_parameter_configurations', [])
        for param in param_configs:
            param_id = param.get('param_set_id', 'N/A')
            is_active = param.get('is_active', True)
            status = "✓" if is_active else "✗"
            print(f"  {status} {param_id}")
        print()
        return 0
    
    # Validate required args
    if not args.image_id or not args.param_set_id:
        print("Error: --image_id and --param_set_id required (or use --list to see available options)")
        parser.print_help()
        return 1
    
    try:
        # Find mask and image config
        results_dir = Path(args.results_dir) if args.results_dir else None
        mask_path, image_config, scale_factor = find_mask_from_config(
            config_path, args.image_id, args.param_set_id, results_dir
        )
        
        print(f"Found mask: {mask_path}")
        
        # Find background image
        background_image_path = find_background_image_path(image_config, args.image_id)
        
        # Resolve the path using the same logic as the pipeline
        bg_path_str = str(background_image_path)
        resolved_path = resolve_image_path(bg_path_str, str(project_root))
        background_image_path = Path(resolved_path)
        
        # If the resolved path doesn't exist, try treating the original path as relative
        # This handles cases where paths like "\data\..." are incorrectly treated as absolute
        if not background_image_path.exists():
            # Check if original path starts with backslash (Windows absolute to drive root)
            # but should be treated as relative to project root
            if bg_path_str.startswith('\\') or bg_path_str.startswith('/'):
                # Try as relative path (remove leading slash)
                relative_path = bg_path_str.lstrip('\\/')
                candidate = project_root / relative_path
                if candidate.exists():
                    background_image_path = candidate
                    print(f"  Resolved as relative path: {background_image_path}")
                else:
                    print(f"Error: Background image not found: {background_image_path}")
                    print(f"  Also tried: {candidate}")
                    print(f"  Original path from config: {bg_path_str}")
                    return 1
            else:
                print(f"Error: Background image not found: {background_image_path}")
                print(f"  Original path from config: {bg_path_str}")
                return 1
        
        print(f"Using background image: {background_image_path}")
        
        # Load images
        print("Loading images...")
        background_image = load_image(background_image_path)
        mask = load_image(mask_path)
        
        if background_image is None or mask is None:
            print("Error: Failed to load image or mask")
            return 1
        
        # Create overlay
        print("Creating overlay...")
        overlay = create_colored_overlay(
            background_image, mask,
            color_mode=args.color,
            alpha=args.alpha,
            force_8bit_conversion=args.force_8bit
        )
        
        if overlay is None:
            print("Error: Failed to create overlay")
            return 1
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(f"{args.image_id}_{args.param_set_id}_overlay.png")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save overlay (convert RGB to BGR for OpenCV)
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"\n✓ Overlay saved to: {output_path.resolve()}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())