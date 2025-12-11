# Google Colab Code Block - Batch Stitch Tiled Masks
# Copy this entire block into a Google Colab cell

import os
import sys

# Set environment variable to handle OpenMP warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Update this path to match your Google Drive mount point
project_root = '/content/drive/MyDrive/Github/PYcellsegmentation'

# Change to project directory
if os.path.exists(project_root):
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Working directory: {os.getcwd()}")
else:
    print(f"Warning: Project root not found at {project_root}")
    print("Please update the project_root path to match your Google Drive location")
    print("Current directory:", os.getcwd())
    raise FileNotFoundError(f"Project directory not found: {project_root}")

# Import and run the batch stitching script
try:
    from src.stitch_masks_batch import main
    print("=" * 80)
    print("Starting batch stitching of tiled masks...")
    print("=" * 80)
    main()
    print("\n" + "=" * 80)
    print("Batch stitching completed!")
    print("=" * 80)
except Exception as e:
    print(f"Error running batch stitching: {e}")
    import traceback
    traceback.print_exc()

