import os

# Determine project root assuming this file (file_paths.py) is in the src directory.
# This means PROJECT_ROOT will be the parent directory of src/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR_BASE_NAME = "images"
RESULTS_DIR_BASE_NAME = "results"

# Base directories relative to PROJECT_ROOT
IMAGE_DIR_BASE = os.path.join(PROJECT_ROOT, IMAGE_DIR_BASE_NAME)
RESCALED_IMAGE_CACHE_DIR = os.path.join(IMAGE_DIR_BASE, "rescaled_cache")
TILED_IMAGE_OUTPUT_BASE = os.path.join(IMAGE_DIR_BASE, "tiled_outputs")
RESULTS_DIR_BASE = os.path.join(PROJECT_ROOT, RESULTS_DIR_BASE_NAME)

# Exported names for clarity when importing
__all__ = [
    "PROJECT_ROOT",
    "IMAGE_DIR_BASE",
    "RESCALED_IMAGE_CACHE_DIR",
    "TILED_IMAGE_OUTPUT_BASE",
    "RESULTS_DIR_BASE"
] 