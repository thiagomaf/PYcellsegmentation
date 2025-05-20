import os

# Determine project root assuming this file (file_paths.py) is in the src directory.
# This means PROJECT_ROOT will be the parent directory of src/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR_BASE_NAME = "images"
RESULTS_DIR_RELATIVE_TO_PROJECT = "results"
DEFAULT_CONFIG_FILENAME = "processing_config.json"
VISUALIZATION_CONFIG_FILENAME = "visualization_config.json"
PROCESSING_CONFIG_FILENAME = "processing_config.json"

# Base directories relative to PROJECT_ROOT
IMAGE_DIR_BASE = os.path.join(PROJECT_ROOT, IMAGE_DIR_BASE_NAME)
RESCALED_IMAGE_CACHE_DIR = os.path.join(IMAGE_DIR_BASE, "rescaled_cache")
TILED_IMAGE_OUTPUT_BASE = os.path.join(IMAGE_DIR_BASE, "tiled_outputs")
RESULTS_DIR_BASE = os.path.join(PROJECT_ROOT, RESULTS_DIR_RELATIVE_TO_PROJECT)

# Exported names for clarity when importing
__all__ = [
    "PROJECT_ROOT",
    "IMAGE_DIR_BASE_NAME",
    "IMAGE_DIR_BASE",
    "RESCALED_IMAGE_CACHE_DIR",
    "TILED_IMAGE_OUTPUT_BASE",
    "RESULTS_DIR_RELATIVE_TO_PROJECT",
    "RESULTS_DIR_BASE",
    "DEFAULT_CONFIG_FILENAME",
    "VISUALIZATION_CONFIG_FILENAME",
    "PROCESSING_CONFIG_FILENAME"
] 