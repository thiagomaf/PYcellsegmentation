# Configuration Guide

This guide details how to configure the PYcellsegmentation pipeline. You can create configuration files automatically using the provided script or manually for custom setups.

## 1. Automated Configuration (Recommended)

The easiest way to generate configuration files is to use the `scripts/create_config_files.py` script. This tool scans your raw data directories, extracts image metadata (like DPI/resolution), and creates ready-to-use JSON files.

### Usage

1.  **Organize your data**: Ensure your raw images are in the correct directories (e.g., `data/raw/images/xenium_HE/xenium`, `data/raw/images/xenium_HE/HE`, etc.).
2.  **Run the script**:
    ```bash
    python scripts/create_config_files.py
    ```

### What it does
*   **Scans Directories**: Looks for `.tif` and `.ome.tif` files in the standard data folders.
*   **Extracts Metadata**: Reads OME-XML or TIFF tags to find DPI/resolution. If missing, it attempts to parse DPI from the filename (e.g., `image_14DPI.tif`).
*   **Calculates Parameters**:
    *   **MPP (Microns Per Pixel)**: Converted from DPI.
    *   **Scale Factor**: Calculates a downscaling factor to keep images within a manageable size (target ~275MB) for efficient processing.
*   **Generates JSONs**: Creates batch configuration files in `config/` (e.g., `config/xenium_HE/processing_config_xenium_14DPI_batch_1.json`).
*   **Tracking**: Updates `config/xenium_HE/PROCESSING_STATUS.md` to track which batches are pending/complete.

---

## 2. Configuration File Structure

The pipeline uses JSON files to control execution. The main file is `processing_config.json`.

### `processing_config.json`

This file controls the segmentation and mapping steps. It contains four main sections:

#### A. `global_segmentation_settings`
Global settings for the pipeline execution.
```json
"global_segmentation_settings": {
    "default_log_level": "INFO",      // Logging verbosity
    "max_processes": 1,               // Number of parallel processes
    "FORCE_GRAYSCALE": true,          // Convert RGB to grayscale before processing
    "USE_GPU_IF_AVAILABLE": true      // Use NVIDIA GPU if detected
}
```

#### B. `image_configurations`
A list of images to process.
```json
{
    "image_id": "sample_123",                 // Unique identifier for the image
    "original_image_filename": "images/sample_123.tif", // Path relative to project root
    "is_active": true,                        // Set to false to skip this image
    "mpp_x": 0.2125,                          // Microns per pixel (X axis)
    "mpp_y": 0.2125,                          // Microns per pixel (Y axis)
    "segmentation_options": {
        "apply_segmentation": true,           // Whether to run segmentation
        "rescaling_config": {
            "scale_factor": 0.5,              // Downscale image (0.5 = 50% size)
            "interpolation": "INTER_LINEAR"
        },
        "tiling_parameters": {
            "apply_tiling": false,            // Split large images into tiles?
            "tile_size": 2000,                // Size of tiles in pixels (if tiling)
            "overlap": 100                    // Overlap between tiles
        }
    }
}
```

#### C. `cellpose_parameter_configurations`
Defines the models and parameters to use. You can define multiple sets to compare results (e.g., `comparison_60` vs `comparison_90`).
```json
{
    "param_set_id": "standard_cyto",
    "is_active": true,
    "cellpose_parameters": {
        "MODEL_CHOICE": "cyto3",         // Model type: cyto3, nuclei, or path to custom model
        "DIAMETER": 60,                  // Expected cell diameter in pixels (BEFORE scaling)
        "MIN_SIZE": 15,                  // Minimum cell size in pixels
        "CELLPROB_THRESHOLD": 0.0,       // Threshold for cell probability
        "FORCE_GRAYSCALE": true,
        "Z_PROJECTION_METHOD": "max",    // How to flatten 3D stacks: max, mean, or none
        "CHANNEL_INDEX": 0               // Channel to segment (0 = first channel)
    }
}
```
**Note**: If you apply a `scale_factor` to an image, the pipeline automatically adjusts the `DIAMETER` passed to Cellpose. For example, if `DIAMETER` is 60 and `scale_factor` is 0.5, Cellpose will look for cells of diameter 30.

#### D. `mapping_tasks` (Optional)
Used for the transcript mapping step.
```json
"mapping_tasks": [
    {
        "image_id": "sample_123",
        "transcript_file": "data/raw/transcripts/sample_123_transcripts.parquet"
    }
]
```

### `visualization_config.json`

Controls the output plots.
```json
{
    "default_genes_to_visualize": ["ACTB", "GAPDH"], // Fallback list
    "tasks": [
        {
            "image_id": "sample_123",
            "genes": ["GeneA", "GeneB", "GeneC"],
            "colors": ["#FF0000", "#00FF00", "#0000FF"]
        }
    ]
}
```

## 3. Key Parameters Explained

*   **MPP (Microns Per Pixel)**: Critical for spatial calculations. If your image metadata is missing, you must calculate this manually (1 inch = 25400 microns).
    *   Formula: `25400 / DPI`
*   **Scale Factor**: Reduces memory usage. 
    *   1.0 = Original size.
    *   0.5 = Half width/height (1/4 area).
    *   **Important**: This affects the `DIAMETER` parameter. Always set `DIAMETER` based on the *original* image resolution; the pipeline scales it for you.
*   **Tiling**: For extremely large images (e.g., >2GB) that don't fit in GPU memory even after scaling.
    *   `apply_tiling`: Set to true.
    *   `tile_size`: Typically 2000-4000 pixels.
    *   `overlap`: Ensures cells on the border are not lost (typically ~2x cell diameter).
