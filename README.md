# Advanced Cell Segmentation & Analysis Pipeline for Spatio-Transcriptomics

This project provides a Python-based pipeline for segmenting cells in microscopy images using Cellpose. It is highly configurable via a central JSON file, allowing for batch processing of multiple images with various Cellpose parameters, optional on-the-fly tiling and rescaling for very large images. It also includes tools for pre-processing OME-TIFFs, mapping transcripts to segmented cells, visualizing gene expression, and generating comparative summary visualizations.

The primary data organization for inputs, intermediate processed files, and final results is managed within the `data/` directory (see "Project Directory Structure" below).

## Core Components & Workflow

The pipeline generally follows these steps:

1.  **Data Preparation & Configuration**:
    *   Place your raw images (e.g., `.ome.tif`) in `data/raw/images/` and transcript files (e.g., `.parquet`) in `data/raw/transcripts/`.
    *   Configure `parameter_sets.json` located in the project root. This central JSON file defines the entire batch processing and visualization run.

2.  **`(Optional)` Image Pre-processing (`src/preprocess_ometiff.py`)**:
    *   Extracts relevant 2D planes from complex OME-TIFFs (e.g., Xenium) as standard TIFF files. Output these to a working directory, ideally within `data/processed/` or a subfolder of `data/raw/images/` before referencing them in `parameter_sets.json`.
    *   **Example Usage:**
        ```bash
        python src/preprocess_ometiff.py path/to/your/input.ome.tif path/to/output_directory --channel 0 --zplane 5 --prefix my_experiment
        ```

3.  **Main Configuration in `parameter_sets.json`**:
    *   **`image_configurations`**: Defines original images, their unique IDs, paths (pointing to files in `data/raw/images/` or pre-processed locations), activity status, microns-per-pixel (`mpp_x`, `mpp_y`), and `segmentation_options` (including `rescaling_config` and `tiling_parameters`).
    *   **`cellpose_parameter_configurations`**: Defines different sets of Cellpose parameters.
    *   **`global_segmentation_settings`**: Global settings like `max_processes` for segmentation, `default_log_level`, `FORCE_GRAYSCALE`, and `USE_GPU_IF_AVAILABLE`.
    *   **`mapping_tasks`**: Defines transcript-to-cell mapping tasks.
    *   **`visualization_tasks`**: Defines gene expression visualization tasks, specifying source segmentation, transcript data, genes, and output parameters. It typically uses the output from `mapping_tasks`.

4.  **Segmentation & Tiling (`src/segmentation_pipeline.py`)**:
    *   Reads the `parameter_sets.json`.
    *   For each active image configuration:
        *   If rescaling is specified, the image is rescaled. Cached rescaled images might be stored (e.g., in `src/file_paths.py` defined `RESCALED_IMAGE_CACHE_DIR` which defaults to `images/rescaled_cache/`).
        *   If tiling is specified, it generates tiles. Cached tile images might be stored (e.g., in `src/file_paths.py` defined `TILED_IMAGE_OUTPUT_BASE` which defaults to `images/tiled_outputs/`).
    *   For each job (image/tile + Cellpose parameter set):
        *   Performs segmentation.
        *   Outputs (masks, summaries) are saved in structured directories, typically under `data/processed/segmentation/<experiment_id_final>/` (the exact base path is defined by `RESULTS_DIR_BASE` in `src/file_paths.py`, which can be configured to point into `data/processed/segmentation/`).
        *   The `experiment_id_final` incorporates image ID, parameter set ID, scaling, and tile information.
    *   A main `run_log.json` is saved (e.g., in `data/processed/segmentation/run_log.json` if `RESULTS_DIR_BASE` points there).

5.  **`(Optional)` Stitch Tiled Segmentations (`src/stitch_masks.py`)**:
    *   Combines individual tile masks into a single coherent mask. Input tiles are typically from the tile cache, and the output stitched mask should be saved to a relevant path in `data/processed/segmentation/`.

6.  **`(Optional)` Transcript Mapping (`src/map_transcripts_to_cells.py`)**:
    *   Maps transcript locations from `data/raw/transcripts/` (or a processed version) to segmented cell IDs using a chosen mask (from `data/processed/segmentation/`).
    *   Outputs (e.g., mapped transcripts CSV, feature-cell matrix) should be saved to `data/processed/mapped/`.

7.  **`(Optional)` Gene Expression Visualization (`src/visualize_gene_expression.py`)**:
    *   Driven by `visualization_tasks` in `parameter_sets.json`.
    *   Loads segmentation masks, background images, and mapped transcripts.
    *   Generates PNG images, typically saved to `data/results/visualizations/<task_output_subfolder_name>/`.

8.  **`(Optional)` Summary Visualization (`src/create_summary_image.py`)**:
    *   Analyzes segmentation results to generate consensus/consistency images, typically saved to `data/results/visualizations/` or `data/processed/segmentation/`.

## Features
*   **Centralized JSON Configuration:** All parameters for segmentation, tiling, rescaling, and visualization are managed in `parameter_sets.json`.
*   **Structured Data Management:** Clear organization for raw data, processed files, and results using the `data/` directory.
*   Flexible batch processing.
*   Automatic on-the-fly image rescaling and tiling.
*   OME-TIFF pre-processing utility.
*   Configurable Cellpose models, parameters, and GPU usage.
*   Stitching of tiled segmentation masks.
*   Transcript mapping and feature-cell matrix generation.
*   Configurable gene expression visualization.
*   Advanced summary visualization of segmentation consistency.
*   **Comprehensive Logging:** Configurable log levels, job summaries, and a main `run_log.json`.
*   Internal path management defaults are in `src/file_paths.py` but outputs are intended for the `data/` directory structure.

## Installation

1.  **Clone the Repository.**
2.  **Create and Activate a Python Virtual Environment** (e.g., using `venv` or `conda`).
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Key dependencies include `cellpose`, `opencv-python-headless`, `numpy`, `matplotlib`, `tifffile`, `pandas`, `scipy`, `pyarrow`, `pytest`, `pytest-mock`).

## Project Directory Structure

The project is organized as follows:

```
project_root/
├── data/                     # All user data, processed files, and results
│   ├── raw/                  # Original, unmodified input data
│   │   ├── images/           # Raw microscopy images (e.g., *.ome.tif)
│   │   │   └── experiment1/  # Example experiment subfolder
│   │   └── transcripts/      # Raw transcript data (e.g., *.parquet)
│   │       └── experiment1/  # Example experiment subfolder
│   ├── processed/            # Intermediate files generated by the pipeline
│   │   ├── segmentation/     # Segmentation masks, summaries, run logs
│   │   │   └── experiment1/
│   │   ├── mapped/           # Transcript-to-cell mapping outputs
│   │   │   └── experiment1/
│   │   └── ...               # Other intermediate processing outputs
│   └── results/              # Final analysis results and visualizations
│       └── experiment1/
│           ├── visualizations/
│           └── statistics/
├── src/                      # Python source code for the pipeline
│   ├── segmentation_pipeline.py  # Main script for running segmentation
│   ├── visualize_gene_expression.py # Script for gene expression visualization
│   ├── pipeline_config_parser.py # Handles loading and parsing parameter_sets.json
│   ├── segmentation_worker.py    # Core Cellpose segmentation logic for a single job
│   ├── tile_large_image.py     # Logic for tiling and rescaling images
│   ├── pipeline_utils.py       # Utility functions used across the pipeline
│   ├── file_paths.py         # Defines default base paths (can be configured)
│   ├── stitch_masks.py         # Stitches tiled segmentation masks
│   ├── map_transcripts_to_cells.py # Maps transcripts to segmented cells
│   ├── create_summary_image.py   # Creates summary visualizations of segmentations
│   └── preprocess_ometiff.py   # Utility for pre-processing OME-TIFFs
├── tests/                    # Automated tests
│   ├── unit/
│   ├── integration/
│   └── functional/
├── parameter_sets.json       # Central JSON configuration file
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── .gitignore                # Specifies intentionally untracked files by Git
```
*Note: The `images/` and `results/` directories at the project root (if present from older versions) are generally used as default cache locations by scripts (e.g., for rescaled images, tiled images before stitching) as defined in `src/file_paths.py`. The primary, organized data storage is intended for the `data/` directory.* A `data/README.md` provides more detail on the `data/` directory.


## Detailed Workflow Steps

### Step 0: Prepare Input Data and Environment
1.  Place your raw microscopy images (e.g., `.ome.tif`, `.tif`) into a subdirectory within `data/raw/images/` (e.g., `data/raw/images/my_experiment/`).
2.  Place corresponding raw transcript data (e.g., `.parquet`, `.csv`) into `data/raw/transcripts/my_experiment/`.
3.  Ensure your Python environment is set up and dependencies from `requirements.txt` are installed.

### Step 1: Configure `parameter_sets.json`
This is the main control file, located in the project root. Update it to point to your data in `data/raw/` and define your processing parameters.

*   **Structure Overview:**
    ```json
    {
      "global_segmentation_settings": { /* ... */ },
      "image_configurations": [ /* ... */ ],
      "cellpose_parameter_configurations": [ /* ... */ ],
      "mapping_tasks": [ /* ... */ ],
      "visualization_tasks": [ /* ... */ ]
    }
    ```

*   **`global_segmentation_settings` (Object, Optional):**
    Settings that apply globally to `segmentation_pipeline.py`.
    
    *   `default_log_level`: (String) "DEBUG", "INFO", etc. Overridden by CLI `--log_level`.
    *   `max_processes`: (Integer) Max parallel jobs. Overridden by CLI `--max_processes`.
    *   `FORCE_GRAYSCALE`: (Boolean) Global default for grayscale conversion.
    *   `USE_GPU_IF_AVAILABLE`: (Boolean) Global default for Cellpose GPU usage.

    ```json
    {
      "global_segmentation_settings": {
        "default_log_level": "INFO",
        "max_processes": 1,
        "FORCE_GRAYSCALE": true,
        "USE_GPU_IF_AVAILABLE": true
      }
    }
    ```

*   **`image_configurations` (List of Objects):**
    Define each source image to be processed by the pipeline. Each object in this list represents one image and its specific processing settings.
    *   `"image_id"`: (String) A unique identifier for this image configuration (e.g., `"exp1_dapi_channel"`, `"slide2_tile_grid"`). This ID is used throughout the pipeline for logging, output file naming, and linking to other configurations (like `visualization_tasks`).
    *   `"original_image_filename"`: (String) The path to the source image file, relative to the project root directory (e.g., `"data/raw/images/my_experiment/image1.ome.tif"`). This can be an OME-TIFF, standard TIFF, or other formats readable by `tifffile`.
    *   `"is_active"`: (Boolean) Set to `true` to include this image configuration in the current pipeline run. If `false`, this image will be skipped.
    *   `"mpp_x"`, `"mpp_y"`: (Float) Microns per pixel for the image in the X and Y dimensions, respectively. These values are crucial for accurate measurements if tiling is based on physical units (microns) and for some visualization scaling. If not applicable or unknown, you can use `1.0`.
    *   `"segmentation_options"`: (Object, Optional) Contains parameters related to how this specific image should be segmented, including whether to segment it at all, and if rescaling or tiling should be applied before segmentation.
        *   `"apply_segmentation"`: (Boolean) If `true`, Cellpose segmentation will be performed on this image (or its rescaled/tiled versions). If `false`, segmentation is skipped for this image configuration, though other operations might still apply.
        *   `"rescaling_config"`: (Object or `null`, Optional) Defines parameters for on-the-fly image rescaling before segmentation. Set to `null` or omit if no rescaling is needed.
            *   `"scale_factor"`: (Float) The factor by which to rescale the image. E.g., `0.5` halves the image dimensions, `2.0` doubles them. Rescaling can help if object sizes are too large or too small for the chosen Cellpose model or diameter, or to reduce processing time for very large images. It affects the effective microns-per-pixel of the image being segmented.
            *   `"interpolation"`: (String) The interpolation method to use for rescaling. Common values include `"INTER_NEAREST"` (fast, preserves sharp edges, good for masks), `"INTER_LINEAR"` (default, good for general downscaling), `"INTER_CUBIC"` (good for upscaling, slower), `"INTER_AREA"` (good for downscaling, preserves area). Refer to OpenCV documentation for more options.
        *   `"tiling_parameters"`: (Object or `null`, Optional) Defines parameters for dividing the image into smaller tiles for segmentation. Useful for very large images that don't fit in memory or to apply segmentation locally. Set to `null` or omit if no tiling is needed.
            *   `"apply_tiling"`: (Boolean) If `true`, tiling will be performed. If `false`, the entire image (possibly after rescaling) is processed at once.
            *   `"tile_size_xy_microns"`: (Integer) The desired size of each tile in microns (both width and height). This requires `mpp_x` and `mpp_y` to be set correctly. Alternatively, a pixel-based tile size could be implemented if needed (not current default).
            *   `"overlap_microns"`: (Integer) The overlap between adjacent tiles in microns. Overlap helps to avoid edge artifacts during segmentation and ensures objects spanning tile boundaries are captured correctly. Cellpose's stitching mechanism (if used) benefits from this.
    **Example:**
    ```json
    "image_configurations": [
      {
        "image_id": "experiment1_image_dapi",
        "original_image_filename": "data/raw/images/experiment1/image_channel_0.tif",
        "is_active": true,
        "mpp_x": 0.21,
        "mpp_y": 0.21,
        "segmentation_options": {
          "apply_segmentation": true,
          "rescaling_config": {
            "scale_factor": 0.5,
            "interpolation": "INTER_NEAREST"
          },
          "tiling_parameters": {
            "apply_tiling": false
          }
        }
      },
      {
        "image_id": "experiment1_image_cells",
        "original_image_filename": "data/raw/images/experiment1/image_channel_1.tif",
        "is_active": true,
        "mpp_x": 0.21,
        "mpp_y": 0.21,
        "segmentation_options": {
          "apply_segmentation": true,
          "rescaling_config": null,
          "tiling_parameters": {
            "apply_tiling": true,
            "tile_size_xy_microns": 500,
            "overlap_microns": 50
          }
        }
      },
      {
        "image_id": "experiment2_image_simple",
        "original_image_filename": "data/raw/images/experiment2/another_image.ome.tif",
        "is_active": false,
        "mpp_x": 1.0,
        "mpp_y": 1.0,
        "segmentation_options": { 
            "apply_segmentation": false 
        }
      }
    ]
    ```

*   **`cellpose_parameter_configurations` (List of Objects):**
    Define sets of Cellpose parameters to be applied to images. Each object in this list represents a distinct configuration.
    *   `"param_set_id"`: (String) A unique identifier for this specific set of Cellpose parameters (e.g., `"cyto2_default_diam30"`, `"nuclei_custom_flow0.8"`). This ID is used to link `image_configurations` to these settings, allowing different images or the same image to be processed with multiple parameter sets.
    *   `"is_active"`: (Boolean) Set to `true` to enable this parameter set for processing, or `false` to disable and skip it during a pipeline run.
    *   `"cellpose_parameters"`: (Object) An object containing key-value pairs that correspond to the arguments for the Cellpose `model.eval()` method. These parameters directly control the segmentation behavior. For a comprehensive list and detailed explanations of all available Cellpose parameters, please refer to the official Cellpose documentation. Common parameters include:
        *   `"MODEL_CHOICE"`: (String) Specifies the Cellpose model to use (e.g., `"cyto2"` for cytoplasm, `"nuclei"` for nuclei, `"livecell"`, or a path to a custom-trained model).
        *   `"USE_GPU"`: (Boolean) If `true`, Cellpose will attempt to use a compatible GPU if available, which can significantly speed up segmentation. Set to `false` to force CPU usage.
        *   `"CHANNELS"`: (List of Integers) Defines the channels for segmentation. For grayscale images, use `[0,0]`. For multi-channel images where, for example, the cytoplasm is green (channel 2) and nucleus is blue (channel 1), you might use `[2,1]` (cytoplasm channel first, nucleus channel second for cyto2 model) or `[1,0]` (nucleus channel first for nuclei model, second channel is ignored or set to 0).
        *   `"DIAMETER"`: (Integer) Estimated average diameter of the objects (cells or nuclei) in pixels. If set to `0` or `null`, Cellpose will attempt to automatically estimate the diameter from the image. Providing an accurate estimate can improve segmentation quality.
        *   `"FLOW_THRESHOLD"`: (Float) Threshold for the flow field prediction from the model. Default is typically `0.4`. Higher values generally lead to fewer, larger, and more merged objects, while lower values can result in more segmented objects, potentially including noise.
        *   `"CELLPROB_THRESHOLD"`: (Float) Threshold for the cell probability map. Default is typically `0.0`. Increasing this (e.g., towards positive values) makes segmentation more conservative (fewer objects), while decreasing it (e.g., towards negative values) makes it more sensitive (more objects).
        *   `"MIN_SIZE"`: (Integer) The minimum number of pixels a segmented object must have to be included in the final masks. Objects smaller than this are discarded. Default is often around `15`.
        *   `"STITCH_THRESHOLD"`: (Float, Optional) When segmenting tiled images, this threshold (ranging from 0.0 to 1.0) is used to stitch together masks from adjacent tiles. Only relevant if tiling is enabled in `image_configurations`. Default is typically `0.0` if not specified by Cellpose, but can be adjusted.
    **Example:**
    ```json
    "cellpose_parameter_configurations": [
      {
        "param_set_id": "cyto2_default_diam30",
        "is_active": true,
        "cellpose_parameters": {
          "MODEL_CHOICE": "cyto2",
          "USE_GPU": true,
          "CHANNELS": [0, 0],
          "DIAMETER": 30,
          "FLOW_THRESHOLD": 0.4,
          "CELLPROB_THRESHOLD": 0.0,
          "MIN_SIZE": 15
        }
      },
      {
        "param_set_id": "nuclei_custom_diam15",
        "is_active": true,
        "cellpose_parameters": {
          "MODEL_CHOICE": "nuclei",
          "USE_GPU": true,
          "CHANNELS": [1, 0],
          "DIAMETER": 15,
          "FLOW_THRESHOLD": 0.6,
          "CELLPROB_THRESHOLD": -2.0,
          "MIN_SIZE": 10,
          "STITCH_THRESHOLD": 0.1 
        }
      },
      {
        "param_set_id": "livecell_diam0_autodetect",
        "is_active": false,
        "cellpose_parameters": {
          "MODEL_CHOICE": "livecell",
          "USE_GPU": false,
          "CHANNELS": [0,0],
          "DIAMETER": 0 
        }
      }
    ]
    ```

*   **`mapping_tasks` (List of Objects):**
    Define transcript-to-cell mapping tasks. Each task processes one transcript file against one segmentation mask.
    *   `"task_id"`: (String) Unique identifier for this mapping task.
    *   `"is_active"`: (Boolean) Set to `true` to run this mapping task. If `false`, it will be skipped.
    *   `"description"`: (String, Optional) User-friendly description of the task.
    *   `"source_image_id"`: (String) ID of the original image from `image_configurations`. This is used to find the relevant image metadata (like original MPP, scaling, and tiling settings).
    *   `"source_param_set_id"`: (String) ID of the Cellpose parameters from `cellpose_parameter_configurations` used for the segmentation that produced the mask.
    *   `"source_processing_unit_display_name"`: (String, Optional for non-tiled segmentations) The filename (without path) of the specific image unit that was segmented and whose mask will be used.
        *   **For non-tiled segmentations:** If omitted, this name is automatically derived. If the image was rescaled (per `rescaling_config` in `image_configurations`), the name is constructed as `original_basename_scaled_X_Y.ext`. If not rescaled, it defaults to the base name of the `original_image_filename`.
        *   **For tiled segmentations:** This field is **mandatory** and must specify the exact tile filename (e.g., `"tile_r0_c0.tif"`). Whether a source image is considered tiled is determined by the `apply_tiling` setting in its `image_configuration`.
    *   `"input_transcripts_path"`: (String) Path to the input transcript file (e.g., `.parquet` or `.csv`), typically under `data/raw/transcripts/`.
    *   `"output_base_dir"`: (String) Directory where mapping results (mapped transcripts CSV, feature-cell matrix CSVs) will be saved, typically under `data/processed/mapped/`.
    *   `"output_prefix"`: (String) Prefix for output filenames (e.g., `"mapped_transcripts_run1"`).
    *   `"mask_path_override"`: (String, Optional) If provided, this full path to the mask TIFF file will be used directly, bypassing all mask path derivation logic based on source IDs and names.
    *The following parameters are automatically derived from the linked `image_configuration` and are no longer needed directly in the task definition: `source_segmentation_scale_factor`, `source_segmentation_is_tile`, `mpp_x_of_mask`, `mpp_y_of_mask`.*
    **Example:**
    ```json
    "mapping_tasks": [
      {
        "task_id": "map_exp1_non_tiled_rescaled",
        "is_active": true,
        "description": "Map transcripts for experiment1 using segmentation from pset1 on a rescaled image (unit name derived).",
        "source_image_id": "experiment1_image_dapi", // Assumes this image_config has rescaling defined
        "source_param_set_id": "cyto2_default_diam30",
        // "source_processing_unit_display_name" is omitted, will be derived e.g., to "image_channel_0_scaled_0_5.tif"
        "input_transcripts_path": "data/raw/transcripts/experiment1/transcripts.parquet",
        "output_base_dir": "data/processed/mapped/exp1_dapi_map/",
        "output_prefix": "mapped_transcripts_dapi"
      },
      {
        "task_id": "map_exp1_tiled_specific_tile",
        "is_active": true,
        "description": "Map transcripts for experiment1 using segmentation from pset2 on a specific tile.",
        "source_image_id": "experiment1_image_cells", // Assumes this image_config has apply_tiling = true
        "source_param_set_id": "nuclei_custom_diam15",
        "source_processing_unit_display_name": "tile_r1_c2.tif", // Specific tile name is mandatory here
        "input_transcripts_path": "data/raw/transcripts/experiment1/transcripts_other.parquet",
        "output_base_dir": "data/processed/mapped/exp1_cells_tile_r1_c2_map/",
        "output_prefix": "mapped_transcripts_cells_tile_r1_c2"
      }
    ]
    ```

*   **`visualization_tasks` (Object):**
    Configures all gene expression visualization operations. This object contains a global default for genes to visualize and a list of specific visualization tasks.
    *   `"default_genes_to_visualize"`: (List of Strings, Optional, e.g. `["GeneX", "GeneY"]`) If provided, this list of genes will be used for any task in the `tasks` list below that does not specify its own `genes_to_visualize`.
    *   `"tasks"`: (List of Objects) Each object in this list defines a specific visualization task.
        *   `"task_id"`: (String) A unique identifier for this visualization task (e.g., `"exp1_dapi_gene_set1_viz"`).
        *   `"is_active"`: (Boolean) Set to `true` to run this visualization task. If `false`, it will be skipped.
        *   `"source_image_id"`: (String) Refers to the `image_id` from an `image_configurations` entry. This specifies the original image context (e.g., for background, scale) and is used to derive the segmentation scale factor.
        *   `"source_param_set_id"`: (String) Refers to the `param_set_id` from a `cellpose_parameter_configurations` entry. This identifies the segmentation parameters used to generate the mask that will be visualized.
        *   `"source_processing_unit_display_name"`: (String, Optional for non-tiled segmentations) The filename (without path) of the specific image unit that was segmented. 
            *   **For non-tiled segmentations:** If omitted, this name will be automatically derived. If the image was rescaled via `rescaling_config` in `image_configurations`, the name is constructed from the original image filename and the scale factor (e.g., `original_basename_scaled_0_5.tif`). If not rescaled, it defaults to the base name of the `original_image_filename`.
            *   **For tiled segmentations:** This field is **mandatory** and must specify the exact tile filename (e.g., `"tile_r0_c0.tif"`) whose mask is to be used. (Whether a segmentation is considered tiled is determined by the `apply_tiling` setting in the `segmentation_options` of the linked `image_configuration`).
            *   This name is crucial for locating the correct `_mask.tif` file.
        *   `"mapped_transcripts_csv_path"`: (String) Path to the CSV file containing mapped transcript data. This file should typically include columns for transcript coordinates (e.g., 'x', 'y' or 'global_x', 'global_y'), gene names (e.g., 'gene'), and the cell ID to which each transcript was assigned (e.g., 'cell_id'). An example path: `"data/processed/mapped/my_experiment/mapped_transcripts.csv"`.
        *   `"genes_to_visualize"`: (List of Strings, Optional) A list of gene names (e.g., `["GeneA", "GeneB", "GeneC"]`) for which expression data will be plotted. These gene names must exist in the provided `mapped_transcripts_csv_path`. If not provided for a specific task, the `default_genes_to_visualize` (defined at the parent `visualization_tasks` level) will be used. If neither is defined, no genes will be plotted for that task.
        *   `"output_subfolder_name"`: (String) The name of the subfolder where the generated visualization images for this task will be saved. This folder will be created under the main visualization output directory (typically `data/results/visualizations/`). E.g., `"exp1_dapi_channel_genes"`.
        *   `"visualization_params"`: (Object, Optional) Additional parameters to customize the visualization appearance.
            *   `"background_image_path_override"`: (String or `null`, Optional) Full path to a specific image to use as the background for the visualization. If `null` or omitted, the script will attempt to use the `original_image_filename` from the linked `image_configurations`.
            *   `"mask_alpha"`: (Float, 0.0-1.0, Optional) Opacity of the cell segmentation mask outlines. Default: `0.3`.
            *   `"dot_size"`: (Integer, Optional) Size of the dots representing transcripts. Default: `5`.
            *   `"dot_alpha"`: (Float, 0.0-1.0, Optional) Opacity of the transcript dots. Default: `0.5`.
            *   `"image_brightness_factor"`: (Float, Optional) Multiplier to adjust the brightness of the background image. Default: `1.0`.
            *   `"crop_to_segmentation_area"`: (Boolean, Optional) If `true`, the visualization will be cropped to the extent of the segmentation mask. Default: `false`.
    **Example:**
    ```json
    "visualization_tasks": {
      "default_genes_to_visualize": ["GlobalGeneX", "GlobalGeneY"],
      "tasks": [
        {
          "task_id": "vis_exp1_image_dapi_genes_ab", // This task will use its own gene list
          "is_active": true,
          "source_image_id": "experiment1_image_dapi",
          "source_param_set_id": "cyto2_default_diam30",
          "source_processing_unit_display_name": "image_channel_0_scaled_0_5.tif", 
          "mapped_transcripts_csv_path": "data/processed/mapped/experiment1_image_dapi_cyto2_default_diam30/mapped_transcripts.csv",
          "genes_to_visualize": ["GeneA", "GeneB"], // Overrides default
          "output_subfolder_name": "exp1_dapi_genes_ab_on_scaled_seg",
          "visualization_params": {
            "mask_alpha": 0.4,
            "dot_size": 3,
            "image_brightness_factor": 1.2
          }
        },
        {
          "task_id": "vis_exp1_image_cells_global_genes", // This task will use the default_genes_to_visualize
          "is_active": true,
          "source_image_id": "experiment1_image_cells",
          "source_param_set_id": "nuclei_custom_diam15",
          "source_processing_unit_display_name": "tile_r0_c0.tif", 
          "mapped_transcripts_csv_path": "data/processed/mapped/experiment1_image_cells_tile_r0_c0_nuclei_custom_diam15/mapped_transcripts_tile_r0_c0.csv",
          // "genes_to_visualize" is omitted, so it will use ["GlobalGeneX", "GlobalGeneY"]
          "output_subfolder_name": "exp1_cells_global_genes_tile_r0_c0",
          "visualization_params": {
            "background_image_path_override": "data/raw/images/experiment1/tiles/tile_r0_c0_background.tif", 
            "dot_alpha": 0.7,
            "crop_to_segmentation_area": true
          }
        }
      ]
    }
    ```

### Step 2: Run Segmentation Pipeline
Execute from the project root:
```bash
python -m src.segmentation_pipeline --config parameter_sets.json
```
**Command-line arguments:**
*   `--config <path>`: Path to your JSON configuration. Default: `parameter_sets.json`.
*   `--log_level <LEVEL>`: Overrides JSON `default_log_level`.
*   `--max_processes <N>`: Overrides JSON `max_processes`.