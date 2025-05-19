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

3.  **Main Configuration in `parameter_sets.json`**:
    *   **`image_configurations`**: Defines original images, their unique IDs, paths (pointing to files in `data/raw/images/` or pre-processed locations), activity status, microns-per-pixel (`mpp_x`, `mpp_y`), and `segmentation_options` (including `rescaling_config` and `tiling_parameters`).
    *   **`cellpose_parameter_configurations`**: Defines different sets of Cellpose parameters.
    *   **`global_segmentation_settings`**: Global settings like `max_processes` for segmentation, `default_log_level`, `FORCE_GRAYSCALE`, and `USE_GPU_IF_AVAILABLE`.
    *   **`visualization_tasks`**: Defines gene expression visualization tasks, specifying source segmentation, transcript data, genes, and output parameters.

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
      "visualization_tasks": [ /* ... */ ]
    }
    ```

*   **`global_segmentation_settings` (Object, Optional):**
    Settings that apply globally to `segmentation_pipeline.py`.
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
    *   `default_log_level`: (String) "DEBUG", "INFO", etc. Overridden by CLI `--log_level`.
    *   `max_processes`: (Integer) Max parallel jobs. Overridden by CLI `--max_processes`.
    *   `FORCE_GRAYSCALE`: (Boolean) Global default for grayscale conversion.
    *   `USE_GPU_IF_AVAILABLE`: (Boolean) Global default for Cellpose GPU usage.

*   **`image_configurations` (List of Objects):**
    Define each source image to process.
    *   `"image_id"`: Unique string.
    *   `"original_image_filename"`: Path to the image file, relative to `PROJECT_ROOT` (e.g., `"data/raw/images/my_experiment/image1.tif"`).
    *   `"is_active"`: Boolean.
    *   `"mpp_x"`, `"mpp_y"`: Microns per pixel (required for some visualization/tiling).
    *   `"segmentation_options"`: (Object, Optional)
        *   `"apply_segmentation"`: Boolean.
        *   `"rescaling_config"`: { `"scale_factor"`, `"interpolation"` }
        *   `"tiling_parameters"`: { `"apply_tiling"`, `"tile_size_xy_microns"`, `"overlap_microns"` }

*   **`cellpose_parameter_configurations` (List of Objects):**
    Define sets of Cellpose parameters.
    *   `"param_set_id"`: Unique string.
    *   `"is_active"`: Boolean.
    *   `"cellpose_parameters"`: { `"MODEL_CHOICE"`, `"DIAMETER"`, `"FLOW_THRESHOLD"`, etc. }

*   **`visualization_tasks` (List of Objects):**
    Define gene expression visualization tasks.
    *   `"task_id"`: Unique string.
    *   `"is_active"`: Boolean.
    *   `"source_image_id"`: Links to an `image_id`.
    *   `"source_param_set_id"`: Links to a `param_set_id`.
    *   `"source_processing_unit_name"`: Filename of the specific segmented unit (e.g., rescaled image name or tile name).
    *   `"mapped_transcripts_csv_path"`: Path to mapped transcripts (e.g., `"data/processed/mapped/my_experiment/mapped_data.csv"`).
    *   `"genes_to_visualize"`: List of gene names.
    *   `"output_subfolder_name"`: Subfolder under `data/results/visualizations/` for this task's output.

### Step 2: Run Segmentation Pipeline
Execute from the project root:
```bash
python -m src.segmentation_pipeline --config parameter_sets.json
```
**Command-line arguments:**
*   `--config <path>`: Path to your JSON configuration. Default: `parameter_sets.json`.
*   `--log_level <LEVEL>`: Overrides JSON `default_log_level`.
*   `--max_processes <N>`: Overrides JSON `max_processes`.

Outputs (masks, summaries) are typically saved to `data/processed/segmentation/`. Check `run_log.json` in the same base output directory.

### Step 3: Stitch Tiled Segmentations (If Tiling Was Used)
```bash
python -m src.stitch_masks <original_image_id> <param_set_id> <output_directory_for_stitched_mask> --config parameter_sets.json
# Example output directory: data/processed/segmentation/my_experiment/stitched_masks/
# Add --scale_factor_applied if appropriate.
```

### Step 4: Map Transcripts to Cells
```bash
python -m src.map_transcripts_to_cells \
    "data/raw/transcripts/my_experiment/transcripts.parquet" \
    "data/processed/segmentation/my_experiment/experiment_id_final/mask_file_name.tif" \
    "data/processed/mapped/my_experiment/" \
    --mpp_x 0.2125 --mpp_y 0.2125 \
    --output_prefix "my_mapping"
```
*Adjust paths and MPP values as needed.*

### Step 5: Visualize Gene Expression
Ensure `visualization_tasks` are defined in `parameter_sets.json`.
```bash
python -m src.visualize_gene_expression --config parameter_sets.json
# Or --task_id <your_task_id> for a specific task.
```
Outputs are saved in `data/results/visualizations/<task_output_subfolder_name>/`.

### Step 6: Generate Summary Image of Segmentations (Optional)
```bash
python -m src.create_summary_image --config parameter_sets.json
```
*Output typically to `data/results/visualizations/` or `data/processed/segmentation/`.*


## Troubleshooting
*   **Import Errors:** Always run scripts as Python modules from the project root (e.g., `python -m src.script_name`).
*   **Configuration Issues:** Double-check paths in `parameter_sets.json` (especially `original_image_filename`, `mapped_transcripts_csv_path`) and ensure they correctly point to your files within the `data/` directory structure or other specified locations.
*   **Logging:** Use `--log_level DEBUG` for verbose output. Check console, `run_log.json`, and individual job summary files.
*   **File Not Found:** Verify all paths. Default output locations for segmentation are influenced by `RESULTS_DIR_BASE` in `src/file_paths.py` – ensure this points where you expect (e.g., into `data/processed/segmentation/`).
*   **GPU Usage:** For Cellpose, ensure correct installation and settings for `USE_GPU` (in parameter sets) and `USE_GPU_IF_AVAILABLE` (global). For single GPU, set `max_processes: 1`.

## Testing

This project uses `pytest`. Ensure development dependencies are installed.

To run all tests, from the project root:
```bash
pytest
```
Verbose output: `pytest -v`
Specific file: `pytest tests/functional/test_segmentation_pipeline_flow.py`
Specific test: `pytest -k "test_function_name"`
