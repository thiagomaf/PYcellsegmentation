# Advanced Cell Segmentation & Analysis Pipeline for Spatio-Transcriptomics

This project provides a Python-based pipeline for segmenting cells in microscopy images using Cellpose. It allows applying multiple Cellpose parameter configurations to a list of input images, with optional on-the-fly tiling for very large images. It also includes tools for pre-processing OME-TIFFs, mapping transcripts to segmented cells, and generating comparative summary visualizations.

## Core Components & Workflow

The pipeline generally follows these steps:

1.  **`(Optional)` Image Pre-processing (`src/preprocess_ometiff.py`)**:
    *   If your source images are complex OME-TIFFs (e.g., from Xenium), use this script to extract relevant 2D planes as standard TIFF files.
    *   These extracted 2D TIFFs become inputs for the main pipeline.

2.  **Configuration (`parameter_sets.json`)**:
    *   This central JSON file defines the entire batch processing run. It specifies:
        *   **`image_configurations`**: A list defining each original image to be processed, its ID, and an optional `tiling_config` if the image is large and needs to be tiled before segmentation.
        *   **`cellpose_parameter_configurations`**: A list defining different sets of Cellpose parameters to be tested.
    *   The pipeline will run every active Cellpose parameter configuration on every active image configuration (or its generated tiles).

3.  **Segmentation & Tiling (`src/segmentation_pipeline.py`)**:
    *   Reads `parameter_sets.json`.
    *   For each active image configuration:
        *   If tiling is specified (via `tiling_config`), it calls `src.tile_large_image.tile_image` to generate tiles and a manifest. Tiles are stored in `images/tiled_outputs/<image_id_from_json>/`.
        *   It then creates jobs for each image (or each generated tile).
    *   For each job (image/tile + Cellpose parameter set):
        *   Performs segmentation using the specified Cellpose model and parameters.
        *   Outputs are saved in a unique subfolder: `results/<image_id>_<param_set_id>/` (for non-tiled) or `results/<image_id>_<param_set_id>_<cleaned_tile_name>/` (for tiled).
        *   Outputs per job: `_mask.tif` (integer-labeled mask) and `_coords.json` (cell centroids).
    *   Logs all executed jobs, their parameters, and status to `results/run_log.json`.
    *   Supports CPU-based parallel processing for these jobs.

4.  **`(Optional)` Transcript Mapping (`src/map_transcripts_to_cells.py`)**:
    *   After segmentation, choose a specific `_mask.tif` from a successful job.
    *   Use this script to map transcript locations (e.g., from a Xenium `transcripts.parquet` file) to the segmented cell IDs.
    *   Outputs a CSV of transcripts with assigned cell IDs and a feature-cell matrix in MTX format.

5.  **`(Optional)` Summary Visualization (`src/create_summary_image.py`)**:
    *   Reads `results/run_log.json` to find successfully processed segmentation jobs.
    *   Generates a consensus probability map.
    *   Calculates Dice scores for each job against the consensus.
    *   Creates `results/segmentation_summary_consistency.png` showing all processed segmentations (tiles or full images) overlaid on their respective original images, with cells colored by consistency and Dice scores displayed.

## Features
*   Flexible batch processing: multiple parameter sets across multiple images.
*   Automatic on-the-fly tiling for large images, configured per image.
*   OME-TIFF pre-processing utility.
*   Choice of Cellpose models and detailed parameter tuning.
*   Configurable GPU usage per Cellpose parameter set.
*   Transcript mapping and feature-cell matrix generation.
*   Advanced summary visualization of segmentation results with cell consistency coloring and Dice scores.
*   Comprehensive logging of all jobs in `results/run_log.json`.

## Installation

1.  **Clone the Repository (if applicable).**

2.  **Create a Python Virtual Environment:**
    (Recommended) In the project's root directory:
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    *   Windows (CMD): `.venv\Scripts\activate`
    *   Windows (PowerShell): `.venv\Scripts\Activate.ps1`
    *   Linux/macOS: `source .venv/bin/activate`

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Ensures `cellpose`, `opencv-python`, `numpy`, `matplotlib`, `tifffile`, `pandas`, `scipy`, and `pyarrow` are installed).

5.  **Handling Potential OpenMP Errors (Windows):**
    `src/segmentation_pipeline.py` includes `os.environ['KMP_DUPLICATE_LIB_OK']='True'` to help mitigate OpenMP runtime conflicts.

## Detailed Workflow Steps

### Step 0: Prepare Input Images
*   Place your primary 2D TIFF images in the `images/` folder.
*   If using OME-TIFFs, first run `src/preprocess_ometiff.py` to extract 2D TIFFs and place them in `images/`.
    ```bash
    python src/preprocess_ometiff.py path/to/your.ome.tif images/extracted_planes --channel X --zplane Y --prefix ome_extract
    ```
    *(Ensure `images/extracted_planes` exists or adjust output path; these extracted TIFFs will be listed in `parameter_sets.json`)*

### Step 1: Configure `parameter_sets.json`
This is the main control file, located in the project root.

*   **Structure:**
    ```json
    {
      "image_configurations": [ /* ... list of image setups ... */ ],
      "cellpose_parameter_configurations": [ /* ... list of Cellpose param sets ... */ ]
    }
    ```

*   **`image_configurations` (List of Objects):**
    *   `"image_id"`: (String) Unique identifier for this image setup (e.g., "XeniumSlide1_RegionA"). Used in output folder naming.
    *   `"original_image_filename"`: (String) Filename (from `images/` folder) for the source image.
    *   `"is_active"`: (Boolean, Optional) `true` (or omitted) to process this image; `false` to skip.
    *   `"tiling_config"`: (Object, Optional) If present and contains `tile_size`, tiling is enabled for this image.
        *   `"tile_size"`: (Integer) e.g., `2048`.
        *   `"overlap"`: (Integer) e.g., `200`.
        *   `"tile_output_prefix_base"`: (String) Base for naming generated tile files and their manifest. Tiles will be stored in `images/tiled_outputs/<image_id>/`.

*   **`cellpose_parameter_configurations` (List of Objects):**
    *   `"param_set_id"`: (String) Unique identifier for this set of Cellpose parameters (e.g., "DefaultCyto3", "AggressiveSmallCells"). Used in output folder naming.
    *   `"is_active"`: (Boolean, Optional) `true` (or omitted) to use this parameter set; `false` to skip.
    *   `"MODEL_CHOICE"`: (String) e.g., `"cyto3"`.
    *   `"DIAMETER"`: (Number or `null`).
    *   `"FLOW_THRESHOLD"`: (Number or `null`).
    *   `"MIN_SIZE"`: (Number or `null`, will default to 15 in script if `null`).
    *   `"CELLPROB_THRESHOLD"`: (Number).
    *   `"FORCE_GRAYSCALE"`: (Boolean).
    *   `"USE_GPU"`: (Boolean).

### Step 2: Run Segmentation Pipeline
1.  **Configure Parallel Processing:** Edit `MAX_PARALLEL_PROCESSES` at the top of `src/segmentation_pipeline.py`.
    *   **GPU Users:** If any active Cellpose parameter configuration uses GPU, set `MAX_PARALLEL_PROCESSES = 1` for stability on single-GPU systems.
2.  **Execute:**
    ```bash
    python src/segmentation_pipeline.py
    ```
    *   This will first tile images if configured (tiles stored in `images/tiled_outputs/<image_id>/`), then run Cellpose segmentation for every active image/tile combined with every active parameter set.
    *   Outputs are saved in `results/<image_id>_<param_set_id>/` (for non-tiled images) or `results/<image_id>_<param_set_id>_<cleaned_tile_name>/` (for tiles).
    *   `results/run_log.json` is created/updated with details of all individual segmentation jobs.

### Step 3: Map Transcripts to Cells (Optional, Per Job)
1.  After Step 2, choose a specific segmentation output (a `_mask.tif` file from one of the job folders in `results/`).
2.  Run `src/map_transcripts_to_cells.py`:
    ```bash
    python src/map_transcripts_to_cells.py \
        "path/to/transcripts.parquet" \
        "results/ImageID_ParamSetID_TileNameOpt/image_mask.tif" \
        "results/ImageID_ParamSetID_TileNameOpt/mapping_output" \
        --mpp_x <val> --mpp_y <val> 
        # ... other arguments ...
    ```

### Step 4: Generate Summary Image of Segmentations (Optional)
1.  Run `src/create_summary_image.py`:
    ```bash
    python src/create_summary_image.py
    ```
    *   It reads `results/run_log.json` to find successfully completed, active segmentation jobs (tiles or full images) and includes them in the summary.
2.  View `results/segmentation_summary_consistency.png`.

## Troubleshooting

*   **Tiling:** Check `images/tiled_outputs/<image_id>/` for generated tiles and `_manifest.json`.
*   **OME-TIFF Pre-processing:** If `preprocess_ometiff.py` outputs black or incorrect images, double-check the `--channel`, `--zplane`, and `--series` arguments against your OME-TIFF's actual structure.
*   **Segmentation Errors (`segmentation_pipeline.py`):**
    *   Check `results/run_log.json` for the status of all jobs.
    *   For failed jobs, check `error_log.txt` in the specific job's output folder in `results/`.
*   **Summary Image Issues (`create_summary_image.py`):**
    *   Ensure `_mask.tif` files exist for successfully processed jobs listed in `run_log.json`.
    *   Verify original images (or tiles if applicable) are accessible.
*   **GPU Errors:** If using GPU, ensure `MAX_PARALLEL_PROCESSES = 1` for single-GPU systems.
*   **Transcript Mapping:** Ensure correct `--mpp_x`, `--mpp_y`, and offset values. Verify `feature_name` and `qv` columns in your transcript file.

