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

4.  **`(Optional)` Stitch Tiled Segmentations (`src/stitch_masks.py`)**:
    *   If tiling was used in Step 3, this script combines the individual tile masks from a specific run (original image + Cellpose parameters) into a single, coherent segmentation mask for the entire original large image.
    *   It uses the `run_log.json` to find the relevant tile masks and their original positions.
    *   Outputs a `_stitched_mask.tif` file.

5.  **`(Optional)` Transcript Mapping (`src/map_transcripts_to_cells.py`)**:
    *   After segmentation (and optional stitching), choose a specific `_mask.tif` (either from a single job or a stitched mask).
    *   Use this script to map transcript locations (e.g., from a Xenium `transcripts.parquet` file) to the segmented cell IDs.
    *   Outputs a CSV of transcripts with assigned cell IDs and a feature-cell matrix in MTX format.

6.  **`(Optional)` Summary Visualization (`src/create_summary_image.py`)**:
    *   Analyzes the segmentation results from `segmentation_pipeline.py` by reading `results/run_log.json`.
    *   Generates a consensus probability map from all processed masks (from active jobs).
    *   Calculates Dice similarity scores for each job's mask against the consensus.
    *   Creates `results/segmentation_summary_consistency.png` showing all processed segmentations (tiles or full images) overlaid on their respective original images, with cells colored by consistency and Dice scores displayed.

## Features
*   Flexible batch processing: multiple parameter sets across multiple images.
*   Automatic on-the-fly tiling for large images, configured per image.
*   OME-TIFF pre-processing utility.
*   Choice of Cellpose models and detailed parameter tuning.
*   Configurable GPU usage per Cellpose parameter set.
*   Stitching of tiled segmentation masks.
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
    python -m src.preprocess_ometiff path/to/your.ome.tif images/extracted_planes --channel X --zplane Y --prefix ome_extract
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
    *   `"rescaling_config"`: (Object, Optional) If present, image (before tiling) will be rescaled.
        *   `"scale_factor"`: (Float) e.g., `0.5` for 50%. Values > 0 and <= 1.0.
        *   `"interpolation"`: (String, Optional) OpenCV interpolation: "INTER_NEAREST", "INTER_LINEAR", "INTER_AREA" (default), "INTER_CUBIC", "INTER_LANCZOS4".

*   **`cellpose_parameter_configurations` (List of Objects):**
    *   `"param_set_id"`: (String) Unique identifier for this set of Cellpose parameters (e.g., "DefaultCyto3", "AggressiveSmallCells"). Used in output folder naming.
    *   `"is_active"`: (Boolean, Optional) `true` (or omitted) to use this parameter set; `false` to skip.
    *   `"MODEL_CHOICE"`, `"DIAMETER"`, `"FLOW_THRESHOLD"`, `"MIN_SIZE"`, `"CELLPROB_THRESHOLD"`, `"FORCE_GRAYSCALE"`, `"USE_GPU"`.

### Step 2: Run Segmentation Pipeline
1.  **Configure Parallel Processing:** Edit `MAX_PARALLEL_PROCESSES` at the top of `src/segmentation_pipeline.py`.
    *   **GPU Users:** If any active Cellpose parameter configuration uses GPU, set `MAX_PARALLEL_PROCESSES = 1` for stability on single-GPU systems.
2.  **Execute:** (Run from project root)
    ```bash
    python -m src.segmentation_pipeline
    ```
    *   This will first rescale/tile images if configured (tiles stored in `images/tiled_outputs/<image_id>/`), then run Cellpose segmentation.
    *   Outputs are saved in `results/<image_id>_<param_set_id>/` (for non-tiled) or `results/<image_id>_<param_set_id>_<cleaned_tile_name>/` (for tiles).
    *   `results/run_log.json` is created/updated.

### Step 3: Stitch Tiled Segmentations (If Tiling Was Used)
If tiling was applied to an image for a particular parameter set:
1.  **Run `stitch_masks.py` Script:** (Run from project root)
    ```bash
    python -m src.stitch_masks <original_image_id> <param_set_id> <output_dir_for_stitched> --output_filename <stitched_mask.tif>
    ```
    *   Example:
        ```bash
        python -m src.stitch_masks XeniumImage1_Scaled50 Cellpose_DefaultDiam results/XeniumImage1_Scaled50_Cellpose_DefaultDiam_STITCHED
        ```
    *   This saves the combined mask (e.g., `results/XeniumImage1_Scaled50_Cellpose_DefaultDiam_STITCHED/stitched_mask.tif`).

### Step 4: Map Transcripts to Cells (Optional)
1.  **Prepare Inputs:** Path to transcript file, chosen segmentation mask (`_mask.tif` from a job or a `stitched_mask.tif`), MPP values (corresponding to the chosen mask's resolution!), and any offsets.
2.  **Run Transcript Mapping Script:** (Run from project root)
    ```bash
    python -m src.map_transcripts_to_cells \
        "path/to/transcripts.parquet" \
        "path/to/your_chosen_mask.tif" \
        "path/to/mapping_output_directory" \
        --mpp_x <val> --mpp_y <val> 
        # ... other arguments ...
    ```

### Step 5: Generate Summary Image of Segmentations (Optional)
(Compares individual segmentation jobs, which may include tiles)
1.  **Run Summary Script:** (Run from project root)
    ```bash
    python -m src.create_summary_image
    ```
2.  View `results/segmentation_summary_consistency.png`.

## Troubleshooting
*   **Import Errors:** Always run scripts as modules from the project root, e.g., `python -m src.script_name`.
*   **Tiling:** Check `images/tiled_outputs/<image_id>/` for generated tiles and `_manifest.json`.
*   **OME-TIFF Pre-processing:** Verify `--channel`, `--zplane`, `--series` arguments.
*   **Segmentation Errors:** Check `results/run_log.json` and `error_log.txt` in specific job output folders.
*   **Stitching Errors:** Ensure `run_log.json` has successful tile jobs for the target IDs. Original image dimension inference might need adjustment.
*   **Transcript Mapping:** Crucially, ensure `--mpp_x`, `--mpp_y` (and offsets) match the resolution and coordinate system of the *mask file being used*.
*   **GPU Errors:** If using GPU, set `MAX_PARALLEL_PROCESSES = 1` in `segmentation_pipeline.py` for single-GPU systems.

