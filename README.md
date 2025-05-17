# Advanced Cell Segmentation & Analysis Pipeline for Spatio-Transcriptomics

This project provides a Python-based pipeline for segmenting cells in microscopy images using Cellpose, followed by mapping transcriptomics data (e.g., from Xenium) to these segmented cells. It features batch processing for parameter optimization and a sophisticated summary tool for comparative analysis.

## Core Components

1.  **`(Optional) src/preprocess_ometiff.py`**:
    *   Pre-processes complex OME-TIFF images (common in microscopy, e.g., Xenium outputs).
    *   Extracts specific 2D image planes based on user-defined series, channel, and Z-plane.
    *   Outputs standard 2D TIFF files suitable for the segmentation pipeline.

2.  **`src/segmentation_pipeline.py`**:
    *   Performs cell segmentation using various Cellpose models and parameters.
    *   Reads a `parameter_sets.json` file which defines:
        *   A global list of input image filenames (`global_image_filename_list`) (these images are typically the output of `preprocess_ometiff.py` or other 2D TIFFs).
        *   A list of parameter configurations (`parameter_configurations`).
    *   Applies each active parameter configuration to every image in the global list.
    *   Outputs for each job (parameter config + image combination), saved in a unique subfolder:
        *   Raw integer-labeled segmentation mask (`_mask.tif`).
        *   Cell coordinates (centroids) in JSON format (`_coords.json`).
    *   Logs all executed jobs and their status to `results/run_log.json`.
    *   Supports CPU-based parallel processing for multiple jobs.

3.  **`src/map_transcripts_to_cells.py`**:
    *   Takes a segmentation mask (e.g., a `_mask.tif` from a chosen segmentation run) and a transcript locations file (e.g., Xenium's `transcripts.parquet`).
    *   Assigns each transcript to a segmented cell ID based on its spatial coordinates.
    *   Requires micron-per-pixel conversion factors and optional coordinate offsets.
    *   Filters transcripts by quality score (QV).
    *   Outputs:
        *   A CSV file of transcripts with their assigned cell IDs (`_transcripts_with_cell_ids.csv`).
        *   A feature-cell matrix in MTX format (`_matrix.mtx`, `_features.tsv`, `_barcodes.tsv`) for downstream analysis in tools like Seurat or Scanpy.

4.  **`src/create_summary_image.py`**:
    *   Analyzes the segmentation results from `segmentation_pipeline.py` by reading `results/run_log.json`.
    *   Generates a consensus probability map from all processed masks (from active jobs).
    *   Calculates Dice similarity scores for each job's mask against the consensus.
    *   Creates a summary image (`segmentation_summary_consistency.png`) where:
        *   Each job's segmentation is overlaid on its original image.
        *   Segmented cells are colored based on their consistency with the consensus.
        *   Dice scores are displayed for each job.

## Features

*   Batch segmentation: apply multiple parameter sets to multiple images.
*   Centralized image list for batch runs via `parameter_sets.json`.
*   OME-TIFF pre-processing utility.
*   Choice of Cellpose models (`MODEL_CHOICE`).
*   Adjustable segmentation parameters: `DIAMETER`, `FLOW_THRESHOLD`, `MIN_SIZE`, `CELLPROB_THRESHOLD`.
*   Configurable GPU usage per parameter configuration.
*   Option to force grayscale processing.
*   Transcript mapping to segmented cells, including QV filtering.
*   Generation of feature-cell count matrices (MTX format).
*   Advanced summary visualization of segmentation results with cell consistency coloring and Dice scores.
*   Detailed logging of executed jobs in `results/run_log.json`.

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
    (Ensures `cellpose`, `opencv-python`, `numpy`, `matplotlib`, `tifffile`, `pandas`, `scipy`, and `pyarrow` (or `fastparquet`) are installed).

5.  **Handling Potential OpenMP Errors (Windows):**
    `src/segmentation_pipeline.py` includes `os.environ['KMP_DUPLICATE_LIB_OK']='True'` to help mitigate OpenMP runtime conflicts.

## Workflow

### (Optional) Step 0: Pre-process OME-TIFFs
If your input images are in complex OME-TIFF format:
1.  **Inspect OME-TIFF:** Use tools like ImageJ/Fiji or `tifffile` in Python to understand its structure (series, axes order, channel for segmentation, Z-plane).
    ```python
    import tifffile
    with tifffile.TiffFile('path/to/your.ome.tif') as tif:
        # print(tif.ome_metadata) # For full XML
        for i, s in enumerate(tif.series):
            print(f"Series {i}: axes='{s.axes}', shape={s.shape}, dtype={s.dtype}")
    ```
2.  **Run `preprocess_ometiff.py`:**
    ```bash
    python src/preprocess_ometiff.py path/to/your.ome.tif output_folder_for_2d_tiffs --channel X --zplane Y --series Z --prefix your_prefix
    ```
    (Replace `X,Y,Z` with correct indices). This saves standard 2D TIFFs.
3.  **Prepare for Segmentation:** Copy these 2D TIFFs into the `images/` folder. List their new filenames in the `"global_image_filename_list"` of your `parameter_sets.json`.

### Step 1: Run Batch Segmentations

1.  **Place Input Images:**
    *   Ensure 2D TIFF images (either original or from Step 0) are in the `images/` folder.

2.  **Configure Experiments (`parameter_sets.json`):**
    *   Edit `parameter_sets.json` with a `global_image_filename_list` and `parameter_configurations` as described previously.
    *   Ensure `is_active` flags and other parameters like `USE_GPU` are set correctly.
    *   **GPU & Parallelism:** If any active job uses GPU, set `MAX_PARALLEL_PROCESSES = 1` in `src/segmentation_pipeline.py` for stability on single-GPU systems.

3.  **Execute Segmentation Pipeline:**
    ```bash
    python src/segmentation_pipeline.py
    ```
    Outputs: `results/<experiment_id_base>_<cleaned_image_name>/` folders with `_mask.tif` and `_coords.json`. Also `results/run_log.json`.

### Step 2: Map Transcripts to Cells (Per Segmentation Run)

1.  **Prepare Inputs:**
    *   Identify the path to your Xenium `transcripts.parquet` file (or similar).
    *   Choose a specific segmentation mask (`_mask.tif`) from one of the successful runs in the `results/` subfolders that you want to use for mapping.
    *   Determine the microns-per-pixel values (e.g., from Xenium metadata like `0.2125`).
    *   Determine any necessary X/Y coordinate offsets between the transcript coordinate system and the image pixel coordinate system.

2.  **Run Transcript Mapping Script:**
    ```bash
    python src/map_transcripts_to_cells.py \
        "path/to/your/transcripts.parquet" \
        "results/chosen_experiment_id/chosen_image_mask.tif" \
        "results/chosen_experiment_id/mapping_output" \
        --mpp_x <microns_per_pixel_x> \
        --mpp_y <microns_per_pixel_y> \
        --qv_threshold <your_qv_threshold> \
        --output_prefix "mapped_data" \
        # --x_offset <offset_x_microns> \
        # --y_offset <offset_y_microns>
    ```
    Replace placeholders. The output directory (e.g., `results/chosen_experiment_id/mapping_output/`) will contain:
    *   `mapped_data_transcripts_with_cell_ids.csv`
    *   `mapped_data_matrix.mtx`, `mapped_data_features.tsv`, `mapped_data_barcodes.tsv`

### Step 3: Generate Summary Image of Segmentations

1.  **Run Summary Script:**
    (Typically run after Step 1 to compare different segmentation parameter sets)
    ```bash
    python src/create_summary_image.py
    ```
    Reads `results/run_log.json` for successfully completed segmentation jobs.

2.  **View Outputs:**
    *   `results/segmentation_summary_consistency.png`: Visual comparison of segmentation runs.
    *   Console output lists Dice scores for each run against the consensus.

## Troubleshooting

*   **OME-TIFF Pre-processing:** If `preprocess_ometiff.py` outputs black or incorrect images, double-check the `--channel`, `--zplane`, and `--series` arguments against your OME-TIFF's actual structure. The `axes_order` printed by the script is key to refining its internal slicing logic if needed.
*   **Segmentation Errors (`segmentation_pipeline.py`):**
    *   Check `results/run_log.json` for the status of all jobs.
    *   For failed jobs, check `error_log.txt` in the specific `results/<experiment_id_base>_<cleaned_image_name>/` folder.
    *   Ensure parameters in `parameter_sets.json` are valid.
*   **Summary Image Issues (`create_summary_image.py`):**
    *   Ensure `_mask.tif` files exist for all successfully processed and logged jobs.
    *   Verify original images are accessible in the `images/` folder.
*   **GPU Errors:**
    *   If using GPU (`"USE_GPU": true`), set `MAX_PARALLEL_PROCESSES = 1` in `segmentation_pipeline.py` for single-GPU systems.
    *   Confirm CUDA and PyTorch GPU setup.
*   **No Cells Segmented (in `_mask.tif`):**
    *   Adjust `CELLPROB_THRESHOLD`, `DIAMETER`, `FLOW_THRESHOLD`, `MIN_SIZE` in `parameter_sets.json`.
    *   Try different `MODEL_CHOICE` or `FORCE_GRAYSCALE` settings.
*   **Transcript Mapping:** Ensure `--mpp_x`, `--mpp_y`, and any `--x_offset`, `--y_offset` values accurately reflect the relationship between your transcriptomic data's micron coordinates and your imaging data's pixel coordinates. Incorrect values will lead to mis-mapping. Check that the `feature_name` and `qv` columns exist in your transcript file.

