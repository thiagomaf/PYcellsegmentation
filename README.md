# Advanced Cell Segmentation & Analysis Pipeline using Cellpose

This project provides a Python-based pipeline for segmenting cells in microscopy images using Cellpose. It supports applying multiple parameter sets across a common list of images, configurable GPU usage, and outputs raw segmentation masks, cell coordinates, and a sophisticated summary image for comparative analysis.

## Core Components

1.  **`src/segmentation_pipeline.py`**:
    *   Performs cell segmentation using various Cellpose models and parameters.
    *   Reads a `parameter_sets.json` file which defines:
        *   A global list of input image filenames (`global_image_filename_list`).
        *   A list of parameter configurations (`parameter_configurations`).
    *   Applies each active parameter configuration to every image in the global list.
    *   Outputs for each job (parameter config + image combination), saved in a unique subfolder:
        *   Raw integer-labeled segmentation mask (`_mask.tif`).
        *   Cell coordinates (centroids) in JSON format (`_coords.json`).
    *   Logs all executed jobs and their status to `results/run_log.json`.
    *   Supports CPU-based parallel processing for multiple jobs.

2.  **`src/create_summary_image.py`**:
    *   Analyzes the results from `segmentation_pipeline.py` by reading `results/run_log.json`.
    *   Generates a consensus probability map from all processed masks (from active jobs).
    *   Calculates Dice similarity scores for each job's mask against the consensus.
    *   Creates a summary image (`segmentation_summary_consistency.png`) where:
        *   Each job's segmentation is overlaid on its original image.
        *   Segmented cells are colored based on their consistency with the consensus.
        *   Dice scores are displayed for each job.

## Features

*   Batch segmentation: apply multiple parameter sets to multiple images.
*   Centralized image list for batch runs via `parameter_sets.json`.
*   Choice of Cellpose models (`MODEL_CHOICE`).
*   Adjustable segmentation parameters: `DIAMETER`, `FLOW_THRESHOLD`, `MIN_SIZE`, `CELLPROB_THRESHOLD`.
*   Configurable GPU usage per parameter configuration.
*   Option to force grayscale processing.
*   Advanced summary visualization with cell consistency coloring and Dice scores.
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
    (Ensures `cellpose`, `opencv-python`, `numpy`, `matplotlib`, `tifffile` are installed).

5.  **Handling Potential OpenMP Errors (Windows):**
    `src/segmentation_pipeline.py` includes `os.environ['KMP_DUPLICATE_LIB_OK']='True'` to help mitigate OpenMP runtime conflicts.

## Workflow

### Step 1: Run Batch Segmentations

1.  **Place Input Images:**
    *   Ensure images are in the `images/` folder (project root).
    *   These filenames should be listed in `global_image_filename_list` within `parameter_sets.json`.

2.  **Configure Experiments (`parameter_sets.json`):**
    *   Create or edit `parameter_sets.json` in the project root. This file now has a specific structure:
        *   **`"global_image_filename_list"`**: (List of Strings) A list of all image filenames (from the `images/` folder) that will be processed by each active parameter configuration.
        *   **`"parameter_configurations"`**: (List of Objects) Each object in this list defines one set of segmentation parameters.
            *   `"experiment_id_base"`: (String) A base name for this parameter set. The actual output folder for a specific image will be `results/<experiment_id_base>_<cleaned_image_name>/`.
            *   `"MODEL_CHOICE"`: (String) E.g., `"cyto3"`, `"cyto2"`.
            *   `"DIAMETER"`: (Number or `null`).
            *   `"FLOW_THRESHOLD"`: (Number or `null`).
            *   `"MIN_SIZE"`: (Number or `null`).
            *   `"CELLPROB_THRESHOLD"`: (Number).
            *   `"FORCE_GRAYSCALE"`: (Boolean).
            *   `"USE_GPU"`: (Boolean).
            *   `"is_active"`: (Boolean, Optional) `true` (or omitted) to run this entire parameter configuration across all global images; `false` to skip it.

    *   **Example `parameter_sets.json` structure:**
        ```json
        {
          "global_image_filename_list": [
            "image1.tif",
            "image2.tif"
          ],
          "parameter_configurations": [
            {
              "experiment_id_base": "params_set_A",
              "MODEL_CHOICE": "cyto3", 
              "DIAMETER": null, 
              "is_active": true
              // ... other parameters ...
            },
            {
              "experiment_id_base": "params_set_B_lowprob",
              "MODEL_CHOICE": "cyto2", 
              "CELLPROB_THRESHOLD": -1.0, 
              "is_active": true
              // ... other parameters ...
            }
          ]
        }
        ```

3.  **Configure Parallel Processing (Optional for Segmentation):**
    *   Edit `MAX_PARALLEL_PROCESSES` at the top of `src/segmentation_pipeline.py`.
    *   **GPU & Parallelism:** If any active job uses GPU, set `MAX_PARALLEL_PROCESSES = 1` for stability on single-GPU systems.

4.  **Execute Segmentation Pipeline:**
    ```bash
    python src/segmentation_pipeline.py
    ```
    This generates output folders like `results/params_set_A_image1/`, `results/params_set_A_image2/`, etc., each containing `_mask.tif` and `_coords.json`. It also creates/updates `results/run_log.json`.

### Step 2: Generate Summary Image and Analysis

1.  **Run Summary Script:**
    ```bash
    python src/create_summary_image.py
    ```
    *   The summary script now reads `results/run_log.json` to find all successfully processed and active jobs to include in the summary. It will only process jobs that were actually run and logged as successful.

2.  **View Outputs:**
    *   **`results/segmentation_summary_consistency.png`**: Visual summary.
    *   **`results/run_log.json`**: A detailed log of all jobs created and their status.
    *   Individual job output folders in `results/` with masks and coordinates.

### (Optional) Step 0: Pre-process OME-TIFFs

If your input images are in complex OME-TIFF format (e.g., from Xenium transcriptomics), use `src/preprocess_ometiff.py` first to extract the correct 2D planes.

1.  **Inspect your OME-TIFF:** Use tools like ImageJ/Fiji or `tifffile` in Python to understand its structure (series, axes order, channel for segmentation, Z-plane).
    ```python
    # Example Python inspection
    import tifffile
    with tifffile.TiffFile('path/to/your.ome.tif') as tif:
        print(tif.ome_metadata)
        for i, s in enumerate(tif.series):
            print(f"Series {i}: axes='{s.axes}', shape={s.shape}")
    ```
2.  **Run `preprocess_ometiff.py`:**
    ```bash
    python src/preprocess_ometiff.py path/to/your.ome.tif output_folder_for_2d_tiffs --channel X --zplane Y --series Z --prefix your_prefix
    ```
    Replace `X`, `Y`, `Z` with the correct indices for your data. This will save standard 2D TIFFs.
3.  **Use Processed TIFFs:** Copy these generated 2D TIFFs into the `images/` folder and list their new filenames in the `"global_image_filename_list"` of your `parameter_sets.json`.

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

