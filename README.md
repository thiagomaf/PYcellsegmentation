# Advanced Cell Segmentation & Analysis Pipeline using Cellpose

This project provides a Python-based pipeline for segmenting cells in microscopy images using Cellpose. It features batch processing for parameter optimization and a sophisticated summary tool for comparative analysis, including consistency-based cell coloring and Dice scores against a consensus segmentation.

## Core Components

1.  **`src/segmentation_pipeline.py`**:
    *   Performs cell segmentation using various Cellpose models and parameters.
    *   Processes a batch of experiments defined in `parameter_sets.json`.
    *   Outputs for each experiment:
        *   Raw integer-labeled segmentation mask (`_mask.tif`).
        *   Cell coordinates (centroids) in JSON format (`_coords.json`).
    *   Supports CPU-based parallel processing for multiple experiments.

2.  **`src/create_summary_image.py`**:
    *   Analyzes the results from `segmentation_pipeline.py`.
    *   Generates a consensus probability map from all experiment masks.
    *   Calculates Dice similarity scores for each experiment against the consensus.
    *   Creates a summary image (`segmentation_summary_consistency.png`) where:
        *   Each experiment's segmentation is overlaid on the original image.
        *   Segmented cells are colored based on their consistency with the consensus (red for low, green for high).
        *   Dice scores are displayed for each experiment.

## Features

*   Batch segmentation with configurable parameters per run (`parameter_sets.json`).
*   Choice of Cellpose models (`MODEL_CHOICE`).
*   Adjustable segmentation parameters: `DIAMETER`, `FLOW_THRESHOLD`, `MIN_SIZE`, `CELLPROB_THRESHOLD`.
*   Configurable GPU usage per experiment (with caveats for parallel processing on a single GPU).
*   Option to force grayscale processing.
*   Advanced summary visualization with cell consistency coloring and Dice scores.

## Installation

1.  **Clone the Repository (if applicable).**

2.  **Create a Python Virtual Environment:**
    (Recommended) In the project's root directory:
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    `Set-ExecutionPolicy Unrestricted -Scope Process`
    
    *   Windows (CMD): `.venv\Scripts\activate`
    *   Windows (PowerShell): `.venv\Scripts\Activate.ps1`
    *   Linux/macOS: `source .venv/bin/activate`

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Ensures `cellpose`, `opencv-python`, `numpy`, `matplotlib` are installed).

5.  **Handling Potential OpenMP Errors (Windows):**
    `src/segmentation_pipeline.py` includes `os.environ['KMP_DUPLICATE_LIB_OK']='True'` to help mitigate OpenMP runtime conflicts.

## Workflow

### Step 1: Run Batch Segmentations

1.  **Place Input Images:**
    *   Ensure images are in the `images/` folder (project root).
    *   Image filenames must match those specified in `parameter_sets.json`.

2.  **Configure Experiments (`parameter_sets.json`):**
    *   Create or edit `parameter_sets.json` in the project root. Each JSON object in the list defines one experiment.
    *   **Key parameters per experiment:**
        *   `"experiment_id"`: (String) Unique name for the experiment's results subfolder.
        *   `"image_filename"`: (String) Image filename from the `images/` folder.
        *   `"MODEL_CHOICE"`: (String) E.g., `"cyto3"`, `"cyto2"`, `"cyto"`.
        *   `"DIAMETER"`: (Number or `null`) Cell diameter in pixels; `null` or `0` for auto-estimate.
        *   `"FLOW_THRESHOLD"`: (Number or `null`) Cellpose flow threshold; `null` for default (e.g., 0.4).
        *   `"MIN_SIZE"`: (Number or `null`) Minimum pixels per mask; `null` for default (e.g., 15).
        *   `"CELLPROB_THRESHOLD"`: (Number) Cell probability threshold.
        *   `"FORCE_GRAYSCALE"`: (Boolean) `true` for grayscale, `false` for Cellpose default channel handling.
        *   `"USE_GPU"`: (Boolean) `true` to use GPU.
            *   **GPU & Parallelism:** If any experiment uses GPU, set `MAX_PARALLEL_PROCESSES = 1` in `src/segmentation_pipeline.py` for stability on single-GPU systems.

3.  **Configure Parallel Processing (Optional for Segmentation):**
    *   Edit `MAX_PARALLEL_PROCESSES` at the top of `src/segmentation_pipeline.py`.

4.  **Execute Segmentation Pipeline:**
    From the project root (with virtual environment active):
    ```bash
    python src/segmentation_pipeline.py
    ```
    This populates `results/<experiment_id>/` with `_mask.tif` and `_coords.json` files.

### Step 2: Generate Summary Image and Analysis

1.  **Run Summary Script:**
    After the segmentation pipeline completes for all desired experiments:
    ```bash
    python src/create_summary_image.py
    ```

2.  **View Outputs:**
    *   **`results/segmentation_summary_consistency.png`**: The main visual output showing all experiments with consistency-colored cells and Dice scores.
    *   Individual experiment folders in `results/` will contain the raw masks and coordinate files.
    *   Console output from the summary script will also list Dice scores.

## Troubleshooting

*   **Segmentation Errors (`segmentation_pipeline.py`):**
    *   Check `error_log.txt` in the specific `results/<experiment_id>/` folder.
    *   Review console output for messages.
    *   Ensure parameters in `parameter_sets.json` are valid.
*   **Summary Image Issues (`create_summary_image.py`):**
    *   Ensure `_mask.tif` files exist for all experiments listed in `parameter_sets.json`.
    *   Verify original images are accessible in the `images/` folder.
*   **GPU Errors:**
    *   If using GPU (`"USE_GPU": true`), set `MAX_PARALLEL_PROCESSES = 1` in `segmentation_pipeline.py` for single-GPU systems.
    *   Confirm CUDA and PyTorch GPU setup.
*   **No Cells Segmented (in `_mask.tif`):**
    *   Adjust `CELLPROB_THRESHOLD`, `DIAMETER`, `FLOW_THRESHOLD`, `MIN_SIZE` in `parameter_sets.json`.
    *   Try different `MODEL_CHOICE` or `FORCE_GRAYSCALE` settings.

