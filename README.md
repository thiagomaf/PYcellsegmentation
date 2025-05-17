# Cell Segmentation Pipeline using Cellpose

This project provides a Python-based pipeline for segmenting cells in microscopy images using the Cellpose library. It supports batch processing of multiple parameter combinations, configurable GPU usage, and outputs raw segmentation masks, cell coordinates, and visual overlay images.

## Features

*   Cell segmentation using pre-trained Cellpose models.
*   Batch processing of multiple experiments with different parameter combinations via a JSON configuration file.
*   Configurable GPU usage per experiment.
*   Option to force grayscale processing for multi-channel images.
*   Adjustable parameters for optimizing segmentation:
    *   Cell probability threshold (`CELLPROB_THRESHOLD`)
    *   Cell diameter (`DIAMETER`)
    *   Flow threshold (`FLOW_THRESHOLD`)
    *   Minimum mask size (`MIN_SIZE`)
    *   Choice of Cellpose model (`MODEL_CHOICE`)
*   Outputs for each experiment (saved in a unique subfolder):
    *   Integer-labeled segmentation mask (`_mask.tif`).
    *   Cell coordinates (centroids) in JSON format (`_coords.json`).
    *   Visual overlay of segmentations on the original image (`_overlay.png`).
*   Basic CPU-based parallel processing for running multiple experiments.

## Installation

1.  **Clone the Repository (if applicable):**
    If this code is in a repository, clone it first.

2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment. In the project's root directory:
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    *   Windows (CMD): `.venv\Scripts\activate`
    *   Windows (PowerShell): `.venv\Scripts\Activate.ps1` (May require `Set-ExecutionPolicy Unrestricted -Scope Process`)
    *   Linux/macOS: `source .venv/bin/activate`
    Your terminal prompt should change (e.g., `(.venv)`).

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This installs `cellpose`, `opencv-python`, `numpy`, etc.

5.  **Handling Potential OpenMP Errors (Windows):**
    The script `src/segmentation_pipeline.py` includes `os.environ['KMP_DUPLICATE_LIB_OK']='True'` at the top to help mitigate OpenMP runtime conflicts on Windows.

## Usage

1.  **Place Input Images:**
    *   The script looks for images in the `images/` folder in the project root.
    *   Ensure the image filenames specified in your `parameter_sets.json` exist in this folder.

2.  **Configure Experiments in `parameter_sets.json`:**
    *   A `parameter_sets.json` file in the project root defines the batch of experiments to run. If it doesn't exist, create one based on the example below.
    *   Each object in the JSON array represents one segmentation experiment with its own set of parameters.
    *   **Key parameters per experiment:**
        *   `"experiment_id"`: (String) A unique name for the experiment. Output files will be saved in `results/<experiment_id>/`.
        *   `"image_filename"`: (String) Filename of the image in the `images/` folder to process for this experiment.
        *   `"MODEL_CHOICE"`: (String) Cellpose model to use (e.g., `"cyto3"`, `"cyto2"`, `"cyto"`, or a path to a custom model). Defaults to `"cyto3"` if omitted.
        *   `"DIAMETER"`: (Number or `null`) Approximate cell diameter in pixels. Set to `null` or `0` for auto-estimation by Cellpose. Example: `30`.
        *   `"FLOW_THRESHOLD"`: (Number or `null`) Cellpose flow threshold. Cellpose default (e.g., 0.4) is used if `null`. Lower values (e.g., `0.2`) might detect more cells with weaker flows.
        *   `"MIN_SIZE"`: (Number or `null`) Minimum number of pixels for a mask. Cellpose default (e.g., 15) is used if `null`.
        *   `"CELLPROB_THRESHOLD"`: (Number) Threshold for cell probability. Lower values (e.g., `-1.0`, `-2.0`) detect more/fainter cells. Defaults to `0.0`.
        *   `"FORCE_GRAYSCALE"`: (Boolean) `true` to force grayscale processing (uses first channel); `false` to let Cellpose handle multi-channel images. Defaults to `true`.
        *   `"USE_GPU"`: (Boolean) `true` to use GPU for this experiment; `false` to use CPU. Defaults to `false`.
            *   **Important for GPU Users:** If any experiment has `"USE_GPU": true` and you have only one GPU, it is **highly recommended** to set `MAX_PARALLEL_PROCESSES = 1` at the top of `src/segmentation_pipeline.py` to run experiments sequentially and avoid GPU memory issues.

    *   **Example `parameter_sets.json` entry:**
        ```json
        {
          "experiment_id": "my_gpu_experiment",
          "image_filename": "sample_image.tif",
          "MODEL_CHOICE": "cyto2",
          "DIAMETER": 35,
          "FLOW_THRESHOLD": 0.3,
          "MIN_SIZE": 20,
          "CELLPROB_THRESHOLD": -0.5,
          "FORCE_GRAYSCALE": true,
          "USE_GPU": true
        }
        ```
        (Your `parameter_sets.json` will be a list `[` ... `]` of such objects).

3.  **Configure Parallel Processing (Optional):**
    *   Open `src/segmentation_pipeline.py`.
    *   At the top, you can adjust `MAX_PARALLEL_PROCESSES`. It defaults to half your CPU cores.
    *   **If using GPU for any experiment, set `MAX_PARALLEL_PROCESSES = 1` for stability on single-GPU systems.**

4.  **Run the Segmentation Pipeline:**
    *   Ensure your virtual environment is activated.
    *   Navigate to the project's root directory.
    *   Execute the script:
        ```bash
        python src/segmentation_pipeline.py
        ```
    The script will process each set of parameters defined in `parameter_sets.json`.

5.  **Outputs:**
    *   A base `results/` folder will be created.
    *   Inside `results/`, a subfolder for each `experiment_id` will contain:
        *   `image_filename_mask.tif`: Integer-labeled segmentation mask. (View with ImageJ/Fiji).
        *   `image_filename_coords.json`: Cell coordinates.
        *   `image_filename_overlay.png`: Visual overlay for quick feedback.
        *   `error_log.txt` (only if an error occurred for that specific experiment).

## Troubleshooting

*   **No cells segmented / Black mask output:**
    *   Adjust `CELLPROB_THRESHOLD`, `DIAMETER`, `FLOW_THRESHOLD`, and `MIN_SIZE` in `parameter_sets.json`.
    *   Experiment with `MODEL_CHOICE`.
    *   Toggle `FORCE_GRAYSCALE`.
    *   Check image quality and contrast.
    *   Use ImageJ/Fiji for `_mask.tif` files.
*   **GPU errors / Out of memory:**
    *   If using GPU (`"USE_GPU": true`), ensure `MAX_PARALLEL_PROCESSES = 1` in `src/segmentation_pipeline.py` if you have a single GPU.
    *   Ensure CUDA drivers and PyTorch with GPU support are correctly installed in your environment.
*   **Errors during execution:**
    *   Check console output and any `error_log.txt` files in experiment subfolders.
    *   Ensure dependencies are installed.

