# Cell Segmentation Pipeline using Cellpose

This project provides a Python-based pipeline for segmenting cells in microscopy images using the Cellpose library. It's designed to be configurable and outputs raw segmentation masks, cell coordinates, and visual overlay images.

## Features

*   Cell segmentation using pre-trained Cellpose models.
*   Configurable GPU usage.
*   Option to force grayscale processing for multi-channel images.
*   Adjustable cell probability threshold for tuning segmentation sensitivity.
*   Outputs:
    *   Integer-labeled segmentation mask (`_mask.tif`).
    *   Cell coordinates (centroids) in JSON format (`_coords.json`).
    *   Visual overlay of segmentations on the original image (`_overlay.png`).

## Installation

1.  **Clone the Repository (if applicable):**
    If this code is in a repository, clone it first.

2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    Open your terminal in the project's root directory and run:
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    *   **Windows (Command Prompt):**
        ```cmd
        .venv\Scripts\activate
        ```
    *   **Windows (PowerShell):**
        ```powershell
        .venv\Scripts\Activate.ps1
        ```
        (If you encounter issues, you might need to run: `Set-ExecutionPolicy Unrestricted -Scope Process` in PowerShell first.)
    *   **Linux/macOS:**
        ```bash
        source .venv/bin/activate
        ```
    Your terminal prompt should change to indicate the active environment (e.g., `(.venv)`).

4.  **Install Dependencies:**
    Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    This will install `cellpose`, `opencv-python`, `numpy`, and other necessary libraries.

5.  **Handling Potential OpenMP Errors (Windows):**
    If you encounter an error like `OMP: Error #15: Initializing libiomp5md.dll...`, it means multiple OpenMP runtimes are conflicting. The script `src/segmentation_pipeline.py` includes a line at the very beginning to help mitigate this:
    ```python
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    ```
    This line should generally handle the issue. If problems persist, ensure it's the first effective line of code executed.

## Usage

1.  **Place Input Images:**
    *   Create an `images` folder in the project's root directory (if it doesn't exist).
    *   Place your input TIFF images into this `images` folder.
    *   By default, the script looks for an image named `test_image.tif`. You can change this in the script.

2.  **Configure the Pipeline (Optional):**
    Open `src/segmentation_pipeline.py` and modify the configuration variables at the top as needed:
    *   `USE_GPU = False`: Set to `True` if you have a compatible NVIDIA GPU with CUDA installed and want to use it for faster processing.
    *   `FORCE_GRAYSCALE = True`:
        *   Set to `True` if your images are grayscale or if you want to process only the first channel of a multi-channel image as grayscale.
        *   Set to `False` to let Cellpose 4.x use its default channel handling (e.g., for RGB images).
    *   `CELLPROB_THRESHOLD = 0.0`: Adjust this to control segmentation sensitivity.
        *   Lower values (e.g., -1.0, -2.0) detect more, potentially fainter, cells.
        *   Higher values make segmentation more conservative.
    *   `DEFAULT_IMAGE_NAME = "test_image.tif"`: Change this to the filename of the image you want to process if it's different.

3.  **Run the Segmentation Script:**
    *   Ensure your virtual environment is activated.
    *   Navigate to the project's root directory in your terminal.
    *   Execute the script:
        ```bash
        python src/segmentation_pipeline.py
        ```

4.  **Outputs:**
    The script will create a `results` folder (if it doesn't exist) and save the following files in it, prefixed with the original image name:
    *   **`_mask.tif`**: An integer-labeled segmentation mask. Each cell is assigned a unique integer ID.
        *   *Note:* This image might appear black in standard image viewers due to its bit depth and value range. Use software like ImageJ/Fiji or QuPath for proper visualization.
    *   **`_coords.json`**: A JSON file containing the `cell_id` and `x`, `y` coordinates of the centroid for each detected cell.
    *   **`_overlay.png`**: A PNG image showing the segmentation outlines drawn on top of the original image, providing direct visual feedback.

## Troubleshooting

*   **No cells segmented / Black mask:**
    *   Try adjusting `CELLPROB_THRESHOLD` to a lower value (e.g., -1.0, -2.0).
    *   Experiment with the `FORCE_GRAYSCALE` setting based on your image type. If `True`, ensure the primary cell signal is in the first channel.
    *   Check the contrast and quality of your input image.
    *   For the `_mask.tif`, use appropriate software like ImageJ/Fiji to view it correctly.
*   **Errors during execution:**
    *   Ensure all dependencies are installed correctly in the active virtual environment.
    *   Check the console output for specific error messages from Cellpose or other libraries.

