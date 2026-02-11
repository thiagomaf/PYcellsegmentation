# PYcellsegmentation: Advanced Spatial Transcriptomics Analysis

Welcome to **PYcellsegmentation**! 

This project is a powerful, flexible pipeline designed to help researchers analyze spatial transcriptomics data. It bridges the gap between raw microscopy images and biological insights by automating the complex process of identifying cells and mapping gene expression data to them.

Whether you are working with standard tissue slides or large-scale OME-TIFFs, this tool handles the heavy lifting—from image preprocessing and segmentation to transcript mapping and visualization.

![PYcellsegmentation Header](docs/images/image1.png)

---

## 🚀 Key Features

*   **Automated Cell Segmentation**: Uses **Cellpose**, a state-of-the-art deep learning tool, to accurately identify cells and nuclei.
*   **Handles Large Images**: Automatically splits very large images into tiles for processing and stitches them back together seamlessly.
*   **Spatial Mapping**: Assigns transcript locations (e.g., from Xenium or other platforms) to specific cells, creating a feature-cell matrix.
*   **Beautiful Visualizations**: Generates high-quality images overlaying gene expression data onto cell masks.
*   **Batch Processing**: Configure once and run multiple experiments or image channels in parallel.
*    **Modular & Portable Pipeline**: Each individual processing step can be executed either locally or remotely (e.g., Google Colab for GPU acceleration), allowing flexible deployment across different computing environments.

![Pipeline Overview](docs/images/image2.png)

---

## 🏁 Quick Start Guide

Follow these steps to get the pipeline running on your machine.

### 1. Installation

**Prerequisites:** You need Python installed (version 3.9+ recommended).

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/YourUsername/PYcellsegmentation.git
    cd PYcellsegmentation
    ```

2.  **Set up a Virtual Environment**
    It is highly recommended to use a virtual environment to keep dependencies organized.
    *   **Windows:**
        ```powershell
        python -m venv .venv
        Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
        .venv\Scripts\activate
        ```
    *   **Mac/Linux:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### 2. Configuration

**📄 [See the Detailed Configuration Guide](docs/configuration_guide.md)** for full documentation on parameter settings and automated setup.

**Option A: Automated Configuration (Recommended)**
Use the provided script to automatically scan your data folders and generate ready-to-use config files:
```bash
python scripts/create_config_files.py
```

**Option B: Manual Setup**
The pipeline is controlled by two main JSON files in the project root:
*   **`processing_config.json`**: Controls segmentation, tiling, and transcript mapping.
*   **`visualization_config.json`**: Controls which genes are plotted and how they look.

**Basic Manual Workflow:**
1.  Put your images in `data/raw/images/`.
2.  Put your transcripts in `data/raw/transcripts/`.
3.  Edit `processing_config.json` to point to your specific filenames.

![Folder Structure](docs/images/image3.png)

### 3. Running the Pipeline

To run the main segmentation workflow:
```bash
python -m src.segmentation_pipeline --config processing_config.json
```

To map transcripts to the segmented cells:
```bash
python -m src.map_transcripts_to_cells --config processing_config.json
```

To visualize the results:
```bash
python -m src.visualize_gene_expression --proc_config processing_config.json --viz_config visualization_config.json
```

---

## 📂 Project Directory Structure

The project is organized to keep data separate from code:

```
project_root/
├── data/                     # All user data, processed files, and results
│   ├── raw/                  # Original inputs (images, transcripts)
│   ├── processed/            # Intermediate files (segmentation masks, mapping)
│   └── results/              # Final outputs (visualizations, statistics)
├── src/                      # Python source code
├── processing_config.json    # Main configuration file
├── visualization_config.json # Visualization settings
└── ...
```

---

## 📖 Detailed Workflow & Core Components

This pipeline can be run as a cohesive workflow or as individual tools.

### 1. Image Pre-processing (Optional)
Extracts relevant 2D planes from complex OME-TIFFs.
*   **Script:** `src/preprocess_ometiff.py`
*   **Usage:** Useful if your raw data is in multi-stack OME-TIFF format and you need to extract specific channels.

### 2. Segmentation & Tiling
*   **Script:** `src/segmentation_pipeline.py`
*   **What it does:**
    *   Reads `processing_config.json`.
    *   Rescales images if needed (e.g., to match Cellpose training data size).
    *   **Tiling:** If an image is too large, it chops it into overlapping tiles.
    *   **Segmentation:** Runs Cellpose on each image or tile.
    *   Saves masks in `data/processed/segmentation/`.

![Tiling and Stitching](docs/images/image4.png)

### 3. Transcript Mapping
*   **Script:** `src/map_transcripts_to_cells.py`
*   **What it does:**
    *   Takes the X/Y coordinates of transcripts from your `.parquet` or `.csv` files.
    *   Overlays them onto the cell masks created in the previous step.
    *   Assigns each transcript to a cell ID (or marks it as background).
    *   Outputs a **Feature-Cell Matrix** (counts of each gene per cell).

### 4. Visualization
*   **Script:** `src/visualize_gene_expression.py`
*   **What it does:** Creates PNGs showing the cell boundaries and the actual transcript dots for selected genes.
*   **Config:** Use `visualization_config.json` to select which genes to plot and set colors/brightness.

> **[IMAGE5 PLACEHOLDER: VISUALIZATION OUTPUT]**
> *Prompt suggestion: An example output image showing a dark background, bright cell outlines, and colorful dots representing gene expression molecules.*

---

## 🛠️ Command-Line Reference (Advanced)

While the pipeline is designed to be driven by the config files, individual scripts can be run directly for specific tasks or debugging.

### Pre-process OME-TIFFs
```bash
python src/preprocess_ometiff.py <input.ome.tif> <output_dir> --channel 0 --zplane 5
```

### Stitch Tiled Masks
If you ran tiling but need to manually re-stitch:
```bash
python -m src.stitch_masks <image_id> <param_set_id> <output_dir>
```

### Create Individual Overlays
Generate high-quality check images for every segmentation:
```bash
python src/create_individual_overlays.py --config processing_config.json --color "#FFFF00"
```

### Analyze Results (Simple)
Get basic stats (count, area) without complex dependencies:
```bash
python src/simple_cellpose_analysis.py --config processing_config.json
```

---

## ⚙️ Configuration File Details

### `processing_config.json`
*   **`image_configurations`**: List of images to process. Set `is_active: true` to process.
    *   `mpp_x`, `mpp_y`: Microns per pixel (important for tiling).
    *   `tiling_parameters`: Set `apply_tiling: true` for large images.
*   **`cellpose_parameter_configurations`**: Settings for the AI model.
    *   `DIAMETER`: Expected cell size in pixels.
    *   `MODEL_CHOICE`: `cyto2`, `nuclei`, or custom path.
*   **`mapping_tasks`**: Links transcripts to segmentation masks.

### `visualization_config.json`
*   **`default_genes_to_visualize`**: List of genes to plot if not specified in a task.
*   **`tasks`**: Specific visualization jobs (e.g., plotting specific genes on specific images).
