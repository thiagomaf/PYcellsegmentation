# **Chapter Title:**

**Optimizing Xenium _In Situ_ for Spatially Resolved Gene Expression Profiling in _Medicago truncatula_ Roots and Nodules**

# **Authors:**

# **Abstract**

# **Key Words**

_Medicago truncatula_, Xenium _in situ_, spatial transcriptomics, nodulation, gene expression, root nodules, nitrogen fixation, plant tissue sectioning

**

---

**

# **1**     **Introduction**

# **2**     **Materials**

## 2.1  Probe Design

### 2.1.1     Software and Resources

## 2.2  Tissue Fixation and Paraffin Block Preparation

### 2.2.1     Reagents and Solutions

### 2.2.2     Equipment and Tools

## 2.3  Paraffin Sectioning and Section Slide Preparation

### 2.3.1     Reagents and Solutions

### 2.3.2     Equipment and Tools

## 2.4  Probe Hybridization and Processing Xenium Slides

### 2.4.1     Reagents and Solutions

### 2.4.2     Equipment and Tools

# **3**     **Methods**

## 3.1  Probe Design for Xenium Panel

### 3.1.1     Target Gene Selection

### 3.1.2     Panel Submission and Review

## 3.2  Tissue Fixation and Paraffin Block Preparation

### 3.2.1     Buffer Preparation

### 3.2.2     Harvesting and Tissue Fixation

### 3.2.3     Tissue Dehydration

### 3.2.4     Paraffin Infiltration

### 3.2.5     Paraffin Embedding

## 3.3  Paraffin Sectioning and Section Slide Preparation

### 3.3.1     Optional Tissue Block Trimming & Scoring

### 3.3.2     Paraffin sectioning

### 3.3.3     H&E Staining for Quality Check

### 3.3.4     Section Placement on Xenium Slides

## 3.4  Probe Hybridization and Processing Xenium Slides

### 3.4.1     Deparaffinization & Decrosslinking

### 3.4.2     Probe Hybridization

### 3.4.3     Post Hybridization Wash

### 3.4.4     Ligation

### 3.4.5     Amplification

### 3.4.6     Post Amplification Wash

### 3.4.7     Autofluorescence Quenching

### 3.4.8     Nuclei Staining

## 3.5  Post-Xenium Processing

### 3.5.1     Post-Xenium H&E Staining

·         Are these steps performed?

·         If yes for H&E, is there a written protocol elsewhere, or should we outline a standard one?

·         If yes for Confocal Imaging, are these newly acquired images on a confocal after Xenium and H&E? If so, how is image registration to Xenium coordinates handled? If not, and you're just using the Xenium morphology.ome.tif, then 3.5.2 becomes much simpler.

#### 3.5.1.1 Introduction: Purpose of post-Xenium H&E staining (morphological context on the Xenium slide).

#### 3.5.1.2 Materials: Reagents, solutions, equipment.

#### 3.5.1.3 Protocol: Step-by-step staining procedure adapted for Xenium slides.

#### 3.5.1.4 Expected Outcome: Briefly describe the stained slide.

### 3.5.2     Post-Xenium Confocal Imaging

#### 3.5.2.1 Introduction: Purpose (e.g., high-resolution morphology from H&E or other stains).

#### 3.5.2.2 Materials: Microscope type, reagents (if re-staining).

#### 3.5.2.3 Protocol: Image Acquisition: Microscope setup, imaging parameters.

#### 3.5.2.4 Protocol: Image Registration (Crucial if external images are used):

Detail the method used to align these new confocal images with the Xenium coordinate space (e.g., using fiducial markers, image registration software, manual alignment based on DAPI if reimaged). This is a critical gap if new images are taken, as transcript coordinates are from Xenium.

(If only using Xenium's morphology.ome.tif, this subsection might be very brief, stating that the Xenium-provided morphology image is used directly, and then details fall under 3.5.3).

### 3.5.3     Plant Cell Image Segmentation

This section details the protocol for segmenting plant cell nuclei from microscopy images using Cellpose, primarily focusing on the DAPI channel typically provided in Xenium `morphology.ome.tif` files. While the main workflow describes nuclear segmentation (from which cellular regions can be inferred), we will also discuss adaptations for using cell wall stains. The accurate assignment of transcripts to individual cells or nuclei is crucial for downstream spatial transcriptomic analyses.

#### 3.5.3.1 Introduction to Plant Cell Image Segmentation
Spatial transcriptomics in plants requires precise localization of mRNA transcripts within the context of tissue morphology. Segmentation of individual cells (or their nuclei, as a proxy) is a critical step to enable the assignment of these transcripts to their respective cellular or sub-cellular compartments. Plant tissues, however, present unique challenges for image segmentation due to:
*   **Cell Wall Autofluorescence:** Plant cell walls can autofluoresce across a range of wavelengths, potentially interfering with specific stains or increasing background noise.
*   **Irregular Cell Shapes and Sizes:** Unlike some animal tissues with uniform cell types, plant tissues often contain cells with diverse morphologies and sizes, particularly in complex structures like roots and nodules.
*   **Dense Packing and 3D Architecture:** Cells in plant tissues are often densely packed, and their three-dimensional nature requires careful consideration during 2D image acquisition and analysis, or true 3D segmentation approaches.

This protocol leverages Cellpose, a deep learning-based segmentation tool, to address these challenges. We primarily focus on segmenting nuclei using the DAPI stain channel from Xenium `morphology.ome.tif` images. These nuclear masks can then be used directly for transcript assignment to nuclei or serve as seeds for estimating cellular boundaries. The pipeline is designed with flexibility, and we will briefly touch upon adapting it for images where cell walls are directly stained.

#### 3.5.3.2 Prerequisites and Setup

**Software:**
*   **Cellpose:** Version 2.0 or later is recommended. Cellpose can be installed into a dedicated conda environment.
    ```bash
    # Create a new conda environment (e.g., for Python 3.8)
    conda create -n cellpose_env python=3.8
    conda activate cellpose_env
    # Install Cellpose and dependencies
    pip install cellpose
    # For GPU acceleration (if NVIDIA GPU and CUDA are set up):
    # pip install cellpose[gui,mxnet_cu11x] or cellpose[gui,torch_cuda]
    # Refer to official Cellpose documentation for the latest installation instructions.
    ```
*   **Python Environment:** A Python environment (e.g., Python 3.8+) with standard scientific libraries such as:
    *   `tifffile`: For reading and writing TIFF files, including OME-TIFF metadata.
    *   `numpy`: For numerical operations.
    *   `scikit-image`: For image processing tasks (though Cellpose handles much internally).
    *   `pandas`: For handling tabular data if manipulating image metadata or results.
    These are typically installed as dependencies of Cellpose or can be added with `pip install tifffile numpy scikit-image pandas`.
*   **ImageJ/Fiji or QuPath (Optional):** Useful for initial image exploration, defining Regions of Interest (ROIs) if not processing the entire image, or for visual inspection of segmentation results.

**Hardware:**
*   **CPU:** Cellpose can run on a standard CPU. Segmentation times will vary depending on image size and CPU performance.
*   **GPU (Recommended):** For significantly faster processing, especially on large images or batches, an NVIDIA GPU compatible with PyTorch or MXNet (the backends Cellpose uses) is highly recommended. Ensure CUDA drivers are correctly installed.
*   **Memory (RAM):** For large 2D images or 3D Z-stacks, ensure sufficient RAM is available. This can range from 8GB to 32GB+ depending on the data. Processing very large images might require tiling (see 3.5.3.3).

**Input Data:**
*   **`morphology.ome.tif` file:** The primary input is typically the multi-page OME-TIFF file generated by the Xenium analyzer, which contains DAPI staining (for nuclei) and potentially other morphological information. This protocol assumes the DAPI channel will be used for nuclear segmentation.
*   **Directory Structure:** While not strictly enforced by Cellpose itself, organizing your data as suggested in the main project `README.md` (e.g., raw images in `data/raw/images/<experiment_name>/`) is good practice, especially when using the provided batch processing scripts.

#### 3.5.3.3 Image Preparation for Cellpose

Raw `morphology.ome.tif` files from Xenium are often image pyramids, meaning they contain multiple resolutions (levels) of the same field of view. Cellpose, however, generally operates on a single 2D image or a 3D stack (Z, Y, X). Therefore, an initial preparation step is usually required.

*   **Understanding OME-TIFF Image Pyramids:**
    *   Xenium `morphology.ome.tif` files store images in a pyramidal format. Level 0 is the highest resolution, Level 1 is downsampled by a factor of 2 (half resolution, 1/4 pixels), Level 2 by a factor of 4, and so on.
    *   This structure allows for efficient viewing at different zoom levels but requires selecting a specific level for processing with tools like Cellpose.
    *   The choice of level depends on a trade-off:
        *   **Higher resolution (e.g., Level 0, 1):** Provides more detail, potentially better for small nuclei, but results in larger image files, increased computational load, and longer processing times. May require tiling.
        *   **Lower resolution (e.g., Level 2, 3):** Smaller files, faster processing. May be sufficient if nuclei are relatively large and well-separated at this resolution. Risk of losing detail for smaller or faint nuclei.

*   **(Protocol Step) Extracting a Single Resolution Level and Channel:**
    The `src/preprocess_ometiff.py` script can be used to extract a specific resolution level and channel from the Xenium `morphology.ome.tif`. We focus on extracting the DAPI channel.
    *   **Objective:** To create a standard 2D TIFF file (or a Z-stack if performing 3D segmentation across multiple Z-planes from the Xenium output) containing only the DAPI signal at the desired resolution.
    *   **Method:**
        1.  **Identify Target Level and Channel:** Determine the appropriate pyramid level (e.g., Level 2, as used in some internal tests for a balance of detail and speed). The DAPI channel is usually Channel 0 in Xenium `morphology.ome.tif` files.
        2.  **Run `preprocess_ometiff.py`:**
            ```bash
            python src/preprocess_ometiff.py <input_path.ome.tif> <output_directory> --channel <dapi_channel_index> --series <pyramid_level_index> --prefix <output_file_prefix>
            ```
            *   **Example:** To extract Level 2 (series index 2, as pyramid levels are often 0-indexed in `tifffile` access) of the DAPI channel (channel index 0) from `my_sample.ome.tif` and save it to `data/processed/images/`:
                ```bash
                python src/preprocess_ometiff.py data/raw/images/my_sample/my_sample.ome.tif data/processed/images/my_sample/ --channel 0 --series 2 --prefix my_sample_L2_dapi
                ```
            *   This would produce an output like `data/processed/images/my_sample/my_sample_L2_dapi_ch0_series2.tif`.
    *   **Output:** A single-channel TIFF image representing the DAPI stain at the chosen resolution. This image will be the input for Cellpose.
    *   **Note on Z-planes:** If your `morphology.ome.tif` contains multiple Z-planes and you intend to perform 3D segmentation, ensure `preprocess_ometiff.py` is configured to extract all relevant Z-planes for the chosen series. Cellpose can perform 3D segmentation if given a Z-stack. If only a single Z-plane is extracted, 2D segmentation will be performed.

*   **(Protocol Step - Conditional) Image Tiling for Large Datasets:**
    If the extracted image (even at a lower resolution like Level 2) is still very large (e.g., > 4000x4000 pixels, depending on available RAM/VRAM), it might exceed memory limits or be slow to process. In such cases, tiling the image into smaller, manageable overlapping patches is recommended. Cellpose can then be run on each tile, and the resulting masks can be stitched back together (see Section 3.5.4.2).
    *   **Rationale:** Reduces memory footprint per processing job, can enable parallel processing of tiles, and makes handling very high-resolution data feasible.
    *   **Method using project scripts:** The project includes functionality for tiling, often orchestrated by `src/segmentation_pipeline.py` based on `tiling_parameters` within the `processing_config.json` file.
        *   Key parameters in `processing_config.json` under `image_configurations -> <your_image> -> segmentation_options -> tiling_parameters`:
            *   `apply_tiling`: (boolean) Set to `true` to enable tiling.
            *   `tile_size_x`, `tile_size_y`: Dimensions of each tile (e.g., 1024, 1024 pixels).
            *   `overlap_x`, `overlap_y`: Overlap between adjacent tiles (e.g., 100-200 pixels) to ensure seamless stitching and avoid losing cells at tile boundaries.
    *   **Alternative Method (Manual/Other Tools):** Tools like ImageJ/Fiji or QuPath, or custom Python scripts using libraries like `scikit-image`, can also be used to define ROIs and save them as tiles. Ensure pixel size metadata is preserved or correctly communicated to Cellpose.
    *   **Output:** A directory of tiled TIFF images.

#### 3.5.3.4 Determining Cellpose Parameters

Cellpose requires several parameters. The most critical for nuclear segmentation are the model type, estimated object diameter, and channel settings.

*   **(Protocol Step) Obtaining Image Pixel Size (Microns per Pixel - MPP):**
    The physical size of a pixel is crucial for estimating the `diameter` parameter in pixels. This information is usually stored in the OME-XML metadata of the `morphology.ome.tif`.
    *   **Method:**
        1.  The `mpp_x` and `mpp_y` values for the *original Level 0 image* are often known from the Xenium instrument or its output files. These should be recorded in the `image_configurations` section of `processing_config.json` for each image:
            ```json
            // In processing_config.json:
            "image_configurations": [
              {
                "image_id": "MySample",
                "original_image_filename": "path/to/your/image.ome.tif",
                "mpp_x": 0.2125, // Example: MPP for Level 0
                "mpp_y": 0.2125, // Example: MPP for Level 0
                // ... other settings
              }
            ]
            ```
        2.  If you extracted Level `N` (e.g., Level 2), the pixel size for that level will be `MPP_Level0 * (2^N)`. For example, if Level 0 MPP is 0.2125 µm/pixel and you are using Level 2, the MPP for Level 2 is `0.2125 * (2^2) = 0.2125 * 4 = 0.85` µm/pixel. This calculation is typically handled by `src/segmentation_pipeline.py` when preparing jobs.
    *   **Verification (Optional):** Python scripts using `tifffile` can read OME-XML metadata directly:
        ```python
        # Example snippet to read pixel size from an OME-TIFF
        import tifffile
        import xml.etree.ElementTree as ET

        def get_mpp_from_ometiff(image_path, series_index=0):
            with tifffile.TiffFile(image_path) as tif:
                if not tif.ome_metadata:
                    print("No OME metadata found.")
                    return None
                ome_xml = ET.fromstring(tif.ome_metadata)
                # Namespace is often present in OME-XML
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                pixels_element = ome_xml.find(f'.//ome:Image[@ID="Image:{series_index}"]/ome:Pixels', ns)
                if pixels_element is not None:
                    mpp_x = pixels_element.get('PhysicalSizeX')
                    mpp_y = pixels_element.get('PhysicalSizeY')
                    # Units are also available, e.g., PhysicalSizeXUnit
                    return float(mpp_x), float(mpp_y)
            return None

        # mpp = get_mpp_from_ometiff("path/to/your/extracted_level.tif")
        # if mpp:
        # print(f"Pixel size (MPP X, MPP Y): {mpp}")
        ```

*   **(Protocol Step) Calculating `--diameter` Parameter:**
    Cellpose uses an estimated `diameter` of the objects to be segmented, in pixels.
    *   **Estimation:**
        1.  Estimate the average diameter of plant nuclei in your images *in microns*. This can be done by observing images in Fiji/ImageJ and using its measurement tools, or based on prior knowledge of plant cell biology (e.g., typical _Medicago truncatula_ nuclei might be 5-15 µm, but this varies greatly by cell type and tissue).
        2.  Convert this micron diameter to pixels using the MPP for the *specific image level you are segmenting*:
            `diameter_pixels = average_nucleus_diameter_microns / mpp_for_current_level_microns_per_pixel`
    *   **Example:** If average nuclei are 10 µm and you are segmenting a Level 2 image with an MPP of 0.85 µm/pixel:
        `diameter_pixels = 10 µm / 0.85 µm/pixel = ~11.76 pixels`
    *   **Cellpose `diameter` parameter:**
        *   If `diameter` is set to `0` or `None` (as in the example `processing_config.json`'s `Cellpose_DefaultDiam` parameter set, where `DIAMETER: null`), Cellpose will attempt to estimate the diameter from the image. This can work well for some images but may be less reliable for images with varied object sizes or low contrast.
        *   Providing a non-zero `diameter` (in pixels) overrides automatic estimation. This is often recommended for more consistent results, especially in batch processing.
        *   The `processing_config.json` allows setting `DIAMETER` under `cellpose_parameter_configurations`. If set to a value, this value (adjusted for any image rescaling applied by the pipeline) will be used.
    *   **Note:** Experimentation is key. Try segmenting a few representative images or tiles with different `diameter` values (or `diameter=0`) to find what works best. For 3D segmentation (`--do_3D`), Cellpose does not automatically estimate diameter, so it *must* be provided.

*   **Choosing a Pre-trained Model (`--pretrained_model` or `MODEL_CHOICE`):**
    Cellpose offers several pre-trained models.
    *   **For DAPI-based nuclear segmentation:**
        *   `nuclei`: This model is specifically trained to segment nuclei. It is the recommended starting point.
    *   **For cell wall/cytoplasmic segmentation (if adapting the protocol):**
        *   `cyto`: Original general cytoplasm model.
        *   `cyto2`: Improved cytoplasm model, often better for images with less distinct cell boundaries.
        *   `cyto3`: Latest cytoplasm model (as of early 2024), generally robust. The example `processing_config.json` uses `cyto3`. If using this for DAPI, it might work but `nuclei` is more specialized.
    *   **Custom Models:** Cellpose also allows training custom models on your own annotated data, which can provide the best performance for very specific image types but is beyond the scope of this basic protocol.
    *   **Configuration:** This is set via the `MODEL_CHOICE` key in `processing_config.json` under `cellpose_parameter_configurations`.

*   **Channel Settings (`--chan` and `--chan2` for Cellpose command line):**
    These parameters tell Cellpose which image channel(s) to use.
    *   **For single-channel grayscale DAPI images (primary focus of this protocol):**
        *   `--chan 0` (or `1` if using 1-based indexing in a direct command): Specifies the grayscale channel to segment.
        *   `--chan2 0` (or `None`): Indicates no secondary channel (e.g., no cytoplasmic stain is being used in conjunction with a nuclear model for DAPI segmentation).
    *   **Pipeline Handling:** The `src/segmentation_worker.py` script, when `FORCE_GRAYSCALE` is true (default in provided config), prepares a grayscale image. It then internally sets `channels = [0,0]` for Cellpose `model.eval()`, which is appropriate for single-channel segmentation. The parameters `CHAN_FOR_CYTO` and `CHAN_FOR_NUCLEUS` in `processing_config.json` are relevant if you were using a model that takes two channels (e.g. a cytoplasm model with a nuclear channel to help). For a simple DAPI segmentation with the `nuclei` model or a `cyto` model on grayscale DAPI, these specific `CHAN_FOR_CYTO/NUCLEUS` settings in the config might be less critical as the worker defaults to `[0,0]` for the already prepared grayscale image.
    *   **If adapting for a cell wall stain (grayscale):** `channels = [0,0]` (handled by the worker) with `MODEL_CHOICE = "cyto2"` or `"cyto3"` would be appropriate.
    *   **If adapting for a two-channel input (e.g., cell wall stain as main channel, DAPI as secondary):** This would require `FORCE_GRAYSCALE: false` and modifications to how `channels` are derived from `processing_config.json` and passed to `model.eval()`. For example, you might set `channels = [1,2]` (1-indexed for Cellpose CLI: segment channel 1, use channel 2 as nuclear). This is an advanced customization not directly covered by the default pipeline behavior.

*   **Other Key Parameters (Cellpose command line / `processing_config.json` equivalent):**
    *   `--do_3D`: Set if segmenting a Z-stack. The `processing_config.json` would need a way to specify this per parameter set if it's not a global decision. The current `segmentation_worker.py` does not explicitly show a `do_3D` parameter being passed from `job_params_dict` to `model.eval()`, but Cellpose models can be run in 3D if `img` is a 3D array and diameter is given. If 3D is intended, ensure the input `img` to `model.eval()` is indeed 3D (Z,Y,X) and `diameter` is specified.
    *   `--use_gpu`: Enable GPU acceleration. Handled by `USE_GPU_IF_AVAILABLE` in `global_segmentation_settings` and `USE_GPU` in the worker.
    *   `--flow_threshold`: (Default 0.4) Cellpose internal parameter. Can be tuned if segmentation is poor. Accessible in `processing_config.json` (e.g. `FLOW_THRESHOLD`).
    *   `--cellprob_threshold`: (Default 0.0) Threshold for pixel probability of being part of a cell. Can be tuned. Accessible in `processing_config.json` (e.g. `CELLPROB_THRESHOLD`).
    *   `--min_size`: Minimum number of pixels per object. Useful for filtering out small spurious detections. Accessible in `processing_config.json` (e.g. `MIN_SIZE`).

#### 3.5.3.5 Running Cellpose for Segmentation

Segmentation can be performed on individual images/tiles directly using Cellpose from the command line, or in batch using the project's `src/segmentation_pipeline.py` script with a `processing_config.json`.

*   **(Protocol Step) Cellpose Command-Line Execution (for single images/tiles or small batches):**
    This is useful for testing parameters on a single image or tile.
    *   **General Command Structure (for a grayscale DAPI image):**
        ```bash
        python -m cellpose \\
          --image_path /path/to/your/dapi_image.tif \\
          --pretrained_model nuclei \\
          --chan 0 \\
          --chan2 0 \\
          --diameter <calculated_diameter_in_pixels_OR_0_to_autoestimate> \\
          --use_gpu \\
          --save_tif \\
          --verbose
          # Add --do_3D if image_path is a Z-stack and 3D segmentation is desired
          # Add other parameters like --flow_threshold, --cellprob_threshold, --min_size as needed
        ```
    *   **Running on a directory of tiles:**
        ```bash
        python -m cellpose \\
          --dir /path/to/your/tiles_directory/ \\
          --pretrained_model nuclei \\
          --chan 0 \\
          --chan2 0 \\
          --diameter <calculated_diameter_in_pixels_OR_0_to_autoestimate> \\
          --use_gpu \\
          --save_tif \\
          --img_filter _dapi # Example if tiles are named like tile_01_dapi.tif
          --verbose
        ```

*   **(Protocol Step) Batch Processing using `src/segmentation_pipeline.py`:**
    For processing multiple images, different parameter sets, or integrating tiling and rescaling automatically, the `segmentation_pipeline.py` script driven by `processing_config.json` is the recommended method.
    1.  **Configure `processing_config.json`:**
        *   Define `image_configurations` for each raw image, including `image_id`, path, `mpp_x/y`, and `segmentation_options` (rescaling, tiling).
        *   Define `cellpose_parameter_configurations` with different `param_set_id`s, specifying `MODEL_CHOICE`, `DIAMETER` (can be null for auto-estimate if not 3D), `MIN_SIZE`, `CELLPROB_THRESHOLD`, etc.
    2.  **Run the pipeline:**
        ```bash
        python -m src.segmentation_pipeline --config config/processing_config.json
        ```
        *   The pipeline will iterate through active image configurations and active Cellpose parameter sets, performing rescaling and tiling if configured, and then running Cellpose segmentation via `src/segmentation_worker.py`.
        *   Output directories and file naming are handled systematically based on `image_id`, `param_set_id`, scaling, and tile information.

*   **Expected Output from Cellpose/Pipeline:**
    *   **`_seg.npy` files:** These NumPy-format files are the primary output from Cellpose. They contain not only the integer mask array (`masks['masks']`) but also other information like flow fields (`masks['flows']`), styles (`masks['styles']`), and potentially extracted image planes. The mask array is a 2D (or 3D) array where each segmented object (nucleus) is labeled with a unique integer ID, and background is 0.
    *   **TIFF masks (`_mask.tif` or similar):** If `--save_tif` is used (Cellpose CLI) or if the pipeline scripts save TIFFs (which `src/segmentation_worker.py` does, naming them like `*_mask.tif`), these are integer label masks directly viewable in image software.
    *   **Log files and summaries:** The pipeline generates log files and, as seen in `src/segmentation_worker.py`, individual JSON summaries for each segmentation job, which can be useful for tracking parameters and results.

*   **Note on Adapting for Cell Wall Stains:**
    If segmenting grayscale images of cell walls:
    1.  In `processing_config.json`, under the relevant `cellpose_parameter_configurations`, set `MODEL_CHOICE` to `"cyto2"` or `"cyto3"`.
    2.  Ensure the input image provided to Cellpose (after any preprocessing like channel extraction) is a grayscale image representing the cell wall stain. The `FORCE_GRAYSCALE: true` setting in the pipeline helps ensure this.
    3.  The `channels` parameter will be handled as `[0,0]` by `segmentation_worker.py`, which is correct for single-channel grayscale cytoplasmic models.
    4.  Adjust `DIAMETER` and other parameters (`MIN_SIZE`, `CELLPROB_THRESHOLD`, `FLOW_THRESHOLD`) appropriately for the size and appearance of cells, not nuclei.

    For segmenting specific channels from multi-channel color images (e.g., H&E) or using dual-channel Cellpose models (e.g., cytoplasm + nucleus from different image channels), more significant modifications to the `FORCE_GRAYSCALE` settings and channel selection logic within the Python scripts would be required, along with corresponding changes in `processing_config.json` to specify these channel choices. This constitutes an advanced customization.

    Image registration is a prerequisite if these alternative stains are from a different imaging modality or coordinate system than the Xenium transcript data and must be performed *before* this segmentation protocol.

#### 3.5.3.6 Troubleshooting

Common issues encountered during segmentation, particularly with large datasets or specific Python environment configurations, are addressed below.

*   **(Protocol Step) NumPy Pickle Protocol Issue for Large `_seg.npy` Files:**
    *   **Symptom:** When Cellpose saves very large `_seg.npy` files (containing masks, flows, etc.), especially with older NumPy versions or on certain systems, you might encounter an `OverflowError` or `ValueError` related to object size exceeding 2GiB or 4GiB limits. For example:
        `OverflowError: serializing a bytes object larger than 4 GiB is not supported` or
        `ValueError: Object too large to be pickled`
    *   **Cause:** This is often due to NumPy using an older pickle protocol (e.g., protocol 3) by default, which has size limitations. Cellpose uses `numpy.save` which in turn uses `pickle`.
    *   **Fix/Workaround:** Modify NumPy's internal `format.py` to use a higher pickle protocol (e.g., `pickle.HIGHEST_PROTOCOL` which is typically 4 or 5, supporting larger objects). This is a system-level change to your Python environment's NumPy installation.
        1.  **Locate `format.py`:** In your activated conda environment, this file is typically found at a path similar to:
            `<conda_env_path>/lib/pythonX.Y/site-packages/numpy/lib/format.py`
            (Replace `X.Y` with your Python version, e.g., `python3.8`).
        2.  **Edit the file:** Open `format.py` in a text editor. Search for the line (around line 630-680 in NumPy 1.21.x - 1.23.x, line number may vary):
            `pickle.dump(array, fp, protocol=3, fix_imports=fix_imports)`
        3.  **Modify the protocol:** Change `protocol=3` to `protocol=pickle.HIGHEST_PROTOCOL`:
            `pickle.dump(array, fp, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=fix_imports)`
        4.  **Import pickle:** Ensure `pickle` is imported at the top of `format.py` if it's not already (it usually is):
            `import pickle`
        5.  **Save the file.**
    *   **Caution:** Modifying installed library code is generally not ideal. This workaround addresses a specific known issue. Future versions of NumPy or Cellpose might handle this differently. Always back up files before editing if unsure.

*   **(Optional) `MemoryError` for Very Large Arrays:**
    *   **Symptom:** `numpy.core._exceptions.MemoryError: Unable to allocate ... GiB for an array with shape ... and data type ...`
    *   **Cause:** The image (or an intermediate array during processing) is too large to fit into available RAM.
    *   **Solutions:**
        *   **Tiling:** The most effective solution is to tile large images into smaller segments, as described in Section 3.5.3.3. Process each tile individually.
        *   **Lower Resolution:** Use a lower resolution pyramid level if acceptable for segmentation quality.
        *   **Close Other Applications:** Free up system memory by closing unnecessary programs.
        *   **Increase RAM/Swap:** If chronically facing memory issues with necessary data sizes, consider increasing physical RAM or system swap space (though swap is much slower).

#### 3.5.3.7 (Optional/Brief) Visualization of Segmentation Results

Visualizing the segmentation masks overlaid on the original image is crucial for quality control and parameter tuning.
*   **Recommended Tools:**
    *   **ImageJ/Fiji:** Widely used in bioimaging. Can open the output `_mask.tif` files and overlay them on the corresponding DAPI or cell wall stain image (e.g., using the image calculator or overlay functions).
    *   **QuPath:** Excellent for digital pathology and whole-slide imaging; handles large images well. Can import masks and display them as annotation layers.
    *   **Napari:** A Python-based, multi-dimensional image viewer. Highly extensible and suitable for interactive visualization of images and segmentation masks, especially within a Python workflow.
        ```python
        # Example Napari snippet (requires napari and tifffile installed)
        # import napari
        # import tifffile
        # viewer = napari.Viewer()
        # original_image = tifffile.imread('/path/to/your/dapi_image.tif')
        # mask_image = tifffile.imread('/path/to/your/output_mask.tif')
        # viewer.add_image(original_image, name='Original Image', colormap='gray')
        # viewer.add_labels(mask_image, name='Segmentation Mask')
        # napari.run()
        ```
    *   **Cellpose GUI:** The Cellpose graphical interface itself can be used to load images and view segmentation results, and is excellent for testing parameters interactively.
*   **What to check:**
    *   Are nuclei correctly identified and separated?
    *   Are there many false positives (regions segmented that are not nuclei) or false negatives (missed nuclei)?
    *   Do the mask boundaries accurately delineate the nuclei?
    *   Adjust Cellpose parameters (`diameter`, `cellprob_threshold`, `flow_threshold`) and re-segment if results are suboptimal.

### 3.5.4     Post-segmentation Analysis and Transcript Mapping

Once plant cell nuclei (or cells) have been segmented using Cellpose as described in Section 3.5.3, the next critical steps involve refining these segmentations (if necessary, e.g., by stitching masks from tiled images) and then mapping the spatially resolved transcriptomic data to these segmented regions. This section outlines these post-segmentation procedures, enabling the correlation of gene expression with individual cellular units.

#### 3.5.4.1 Introduction

The raw output from Cellpose is a label mask where each identified object (e.g., nucleus) has a unique integer ID. To make this useful for spatial transcriptomics, several processes may be needed:
*   **Stitching (if tiling was used):** If the original image was processed in tiles, the segmentation masks from these individual tiles must be combined into a single, coherent mask for the entire region of interest.
*   **Transcript Mapping:** The spatial coordinates of each detected transcript must be assigned to a specific segmented cell/nucleus. This allows for the creation of cell-by-gene expression matrices.
*   **Quality Control & Visualization:** Visualizing mapped transcripts and segmentation consistency can provide valuable quality control.

This section focuses on the first two steps, with a brief mention of downstream visualization concepts that build upon successful mapping.

#### 3.5.4.2 (Conditional) Stitching Tiled Segmentations

If large images were processed by dividing them into smaller, overlapping tiles (as discussed in Section 3.5.3.3), the resulting segmentation masks for each tile must be stitched together to reconstruct the segmentation for the complete image or ROI.

*   **Rationale:** Tiling allows segmentation of images that would otherwise exceed memory or processing limits. Stitching reassembles these pieces, resolving segmentation identities in the overlap regions to create a continuous map.
*   **(Protocol Step) Using `src/stitch_masks.py`:**
    The project provides `src/stitch_masks.py` for this purpose. This script is designed to work with the output structure of the `segmentation_pipeline.py` when tiling is enabled.
    *   **Inputs:** The script typically requires:
        *   `original_image_id`: The `image_id` from `image_configurations` in `processing_config.json` that corresponds to the image that was tiled.
        *   `param_set_id`: The `param_set_id` from `cellpose_parameter_configurations` in `processing_config.json` that was used for segmenting the tiles.
        *   `run_log`: Path to the `run_log.json` file generated by `segmentation_pipeline.py`. This log contains metadata about the tiles, including their original positions and the specific segmentation jobs.
        *   An output directory for the stitched mask.
    *   **Command-Line Example (conceptual, as it's often run via higher-level orchestration or a dedicated script):**
        The exact command might vary based on how it's integrated. If `stitch_masks.py` were run directly, it might look like:
        ```bash
        python -m src.stitch_masks <original_image_id> <param_set_id> <output_dir_for_stitched_mask> --run_log path/to/run_log.json --output_filename stitched_mask.tif
        ```
        Refer to the `README.md` or the script's argument parser for precise usage.
    *   **Key Parameters & Logic:** The script would use information from the `run_log.json` (or a similar source of tile metadata) to identify all mask files belonging to a specific tiled image and parameter set. It then loads these masks, identifies overlapping regions, and applies logic (e.g., based on Cellpose's `stitch_threshold` if Cellpose's own stitching was run per-tile, or custom logic in `stitch_masks.py`) to resolve cell identities and combine the masks into a single TIFF file.
*   **Expected Output:** A single TIFF file (`stitched_mask.tif` or similar) containing the integer label mask for the entire stitched area.
*   **(Note) Quality Control of Stitching:** Visual inspection of the stitched mask, particularly at the former tile boundaries, is recommended to ensure there are no obvious artifacts or misalignments. Overlap between tiles is crucial for good stitching.

#### 3.5.4.3 Transcript Mapping to Segmented Cells/Nuclei

With a complete segmentation mask (either directly from Cellpose for non-tiled images, or after stitching), the next step is to assign each transcript detected by the Xenium platform to its corresponding segmented nucleus (or cell, if cell boundaries were segmented/inferred).

*   **Introduction:** This process links the molecular data (transcripts) with the spatial data (segmented objects). The primary goal is to count how many transcripts of each gene are located within each segmented region.
*   **Prerequisites:**
    *   **Segmentation Mask:** An integer label TIFF image where each segmented object (nucleus/cell) has a unique positive integer ID, and background is 0. This can be the direct output of Cellpose (e.g., `*_mask.tif`) or the output of the stitching script.
    *   **Transcript Coordinates File:** Typically a `.parquet` or `.csv` file from the Xenium output (e.g., `transcripts.parquet`). This file should contain, at a minimum, the X and Y spatial coordinates for each transcript, the gene identity, and ideally a quality score (e.g., `qv`). Z-coordinates are also relevant if performing 3D analysis.
        *   Key columns often include: `x_location`, `y_location`, `z_location`, `feature_name` (gene name), `qv` (quality value).
*   **(Protocol Step) Using `src/map_transcripts_to_cells.py`:**
    The `src/map_transcripts_to_cells.py` script is designed to perform this mapping. It is typically configured and run as part of the batch processing workflow defined in `processing_config.json`.
    *   **Configuration in `processing_config.json` (`mapping_tasks`):**
        A `mapping_tasks` array in `processing_config.json` defines each mapping job. Each task object specifies:
        ```json
        {
          "task_id": "map_MySample_run1",
          "is_active": true,
          "description": "Map transcripts for MySample using default segmentation.",
          "source_image_id": "MySample", // Links to image_configurations for metadata
          "source_param_set_id": "Cellpose_DefaultDiam", // Links to cellpose_parameter_configurations
          "source_segmentation_is_tile": false, // Or true if mapping to individual tiles before stitching (less common)
          "input_transcripts_path": "data/raw/transcripts/MySample/transcripts.parquet",
          "output_base_dir": "data/processed/mapped/MySample_run1",
          "mask_path_override": null // Optional: directly specify mask path if not following convention
          // Potentially other parameters like nuclear expansion distance, qv_cutoff can be added here or be global
        }
        ```
    *   **Execution:** The mapping is usually invoked by running the main pipeline script, which then calls `map_transcripts_to_cells.py` for active tasks, or by running `map_transcripts_to_cells.py` with specific task IDs.
        ```bash
        python -m src.map_transcripts_to_cells --config config/processing_config.json --task_id map_MySample_run1
        ```
        Or, if running all active mapping tasks:
        ```bash
        python -m src.map_transcripts_to_cells --config config/processing_config.json
        ```
    *   **Mapping Logic (Conceptual):**
        1.  Load the segmentation mask and the transcript coordinates.
        2.  For each transcript, use its X, Y (and Z, if applicable) coordinates to query the segmentation mask and find the integer ID of the segmented object at those coordinates.
        3.  Assign the transcript to that object ID.
        4.  (Optional) Nuclear Expansion: If segmenting nuclei but aiming to capture transcripts in the immediate peri-nuclear cytoplasm, the nuclear masks can be virtually expanded by a defined distance (e.g., a few microns) before transcript assignment. This requires careful parameterization and justification.
        5.  (Optional) Quality Filtering: Transcripts below a certain quality threshold (e.g., `qv < 20`) might be excluded from mapping.
*   **Expected Output:**
    *   **Mapped Transcript Table:** A table (e.g., CSV or Parquet file) listing each transcript, its assigned cell/nucleus ID, its original coordinates, and gene identity. This is useful for detailed inspection and custom analyses.
    *   **Cell-by-Gene Matrix (Feature-Cell Matrix):** A matrix where rows are genes (features) and columns are cells/nuclei (barcodes, derived from the segmentation IDs). Each entry `(i, j)` in the matrix contains the count of transcripts of gene `i` found in cell/nucleus `j`.
        *   This matrix is often output in formats compatible with downstream single-cell analysis packages, such as the MTX (Matrix Market) format, often accompanied by `barcodes.tsv` (listing cell IDs) and `features.tsv` (listing gene names).
    *   Output files are typically saved in the directory specified by `output_base_dir` in the mapping task configuration (e.g., `data/processed/mapped/MySample_run1/`).
*   **(Note) Considerations for Mapping Accuracy:**
    *   **Segmentation Quality:** The accuracy of transcript mapping is fundamentally dependent on the quality of the segmentation. Poor segmentation will lead to incorrect transcript assignments.
    *   **Z-Resolution:** In 2D segmentation of 3D tissue, transcripts above or below the focal plane of segmentation might be misassigned or missed. True 3D segmentation and mapping can mitigate this if Z-stack images and 3D transcript coordinates are available.
    *   **Nuclear vs. Cellular Assignment:** If only nuclei are segmented, assigning all transcripts within that nuclear boundary attributes them to the nucleus. If cell boundaries are desired, methods to infer cell shapes from nuclear segmentations (e.g., Voronoi tessellation, or fixed-radius expansion) might be applied prior to or during mapping, but these are approximations.

#### 3.5.4.4 (Briefly) Downstream Applications and Visualization

With a cell-by-gene matrix and mapped transcript locations, a wide range of spatial transcriptomic analyses can be performed. While detailed methods for these analyses are beyond the scope of this specific segmentation protocol, it is important to note the typical next steps:
*   **Data Exploration and Quality Control:** Examining read counts, gene detection rates per cell, and overall data quality using standard single-cell RNA-seq analysis packages (e.g., Seurat in R, Scanpy in Python).
*   **Clustering and Cell Type Identification:** Grouping cells based on their gene expression profiles to identify known cell types or discover novel ones.
*   **Differential Gene Expression Analysis:** Identifying genes that are differentially expressed between cell types or conditions.
*   **Spatial Pattern Recognition:** Analyzing the spatial distribution of cell types and gene expression patterns within the tissue context.
*   **Visualization of Gene Expression:** The `src/visualize_gene_expression.py` script in this project (configured via `visualization_config.json`) provides examples of how to generate images overlaying gene expression data onto the original microscopy images or segmentation masks. This allows for visual confirmation of mapping results and exploration of spatial gene expression patterns.
    *   This typically involves loading the original image, the segmentation mask, and the mapped transcript data. Specific genes can then be selected, and their expression (e.g., transcript counts per cell, or individual transcript dots) can be color-coded or overlaid onto the image.

#### 3.5.4.5 (Optional but Recommended) Generating Summary Visualizations of Segmentation

Before or after transcript mapping, it can be insightful to generate summary visualizations that assess the consistency and quality of the segmentation itself, especially if multiple parameter sets were tested for Cellpose.
*   **Introduction:** Assessing how different Cellpose parameters (e.g., `diameter`, model choice) affect the segmentation outcome on the same image, or how a single parameter set performs across different images.
*   **(Protocol Step) Using `src/create_segmentation_summary.py`:**
    This script is designed to generate a comparative image of segmentations from different parameter sets.
    *   **Inputs:** It typically requires the `processing_config.json` (to identify image configurations and Cellpose parameters) and the base directory where segmentation results (masks from different runs) are stored.
    *   **Command-Line Example:**
        ```bash
        python -m src.create_segmentation_summary --config config/processing_config.json --results_dir data/processed/segmentation/ --output_filename segmentation_comparison.png
        ```
        (The default `results_dir` in the script might be `results/` at the project root, ensure this points to where the segmentation masks are actually saved, e.g., within `data/processed/segmentation/`).
    *   **Expected Output:** A PNG image that juxtaposes or overlays segmentation masks from different parameter sets, allowing for visual comparison of their consistency, object counts, and boundary definitions.

This concludes the core protocol for plant cell image segmentation and transcript mapping. The resulting data provides a rich foundation for exploring spatially resolved gene expression in plant tissues.