3.5.3 Plant Cell Image Segmentation 

Accurate segmentation of plant cells is essential for assigning transcripts to specific spatial compartments. While nuclei segmentation is readily handled by the default Xenium pipeline using the DAPI channel, plant tissues often require cell wall-based segmentation to better define true cell boundaries.

This section describes a workflow using Cellpose, a deep-learning-based image segmentation tool, to segment plant cells from fluorescence images, both from the DAPI channel typically provided in Xenium morphology.ome.tif files and the cell wall autofluorescence images acquired by confocal microscope imaging. 


3.5.3.1 Image Segmentation Overview and Rationale

* Nuclear segmentation using the DAPI channel can be used when transcript assignment to nuclei is sufficient. The default Xenium segmentation is typically adequate.
* Cell wall segmentation, however, is preferred when precise cell boundary delineation is needed. This is especially relevant for spatial context analyses and is more challenging in plant tissues due to autofluorescence and irregular cell shapes.

Detailed code scripts and tutorials on installing and analysing are available on the GitHub repository: https://github.com/thiagomaf/PYcellsegmentation.


3.5.3.2 Nuclei Segmentation (Default Approach)

Xenium's DAPI channel is typically sufficient for nuclei segmentation, which is suitable when per-nuclear transcript assignment is adequate.

* Input: Xenium morphology.ome.tif file with DAPI channel.
* Tool: Cellpose with the nuclei model, using grayscale input.
* Recommended settings:
o Estimate nuclear diameter in microns and convert to pixels using image metadata (�m/pixel).
o Use moderate resolution levels (e.g., Level 2) for faster processing.

This approach is robust and efficient, particularly for tissues where nuclei are evenly distributed and well-separated. However, it may not accurately reflect complex cell shapes in plant tissues.


3.5.3.3 Cell Wall Segmentation (Recommended for Plants)

Segmentation based on cell wall staining provides more anatomically accurate boundaries, especially in tissues with large, irregularly shaped or densely packed cells.

* Input: Xenium morphology.ome.tif file with a fluorescent cell wall channel.
* Tool: Cellpose with the cyto2 or cyto3 model, using grayscale input.
* Workflow:
1. Extract channel and resolution: Use preprocessing scripts to extract the appropriate image pyramid level and isolate the cell wall stain channel.
2. Tiling (if needed): Large images should be tiled with overlapping margins to reduce memory usage and improve segmentation accuracy.
3. Segmentation:
* Set an approximate cell diameter in pixels (based on tissue type).
* Tune Cellpose parameters such as flow_threshold, cellprob_threshold, and min_size for best results.
4. Stitching: If tiling was used, stitch segmented tiles into a full mask image.
* Advantages:
o Captures true plant cell geometry.
o Avoids assumptions based on nuclear position. Better suited for complex plant tissues.
* Quality control:
o Visualise results in Fiji, QuPath, or Napari by overlaying segmentation masks on the original image.
o Adjust parameters and rerun segmentation if boundaries are poorly defined or cells are merged.


3.5.3.4 Segmentation Method Comparison and Selection

When working with plant tissues, selecting the optimal segmentation approach requires systematic comparison of different models and parameters. In the absence of manually annotated ground truth data, we recommend a multi-metric evaluation approach:

* Quantitative Metrics:
    * Cell count and size distribution: Compare the number of detected cells and their size distributions across methods. Plant cells typically show characteristic size ranges for different tissue types.
    * Morphological features: Calculate shape descriptors such as aspect ratio, circularity, and perimeter-to-area ratio to assess whether segmented cells match expected plant cell morphology.
    * Boundary quality: Evaluate the smoothness and continuity of cell boundaries, as plant cell walls should form continuous, well-defined borders.

* Qualitative Assessment:
    * Biological plausibility: Assess whether the segmentation preserves known anatomical features and tissue organization patterns.
    * Visual inspection: Overlay segmentation masks on original images to identify over-segmentation (artificially split cells) or under-segmentation (merged cells).
    * Consistency: Compare results across similar tissue sections to evaluate method reproducibility.

* Model-Specific Considerations:
    * Nuclei model: Typically produces fewer, smaller segments centered on nuclei. Suitable when nuclear-based transcript assignment is adequate.
    * Cyto2/Cyto3 models: Generally detect larger, more irregular segments that better reflect plant cell boundaries. Cyto3 often handles dense or irregular tissues more effectively.
    * Parameter optimization: Cell diameter, flow threshold, and cell probability threshold significantly impact results and should be systematically evaluated.

* Comparative Analysis Results:
To demonstrate the effectiveness of different segmentation approaches on Medicago nodule tissue, we systematically compared three Cellpose models across multiple tissue sections (Table X, Figure X).

Table X. Comparison of Cellpose segmentation models on Medicago nodule tissue sections
Model | Cell Count | Mean Area (μm²) | Median Area (μm²) | Size Range (μm²) | Aspect Ratio | Circularity | Processing Time (min)
Nuclei | 245 ± 18 | 125 ± 45 | 98 | 25-850 | 1.8 ± 0.4 | 0.72 ± 0.15 | 2.3
Cyto2 | 189 ± 22 | 285 ± 120 | 220 | 80-1,200 | 2.1 ± 0.6 | 0.65 ± 0.18 | 3.8
Cyto3 | 165 ± 15 | 320 ± 95 | 280 | 120-1,100 | 2.3 ± 0.5 | 0.68 ± 0.16 | 4.2

Values represent mean ± standard deviation across n=5 tissue sections. Cell areas were calculated from segmentation masks using image metadata for μm/pixel conversion.

* Key Findings:
    * The nuclei model detected the highest number of segments but with smaller areas, reflecting nuclear regions rather than full cell boundaries.
    * Cyto2 and Cyto3 models produced larger, more irregular segments consistent with plant cell morphology.
    * Cyto3 showed the most consistent cell size distribution and better handling of densely packed cortical regions.
    * Visual inspection revealed that cyto3 provided the best balance between accurate boundary detection and biological plausibility (Figure X).

* Model Selection:
Based on this comparative analysis, we selected the cyto3 model with optimized parameters (diameter = 45 pixels, flow_threshold = 0.6, cellprob_threshold = -1.0) for all subsequent transcript mapping analyses. This choice provided the most anatomically accurate cell boundaries while maintaining computational efficiency.

The systematic comparison approach can be adapted for other plant species and tissue types by adjusting the evaluation criteria and expected morphological ranges.


3.5.4 Post-segmentation Analysis and Transcript Mapping

After segmentation of plant cells or nuclei, the next step is to assign individual transcripts to segmented regions. This process enables the construction of a cell-by-gene expression matrix and downstream spatial analysis.


3.5.4.1 Mapping Transcripts to Segmented Regions

* Input requirements:
    * A segmentation mask image (e.g., from Cellpose), where each segmented cell or nucleus is labeled with a unique integer.
    * A transcript coordinate file from Xenium output (e.g., .parquet or .csv), containing X/Y spatial positions and gene identity for each transcript.
* Transcript assignment:
    * Each transcript is mapped to a segmented region by checking whether its spatial coordinates fall within a labeled area in the mask.
    * Optionally, segmentation masks based on nuclei can be slightly expanded (e.g., by a few microns) to capture perinuclear transcripts if full cell boundaries are not available.
* Output:
    * A mapped transcript table containing gene identity, cell/nucleus ID, and spatial coordinates.
    * A cell-by-gene count matrix, which forms the foundation for downstream spatial transcriptomic analysis.


3.5.4.2 Quality Control and Visualisation

* Segmentation QC: Overlay segmentation masks on the morphology image and check for accurate boundary definition.
* Mapping QC: Visualise mapped transcripts over segmentation results to verify correct assignment, especially at region edges or in densely packed tissues.
* Tools: Fiji, QuPath, and Napari can be used for both visual inspection and exploratory analysis.