# **Comprehensive Guide to Quantitative Evaluation of Plant Tissue Segmentation: From Naive Statistics to Biological Priors**

## **1\. Introduction: The Unique Challenge of Plant Morphometry**

The quantitative analysis of plant tissue morphogenesis relies fundamentally on the accurate segmentation of individual cells from volumetric or two-dimensional microscopy data. As you have correctly identified, the transition from qualitative visual inspection to rigorous, data-driven quality control is a pivotal step in any high-throughput phenotyping pipeline. While the intuition to employ basic statistical measures—such as total pixel area, average cell size, and variance—provides a necessary baseline, the complex, space-filling topology of plant tissues demands a more sophisticated approach to "goodness-of-segmentation."

Unlike animal cell cultures, which often exist as discrete objects separated by a fluid medium, plant tissues represent a dense, tessellated mesh where cell walls are shared boundaries. This topological reality fundamentally alters the definition of segmentation quality. In a sparse cell culture, a segmentation error typically results in a missed object (False Negative) or a hallucinated object (False Positive). In a dense plant tissue, however, an error almost invariably violates the physical laws of the tissue structure: a missed wall merges two cells into a biologically impossible "super-cell," while a false wall splits a functional unit into fragments. Consequently, the evaluation of segmentation quality must move beyond simple pixel counting to encompass topological integrity, morphometric consistency, and distributional stability.1

This report synthesizes the current state-of-the-art (2020–2025) in unsupervised and reference-free segmentation evaluation. It addresses your specific requirements for implementing quantitative scores by bridging your initial "naive" intuitions—such as area sums and size distributions—with their rigorous mathematical counterparts found in advanced literature. We will explore how "total pixel sum" evolves into **Tiling Efficiency** and **Foreground Coverage**, how "average cell size" is refined by **Stereological Priors** and **Log-Normal Physics**, and how "distribution consistency" is rigorously measured using **Earth Mover’s Distance** and **Information Theory**. Furthermore, we examine emerging techniques such as **Reverse Classification Accuracy (RCA)** and **Ensemble Consensus**, which allow for high-precision quality estimation even in the total absence of ground truth annotations.3

## **2\. Statistical Moments and Distributional Consistency**

Your query posits that measures of central tendency and variance (e.g., "average cell size," "cell size variance") could serve as indicators of segmentation quality. This intuition is sound, yet in the context of plant biology, these simple moments can be misleading if the underlying statistical distributions are not accounted for. Plant cell growth is a multiplicative process governed by exponential expansion and distinct division rules, which rarely results in Gaussian (normal) distributions. Therefore, evaluating the "goodness" of a segmentation requires analyzing the full shape of these distributions rather than just their means.

### **2.1 From "Average Size" to Log-Normal Validity**

In uniform plant tissues, such as the meristematic zone of a root or the hypocotyl, cell sizes typically follow a **Log-Normal distribution** rather than a normal distribution. This phenomenon arises because the growth rate of a cell is proportional to its current size (exponential growth), and random fluctuations in growth rates accumulate multiplicatively.5

#### **2.1.1 The Goodness-of-Fit (GoF) Test**

A robust quality metric involves testing the hypothesis that the segmented cell sizes conform to this expected biological distribution. If a segmentation algorithm fails—producing, for example, a large number of tiny "noise" fragments or a few massive under-segmented regions—the resulting size distribution will deviate significantly from log-normality.

To implement this as a quantitative score:

1. **Log-Transformation:** Compute the natural logarithm of the area ($A$) for every segmented instance: $X\_i \= \\ln(A\_i)$.  
2. **Normality Test:** Apply a **Shapiro-Wilk** or **Kolmogorov-Smirnov (KS)** test to the transformed data $X$.  
3. **The Score:** The $p$-value of this test serves as a "Biological Plausibility Score." A high $p$-value ($\>0.05$) indicates that the segmentation preserves the natural variability of the tissue. A vanishingly small $p$-value suggests the presence of non-biological artifacts.

The detection of specific distributional artifacts provides diagnostic insight into the *type* of segmentation error:

* **Left-Skewed Anomalies (Low Area):** A heavy tail on the left side of the distribution (or a secondary peak of very small objects) is a hallmark of **Over-Segmentation**. This occurs when the algorithm incorrectly splits single cells into multiple fragments, often triggered by intracellular organelles or textured vacuoles being mistaken for cell boundaries.1  
* **Right-Skewed Anomalies (High Area):** A heavy tail on the right, or the presence of "outliers" that are integer multiples of the median size, indicates **Under-Segmentation**. This suggests that cell walls were missed, causing neighbors to merge.

#### **2.1.2 Geometric Standard Deviation**

Your suggestion to monitor "cell size variance" is crucial, but the standard deviation ($\\sigma$) is scale-dependent. A more robust metric for log-normal populations is the **Geometric Coefficient of Variation (GCV)** or the standard deviation of the log-transformed data. In stable tissues, the variability of cell size is often tightly regulated. A sudden spike in cell size variance across adjacent image slices (in a z-stack) or time points is rarely biological; it is almost always a sign of segmentation instability, where the algorithm flickers between merging and splitting cells due to minor fluctuations in image contrast.5

### **2.2 Distributional Consistency via Earth Mover's Distance**

You asked for a measure of "how consistent is the distributions of sizes and shapes." The most rigorous metric for this in the current literature is the **Earth Mover's Distance (EMD)**, also known as the Wasserstein Metric.7 Unlike simple histogram comparisons (like the Chi-Square test) or bin-to-bin subtraction, EMD measures the minimal "work" required to transform one distribution into another. This is physically intuitive: if the segmentation shifts slightly (e.g., all cells are 5% larger due to a dilation artifact), EMD reports a small error, whereas histogram metrics might report zero overlap (maximum error) if the bins do not align.

#### **2.2.1 Application in Unsupervised Quality Control**

To use EMD as a "goodness" score without ground truth:

1. **Reference Construction:** Create a "Gold Standard" distribution by aggregating cell sizes/shapes from a small, manually verified subset of data (e.g., 5-10 high-quality images). Alternatively, derive a theoretical distribution from literature parameters for the specific tissue type.  
2. **Distance Calculation:** For every new automated segmentation, calculate the EMD between its size histogram and the Reference Distribution.  
3. **Interpretation:** A low EMD indicates that the segmentation produces a population of cells that is statistically indistinguishable from the verified baseline. A high EMD alerts the user to a "Distributional Shift," which could stem from algorithmic failure (e.g., severe over-segmentation shifting mass to the left) or a genuine biological phenotype (e.g., a mutant with giant cells).9

Recent advances have extended this to **Hierarchical EMD (H-EMD)**, which allows for checking consistency across multiple scales (e.g., comparing the distribution of nuclei sizes simultaneously with cell sizes), ensuring that the segmentation is consistent not just at the cellular level but also at the organelle level.9

### **2.3 The Coefficient of Variation (CV) as a Homogeneity Index**

For tissues expected to be relatively homogeneous (e.g., a specific layer of the root cortex), the **Coefficient of Variation** ($CV \= \\sigma / \\mu$) acts as a potent quality filter.

* **Quality Control Logic:** If the segmentation is accurate, the CV of geometric features (Area, Circularity) should remain stable. A segmentation algorithm that fails on a subset of the image (e.g., due to uneven lighting) will introduce a sub-population of artifacts, inflating the standard deviation relative to the mean and causing a spike in the CV.11  
* **Thresholding:** In high-throughput screens, images with a CV exceeding a pre-defined threshold (derived from control samples) are often automatically flagged for manual review or discarded, as this metric effectively captures "noisy" segmentation results.13

### **Table 1: Comparative Analysis of Naive vs. Advanced Statistical Metrics**

| Naive Metric (User Request) | Advanced Metric (Literature Best Practice) | Mathematical Formulation | Biological Insight / Implementation Benefit |
| :---- | :---- | :---- | :---- |
| **Total Pixel Sum** | **Foreground Coverage Ratio (FCR)** | $\\frac{\\sum \\text{Area}\_{segments}}{\\text{Total Tissue Area}}$ | Detects "Coverage Gaps" where cells are missed entirely or "Overlaps" which are physically impossible in single-layer segmentation. |
| **Average Cell Size** | **Log-Normal Location Parameter ($\\mu$)** | Median of $\\ln(\\text{Area})$ | Robust against outliers; aligns with the multiplicative nature of cell division and expansion. |
| **Size Variance** | **Geometric Coefficient of Variation** | $\\sqrt{e^{\\sigma^2} \- 1}$ | Scale-independent measure of heterogeneity; spikes indicate segmentation instability or "flickering" errors. |
| **Distribution Consistency** | **Earth Mover's Distance (EMD)** | $\\inf E$ | Quantifies the "cost" to transform the segmented population into a reference population; robust to small calibration shifts. |
| **Shape Consistency** | **PCA Shape Space Projection** | Mahalanobis Distance in PCA space | Identifies multivariate outliers (e.g., objects that are the right size but wrong shape) as segmentation artifacts. |

## **3\. Tiling, Topology, and Space-Filling Constraints**

Your initial thought to look at the "total px sum of segments area" touches upon a fundamental property of plant tissues: they are **tessellations**. Unlike a petri dish of disjoint animal cells, where the "background" is a real physical space, the "background" in a healthy plant tissue section (excluding air spaces in spongy mesophyll) is essentially non-existent. Cells fill the available space completely. This "Space-Filling Prior" is one of the most powerful checks for segmentation quality.2

### **3.1 Foreground Coverage and Partition Completeness**

The "total pixel sum" you mentioned is effectively a measure of **coverage**. In a perfect segmentation of a confocal section of a root tip, the sum of the areas of all segmented cells should approximately equal the area of the Region of Interest (ROI).

#### **3.1.1 The Coverage Ratio**

We can define the Foreground Coverage Ratio (FCR) as:

$$FCR \= \\frac{\\sum\_{i=1}^{N} Area(Cell\_i)}{Area(ROI)}$$

* **Under-Segmentation (Gaps):** An FCR significantly less than 1.0 (e.g., \< 0.90) implies "voids" in the segmentation—regions of the tissue that were not assigned to any cell. In watershed-based algorithms, this often happens if the signal-to-noise ratio is too low to define a boundary, leading the algorithm to discard the region as background.15  
* **Over-Segmentation (Overlap):** An FCR \> 1.0 implies that segmented regions are overlapping. In 2D instance segmentation of a single focal plane, overlap is physically impossible for plant cells sharing a wall. Any overlap is a definitive segmentation error (e.g., double-counting a thick cell wall).

### **3.2 Topological Integrity: The Euler Characteristic**

Beyond area, the **topology** of the cells is a rigorous indicator of quality. A valid 2D plant cell segmentation mask should consist of simple, simply connected polygons.

* **The Euler Number ($\\chi$):** For a 2D binary object, $\\chi \= (\\text{Number of Objects}) \- (\\text{Number of Holes})$.  
* **The "Goodness" Condition:** For a single segmented plant cell mask, we expect exactly **one object** and **zero holes**, yielding $\\chi \= 1$.  
* **Common Failure Mode ("Swiss Cheese"):** A common issue in fluorescence microscopy is that the vacuole (occupying 90% of the cell volume) is not stained, appearing dark. Naive intensity thresholding often segments the cytoplasm (bright) but leaves the vacuole as a "hole" in the mask. This results in $\\chi \\le 0$.  
* **Automated Scoring:** By calculating the Euler number for every segmented instance (using scikit-image.measure.regionprops), one can generate a "Topological Integrity Score": the percentage of cells with $\\chi \= 1$. A low score instantly flags a failure to handle intracellular heterogeneity.17

### **3.3 Junction Dynamics and Vertex Analysis**

The geometry of cell junctions provides a sophisticated "goodness" metric derived from the physics of cell walls.

* **Plateau's Laws / The Rule of Three:** In stable tissues, surface tension minimization dictates that cell walls meet at **trivalent vertices** (three walls meeting at approximately 120 degrees).  
* **Vertex Degree Metric:** By skeletonizing the segmentation mask (reducing walls to 1-pixel lines) and analyzing the graph nodes:  
  * **Degree-3 Nodes:** Expected biological standard.  
  * **Degree-4 Nodes:** Rare/Unstable. A high prevalence indicates a "Grid Artifact," often caused by pixel-grid bias in graph-cut algorithms that favor vertical/horizontal cuts.  
  * **Degree-1 Nodes (Dead Ends):** These represent **broken walls**. A wall that simply ends in the middle of a cell is a topological impossibility in healthy tissue. It indicates a failure to close a contour, leading to under-segmentation (two cells merging through the gap).20

### **3.4 Partition Distance and Stability**

The **Partition Distance** measures how different two segmentations are. While typically used to compare to Ground Truth, it can be adapted for **Stability Analysis** (Unsupervised).

* **Perturbation Testing:** By adding a small amount of Gaussian noise to the input image and re-running the segmentation, one can measure the Partition Distance between the original result and the noisy result.  
* **Insight:** A robust ("good") segmentation should be stable; the Partition Distance should be small. If a small amount of noise causes a massive reshuffling of cell boundaries (high Partition Distance), the segmentation is "brittle" and likely over-fitting to image noise rather than detecting true boundaries.21

## **4\. Morphological Priors: Shape as a Quality Filter**

Your request mentions "consistent distributions of sizes and shapes." While size is useful, shape is often more discriminatory for identifying artifacts. Plant cells are constrained by turgor pressure to be convex (or near-convex) polyhedra. This biological prior allows us to define "goodness" based on geometric properties.

### **4.1 Solidity and Convexity**

Solidity is defined as the ratio of the object's area to the area of its convex hull:

$$\\text{Solidity} \= \\frac{\\text{Area}}{\\text{ConvexHullArea}}$$

* **The "Dumbbell" Effect (Under-Segmentation):** When a cell wall is missed, two adjacent cells are merged. Since plant cells are typically convex, the union of two convex shapes is often non-convex (forming a "dumbbell" or "peanut" shape with a narrow waist). This drastically reduces Solidity.  
* **Metric:** A segmentation where the mean Solidity drops below \~0.85 (depending on tissue type) is highly suspect for under-segmentation errors. This serves as a rapid, computation-light filter for quality.23

### **4.2 Circularity and Boundary Roughness**

Circularity (or Isoperimetric Quotient) measures the compactness of the cell:

$$\\text{Circularity} \= \\frac{4\\pi \\cdot \\text{Area}}{\\text{Perimeter}^2}$$

* **The "Fractal" Effect (Over-Segmentation/Noise):** Watershed algorithms can sometimes produce "fractal" boundaries that follow pixel noise rather than smooth walls. This increases the perimeter without adding area, lowering Circularity.  
* **Caveat:** While effective for meristematic cells (which are roughly isodiametric), this metric must be used with caution for specialized cells like pavement cells in the epidermis, which biologically form complex, interlocked jigsaw shapes. For such tissues, "Lobeyness" or Skeleton-based descriptors are preferred.6

### **4.3 PCA-Based Shape Anomaly Detection**

To integrate multiple shape descriptors (Area, Solidity, Eccentricity, Major Axis Length) into a single "goodness" score, literature recommends **Principal Component Analysis (PCA)**.

* **Method:**  
  1. Extract a vector of shape features for every cell.  
  2. Perform PCA to project these vectors into a low-dimensional "Shape Space."  
  3. **Elliptical Envelope:** Define a confidence boundary (e.g., 95%) around the cluster of data points.  
* **The Score:** The percentage of cells falling *outside* this envelope. These outliers represent segmentation failures—objects that are too long, too jagged, or too irregular to be valid cells. This method is essentially an unsupervised anomaly detector trained on the image's own population statistics.5

## **5\. Reference-Free Evaluation: The "Pseudo-Ground Truth" Revolution**

Perhaps the most significant development in segmentation evaluation between 2020 and 2025 is the ability to generate quantitative scores that correlate with accuracy *without* requiring manual ground truth. This directly addresses your need to "infer the quality" in a rigorous way.

### **5.1 Reverse Classification Accuracy (RCA)**

**Reverse Classification Accuracy (RCA)** is a framework that assesses quality by checking if the segmentation contains enough signal to train a model.3

* **The Hypothesis:** If a segmentation mask is accurate, it captures the true relationship between the image features (intensity, texture) and the labels. A machine learning model trained on this data should effectively learn "what a cell looks like." If the segmentation is garbage, the model learns noise.  
* **The Protocol:**  
  1. **Step 1 (Prediction):** Run your segmentation algorithm on Image $I$ to get Mask $M$.  
  2. **Step 2 (Reverse Training):** Train a *new* classifier (the "Reverse Classifier") using $I$ as the input and $M$ as the Ground Truth.  
  3. **Step 3 (Validation):** Apply this Reverse Classifier to a small, separate set of reference images ($I\_{ref}$) for which you *do* have high-quality labels.  
  4. **The Score:** The Dice score of the Reverse Classifier on $I\_{ref}$ is the RCA score for your original segmentation $M$.  
* **Correlation:** Studies have shown that the RCA score correlates strongly ($r \> 0.8$) with the true accuracy of the segmentation. This effectively allows you to estimate the Dice score of your segmentation without knowing the true mask for that specific image.

#### **5.1.1 In-Context RCA (Efficient Estimation)**

Training a model for every image is computationally expensive. **In-Context RCA** utilizes modern "Foundation Models" (like UniverSeg or SAM) to perform this check instantly. Instead of training, you provide the $(I, M)$ pair as a "prompt" or "context" to the model and ask it to segment a reference image. The performance on the reference image reflects the quality of the prompt (your segmentation).26

### **5.2 Ensemble Consensus (The "SEG" Method)**

When a single "Ground Truth" is unavailable, the "Wisdom of the Crowds" can serve as a proxy. This method, often referred to as **SEG (Segmentation Evaluation with no Ground truth)**, is widely used in benchmarking competitions.24

* **Mechanism:**  
  1. Apply $N$ different segmentation algorithms (e.g., PlantSeg, Cellpose, StarDist, Classical Watershed) to the same image.  
  2. Generate a **Consensus Mask** (Pseudo-GT) by pixel-wise voting (e.g., a pixel is "cell boundary" if \>50% of models agree).  
  3. **Evaluation:** Score your primary method against this Consensus Mask.  
* **Rationale:** Different algorithms have different failure modes (e.g., Deep Learning models might hallucinate cells, while Watershed might over-segment texture). It is statistically unlikely that all methods will fail in the exact same way at the same pixel. Therefore, high agreement with the ensemble suggests high reliability.  
* **Uncertainty Maps:** The variance between the models can be plotted as a spatial "Confidence Map," highlighting exactly which regions of the tissue are ambiguous and require manual inspection.

### **5.3 Cycle-Consistency (CycleGAN)**

For advanced users, **Cycle-Consistency** offers a deep-learning-based quality check.

* **Concept:** Train a GAN (Generative Adversarial Network) to translate $Mask \\rightarrow Image$.  
* **The Test:** If the segmentation mask is "good," it should contain all the structural information necessary to reconstruct the original microscopy image.  
* **Score:** The pixel-wise error (L1 or L2 loss) between the *Original Image* and the *Reconstructed Image* serves as a "goodness" map. High reconstruction error implies that the segmentation missed critical visual features (e.g., a cell was missed, so the generator couldn't reconstruct it).29

## **6\. Benchmarking Metrics: Best Practices for Literature Comparison**

While the focus has been on unsupervised scores for your internal use, it is vital to align with the standard metrics used in the literature (e.g., PlantSeg, CREMI, Cell Tracking Challenge). Using these metrics allows you to compare your results against published benchmarks if you generate a small annotated validation set.

### **6.1 Adapted Rand Error (ARand)**

Standard metrics like Intersection-over-Union (IoU) or Dice are often overly punitive for plant tissues, where boundaries are thick and exact pixel alignment is ambiguous. The field has largely standardized on the **Adapted Rand Error (ARand)**.1

* **Definition:** The Rand Index measures the similarity between two data clusterings. It asks: "For every pair of pixels, do they belong to the same cell or different cells?"  
* **Advantage:** ARand is a **topological** metric. It doesn't care if the boundary is shifted by 1 pixel (which ruins IoU), provided the connectivity (who is neighbor to whom) remains correct. This makes it much more robust for comparing segmentation against human annotation, which often varies in boundary thickness.  
* **Metric:** $ARand \= 1 \- F\_1(\\text{Rand Precision}, \\text{Rand Recall})$. Lower is better (0.0 is perfect).

### **6.2 Variation of Information (VOI): Split vs. Merge**

The most actionable metric for tuning segmentation parameters is **Variation of Information (VOI)**, which is based on Information Theory (Entropy).1

$$VOI \= VOI\_{split} \+ VOI\_{merge} \= H(S|GT) \+ H(GT|S)$$

1. **$VOI\_{split}$ (Over-Segmentation Entropy):** Measures the uncertainty of the segmentation given the ground truth.  
   * *High Score:* Means single real cells are being shattered into multiple fragments.  
   * *Action:* Increase the "merge threshold" or "agglomeration parameter" (e.g., in PlantSeg's GASP).  
2. **$VOI\_{merge}$ (Under-Segmentation Entropy):** Measures the uncertainty of the ground truth given the segmentation.  
   * *High Score:* Means multiple real cells are being merged into super-regions.  
   * *Action:* Decrease the threshold or use a more sensitive boundary detector.

This decomposition is critical because "Accuracy" aggregates these errors. A method could have 80% accuracy by over-segmenting, or 80% by under-segmenting. VOI tells you *which* problem to fix.

## **7\. Implementation Strategy: A Hierarchical Pipeline**

To integrate these findings into your workflow, I recommend a hierarchical approach that scales from computationally cheap "sanity checks" to rigorous validation.

### **Level 1: Real-Time Sanity Checks (The "Naive" Plus)**

* **Execute:** For every segmented image, compute **Solidity**, **Euler Number**, and **FCR (Foreground Coverage)**.  
* **Alert:** Flag any image where:  
  * FCR \< 0.90 (Gaps).  
  * Mean Solidity \< 0.85 (Merges).  
  * $\>5\\%$ of cells have Euler Number $\\neq 1$ (Holes).

### **Level 2: Distributional Quality Control**

* **Execute:** Compute the **Log-Normal Fit $p$-value** and **CV of Cell Area**.  
* **Alert:** Flag images with $p \< 0.05$ (Distributional failure) or CV spikes \> 20% compared to neighbors (Instability).

### **Level 3: Deep Validation (For critical data)**

* **Execute:** Run a secondary "Witness Model" (e.g., a pre-trained Cellpose model) and compute the **Consensus IoU**.  
* **Execute:** (If resources allow) **In-Context RCA** using a foundation model to estimate the true Dice score.

### **Software Recommendations**

* **PlantSeg (Python):** Contains built-in scripts for calculating ARand and VOI. It is the gold standard for volumetric plant tissue segmentation.2  
* **scikit-image (Python):** Essential for computing the "naive" statistics (RegionProps, Euler number) and implementing the distributional tests.19  
* **MorphoLibJ (ImageJ/Fiji):** Excellent for interactive topological analysis and watershed tuning.35

## **8\. Conclusion**

Your intuition to employ "naive" statistics was a correct starting point, but the specific constraints of plant tissues—tessellation, log-normal growth, and rigid topology—require transforming these basic measures into more rigorous "goodness" scores. By monitoring **Foreground Coverage** (instead of just area sum), **Log-Normal Goodness-of-Fit** (instead of just average size), and **Topological Euler Numbers** (instead of just shape), you can construct a robust, unsupervised quality control pipeline. For the highest level of confidence without ground truth, adopting **Reverse Classification Accuracy** or **Ensemble Consensus** methods represents the current best practice in the field. These metrics will allow you to infer the quality of your segmentations with a precision that rivals human inspection, scaling effectively to datasets where manual validation is impossible.

#### **Works cited**

1. Accurate and versatile 3D segmentation of plant tissues at cellular resolution | eLife, accessed on December 26, 2025, [https://elifesciences.org/articles/57613](https://elifesciences.org/articles/57613)  
2. PlantSeg introduction \- GitHub Pages, accessed on December 26, 2025, [https://kreshuklab.github.io/plant-seg/2.0.0rc9/](https://kreshuklab.github.io/plant-seg/2.0.0rc9/)  
3. In-Context Reverse Classification Accuracy: Efficient Estimation of Segmentation Quality without Ground-Truth \- arXiv, accessed on December 26, 2025, [https://arxiv.org/html/2503.04522v1](https://arxiv.org/html/2503.04522v1)  
4. \[2503.04522\] Conformal In-Context Reverse Classification Accuracy: Efficient Estimation of Segmentation Quality with Statistical Guarantees \- arXiv, accessed on December 26, 2025, [https://arxiv.org/abs/2503.04522](https://arxiv.org/abs/2503.04522)  
5. Time Series Modeling of Live-Cell Shape Dynamics for Image-based Phenotypic Profiling, accessed on December 26, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5058786/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5058786/)  
6. Exploring the Impact of Variability in Cell Segmentation and Tracking Approaches \- NIH, accessed on December 26, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11842944/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11842944/)  
7. The Earth Mover's Distance as a Metric for Image Retrieval 1 Introduction \- CMU School of Computer Science, accessed on December 26, 2025, [https://www.cs.cmu.edu/\~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf)  
8. The Earth Mover's Distance as a Metric for Image Retrieval, accessed on December 26, 2025, [http://luthuli.cs.uiuc.edu/\~daf/courses/optimization/Combinatorialpapers/fulltext.pdf](http://luthuli.cs.uiuc.edu/~daf/courses/optimization/Combinatorialpapers/fulltext.pdf)  
9. H-EMD: A Hierarchical Earth Mover's Distance Method for Instance Segmentation \- arXiv, accessed on December 26, 2025, [https://arxiv.org/pdf/2206.01309](https://arxiv.org/pdf/2206.01309)  
10. (PDF) H-EMD: A Hierarchical Earth Mover's Distance Method for Instance Segmentation, accessed on December 26, 2025, [https://www.researchgate.net/publication/361107190\_H-EMD\_A\_Hierarchical\_Earth\_Mover's\_Distance\_Method\_for\_Instance\_Segmentation](https://www.researchgate.net/publication/361107190_H-EMD_A_Hierarchical_Earth_Mover's_Distance_Method_for_Instance_Segmentation)  
11. Data Spread and How to Measure It: the Coefficient of Variation (CV) \- Bitesize Bio, accessed on December 26, 2025, [https://bitesizebio.com/22776/data-spread-and-how-to-measure-it-the-coefficient-of-variation-cv/](https://bitesizebio.com/22776/data-spread-and-how-to-measure-it-the-coefficient-of-variation-cv/)  
12. Coefficient of variation \- Wikipedia, accessed on December 26, 2025, [https://en.wikipedia.org/wiki/Coefficient\_of\_variation](https://en.wikipedia.org/wiki/Coefficient_of_variation)  
13. Calculating and Reporting Coefficients of Variation for DIA-Based Proteomics \- PMC \- NIH, accessed on December 26, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11629372/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11629372/)  
14. Calculating Inter- and Intra-Assay Coefficients of Variability \- Salimetrics, accessed on December 26, 2025, [https://salimetrics.com/calculating-inter-and-intra-assay-coefficients-of-variability/](https://salimetrics.com/calculating-inter-and-intra-assay-coefficients-of-variability/)  
15. CoverageTool: A semi-automated graphic software: applications for plant phenotyping \- NIH, accessed on December 26, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6683572/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6683572/)  
16. Background intensity correction for terabyte-sized time-lapse images \- NIST Computational Science in Metrology, accessed on December 26, 2025, [https://isg.nist.gov/deepzoomweb/resources/csmet/pages/gfp\_background\_correction/downloads/chalfoun\_background\_correction.pdf](https://isg.nist.gov/deepzoomweb/resources/csmet/pages/gfp_background_correction/downloads/chalfoun_background_correction.pdf)  
17. Topology Optimization in Medical Image Segmentation With Fast χ Euler Characteristic \- IEEE Xplore, accessed on December 26, 2025, [https://ieeexplore.ieee.org/iel8/42/11279972/11097356.pdf](https://ieeexplore.ieee.org/iel8/42/11279972/11097356.pdf)  
18. Quantitative Assessment of Structural Image Quality \- PMC \- NIH, accessed on December 26, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5856621/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5856621/)  
19. skimage.measure — skimage 0.26.0 documentation, accessed on December 26, 2025, [https://scikit-image.org/docs/stable/api/skimage.measure.html?highlight=regionprops](https://scikit-image.org/docs/stable/api/skimage.measure.html?highlight=regionprops)  
20. Topology Optimization in Medical Image Segmentation with Fast Euler Characteristic | Request PDF \- ResearchGate, accessed on December 26, 2025, [https://www.researchgate.net/publication/394175162\_Topology\_Optimization\_in\_Medical\_Image\_Segmentation\_with\_Fast\_Euler\_Characteristic](https://www.researchgate.net/publication/394175162_Topology_Optimization_in_Medical_Image_Segmentation_with_Fast_Euler_Characteristic)  
21. An evaluation metric for image segmentation of multiple objects | Request PDF, accessed on December 26, 2025, [https://www.researchgate.net/publication/222699951\_An\_evaluation\_metric\_for\_image\_segmentation\_of\_multiple\_objects](https://www.researchgate.net/publication/222699951_An_evaluation_metric_for_image_segmentation_of_multiple_objects)  
22. Measures and Meta-Measures for the Supervised Evaluation of Image Segmentation \- CVL Segmentation, accessed on December 26, 2025, [https://cvlsegmentation.github.io/seism/pdf/PontMarquesCVPR2013.pdf](https://cvlsegmentation.github.io/seism/pdf/PontMarquesCVPR2013.pdf)  
23. Neurons and astrocytes have distinct organelle signatures and responses to stress \- PMC, accessed on December 26, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12631801/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12631801/)  
24. Evaluation of cell segmentation methods without reference segmentations, accessed on December 26, 2025, [https://www.molbiolcell.org/doi/abs/10.1091/mbc.E22-08-0364](https://www.molbiolcell.org/doi/abs/10.1091/mbc.E22-08-0364)  
25. Reverse Classification Accuracy: Predicting Segmentation Performance in the Absence of Ground Truth \- Department of Computing, accessed on December 26, 2025, [http://www.doc.ic.ac.uk/\~bglocker/pdfs/valindria2017tmi.pdf](http://www.doc.ic.ac.uk/~bglocker/pdfs/valindria2017tmi.pdf)  
26. In-Context Reverse Classification Accuracy: Efficient Estimation of Segmentation Quality without Ground-Truth \- ResearchGate, accessed on December 26, 2025, [https://www.researchgate.net/publication/389648345\_In-Context\_Reverse\_Classification\_Accuracy\_Efficient\_Estimation\_of\_Segmentation\_Quality\_without\_Ground-Truth](https://www.researchgate.net/publication/389648345_In-Context_Reverse_Classification_Accuracy_Efficient_Estimation_of_Segmentation_Quality_without_Ground-Truth)  
27. Conformal In-Context Reverse Classification Accuracy: Efficient Estimation of Segmentation Quality with Statistical Guarantees \- arXiv, accessed on December 26, 2025, [https://arxiv.org/html/2503.04522v3](https://arxiv.org/html/2503.04522v3)  
28. SEG: Segmentation Evaluation in absence of Ground truth labels ..., accessed on December 26, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9980141/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9980141/)  
29. Learning unsupervised feature representations for single cell microscopy images with paired cell inpainting | PLOS Computational Biology, accessed on December 26, 2025, [https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007348](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007348)  
30. Accurate and versatile 3D segmentation of plant tissues at cellular resolution \- PMC, accessed on December 26, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7447435/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7447435/)  
31. akhadangi/Segmentation-Evaluation-after-border-thinning: Python implementation of Rand error and Variation of Information \- GitHub, accessed on December 26, 2025, [https://github.com/akhadangi/Segmentation-Evaluation-after-border-thinning](https://github.com/akhadangi/Segmentation-Evaluation-after-border-thinning)  
32. Graph-based active learning of agglomeration (GALA): a Python library to segment 2D and 3D neuroimages \- PMC, accessed on December 26, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3983515/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3983515/)  
33. kreshuklab/plant-seg: A tool for cell instance aware segmentation in densely packed 3D volumetric images \- GitHub, accessed on December 26, 2025, [https://github.com/kreshuklab/plant-seg](https://github.com/kreshuklab/plant-seg)  
34. skimage.metrics — skimage 0.25.2 documentation, accessed on December 26, 2025, [https://scikit-image.org/docs/0.25.x/api/skimage.metrics.html](https://scikit-image.org/docs/0.25.x/api/skimage.metrics.html)  
35. An Open-source Protocol for Deep Learning-based Segmentation of Tubular Structures in 3D Fluorescence Microscopy Images \- JoVE, accessed on December 26, 2025, [https://www.jove.com/t/68004/an-open-source-protocol-for-deep-learning-based-segmentation-tubular](https://www.jove.com/t/68004/an-open-source-protocol-for-deep-learning-based-segmentation-tubular)  
36. MorphoLibJ: integrated library and plugins for mathematical morphology with ImageJ | Bioinformatics | Oxford Academic, accessed on December 26, 2025, [https://academic.oup.com/bioinformatics/article/32/22/3532/2525592](https://academic.oup.com/bioinformatics/article/32/22/3532/2525592)