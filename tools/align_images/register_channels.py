import numpy as np
import SimpleITK as sitk
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim

# ---------- I/O helpers ----------
def read_tiff_with_spacing(path):
    """
    Returns: (img_np float32 2D, spacing_xy or None)
    Tries to parse OME-XML PhysicalSizeX/Y if present (µm/px).
    """
    with tiff.TiffFile(path) as tf:
        arr = tf.asarray()

        # Squeeze simple extraneous dims (e.g., (1,H,W) or (H,W,1))
        arr = np.squeeze(arr)

        # If it’s a stack/hyperstack, pick the first plane (adapt as needed)
        if arr.ndim > 2:
            # heuristics: prefer (C,H,W) where small C
            if arr.shape[0] <= 4:
                arr = arr[0]
            else:
                arr = arr[..., 0]

        spacing = None
        try:
            if tf.ome_metadata:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tf.ome_metadata)
                pixels = root.find('.//{*}Pixels')
                if pixels is not None:
                    sx = pixels.get('PhysicalSizeX')
                    sy = pixels.get('PhysicalSizeY')
                    if sx and sy:
                        spacing = (float(sx), float(sy))  # µm/px
        except Exception:
            pass

    return arr.astype(np.float32), spacing

def save_tiff_float32(path, img):
    tiff.imwrite(path, img.astype(np.float32), photometric='minisblack')

# ---------- Normalization ----------
def norm01(img, p_low=1, p_high=99):
    lo, hi = np.percentile(img, (p_low, p_high))
    den = max(hi - lo, 1e-6)
    x = np.clip((img - lo) / den, 0, 1)
    return x

# ---------- SITK wrappers ----------
def to_sitk(img_np, spacing_xy=None):
    img = sitk.GetImageFromArray(img_np.astype(np.float32), isVector=False)
    if spacing_xy is not None:
        img.SetSpacing((float(spacing_xy[0]), float(spacing_xy[1])))
    return img

def register_multimodal_MI(
    fixed_np, moving_np,
    fixed_spacing=None, moving_spacing=None,
    transform_type='similarity',
    max_iter=500, levels=(4,2,1)
):
    """
    Returns: aligned_moving_np, final_transform (SITK)
    Aligns 'moving_np' onto 'fixed_np' using Mutual Information.
    """
    fixed = to_sitk(fixed_np, fixed_spacing)
    moving = to_sitk(moving_np, moving_spacing)

    # Initial transform
    if transform_type == 'rigid':
        initial_tx = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )
    elif transform_type == 'similarity':
        initial_tx = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Similarity2DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )
    else:  # affine
        initial_tx = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.AffineTransform(2),
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.2)
    R.SetInterpolator(sitk.sitkLinear)

    R.SetShrinkFactorsPerLevel(levels)
    R.SetSmoothingSigmasPerLevel([2,1,0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=max_iter,
                                    convergenceMinimumValue=1e-6, convergenceWindowSize=20)
    R.SetOptimizerScalesFromPhysicalShift()

    R.SetInitialTransform(initial_tx, inPlace=False)
    final_tx = R.Execute(fixed, moving)

    # Resample moving into fixed space
    resampled = sitk.Resample(moving, fixed, final_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    aligned_np = sitk.GetArrayFromImage(resampled)

    return aligned_np, final_tx

def apply_transform_to_mask(mask_np, final_tx, fixed_np, fixed_spacing=None, moving_spacing=None):
    """
    Warp a label/mask from moving->fixed using nearest-neighbor.
    """
    fixed = to_sitk(fixed_np, fixed_spacing)
    mask = to_sitk(mask_np, moving_spacing)
    mask = sitk.Cast(mask, sitk.sitkUInt8)  # labels
    warped = sitk.Resample(mask, fixed, final_tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(warped)

# ---------- QC metrics ----------
def edge_ncc(a, b):
    ea = sobel(norm01(a))
    eb = sobel(norm01(b))
    va = (ea - ea.mean()).ravel()
    vb = (eb - eb.mean()).ravel()
    den = (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
    return float(np.dot(va, vb) / den)

def ssim_on_norm(a, b):
    return float(ssim(norm01(a), norm01(b), data_range=1.0))

# ---------- Demo / CLI ----------
if __name__ == "__main__":
    # >>> CHANGE THESE <<<
    fixed_path  = "5A_morphology_focus.ome.tif"   # reference channel
    moving_path = "20x_28DPI_5A_shadow.tif"       # channel to be aligned onto fixed
    out_img     = "ch2_aligned_to_ch1.tif"
    out_tfm     = "ch2_to_ch1.tfm"
    maybe_mask_from_moving = None   # e.g., "ch2_mask.tif"

    fixed_np, fixed_spacing = read_tiff_with_spacing(fixed_path)
    moving_np, moving_spacing = read_tiff_with_spacing(moving_path)

    # (Optional) light normalization before registration
    fixed_np_n  = norm01(fixed_np)
    moving_np_n = norm01(moving_np)

    aligned_np, tfm = register_multimodal_MI(
        fixed_np_n, moving_np_n,
        fixed_spacing=fixed_spacing, moving_spacing=moving_spacing,
        transform_type='similarity',  # try 'affine' if residual shear remains
        max_iter=600, levels=(4,2,1)
    )

    # Save outputs
    save_tiff_float32(out_img, aligned_np)
    sitk.WriteTransform(tfm, out_tfm)
    print(f"[OK] Wrote: {out_img} and {out_tfm}")

    # QC: metrics & overlay
    print(f"Edge NCC: {edge_ncc(fixed_np, aligned_np):.4f}")
    print(f"SSIM (normalized): {ssim_on_norm(fixed_np, aligned_np):.4f}")

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.title("Fixed");  plt.imshow(norm01(fixed_np), cmap='gray'); plt.axis('off')
    plt.subplot(1,2,2); plt.title("Aligned moving"); plt.imshow(norm01(aligned_np), cmap='gray'); plt.axis('off')
    plt.show()

    # Optional: warp a mask from the moving channel onto fixed
    if maybe_mask_from_moving:
        mask_m = tiff.imread(maybe_mask_from_moving).astype(np.uint8)
        mask_warped = apply_transform_to_mask(mask_m, tfm, fixed_np, fixed_spacing, moving_spacing)
        tiff.imwrite("ch2_mask_warped_to_ch1.tif", mask_warped, photometric='minisblack')
        print("[OK] Wrote: ch2_mask_warped_to_ch1.tif")
