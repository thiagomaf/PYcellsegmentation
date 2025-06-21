import os

# File paths
f1 = 'results/5A_comparisons_45_comparison_nuclei_45_scaled_0_25/5A_morphology_focus.ome_scaled_0_25_mask.tif'
f2 = 'results/5A_comparisons_45_comparison_cyto2_45_scaled_0_25/5A_morphology_focus.ome_scaled_0_25_mask.tif'
f3 = 'results/5A_comparisons_45_comparison_cyto3_45_scaled_0_25/5A_morphology_focus.ome_scaled_0_25_mask.tif'

files = [f1, f2, f3]
models = ['nuclei', 'cyto2', 'cyto3']

print("Checking mask file comparison...")

# Check if files exist
for i, f in enumerate(files):
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    print(f"{models[i]}: exists={exists}, size={size} bytes")

# Check if files are identical
if all(os.path.exists(f) for f in files):
    with open(f1, 'rb') as file1, open(f2, 'rb') as file2, open(f3, 'rb') as file3:
        content1 = file1.read()
        content2 = file2.read()
        content3 = file3.read()
    
    print(f"\nContent comparison:")
    print(f"nuclei == cyto2: {content1 == content2}")
    print(f"nuclei == cyto3: {content1 == content3}")
    print(f"cyto2 == cyto3: {content2 == content3}")
    
    if content1 == content2 == content3:
        print("\n*** WARNING: All three mask files are IDENTICAL! ***")
        print("This suggests a problem with the segmentation process.")
    else:
        print("\nMask files are different as expected.")
else:
    print("Some files don't exist, cannot compare content.") 