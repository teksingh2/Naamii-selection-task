import SimpleITK as sitk
import numpy as np
import os
import random

# --- Parameters ---
random_ratio = 0.5  # Between 0 (no expansion) and 1 (full 2mm expansion)
expanded_path = "output/bone_segmentation_expanded_2mm.nii.gz"
original_path = "output/bone_segmentation_task1_1_2.nii.gz"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# --- Load masks ---
original = sitk.ReadImage(original_path)
expanded = sitk.ReadImage(expanded_path)

original_np = sitk.GetArrayFromImage(original)
expanded_np = sitk.GetArrayFromImage(expanded)

# Function to randomize a label
def get_randomized_label(original_np, expanded_np, label, ratio):
    orig = (original_np == label)
    expd = (expanded_np == label)
    diff = np.logical_and(expd, np.logical_not(orig))  # Band between original and expanded

    indices = np.argwhere(diff)
    num_to_keep = int(len(indices) * ratio)
    selected_indices = indices[np.random.choice(len(indices), num_to_keep, replace=False)]

    randomized = np.copy(orig)
    for z, y, x in selected_indices:
        randomized[z, y, x] = True

    return randomized.astype(np.uint8) * label

# Generate two random masks
for i in range(1, 3):
    randomized_mask = get_randomized_label(original_np, expanded_np, 1, random_ratio) + \
                      get_randomized_label(original_np, expanded_np, 2, random_ratio)

    # Convert to image
    randomized_img = sitk.GetImageFromArray(randomized_mask)
    randomized_img.CopyInformation(original)
    out_path = os.path.join(output_dir, f"bone_segmentation_randomized_mask{i}.nii.gz")
    sitk.WriteImage(randomized_img, out_path)
    print(f"Saved {out_path}")
