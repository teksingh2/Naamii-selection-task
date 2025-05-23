import SimpleITK as sitk
import numpy as np
import os

# --- Parameters ---
expansion_mm = 2.0
input_mask_path = "output/bone_segmentation_task1_1.nii.gz"
output_path = f"output/bone_segmentation_expanded_{int(expansion_mm)}mm.nii.gz"
output_path_1 = f"output/bone_segmentation_expanded_{int(4)}mm.nii.gz"

# --- Load Segmentation ---
seg = sitk.ReadImage(input_mask_path)
spacing = seg.GetSpacing()
print("Voxel spacing (x,y,z):", spacing)

# --- Convert mm to voxel radius ---
def mm_to_voxels(mm, spacing):
    return [int(np.ceil(mm / s)) for s in spacing]

radius_vox = mm_to_voxels(expansion_mm, spacing)
radius_vox_4 = mm_to_voxels(4, spacing)

# --- Expand each label separately ---
femur = sitk.BinaryThreshold(seg, 1, 1, 1, 0)
tibia = sitk.BinaryThreshold(seg, 2, 2, 1, 0)

femur_1 = sitk.BinaryThreshold(seg, 1, 1, 1, 0)
tibia_1 = sitk.BinaryThreshold(seg, 2, 2, 1, 0)

# Use binary dilation
femur_expanded = sitk.BinaryDilate(femur, radius_vox)
tibia_expanded = sitk.BinaryDilate(tibia, radius_vox)

femur_expanded_1 = sitk.BinaryDilate(femur, radius_vox)
tibia_expanded_1 = sitk.BinaryDilate(tibia, radius_vox)


# Combine with separate labels again
expanded_mask = sitk.Cast(femur_expanded, sitk.sitkUInt8) * 1 + \
                sitk.Cast(tibia_expanded, sitk.sitkUInt8) * 2

expanded_mask_1 = sitk.Cast(femur_expanded_1, sitk.sitkUInt8) * 1 + \
                sitk.Cast(tibia_expanded_1, sitk.sitkUInt8) * 2

# --- Save Expanded Mask ---
sitk.WriteImage(expanded_mask, output_path)
sitk.WriteImage(expanded_mask_1, output_path_1)
print(f"Saved expanded mask to {output_path}")
print(f"Saved expanded mask to {output_path_1}")

