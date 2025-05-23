import SimpleITK as sitk
import numpy as np
import os

def extract_landmarks(mask_path, tibia_label=2):
    mask = sitk.ReadImage(mask_path)
    mask_np = sitk.GetArrayFromImage(mask)  # shape: (z, y, x)

    # Get all tibia voxel coordinates
    tibia_coords = np.argwhere(mask_np == tibia_label)
    
    if tibia_coords.size == 0:
        return None, None  # Tibia not found in mask

    # Find the lowest z (most inferior slice)
    lowest_z = np.max(tibia_coords[:, 0])
    lowest_slice_coords = tibia_coords[tibia_coords[:, 0] == lowest_z]

    # Among the lowest slice, find medial (min x) and lateral (max x)
    medial_point = lowest_slice_coords[np.argmin(lowest_slice_coords[:, 2])]
    lateral_point = lowest_slice_coords[np.argmax(lowest_slice_coords[:, 2])]

    # Convert voxel coordinates to physical coordinates
    spacing = mask.GetSpacing()
    origin = mask.GetOrigin()
    direction = np.array(mask.GetDirection()).reshape(3, 3)

    def voxel_to_physical(index):
        return tuple(origin[i] + spacing[i] * (direction[i, 0] * index[2] +
                                               direction[i, 1] * index[1] +
                                               direction[i, 2] * index[0]) for i in range(3))

    medial_phys = voxel_to_physical(medial_point)
    lateral_phys = voxel_to_physical(lateral_point)

    return medial_phys, lateral_phys

masks = {
    "Original": "output/bone_segmentation_task1_1.nii.gz",
    "Expanded 2mm": "output/bone_segmentation_expanded_2mm.nii.gz",
    "Expanded 4mm": "output/bone_segmentation_expanded_4mm.nii.gz",
    "Randomized 1": "output/bone_segmentation_randomized_mask1.nii.gz",
    "Randomized 2": "output/bone_segmentation_randomized_mask2.nii.gz",
}

for name, path in masks.items():
    medial, lateral = extract_landmarks(path)
    print(f"{name}:")
    print(f"  Medial  = {medial}")
    print(f"  Lateral = {lateral}\n")
