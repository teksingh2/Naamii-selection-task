import SimpleITK as sitk
import numpy as np
import os

# Load the CT image
input_path = "image/left_knee.nii"
image = sitk.ReadImage(input_path)
image_array = sitk.GetArrayFromImage(image)
spacing = image.GetSpacing()
print(f"Image shape: {image_array.shape}, spacing: {spacing}")

# --- Step 1: Threshold for bone 
lower_threshold = 250
upper_threshold = 3000
bone_mask = sitk.BinaryThreshold(image, lowerThreshold=lower_threshold, upperThreshold=upper_threshold, insideValue=1, outsideValue=0)

#Region growing is another popular segmentation technique, which involves starting from a seed point and progressively adding neighboring
#pixels to the region if they meet certain criteria. The ConnectedThreshold() function in SimpleITK allows you to perform region growing:

# --- Step 2: Connected component analysis ---
cc_filter = sitk.ConnectedComponentImageFilter()
labeled_mask = cc_filter.Execute(bone_mask)

# --- Step 3: Get sizes of all components ---
stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(labeled_mask)
sizes = {l: stats.GetPhysicalSize(l) for l in stats.GetLabels()}
sorted_labels = sorted(sizes, key=sizes.get, reverse=True)

# Get top 2 largest components (assumed to be femur and tibia)
femur_label = sorted_labels[0]
tibia_label = sorted_labels[1]

# Create labeled mask
segmentation = sitk.Image(labeled_mask.GetSize(), sitk.sitkUInt8)
segmentation.CopyInformation(labeled_mask)
segmentation = sitk.BinaryThreshold(labeled_mask, femur_label, femur_label, 1, 0) + \
               sitk.BinaryThreshold(labeled_mask, tibia_label, tibia_label, 2, 0)

# --- Step 4: Save as NIfTI ---
# no significant difference in result so, 
# morphological hole filling
segmentation = sitk.BinaryFillhole(segmentation == 1) * 1 + sitk.BinaryFillhole(segmentation == 2) * 2

#without morpho
output_path = "output/bone_segmentation_task1_1_2.nii.gz"
#output_path = "output/bone_segmentation_task1_1_1.nii.gz"
sitk.WriteImage(segmentation, output_path)
print(f"Saved segmentation to {output_path}")

