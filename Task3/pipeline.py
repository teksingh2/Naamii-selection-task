import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os 

# --- Load the NIfTI file ---
nifti_file_path = '../Task1/image/left_knee.nii.gz'  # <-- Change this

# --- Print shape info ---
#print(f"Image shape: {image_data.shape}")  # Typically (height, width, depth)

def visualize(image_data,name):
    # --- Choose slices to visualize ---
    num_slices = 9
    slice_indices = np.linspace(0, image_data.shape[2] - 1, num_slices, dtype=int)

    # --- Plot slices ---
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for i, slice_idx in enumerate(slice_indices):
        axes[i].imshow(image_data[:, :, slice_idx], cmap='gray')
        axes[i].set_title(f"Slice {slice_idx}")
        axes[i].axis('off')
    plt.savefig(name)
    plt.tight_layout()
    plt.show()

def load_nifti(file_path):
    """
    Load a NIfTI file and return the data array and affine.
    """
    nifti = nib.load(file_path)
    data = nifti.get_fdata()
    affine = nifti.affine
    return data, affine


def save_nifti(data, affine, output_path):
    """
    Save a NumPy array as a NIfTI file.
    """
    nifti = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(nifti, output_path)

def segment_knee_regions(ct_path, mask_path, output_dir):
    """
    Segment Tibia, Femur, and Background from a CT volume using the mask.
    """
    # Load the CT scan and segmentation mask
    ct_data, ct_affine = load_nifti(ct_path)
    mask_data, _ = load_nifti(mask_path)

    # Define region labels (change if your mask uses different values)
    TIBIA_LABEL = 1
    FEMUR_LABEL = 2
    BACKGROUND_LABEL = 0

    # Generate binary masks
    tibia_mask = (mask_data == TIBIA_LABEL)
    femur_mask = (mask_data == FEMUR_LABEL)
    background_mask = (mask_data == BACKGROUND_LABEL)

    # Apply masks to CT data
    tibia_volume = ct_data * tibia_mask
    femur_volume = ct_data * femur_mask
    background_volume = ct_data * background_mask

    # Save each region as a new .nii.gz file
    os.makedirs(output_dir, exist_ok=True)
    save_nifti(tibia_volume, ct_affine, os.path.join(output_dir, "tibia_volume.nii.gz"))
    save_nifti(femur_volume, ct_affine, os.path.join(output_dir, "femur_volume.nii.gz"))
    save_nifti(background_volume, ct_affine, os.path.join(output_dir, "background_volume.nii.gz"))

    print("Segmentation done. Files saved to:", output_dir)

# ct_file = "../Task1/image/left_knee.nii.gz"
# mask_file = "..//Task1/output/bone_segmentation_task1_1.nii.gz"
# output_folder = "segmented_regions"

def overlay_mask(image_slice, mask_slice, alpha=0.4, cmap_img='gray', cmap_mask='jet'):
    """
    Overlay a segmentation mask on a single image slice.
    
    Parameters:
    - image_slice: 2D numpy array of the image slice.
    - mask_slice: 2D numpy array of the mask slice (same size as image_slice).
    - alpha: float, transparency of the mask overlay.
    - cmap_img: str, colormap for the image.
    - cmap_mask: str, colormap for the mask.
    """
    plt.imshow(image_slice, cmap=cmap_img)
    if mask_slice is not None and np.any(mask_slice):
        plt.imshow(mask_slice, cmap=cmap_mask, alpha=alpha)
    plt.axis('off')

def visualize_nifti_with_mask(image_path, mask_path=None, num_slices=9, axis=2):
    """
    Visualize NIfTI image slices with optional mask overlays.
    
    Parameters:
    - image_path: str, path to the image .nii.gz file.
    - mask_path: str or None, path to the mask .nii.gz file (optional).
    - num_slices: int, number of slices to visualize.
    - axis: int, axis along which to slice (0=sagittal, 1=coronal, 2=axial).
    """
    # Load image and optional mask
    image, ct_affine = load_nifti(image_path)
    mask, _ = load_nifti(mask_path)
    assert mask is None or image.shape == mask.shape, "Image and mask must have the same shape"

    # Select slices evenly along the chosen axis
    slice_indices = np.linspace(0, image.shape[axis] - 1, num_slices, dtype=int)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for i, idx in enumerate(slice_indices):
        ax = axes[i]
        plt.sca(ax)

        if axis == 0:
            img_slice = image[idx, :, :]
            msk_slice = mask[idx, :, :] if mask is not None else None
        elif axis == 1:
            img_slice = image[:, idx, :]
            msk_slice = mask[:, idx, :] if mask is not None else None
        else:  # default to axial
            img_slice = image[:, :, idx]
            msk_slice = mask[:, :, idx] if mask is not None else None

        overlay_mask(img_slice, msk_slice)
        ax.set_title(f"Slice {idx}")
    plt.savefig('slices with masks ')
    plt.tight_layout()
    plt.show()

ct_file = "../Task1/image/left_knee.nii.gz"
mask_file = "..//Task1/output/bone_segmentation_task1_1.nii.gz"
# visualize_nifti_with_mask(
#     ct_file,
#     mask_file,  # Set to None if you don’t have a mask
#     num_slices=9,
#     axis=2  # axial view
# )

def plot_image_mask_overlay(image_path, mask_path, slice_index, axis=2, alpha=0.4):
    """
    Plot image slice, mask slice, and overlay side by side.

    Parameters:
    - image: 3D numpy array (CT volume)
    - mask: 3D numpy array (segmentation mask)
    - slice_index: int, index of the slice to visualize
    - axis: int, slicing axis (0=sagittal, 1=coronal, 2=axial)
    - alpha: float, transparency for overlay
    """
    image, ct_affine = load_nifti(image_path)
    mask, _ = load_nifti(mask_path)
    # Extract slices
    if axis == 0:
        img_slice = image[slice_index, :, :]
        mask_slice = mask[slice_index, :, :]
    elif axis == 1:
        img_slice = image[:, slice_index, :]
        mask_slice = mask[:, slice_index, :]
    else:
        img_slice = image[:, :, slice_index]
        mask_slice = mask[:, :, slice_index]

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img_slice, cmap='gray')
    axs[0].set_title(f'Image Slice {slice_index}')
    axs[0].axis('off')

    axs[1].imshow(mask_slice, cmap='jet')
    axs[1].set_title(f'Mask Slice {slice_index}')
    axs[1].axis('off')

    axs[2].imshow(img_slice, cmap='gray')
    axs[2].imshow(mask_slice, cmap='jet', alpha=alpha)
    axs[2].set_title('Overlay')
    axs[2].axis('off')
    plt.savefig('slice_108_with_maska')
    plt.tight_layout()
    plt.show()


slice_index = 108  
axis = 2  # Axial view

plot_image_mask_overlay(ct_file, mask_file, slice_index, axis=axis)


