import nibabel as nib
import numpy as np
import random
from scipy.ndimage import binary_dilation
from skimage.measure import marching_cubes, mesh_to_contour  # skimage can help with contours

def randomized_contour_adjustment_nifti(
    input_mask_path,
    output_mask_path,
    expansion_mm=2.0,
    max_random_shift_mm=1.5, # Max shift inwards from expanded contour (should be less than expansion_mm)
    seed=None
):
    """
    Applies a randomized inward adjustment to the boundary of an expanded 3D mask,
    respecting original mask boundaries.

    Args:
        input_mask_path (str): Path to the input NIfTI binary mask file (.nii.gz).
        output_mask_path (str): Path to save the output NIfTI randomized mask file (.nii.gz).
        expansion_mm (float): The maximum distance in mm to expand the original mask.
        max_random_shift_mm (float): The maximum distance in mm to randomly shift points
                                      inwards from the expanded contour. Must be <= expansion_mm.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    """
    if max_random_shift_mm > expansion_mm:
        print("Warning: max_random_shift_mm should ideally be less than or equal to expansion_mm.")
        print("Adjusting max_random_shift_mm to be <= expansion_mm.")
        max_random_shift_mm = expansion_mm

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    try:
        # Load the NIfTI image
        nii_img = nib.load(input_mask_path)
        data = nii_img.get_fdata()
        affine = nii_img.affine
        header = nii_img.header
        voxel_sizes = header.get_zooms()[:3] # Get x, y, z voxel dimensions

        # Ensure mask is binary (0 or 1) and integer type
        original_mask = (data > 0).astype(np.uint8)

        # --- 1. Calculate Expansion in Voxels ---
        # Determine the number of voxels needed for the expansion in each dimension
        expansion_voxels = [int(np.ceil(expansion_mm / voxel_sizes[i])) for i in range(3)]
        max_random_shift_voxels = [int(np.ceil(max_random_shift_mm / voxel_sizes[i])) for i in range(3)]

        # --- 2. Expand the Original Mask ---
        # We use binary dilation. The 'structure' determines the shape of the dilation.
        # A 3x3x3 cube approximates isotropic expansion if voxel sizes are similar.
        # More complex structures could be used for more isotropic dilation if needed.
        expanded_mask = binary_dilation(original_mask, iterations=max(expansion_voxels)) # Use the max voxel expansion

        # --- 3. Find Boundary Voxels (Approximate Contours) ---
        # A simple way is to find the difference between dilated masks.
        # Original boundary voxels are 1 in original, 0 in eroded (by 1 voxel) original.
        # Expanded boundary voxels are 1 in expanded, 0 in eroded expanded.

        # Find the boundary of the ORIGINAL mask
        # We dilate by 1 voxel and subtract the original to get the boundary layer
        original_boundary = binary_dilation(original_mask, iterations=1) > original_mask

        # Find the boundary of the EXPANDED mask
        expanded_boundary = expanded_mask > binary_dilation(expanded_mask, iterations=-1) # Erosion by 1 voxel

        # Get coordinates of the expanded boundary voxels
        expanded_boundary_coords = np.argwhere(expanded_boundary)

        # --- 4. Randomly Adjust Expanded Boundary Points Inwards ---
        # Create a new mask starting from the original mask
        randomized_mask_data = original_mask.copy()

        # Consider the expanded boundary voxels that are *not* part of the original mask
        # These are the candidates for the new randomized boundary
        candidate_boundary_coords = np.argwhere(expanded_mask > original_mask)

        # We want to select a random subset of the *expanded* boundary to include,
        # ensuring they are within the expansion limit and outside the original mask.

        # Let's build the new mask by starting with the original and adding points
        # from the expanded mask based on a probabilistic decision at the expanded boundary.

        # Iterate through the expanded boundary voxels
        randomized_mask_data = original_mask.copy()

        # Iterate through the voxels in the expanded region (but outside original)
        expanded_region_coords = np.argwhere(expanded_mask > original_mask)

        for x, y, z in expanded_region_coords:
             # Calculate distance to the original mask (in voxels)
             # This is computationally expensive for every point.
             # A simpler approach: make a decision based on the distance *from* the expanded boundary inwards.

             # Let's use a simpler probabilistic approach based on distance from the *expanded* edge
             # Find the closest point on the expanded boundary
             # (Again, finding closest point is complex in 3D for every voxel)

             # Alternative: Iterate through the expanded boundary, and for each boundary point,
             # decide whether to include it based on a random shift *inwards*.
             # This feels closer to "adjusting the contour".

             # Let's try iterating through the expanded boundary coordinates
             is_in_randomized_mask = np.zeros_like(original_mask, dtype=bool)
             is_in_randomized_mask[original_mask > 0] = True # Start with the original mask

             # Potential points to add are those in the expanded mask but not the original
             potential_add_coords = np.argwhere(expanded_mask > original_mask)

             # For each potential point, check its distance to the *expanded* boundary
             # and randomly decide to include it if the random inward shift allows.

             # This is still complex. A more practical approach might be:
             # 1. Get the expanded mask.
             # 2. Iterate through the voxels *in* the expanded mask.
             # 3. For each voxel, calculate its shortest distance *to the expanded boundary*.
             # 4. Generate a random value representing an inward shift.
             # 5. If the voxel's distance to the expanded boundary is greater than the random shift, include it.
             # 6. ENSURE this voxel is outside the original mask.

             # Let's try the distance-based approach
             from scipy.ndimage import distance_transform_edt

             # Calculate distance from every voxel to the *expanded* mask's interior
             # This gives us the distance to the expanded boundary
             distance_to_expanded_boundary = distance_transform_edt(~expanded_mask) # Distance outside expanded mask, invert for distance *to* boundary

             # Generate a random inward shift for each potential voxel in the expanded region
             for x, y, z in candidate_boundary_coords:
                 # Distance from this voxel to the expanded boundary (in voxels)
                 dist_voxels = distance_to_expanded_boundary[x, y, z]

                 # Generate a random inward shift (in voxels)
                 # Convert max_random_shift_mm to max voxel shift for this point
                 # A simpler way: work with probabilities based on the random shift parameter.
                 # Let's generate a random shift in mm and convert it to voxels for the local dimension.
                 random_inward_shift_mm = random.uniform(0, max_random_shift_mm)
                 random_inward_shift_voxels_x = random_inward_shift_mm / voxel_sizes[0]
                 random_inward_shift_voxels_y = random_inward_shift_mm / voxel_sizes[1]
                 random_inward_shift_voxels_z = random_inward_shift_mm / voxel_sizes[2]

                 # A simplified random decision:
                 # Generate a random value between 0 and max_random_shift_mm for *this voxel*.
                 # If the voxel is 'closer' to the original mask boundary than this random value, keep it out.
                 # If the voxel is 'further' from the original mask boundary than this random value, potentially include it.

                 # Let's try a different angle: Define the new boundary as a random offset *inwards* from the expanded boundary.
                 # This is hard to do precisely without surface meshing.

                 # A more robust approach for 3D volumes and boundaries:
                 # 1. Get the expanded mask.
                 # 2. Get the original mask.
                 # 3. Consider the region `expanded_mask AND NOT original_mask`.
                 # 4. For each voxel in this region, decide probabilistically whether to include it
                 #    in the new mask. The probability could depend on its distance to the original mask.

                 # Let's try this probabilistic approach:
                 # Start with the original mask. Add voxels from the expanded region stochastically.
                 # The probability of adding a voxel could decrease as it gets further from the original mask boundary
                 # (or increase as it gets closer to the expanded boundary - same idea).

                 # Calculate distance from every voxel to the *original* mask's interior
                 distance_to_original_boundary = distance_transform_edt(~original_mask)

                 # Iterate through voxels in the expanded mask but outside the original
                 for x, y, z in np.argwhere(expanded_mask > original_mask):
                     dist_from_original_boundary_voxels = distance_to_original_boundary[x, y, z]

                     # Calculate the maximum possible inward shift in voxels for this point
                     max_inward_shift_voxels_at_this_point = [
                         max_random_shift_mm / voxel_sizes[i] for i in range(3)
                     ]
                     # Let's use the maximum of these for a simple heuristic probability
                     max_inward_shift_voxels = max(max_inward_shift_voxels_at_this_point)


                     # Simple probability model: Probability of *keeping* the voxel (adding it to the original)
                     # is higher if it's further from the original boundary (closer to the expanded boundary).
                     # Or, probability of *removing* it (not adding it to the original) is higher if it's closer
                     # to the original boundary.

                     # Let's define the region where randomization happens:
                     # It's the region between the original boundary and the expanded boundary.
                     # We want the new boundary to lie somewhere in this region.

                     # Consider a voxel in the `expanded_mask AND NOT original_mask` region.
                     # Its distance from the original mask is `dist_from_original_boundary_voxels`.
                     # Its distance from the expanded mask boundary is `distance_to_expanded_boundary[x,y,z]`.
                     # The sum of these distances is approximately the thickness of the expanded region at that point.

                     # Simple random decision based on max_random_shift_mm:
                     # If the voxel is further than `max_random_shift_mm` (converted to voxels)
                     # from the original boundary, *always* include it.
                     # If it's within `max_random_shift_mm` of the original boundary,
                     # include it with a probability that decreases as it gets closer to the original boundary.

                     # Let's simplify: just randomly sample points from the `expanded_mask AND NOT original_mask`
                     # region, but only include them if their distance from the original mask is greater
                     # than a random value between 0 and `max_random_shift_mm`.

                     dist_to_original_boundary_mm = dist_from_original_boundary_voxels * np.mean(voxel_sizes) # Approximate distance in mm

                     # Generate a random threshold for this voxel (in mm)
                     random_threshold_mm = random.uniform(0, max_random_shift_mm)

                     # If the voxel's distance from the original boundary is greater than
                     # the random threshold, *and* it's within the expanded mask, include it.
                     # (We are already iterating within `expanded_mask > original_mask`)
                     if dist_to_original_boundary_mm > random_threshold_mm:
                          randomized_mask_data[x, y, z] = 1 # Include this voxel

        # Ensure the original mask is fully included (redundant with initialization but safe)
        randomized_mask_data[original_mask > 0] = 1


        # --- 5. Create and Save the New NIfTI Mask ---
        # The randomized_mask_data now contains the new binary mask
        randomized_mask_data = randomized_mask_data.astype(np.uint8) # Ensure binary integer type

        # Create a new NIfTI image
        nii_output = nib.Nifti1Image(randomized_mask_data, affine, header)

        # Save the image
        nib.save(nii_output, output_mask_path)
        print(f"Successfully saved randomized mask to {output_mask_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_mask_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy NIfTI file for demonstration
    # This requires nibabel and numpy
    # Create a simple 3D binary mask (e.g., a sphere)
    shape = (64, 64, 64)
    voxel_size = (1.0, 1.0, 1.0) # Assume isotropic voxels for simplicity
    center = np.array(shape) // 2
    radius = 15

    dummy_data = np.zeros(shape, dtype=np.uint8)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2:
                    dummy_data[x, y, z] = 1

    dummy_affine = np.diag(list(voxel_size) + [1]) # Simple affine matrix

    dummy_img = nib.Nifti1Image(dummy_data, dummy_affine)
    dummy_nii_path = "dummy_mask.nii.gz"
    nib.save(dummy_img, dummy_nii_path)
    print(f"Created dummy mask at {dummy_nii_path}")

    # Define input and output paths
    input_nii = dummy_nii_path
    output_nii = "randomized_mask.nii.gz"

    # Parameters for the adjustment
    expansion_distance = 5.0  # Expand by 5 mm
    random_shift_limit = 3.0  # Randomly shift inwards up to 3 mm

    # Apply the randomized adjustment
    randomized_contour_adjustment_nifti(
        input_nii,
        output_nii,
        expansion_mm=expansion_distance,
        max_random_shift_mm=random_shift_limit,
        seed=42 # Use a seed for reproducible results
    )

    print(f"\nCheck the generated file: {output_nii}")
    print(f"You can view NIfTI files using software like FSLeyes, ITK-SNAP, or 3D Slicer.")

    # Optional: Load and display slices (requires matplotlib)
    try:
        import matplotlib.pyplot as plt

        original_img = nib.load(input_nii)
        original_data = original_img.get_fdata()

        randomized_img = nib.load(output_nii)
        randomized_data = randomized_img.get_fdata()

        slice_idx = shape[2] // 2 # Choose a middle slice

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_data[:, :, slice_idx].T, cmap='gray', origin='lower')
        axes[0].set_title("Original Mask (Slice)")
        axes[0].axis('off')

        axes[1].imshow(randomized_data[:, :, slice_idx].T, cmap='gray', origin='lower')
        axes[1].set_title("Randomized Mask (Slice)")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\nMatplotlib not found. Skipping slice visualization.")
        print("Install it with: pip install matplotlib")