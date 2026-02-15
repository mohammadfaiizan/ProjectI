"""
Scikit-learn PatchExtractor, feature extraction patterns
"""
import numpy as np
from sklearn.feature_extraction import image

np.random.seed(42)
img = np.random.rand(10, 10)
print("Image shape:", img.shape)

print("\n--- PatchExtractor ---")
patch_extractor = image.PatchExtractor(patch_size=(3, 3))
img_batch = img.reshape(1, 10, 10)
patches = patch_extractor.fit_transform(img_batch)
print("Patches shape:", patches.shape)
print("First patch:\n", np.round(patches[0], 2))

print("\n--- extract_patches_2d ---")
patches_2d = image.extract_patches_2d(img, patch_size=(3, 3))
print("Patches 2D shape:", patches_2d.shape)
print("Number of patches:", len(patches_2d))

print("\n--- reconstruct_from_patches_2d ---")
reconstructed = image.reconstruct_from_patches_2d(patches_2d, (10, 10))
print("Reconstructed shape:", reconstructed.shape)
print("Reconstruction diff (should be small):", np.abs(reconstructed - img).max())

print("\n--- grid_to_graph (connectivity) ---")
connectivity = image.grid_to_graph(4, 4)
print("Grid graph - n_nodes:", connectivity.shape[0])
print("Grid graph - n_edges:", connectivity.nnz)
