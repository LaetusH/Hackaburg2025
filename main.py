import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

array = np.load("C:\\Users\\henri\\Documents\\Hackaburg2025\\Github\\img01.npy")

# Step 1: Threshold values close to 0
threshold = 0.18
binary_mask = array < threshold

# Step 2: Label connected components
labeled_array, num_features = ndi.label(binary_mask)

# Step 3: Filter for larger regions
min_size = 20
sizes = ndi.sum(binary_mask, labeled_array, range(1, num_features + 1))
large_dot_labels = [i+1 for i, size in enumerate(sizes) if size >= min_size]

# Create a mask of just the large dots
large_dots_mask = np.isin(labeled_array, large_dot_labels)

# Step 4: Compute centroids of large dots
centroids = ndi.center_of_mass(binary_mask, labeled_array, large_dot_labels)

# Visualize
plt.figure(figsize=(8, 6))
plt.title("Large Dots with Centroids")
plt.imshow(large_dots_mask, cmap='gray')

# Plot centroids
for y, x in centroids:
    plt.plot(x, y, 'r+', markersize=10)

plt.tight_layout()
plt.show()

# Print centroid coordinates
print("Centroids of large dots (row, column):")
for i, c in enumerate(centroids):
    print(f"Dot {i+1}: {c}")

