import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Read the JSON file from only_for_illustration folder
json_file = "only_for_illustration/particle_selections_20251016_094958.json"
with open(json_file, 'r') as f:
    data = json.load(f)

# Extract bbox and folder path
image_set = data['image_sets'][0]
# Use the aligned folder instead of the original folder
aligned_folder = "example_data1_aligned"
folder_path = Path(aligned_folder)
bbox = image_set['selected_particles'][0]['bbox']  # [x1, y1, x2, y2]
total_images = image_set['total_images']

# Extract bbox coordinates and enlarge by 10 pixels on each side
x1, y1, x2, y2 = bbox
x1_enlarged = max(0, x1 - 10)  # Ensure we don't go below 0
y1_enlarged = max(0, y1 - 10)
x2_enlarged = x2 + 10
y2_enlarged = y2 + 10

# Calculate center point (relative to the cropped image)
# The center remains at the original bbox center within the enlarged crop
center_x = (x2 - x1) / 2 + 10  # Add 10 to account for the padding
center_y = (y2 - y1) / 2 + 10

# Create figure with 5 subplots in one row
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

# Load and crop each image from aligned folder
for i in range(1, total_images + 1):
    # Use _aligned.tif naming pattern
    image_path = folder_path / f"{i}_aligned.tif"

    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Crop the image using enlarged bbox coordinates
    cropped = img_array[y1_enlarged:y2_enlarged, x1_enlarged:x2_enlarged]

    # Plot on the corresponding subplot
    ax = axes[i-1]
    ax.imshow(cropped, cmap='gray')
    ax.set_title(f'Image {i}', fontsize=12, fontweight='bold')

    # Plot the center point
    ax.plot(center_x, center_y, 'ro', markersize=8, markerfacecolor='red',
            markeredgecolor='white', markeredgewidth=1.5)

    # Add 'center' label near the point
    ax.text(center_x, center_y - 1, 'center', color='red', fontsize=9,
            ha='center', va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='red', alpha=0.7))

    # Remove axes for cleaner look
    ax.axis('off')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
output_file = "cropped_particles_aligned_visualization.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved visualization to: {output_file}")

plt.close()
