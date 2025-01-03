import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# Load dose data
dose = np.load('your_dose_data.npy')  # Replace with your dose data
threshold = 0.1  # Gy threshold for visualization

# Get the 3D dose surface using marching cubes
verts, faces, _, _ = measure.marching_cubes(dose, level=threshold)

# Create a 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Add dose volume with alpha scaled by dose
mesh = Poly3DCollection(verts[faces], alpha=0.3, edgecolor='none')
mesh.set_facecolor((1, 0, 0, 0.5))  # Red color, adjust alpha as needed
ax.add_collection3d(mesh)

# Add critical structure contours
# Example: Add tumor, OARs, and body volume
structures = {
    "Tumor": ("tumor_mask.npy", "blue", 0.5),
    "OAR": ("oar_mask.npy", "green", 0.3),
    "Body": ("body_mask.npy", "gray", 0.2),
}

for name, (file, color, alpha) in structures.items():
    mask = np.load(file)
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    mesh = Poly3DCollection(verts[faces], alpha=alpha, edgecolor='none')
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)

# Adjust plot limits and view angle
ax.set_xlim(0, dose.shape[0])
ax.set_ylim(0, dose.shape[1])
ax.set_zlim(0, dose.shape[2])
ax.view_init(elev=30, azim=135)

# Labels
ax.set_xlabel("X-axis (Voxels)")
ax.set_ylabel("Y-axis (Voxels)")
ax.set_zlabel("Z-axis (Voxels)")

plt.colorbar(mesh, ax=ax, label="Radiation Dose (Gy)")
plt.tight_layout()
plt.show()
