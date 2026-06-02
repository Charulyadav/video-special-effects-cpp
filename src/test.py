import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("C:/Users/Charul/Downloads/ranurte-d3oNTQJVYuw-unsplash.jpg")  # change to your filename
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

R = img_rgb[:,:,0]
G = img_rgb[:,:,1]
B = img_rgb[:,:,2]

# Compute rg chromaticity
total = R + G + B + 1e-6
r = R / total
g = G / total

# --- 2D rg Chromaticity Histogram ---
bins = 100
hist, xedges, yedges = np.histogram2d(r.flatten(), g.flatten(),
                                       bins=bins, range=[[0,1],[0,1]])
hist_log = np.log1p(hist)  # log scale for visibility

# --- rg Chromaticity Image ---
b_chroma = 1 - r - g
b_chroma = np.clip(b_chroma, 0, 1)
chroma_img = np.stack([r, g, b_chroma], axis=2)
chroma_img = np.clip(chroma_img * 200, 0, 255).astype(np.uint8)

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img_rgb.astype(np.uint8))
axes[0].set_title("Original Image")
axes[0].axis('off')

im = axes[1].imshow(hist_log.T, origin='lower', extent=[0,1,0,1],
                    cmap='hot', aspect='equal')
axes[1].set_title("rg Chromaticity Histogram")
axes[1].set_xlabel("r chromaticity")
axes[1].set_ylabel("g chromaticity")
plt.colorbar(im, ax=axes[1], label='log(count+1)')

axes[2].imshow(chroma_img)
axes[2].set_title("rg Chromaticity Image")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("shadow_chromaticity.png", dpi=150, bbox_inches='tight')
plt.show()