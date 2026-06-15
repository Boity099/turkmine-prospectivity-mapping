"""
CODE 1: LINEAMENT EXTRACTION FROM SENTINEL-1 SAR
Purpose: Extract geological lineaments from VV polarisation SAR data
Input: VV_db.img
Output: final_geological_lineaments.tif
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi
from skimage.feature import canny
from skimage.morphology import remove_small_objects, skeletonize, binary_closing, disk
from skimage.measure import label, regionprops

# Input file
vv_file = r"/content/drive/MyDrive/VV_db.img"

# Open raster
print("Opening VV_db raster...")
with rasterio.open(vv_file) as src:
    vv = src.read(1)
    profile = src.profile
print("Raster shape:", vv.shape)

# Convert to float and clean invalid values
vv = vv.astype(np.float32)
vv[vv < -40] = np.nan
vv[vv > 5] = np.nan
vv = np.nan_to_num(vv)

# Robust normalization (2nd and 98th percentiles)
p2, p98 = np.percentile(vv, 2), np.percentile(vv, 98)
vv = np.clip(vv, p2, p98)
vv_norm = (vv - vv.min()) / (vv.max() - vv.min())

# Gaussian smoothing
smooth = gaussian_filter(vv_norm, sigma=1.2)

# Frangi ridge enhancement
ridge = frangi(smooth, sigmas=range(1,5), black_ridges=False)

# Canny edge detection
edges = canny(ridge, sigma=1.5, low_threshold=0.03, high_threshold=0.12)

# Connect fragmented structures
connected = binary_closing(edges, disk(2))

# Remove small noise
clean = remove_small_objects(connected, min_size=25)

# Skeletonize
skeleton = skeletonize(clean)

# Remove short lineaments (<20 pixels)
labels = label(skeleton)
filtered = np.zeros_like(skeleton)
for region in regionprops(labels):
    if region.area >= 20:
        coords = region.coords
        filtered[coords[:,0], coords[:,1]] = 1

# Export
profile.pop("blockxsize", None); profile.pop("blockysize", None)
profile.pop("tiled", None); profile.pop("interleave", None)
profile.update(driver="GTiff", dtype=rasterio.uint8, count=1, compress='lzw')
output = "/content/final_geological_lineaments.tif"
with rasterio.open(output, "w", **profile) as dst:
    dst.write(filtered.astype(rasterio.uint8), 1)

print("Lineament extraction complete. Output:", output)
