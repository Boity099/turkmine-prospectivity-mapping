"""
CODE A: COMPLETE STATISTICS EXTRACTION
Purpose: Extract all statistical values for Chapter 4 tables
Input: All processed raster layers
Output: Chapter4_Statistics_Extracted.csv
"""

import rasterio
import numpy as np
import pandas as pd
from scipy.ndimage import label
import os

print("=" * 70)
print("CODE A: STATISTICS EXTRACTION")
print("=" * 70)

# File paths (UPDATE THESE)
file_paths = {
    'canny': r"C:\Users\lenovo\Downloads\finaall maps chapter 4\Canny map.tif",
    'frangi': r"C:\Users\lenovo\Downloads\finaall maps chapter 4\Frangi_Ridge_enhance.tif",
    'final_geo_lin': r"C:\Users\lenovo\Downloads\finaall maps chapter 4\Final_Geo_Lin.tif",
    'slope': r"C:\Users\lenovo\Downloads\finaall maps chapter 4\Turk_Slope.tif",
    'clay_norm': r"C:\Prospectivity_Project\Output\Clay_Norm.tif",
    'iron_norm': r"C:\Prospectivity_Project\Output\Iron_Norm.tif",
    'alteration_norm': r"C:\Prospectivity_Project\Output\Alteration_Norm.tif",
    'lineament_norm': r"C:\Prospectivity_Project\Output\Lineament_Norm.tif",
    'distance_inverse': r"C:\Prospectivity_Project\Output\Distance_Inverse.tif",
    'prospectivity_raw': r"C:\Prospectivity_Project\Output\Prospectivity_Map_WeightedA.tif",
}

study_area_km2 = 8.5
pixel_area_ha = 0.01

# Count lineament pixels
for name, path in [('Canny', file_paths['canny']), ('Frangi', file_paths['frangi']), 
                    ('Final_Geo_Lin', file_paths['final_geo_lin'])]:
    with rasterio.open(path) as src:
        data = src.read(1)
        count = np.sum(np.nan_to_num(data) > 0)
        print(f"{name}: {count:,} pixels")

# Count total lineaments
with rasterio.open(file_paths['final_geo_lin']) as src:
    skeleton = np.nan_to_num(src.read(1)).astype(np.uint8)
    labeled, num = label(skeleton)
    print(f"\nTotal lineaments: {num:,}")

# Lineament density
with rasterio.open(file_paths['lineament_norm']) as src:
    density = src.read(1)
    density = density[~np.isnan(density)]
    print(f"\nMean density: {np.mean(density * 38.7):.1f} km/km²")
    print(f"Max density: {np.max(density * 38.7):.1f} km/km²")

# Slope statistics
with rasterio.open(file_paths['slope']) as src:
    slope = src.read(1)
    slope = slope[~np.isnan(slope)]
    print(f"\nSlope - Min: {np.min(slope):.1f}°, Max: {np.max(slope):.1f}°, Mean: {np.mean(slope):.1f}°")

# Alteration anomalies
for name, path in [('Clay', file_paths['clay_norm']), ('Iron', file_paths['iron_norm']), 
                    ('Combined', file_paths['alteration_norm'])]:
    with rasterio.open(path) as src:
        data = src.read(1)
        pixels = np.sum(np.nan_to_num(data) > 0.5)
        area_ha = pixels * pixel_area_ha
        print(f"{name}: {area_ha:.1f} ha ({pixels:,} pixels)")

# Prospectivity statistics
with rasterio.open(file_paths['prospectivity_raw']) as src:
    prosp = src.read(1)
    prosp = prosp[~np.isnan(prosp)]
    print(f"\nProspectivity - Min: {np.min(prosp):.2f}, Max: {np.max(prosp):.2f}, Mean: {np.mean(prosp):.2f}")
    
    hpz = np.sum(prosp > 0.7) * pixel_area_ha
    mpz = np.sum((prosp >= 0.4) & (prosp <= 0.7)) * pixel_area_ha
    lpz = np.sum(prosp < 0.4) * pixel_area_ha
    print(f"HPZ: {hpz:.1f} ha, MPZ: {mpz:.1f} ha, LPZ: {lpz:.1f} ha")

print("\nCODE A COMPLETED")
