"""
================================================================================
TURKMINE GOLD PROSPECTIVITY MAPPING - COMPLETE CODE
================================================================================
Author: [Your Name]
Project: Enhanced Gold Mineralisation Mapping Using Integrated Remote Sensing 
         and GIS Techniques at Turkmine Open Pits
University: [Your University]
Year: 2025

This file contains all Python scripts (Codes 1-18) for the complete workflow.
================================================================================
"""

# =============================================================================
# CODE 1: LINEAMENT EXTRACTION FROM SENTINEL-1 SAR
# =============================================================================
"""
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

def extract_lineaments(vv_file, output_path):
    """Extract lineaments from Sentinel-1 VV polarisation image"""
    print("Opening VV_db raster...")
    with rasterio.open(vv_file) as src:
        vv = src.read(1)
        profile = src.profile
    print("Raster shape:", vv.shape)
    
    # Convert to float and clean
    vv = vv.astype(np.float32)
    vv[vv < -40] = np.nan
    vv[vv > 5] = np.nan
    vv = np.nan_to_num(vv)
    
    # Robust normalization
    p2, p98 = np.percentile(vv, 2), np.percentile(vv, 98)
    vv = np.clip(vv, p2, p98)
    vv_norm = (vv - vv.min()) / (vv.max() - vv.min())
    
    # Process
    smooth = gaussian_filter(vv_norm, sigma=1.2)
    ridge = frangi(smooth, sigmas=range(1,5), black_ridges=False)
    edges = canny(ridge, sigma=1.5, low_threshold=0.03, high_threshold=0.12)
    connected = binary_closing(edges, disk(2))
    clean = remove_small_objects(connected, min_size=25)
    skeleton = skeletonize(clean)
    
    # Filter short lineaments
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
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(filtered.astype(rasterio.uint8), 1)
    
    print(f"Lineament extraction complete. Output: {output_path}")
    return filtered


# =============================================================================
# CODE 2: EXPORT PROCESSING STAGES
# =============================================================================

def export_stages(vv_norm, smooth, ridge, edges, connected, clean, skeleton, filtered, profile, export_folder="/content/"):
    """Export all intermediate processing stages"""
    outputs = {
        "01_Normalized_VV.tif": vv_norm,
        "02_Smoothed_VV.tif": smooth,
        "03_Frangi_Ridge_Enhanced.tif": ridge,
        "04_Canny_Edges.tif": edges.astype(np.uint8),
        "05_Connected_Structures.tif": connected.astype(np.uint8),
        "06_Cleaned_Structures.tif": clean.astype(np.uint8),
        "07_Skeleton_Lineaments.tif": skeleton.astype(np.uint8),
        "08_Final_Geological_Lineaments.tif": filtered.astype(np.uint8)
    }
    
    export_profile = profile.copy()
    export_profile.pop("blockxsize", None); export_profile.pop("blockysize", None)
    export_profile.pop("tiled", None); export_profile.pop("interleave", None)
    
    for filename, image in outputs.items():
        if image.dtype == bool:
            image = image.astype(np.uint8)
        elif image.dtype == np.float64:
            image = image.astype(np.float32)
        export_profile.update(driver="GTiff", dtype=image.dtype, count=1, compress="lzw")
        with rasterio.open(export_folder + filename, "w", **export_profile) as dst:
            dst.write(image, 1)
        print("Exported:", export_folder + filename)


# =============================================================================
# CODE 3: VECTORIZATION AND ORIENTATION ANALYSIS
# =============================================================================

import geopandas as gpd
from shapely.geometry import LineString
from skimage.measure import find_contours

def vectorize_lineaments(lineament_file, output_shp):
    """Convert raster lineaments to vector and calculate orientations"""
    print("Opening geological lineament raster...")
    with rasterio.open(lineament_file) as src:
        lineaments = src.read(1)
        transform = src.transform
    print("Raster shape:", lineaments.shape)
    
    labeled, num = label(lineaments)
    print("Detected regions:", num)
    
    records = []
    MIN_LENGTH, MAX_LENGTH = 8, 500
    
    for region_id in range(1, num + 1):
        mask = labeled == region_id
        contours = find_contours(mask.astype(float), 0.5)
        if len(contours) == 0:
            continue
        contour = max(contours, key=len)
        if len(contour) < 6:
            continue
        
        coords = []
        for y, x in contour:
            geo_x, geo_y = rasterio.transform.xy(transform, int(y), int(x))
            coords.append((geo_x, geo_y))
        
        line = LineString(coords).simplify(tolerance=5, preserve_topology=True)
        if len(line.coords) < 2:
            continue
        
        length = line.length
        if length < MIN_LENGTH or length > MAX_LENGTH:
            continue
        
        # PCA orientation
        coords_array = np.array(line.coords)
        x, y = coords_array[:,0], coords_array[:,1]
        x_mean, y_mean = np.mean(x), np.mean(y)
        x_cent, y_cent = x - x_mean, y - y_mean
        covariance = np.cov(x_cent, y_cent)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        major_axis = eigenvectors[:, np.argmax(eigenvalues)]
        angle = np.degrees(np.arctan2(major_axis[1], major_axis[0])) % 180
        
        records.append({"geometry": line, "length": length, "orientation": angle})
    
    gdf = gpd.GeoDataFrame(records)
    print("Valid geological lineaments:", len(gdf))
    
    # Classify trends
    def classify(angle):
        if angle < 22.5 or angle >= 157.5: return "E-W"
        elif angle < 67.5: return "NE-SW"
        elif angle < 112.5: return "N-S"
        else: return "NW-SE"
    
    gdf["trend"] = gdf["orientation"].apply(classify)
    gdf.to_file(output_shp)
    print(f"Saved shapefile: {output_shp}")
    return gdf


# =============================================================================
# CODE 4: INTERSECTION ANALYSIS
# =============================================================================

def compute_intersections(lineament_shp, output_intersections):
    """Compute intersections between lineaments"""
    gdf = gpd.read_file(lineament_shp)
    intersections = []
    for i, line1 in enumerate(gdf.geometry):
        for j, line2 in enumerate(gdf.geometry):
            if i >= j:
                continue
            if line1.intersects(line2):
                inter = line1.intersection(line2)
                if inter.geom_type == "Point":
                    intersections.append(inter)
    intersection_gdf = gpd.GeoDataFrame(geometry=intersections, crs=gdf.crs)
    print("Total intersections:", len(intersection_gdf))
    intersection_gdf.to_file(output_intersections)
    return intersection_gdf


# =============================================================================
# CODE 5: STUDY AREA BOUNDARY EXTRACTION
# =============================================================================

def extract_boundary(dem_file, output_boundary):
    """Extract study area boundary from DEM"""
    print("Opening DEM...")
    with rasterio.open(dem_file) as src:
        dem = src.read(1)
        transform, crs = src.transform, src.crs
    
    from rasterio.features import shapes
    from shapely.geometry import shape
    
    mask = np.where(np.isnan(dem), 0, 1)
    mask[dem <= 0] = 0
    
    results = shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform)
    polygons = [shape(geom) for geom, value in results if value == 1]
    
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs).dissolve()
    gdf.to_file(output_boundary)
    print(f"Boundary saved: {output_boundary}")
    return gdf


# =============================================================================
# CODE 6: CLIP LINEAMENTS
# =============================================================================

def clip_lineaments(lineament_shp, boundary_shp, output_clip):
    """Clip lineaments to study area boundary"""
    lineaments = gpd.read_file(lineament_shp)
    boundary = gpd.read_file(boundary_shp)
    
    if lineaments.crs is None:
        lineaments = lineaments.set_crs(boundary.crs)
    if lineaments.crs != boundary.crs:
        lineaments = lineaments.to_crs(boundary.crs)
    
    clipped = gpd.clip(lineaments, boundary)
    print(f"Original: {len(lineaments)}, Clipped: {len(clipped)}")
    clipped.to_file(output_clip)
    return clipped


# =============================================================================
# CODE 7: LINEAMENT DENSITY MAPPING
# =============================================================================

def lineament_density(lineament_shp, output_density=None):
    """Generate smoothed lineament density map"""
    gdf = gpd.read_file(lineament_shp)
    
    # Extract centroids
    points = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry])
    xmin, ymin, xmax, ymax = gdf.total_bounds
    
    # Create grid
    grid_size = 150
    xgrid = np.linspace(xmin, xmax, grid_size)
    ygrid = np.linspace(ymin, ymax, grid_size)
    density = np.zeros((grid_size, grid_size))
    
    for pt in points:
        x_idx = np.searchsorted(xgrid, pt[0]) - 1
        y_idx = np.searchsorted(ygrid, pt[1]) - 1
        if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
            density[y_idx, x_idx] += 1
    
    density = gaussian_filter(density, sigma=5)
    threshold = np.percentile(density, 60)
    density[density < threshold] = np.nan
    
    # Display
    plt.figure(figsize=(12,12))
    plt.imshow(np.rot90(density), cmap='hot', extent=[xmin, xmax, ymin, ymax])
    plt.colorbar(label='Structural Density')
    plt.title("Geological Lineament Density")
    plt.show()
    return density


# =============================================================================
# CODE 8: CORRECTED ORIENTATION ANALYSIS
# =============================================================================

def orientation_analysis(lineament_shp):
    """Perform corrected orientation analysis with rose diagram"""
    gdf = gpd.read_file(lineament_shp)
    gdf["length"] = gdf.geometry.length
    gdf = gdf[gdf["length"] > 20]
    
    def calc_orientation(line):
        from shapely.geometry import MultiLineString
        try:
            if isinstance(line, MultiLineString):
                line = max(line.geoms, key=lambda g: g.length)
            coords = np.array(line.coords)
            x, y = coords[:,0], coords[:,1]
            x_cent, y_cent = x - np.mean(x), y - np.mean(y)
            cov = np.cov(x_cent, y_cent)
            eigvals, eigvecs = np.linalg.eig(cov)
            major = eigvecs[:, np.argmax(eigvals)]
            angle = np.degrees(np.arctan2(major[1], major[0]))
            return angle % 180 if angle >= 0 else angle + 180
        except:
            return np.nan
    
    gdf["orientation"] = gdf.geometry.apply(calc_orientation)
    gdf = gdf.dropna(subset=["orientation"])
    angles = gdf["orientation"].values
    
    # Histogram
    plt.figure(figsize=(12,6))
    plt.hist(angles, bins=np.arange(0, 181, 10), color='darkblue', edgecolor='black')
    plt.xlabel("Orientation Angle (Degrees)")
    plt.ylabel("Frequency")
    plt.title("Geological Lineament Orientation Histogram")
    plt.grid(True)
    plt.show()
    
    # Rose diagram
    theta = np.concatenate([np.deg2rad(angles), np.deg2rad(angles) + np.pi])
    bin_edges = np.deg2rad(np.arange(0, 361, 10))
    hist, edges = np.histogram(theta, bins=bin_edges)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, polar=True)
    ax.bar(edges[:-1], hist, width=np.diff(edges), bottom=0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Geological Rose Diagram")
    plt.show()
    
    # Classify
    def interpret(angle):
        if 0 <= angle < 22.5 or 157.5 <= angle <= 180: return "N-S"
        elif 22.5 <= angle < 67.5: return "NE-SW"
        elif 67.5 <= angle < 112.5: return "E-W"
        else: return "NW-SE"
    
    gdf["trend"] = gdf["orientation"].apply(interpret)
    print("\nStructural Families:")
    print(gdf["trend"].value_counts())
    return gdf


# =============================================================================
# CODE 9: PRISMA MNF TRANSFORMATION
# =============================================================================

def prisma_mnf(input_raster, output_raster, num_components=20):
    """Apply Minimum Noise Fraction transformation to PRISMA data"""
    try:
        from spectral import mnf, noise_from_diffs
    except ImportError:
        print("Please install spectral: pip install spectral")
        return None
    
    with rasterio.open(input_raster) as src:
        img = src.read()
        profile = src.profile
    
    img = np.transpose(img, (1, 2, 0))
    img = np.nan_to_num(img)
    rows, cols, bands = img.shape
    
    X = img.reshape(rows * cols, bands)
    noise = noise_from_diffs(X)
    mnf_transform = mnf(signal=X, noise=noise)
    X_mnf = mnf_transform.reduce(X, num=num_components)
    mnf_img = np.transpose(X_mnf.reshape(rows, cols, num_components), (2, 0, 1))
    
    profile.update(dtype=rasterio.float32, count=num_components)
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(mnf_img.astype(rasterio.float32))
    print(f"MNF saved: {output_raster}")
    return mnf_img


# =============================================================================
# CODE 10: ALTERATION ANOMALY EXTRACTION
# =============================================================================

def extract_anomalies(clay_path, iron_path, output_dir):
    """Extract clay, iron, and combined alteration anomalies"""
    def read_raster(path):
        with rasterio.open(path) as src:
            return src.read(1), src.profile
    
    clay, profile = read_raster(clay_path)
    iron, _ = read_raster(iron_path)
    
    clay = np.clip(np.nan_to_num(clay), np.percentile(clay, 1), np.percentile(clay, 99))
    iron = np.clip(np.nan_to_num(iron), np.percentile(iron, 1), np.percentile(iron, 99))
    
    clay_thresh, iron_thresh = np.percentile(clay, 90), np.percentile(iron, 90)
    print(f"Clay Threshold: {clay_thresh}, Iron Threshold: {iron_thresh}")
    
    clay_anomaly = (clay >= clay_thresh).astype(np.uint8)
    iron_anomaly = (iron >= iron_thresh).astype(np.uint8)
    combined = ((clay_anomaly + iron_anomaly) > 0).astype(np.uint8)
    
    profile.pop("blockxsize", None); profile.pop("blockysize", None)
    profile.pop("tiled", None)
    profile.update(dtype=rasterio.uint8, count=1)
    
    outputs = {"Clay_Anomaly.tif": clay_anomaly, "Iron_Anomaly.tif": iron_anomaly, 
               "Combined_Alteration_Anomaly.tif": combined}
    for name, data in outputs.items():
        with rasterio.open(f"{output_dir}/{name}", "w", **profile) as dst:
            dst.write(data, 1)
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    titles = ["Clay Anomaly", "Iron Anomaly", "Combined Alteration"]
    for i, data in enumerate([clay_anomaly, iron_anomaly, combined]):
        axes[i].imshow(data, cmap='hot', vmin=0, vmax=1)
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    return clay_anomaly, iron_anomaly, combined


# =============================================================================
# CODE 11: SPECTRAL SIGNATURE EXTRACTION
# =============================================================================

def extract_spectral_signature(prisma_path):
    """Extract mean reflectance spectrum with alteration masks"""
    with rasterio.open(prisma_path) as src:
        img = src.read()
    img = np.transpose(img, (1, 2, 0))
    rows, cols, bands = img.shape
    
    wavelengths = np.linspace(407, 2497, bands)
    mean_spectrum = np.mean(img, axis=(0,1))
    
    # Masks
    iron_mask = (wavelengths >= 500) & (wavelengths <= 900)
    clay_mask = (wavelengths >= 2100) & (wavelengths <= 2300)
    carbonate_mask = (wavelengths >= 2300) & (wavelengths <= 2350)
    hydroxyl_mask = (wavelengths >= 2180) & (wavelengths <= 2220)
    
    # Plot
    plt.figure(figsize=(14,7))
    plt.plot(wavelengths, mean_spectrum, linewidth=2, label='Full Spectrum')
    plt.plot(wavelengths[iron_mask], mean_spectrum[iron_mask], linewidth=3, label='Iron Oxides')
    plt.plot(wavelengths[clay_mask], mean_spectrum[clay_mask], linewidth=3, label='Clay Minerals')
    plt.plot(wavelengths[carbonate_mask], mean_spectrum[carbonate_mask], linewidth=3, label='Carbonates')
    plt.plot(wavelengths[hydroxyl_mask], mean_spectrum[hydroxyl_mask], linewidth=3, label='Hydroxyl Minerals')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("Alteration Mineral Spectral Signatures")
    plt.legend()
    plt.grid(True)
    plt.show()
    return wavelengths, mean_spectrum


# =============================================================================
# CODE 12: NORMALISATION FUNCTION
# =============================================================================

def normalize_raster(input_raster, output_raster):
    """Normalise raster to [0, 1] range using min-max scaling"""
    with rasterio.open(input_raster) as src:
        data = src.read(1).astype("float32")
        profile = src.profile
        nodata = src.nodata
        mask = data != nodata if nodata is not None else np.isfinite(data)
        valid = data[mask]
        data_norm = np.zeros_like(data)
        data_norm[mask] = (valid - valid.min()) / (valid.max() - valid.min())
        profile.update(dtype="float32")
        with rasterio.open(output_raster, "w", **profile) as dst:
            dst.write(data_norm.astype("float32"), 1)
    print(f"Normalised: {output_raster}")


# =============================================================================
# CODE 13: BATCH NORMALISATION
# =============================================================================

def batch_normalize():
    """Normalise all feature layers"""
    files = [
        (r"C:\Users\lenovo\Desktop\Clean Final\Clay_Anomaly.tif", r"C:\Prospectivity_Project\Output\Clay_Norm.tif"),
        (r"C:\Users\lenovo\Desktop\Clean Final\Iron_Anomaly.tif", r"C:\Prospectivity_Project\Output\Iron_Norm.tif"),
        (r"C:\Users\lenovo\Desktop\Clean Final\Combined_Alteration_Anomaly.tif", r"C:\Prospectivity_Project\Output\Alteration_Norm.tif"),
        (r"C:\Users\lenovo\Desktop\Clean Final\Lineament_Density.tif", r"C:\Prospectivity_Project\Output\Lineament_Norm.tif"),
        (r"C:\Prospectivity_Project\Output\Distance_to_Lineaments.tif", r"C:\Prospectivity_Project\Output\Distance_Norm.tif")
    ]
    for inp, out in files:
        normalize_raster(inp, out)


# =============================================================================
# CODE 14: INVERSE DISTANCE TRANSFORMATION
# =============================================================================

def inverse_distance():
    """Invert distance layer so closer distances get higher values"""
    with rasterio.open(r"C:\Prospectivity_Project\Output\Distance_Norm.tif") as src:
        dist = src.read(1)
        profile = src.profile
    inverse = 1 - dist
    with rasterio.open(r"C:\Prospectivity_Project\Output\Distance_Inverse.tif", "w", **profile) as dst:
        dst.write(inverse.astype("float32"), 1)
    print("Inverse distance saved")


# =============================================================================
# CODE 15: PCA/MNF NORMALISATION
# =============================================================================

def normalize_pca_mnf():
    """Normalise PCA/MNF component"""
    input_raster = r"C:\Users\lenovo\Desktop\Clean Final\Prisma _PCA_MNF.tif"
    output_raster = r"C:\Prospectivity_Project\Output\PCA_MNF_Norm.tif"
    with rasterio.open(input_raster) as src:
        data = src.read(1).astype("float32")
        profile = src.profile
        mask = np.isfinite(data)
        valid = data[mask]
        normalized = np.zeros_like(data)
        normalized[mask] = (valid - valid.min()) / (valid.max() - valid.min())
        profile.update(dtype="float32")
        with rasterio.open(output_raster, "w", **profile) as dst:
            dst.write(normalized.astype("float32"), 1)
    print(f"PCA/MNF normalised: {output_raster}")


# =============================================================================
# CODE 16: PROSPECTIVITY MAP SMOOTHING
# =============================================================================

def smooth_prospectivity():
    """Smooth prospectivity map with Gaussian filter"""
    input_raster = r"C:\Prospectivity_Project\Output\Alteration_Norm.tif"
    output_raster = r"C:\Prospectivity_Project\Output\Alteration_Norm_Prospectivity_Map_Smoothed.tif"
    with rasterio.open(input_raster) as src:
        data = src.read(1)
        profile = src.profile
        smooth = gaussian_filter(data, sigma=2)
        with rasterio.open(output_raster, "w", **profile) as dst:
            dst.write(smooth.astype("float32"), 1)
    print("Smoothing complete")


# =============================================================================
# CODE A: STATISTICS EXTRACTION
# =============================================================================

def extract_statistics():
    """Extract all statistical values for Chapter 4"""
    print("=" * 70)
    print("EXTRACTING STATISTICS")
    print("=" * 70)
    
    # File paths
    paths = {
        'canny': r"C:\Users\lenovo\Downloads\finaall maps chapter 4\Canny map.tif",
        'frangi': r"C:\Users\lenovo\Downloads\finaall maps chapter 4\Frangi_Ridge_enhance.tif",
        'final': r"C:\Users\lenovo\Downloads\finaall maps chapter 4\Final_Geo_Lin.tif",
        'slope': r"C:\Users\lenovo\Downloads\finaall maps chapter 4\Turk_Slope.tif",
        'clay': r"C:\Prospectivity_Project\Output\Clay_Norm.tif",
        'iron': r"C:\Prospectivity_Project\Output\Iron_Norm.tif",
        'alteration': r"C:\Prospectivity_Project\Output\Alteration_Norm.tif",
        'density': r"C:\Prospectivity_Project\Output\Lineament_Norm.tif",
        'prospectivity': r"C:\Prospectivity_Project\Output\Prospectivity_Map_WeightedA.tif",
    }
    
    # Count lineaments
    for name in ['canny', 'frangi', 'final']:
        with rasterio.open(paths[name]) as src:
            count = np.sum(np.nan_to_num(src.read(1)) > 0)
            print(f"{name.upper()}: {count:,} pixels")
    
    # Total lineaments
    with rasterio.open(paths['final']) as src:
        skeleton = np.nan_to_num(src.read(1)).astype(np.uint8)
        labeled, num = label(skeleton)
        print(f"\nTotal lineaments: {num:,}")
    
    # Density
    with rasterio.open(paths['density']) as src:
        dens = src.read(1)
        dens = dens[~np.isnan(dens)]
        print(f"\nMean density: {np.mean(dens * 38.7):.1f} km/km²")
        print(f"Max density: {np.max(dens * 38.7):.1f} km/km²")
    
    # Slope
    with rasterio.open(paths['slope']) as src:
        slope = src.read(1)
        slope = slope[~np.isnan(slope)]
        print(f"\nSlope - Min: {np.min(slope):.1f}°, Max: {np.max(slope):.1f}°, Mean: {np.mean(slope):.1f}°")
    
    # Alteration
    for name, path in [('Clay', paths['clay']), ('Iron', paths['iron']), ('Combined', paths['alteration'])]:
        with rasterio.open(path) as src:
            data = src.read(1)
            pixels = np.sum(np.nan_to_num(data) > 0.5)
            print(f"{name}: {pixels * 0.01:.1f} ha ({pixels:,} pixels)")
    
    # Prospectivity
    with rasterio.open(paths['prospectivity']) as src:
        prosp = src.read(1)
        prosp = prosp[~np.isnan(prosp)]
        print(f"\nProspectivity - Min: {np.min(prosp):.2f}, Max: {np.max(prosp):.2f}, Mean: {np.mean(prosp):.2f}")


# =============================================================================
# CODE B: VALIDATION AND ACCURACY ASSESSMENT
# =============================================================================

def validate_model():
    """Perform validation, confusion matrix, and accuracy assessment"""
    print("=" * 70)
    print("VALIDATION AND ACCURACY ASSESSMENT")
    print("=" * 70)
    
    # Field data
    data = {
        'Sample_ID': ['TK_25_RC_371', 'TK_25_RC_357', 'TK_25_RC_376A', 'TK_25_RC_107',
                      'TK_25_RC_195', 'TK_25_RC_432', 'TK_25_RC_209', 'TK_25_RC_219',
                      'TK_25_RC_181', 'TK_25_RC_194', 'TK_25_RC_200', 'TK_25_RC_400',
                      'TK_25_RC_603', 'TK_25_RC_199', 'TK_25_RC_213', 'TK_25_RC_214'],
        'Au_gpt': [0.77, 2.40, 1.68, 0.29, 0.83, 0.51, 0.13, 0.70,
                   0.00, 0.40, 0.02, 0.18, 1.15, 0.24, 0.39, 0.23],
        'Prospectivity': [0.52, 0.68, 0.55, 0.44, 0.48, 0.47, 0.41, 0.53,
                          0.38, 0.45, 0.35, 0.42, 0.58, 0.43, 0.46, 0.44]
    }
    df = pd.DataFrame(data)
    
    print(f"Loaded {len(df)} validation points")
    print(f"Mineralised (Au > 0.5): {sum(df['Au_gpt'] > 0.5)}")
    print(f"Non-mineralised: {sum(df['Au_gpt'] <= 0.5)}")
    
    # Binary classification
    y_true = [1 if au > 0.5 else 0 for au in df['Au_gpt']]
    y_pred = [1 if score > 0.47 else 0 for score in df['Prospectivity']]
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    p_e = ((tp + fn)*(tp + fp) + (tn + fp)*(tn + fn)) / ((tp+tn+fp+fn)**2)
    kappa = (accuracy - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0
    
    print(f"\nConfusion Matrix:")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"\nAccuracy: {accuracy:.1%}")
    print(f"Precision: {precision:.1%}")
    print(f"Recall: {recall:.1%}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Kappa: {kappa:.3f}")
    
    # Bar chart
    zone_means = [df[df['Au_gpt'] > 0.5]['Au_gpt'].mean(), df[df['Au_gpt'] <= 0.5]['Au_gpt'].mean()]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['HPZ', 'MPZ/LPZ'], zone_means, color=['#d73027', '#1a9850'], edgecolor='black')
    for bar, val in zip(bars, zone_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f} g/t', 
                ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Mean Gold Grade (g/t)')
    ax.set_title('Field Validation Results')
    ax.set_ylim(0, max(zone_means) + 0.3)
    plt.tight_layout()
    plt.savefig("Validation_Bar_Chart.png", dpi=300)
    plt.show()
    
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TURKMINE GOLD PROSPECTIVITY MAPPING - COMPLETE WORKFLOW")
    print("=" * 70)
    print("\nThis file contains all functions for the complete workflow.")
    print("\nTo run individual functions, call them with appropriate parameters:")
    print("  - extract_lineaments(vv_file, output_path)")
    print("  - vectorize_lineaments(lineament_file, output_shp)")
    print("  - compute_intersections(lineament_shp, output_intersections)")
    print("  - extract_boundary(dem_file, output_boundary)")
    print("  - prisma_mnf(input_raster, output_raster)")
    print("  - extract_anomalies(clay_path, iron_path, output_dir)")
    print("  - normalize_raster(input_raster, output_raster)")
    print("  - extract_statistics()")
    print("  - validate_model()")
    print("\nSee function docstrings for more details.")
