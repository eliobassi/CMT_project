
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import glob
import re

# -------------------------
# FILE SETTINGS
# -------------------------
shp_path = "swissBOUNDARIES3D_1_5_TLM_BEZIRKSGEBIET.shp"

# Pattern matching files like NDVI_2010.tif ... NDVI_2018.tif
ndvi_files = sorted(glob.glob("NDVI_*.tif"))
no2_files  = sorted(glob.glob("NO2_*.tif"))

# Regex to extract the year from "NDVI_2010.tif"
year_pattern = re.compile(r".*_(\d{4})\.tif$")

# -------------------------
# STEP 1 — Load Shapefile
# -------------------------
regions = gpd.read_file(shp_path)

# -------------------------
# FUNCTION: extract mean raster value for a region
# -------------------------
def extract_mean_per_region(raster_path, regions):
    results = []

    with rasterio.open(raster_path) as src:
        # Reproject shapefile to match raster CRS
        regions_proj = regions.to_crs(src.crs)
        nodata = src.nodata

        for i, row in regions_proj.iterrows():
            geom = [row.geometry]

            try:
                masked, _ = rasterio.mask.mask(src, geom, crop=True)
            except ValueError:
                # Region does not overlap raster
                results.append(np.nan)
                continue

            masked = masked.astype(float)
            if nodata is not None:
                masked[masked == nodata] = np.nan

            results.append(np.nanmean(masked))

    return results

# -------------------------
# STEP 2 — MASTER LIST TO STORE RESULTS
# -------------------------
all_rows = []

# -------------------------
# STEP 3 — LOOP THROUGH YEARS
# -------------------------
for ndvi_path, no2_path in zip(ndvi_files, no2_files):

    # Extract year from filename
    match = year_pattern.match(ndvi_path)
    if not match:
        print(f"Skipping file with no year: {ndvi_path}")
        continue

    year = int(match.group(1))
    print(f"Processing year {year} ...")

    # ---- Extract NDVI for this year
    ndvi_vals = extract_mean_per_region(ndvi_path, regions)

    # ---- Extract NO2 for this year
    no2_vals = extract_mean_per_region(no2_path, regions)

    # ---- Append results
    for region_name, ndvi_val, no2_val in zip(regions["NAME"], ndvi_vals, no2_vals):
        all_rows.append({
            "Region": region_name,
            "Year": year,
            "Mean_NDVI": ndvi_val,
            "Mean_NO2": no2_val
        })

# -------------------------
# STEP 4 — CREATE FINAL CSV
# -------------------------
df = pd.DataFrame(all_rows)
df = df.sort_values(["Region", "Year"])
df.to_csv("NDVI_NO2_timeseries.csv", index=False)

print("NDVI_NO2_timeseries.csv created successfully")
