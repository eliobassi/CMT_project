import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
import geopandas as gpd
import rasterio
import rasterio.mask
import glob
import re

#ello

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


warnings.filterwarnings("ignore", category=OptimizeWarning)

# ------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------
df = pd.read_csv("NDVI_NO2_timeseries.csv")

# Remove rows missing NDVI or NO₂
df = df.dropna(subset=["Mean_NDVI", "Mean_NO2"])

regions = df["Region"].unique()


# ------------------------------------------------------
# Logistic model
# ------------------------------------------------------
def logistic(t, r, K, B0):
    return K / (1 + (K/B0 - 1)*np.exp(-r*t))


# ------------------------------------------------------
# 2. Fit logistic model for each region
# ------------------------------------------------------
rows = []

for region in regions:
    data = df[df.Region == region].sort_values("Year")

    years = data["Year"].values
    t = years - years.min()
    B = data["Mean_NDVI"].values
    P = data["Mean_NO2"].values

    # Remove invalid points
    mask = ~np.isnan(B)
    t = t[mask]
    B = B[mask]
    P = P[mask]

    if len(B) < 4:
        continue  # Not enough data to fit logistic

    # Initial guesses
    B0_guess = B[0]
    K_guess = max(B) + 0.1
    r_guess = 0.1

    # Use bounded fit for stability
    try:
        popt, _ = curve_fit(
            logistic, t, B,
            p0=[r_guess, K_guess, B0_guess],
            bounds=([0.0001, 0.1, 0.0], [2.0, 2.0, 2.0]),  # r, K, B0 bounds
            maxfev=8000
        )
    except:
        continue  # bad logistic fit → skip region

    r_est, K_est, B0_est = popt

    # store one row PER YEAR
    for yr, Pi, Bi in zip(years[mask], P, B):
        rows.append({
            "Region": region,
            "Year": yr,
            "Mean_NO2": Pi,
            "Mean_NDVI": Bi,
            "r_estimated": r_est,
            "K_estimated": K_est,
            "B0_estimated": B0_est,
        })

results_df = pd.DataFrame(rows)

# ------------------------------------------------------
# 3. Fit pollution sensitivity
# ------------------------------------------------------
clean = results_df.dropna(subset=["r_estimated", "Mean_NO2"])

# Keep only positive r (logistic r must be > 0)
clean = clean[clean["r_estimated"] > 0]

if len(clean) < 3:
    raise RuntimeError("Too few valid samples to fit pollution sensitivity α.")

P = clean["Mean_NO2"].values
log_r = np.log(clean["r_estimated"].values)

# Fit: log(r) = log(r0) - α P
coef = np.polyfit(P, log_r, 1)
alpha = -coef[0]
r0 = np.exp(coef[1])

print("-------------------------------------------------")
print("Global Fitted Parameters:")
print(f"  r0     = {r0:.6f}")
print(f"  alpha  = {alpha:.6f}")
print("-------------------------------------------------")


# ------------------------------------------------------
# 4. Save output
# ------------------------------------------------------
results_df["r0_global"] = r0
results_df["alpha_global"] = alpha

results_df.to_csv("fitted_parameters.csv", index=False)
print("saved fitted_parameters.csv")




# Load fitted parameters to get last known NO2 per region
df = pd.read_csv("fitted_parameters.csv")

regions = df["Region"].unique()

# Determine last measured year in the dataset
last_year = df["Year"].max()

# Future prediction range (you can adjust here)
future_years = range(last_year + 1, 2051)

# Helper: build scenario rows
def make_rows(modifier):
    rows = []
    for region in regions:
        # Last observed NO2 for this region
        last_no2 = df[df["Region"] == region]["Mean_NO2"].iloc[-1]

        for i, year in enumerate(future_years):
            if modifier == "constant":
                no2 = last_no2
            
            elif modifier == "minus1":
                no2 = last_no2 * ((0.99) ** (i + 1))
            
            elif modifier == "plus1":
                no2 = last_no2 * ((1.01) ** (i + 1))

            rows.append({
                "Region": region,
                "Year": year,
                "NO2": no2
            })
    return rows

# Scenario A: NO2 stays constant
rows_const = make_rows("constant")
pd.DataFrame(rows_const).to_csv("future_NO2_constant.csv", index=False)

# Scenario B: NO2 decreases 1% per year
rows_minus = make_rows("minus1")
pd.DataFrame(rows_minus).to_csv("future_NO2_minus1percent.csv", index=False)

# Scenario C: NO2 increases 1% per year
rows_plus = make_rows("plus1")
pd.DataFrame(rows_plus).to_csv("future_NO2_plus1percent.csv", index=False)

print("Generated:")
print("  - future_NO2_constant.csv")
print("  - future_NO2_minus1percent.csv")
print("  - future_NO2_plus1percent.csv")


# ------------------------------------------------------

# --- 1. Charger les paramètres fités ---
df_fitted = pd.read_csv("fitted_parameters.csv")

# Colonnes qui doivent disparaître des paramètres
cols_to_drop = ["Mean_NO2", "Mean_NDVI", "Year"]

for col in cols_to_drop:
    if col in df_fitted.columns:
        df_fitted = df_fitted.drop(columns=[col])
        print(f"Supprimé du fitted_parameters : {col}")

# --- 2. Charger les scénarios NO2 ---
scenario_const = pd.read_csv("future_NO2_constant.csv")
scenario_minus = pd.read_csv("future_NO2_minus1percent.csv")
scenario_plus  = pd.read_csv("future_NO2_plus1percent.csv")


# --- 3. Fonction pour merger proprement SANS créer de doublons ---
def merge_and_clean(scenario_df, fitted_df, scenario_name):

    print(f"\n--- Traitement du scénario : {scenario_name} ---")

    # 3A. Retirer du fitted_parameters les colonnes déjà présentes dans le scénario
    overlapping = [c for c in fitted_df.columns if c in scenario_df.columns and c != "Region"]

    if overlapping:
        print(f"Colonnes supprimées pour éviter doublons : {overlapping}")
        fitted_clean = fitted_df.drop(columns=overlapping)
    else:
        fitted_clean = fitted_df.copy()

    # 3B. Merge propre
    merged = scenario_df.merge(fitted_clean, on="Region", how="left")

    # 3C. Supprimer les doublons (le vrai problème !)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["Region", "Year"], keep="first")
    after = len(merged)

    print(f"Doublons supprimés : {before - after}")

    return merged


# --- 4. Appliquer la fonction aux 3 scénarios ---
clean_const = merge_and_clean(scenario_const, df_fitted, "NO2 constant")
clean_minus = merge_and_clean(scenario_minus, df_fitted, "NO2 -1%/an")
clean_plus  = merge_and_clean(scenario_plus, df_fitted, "NO2 +1%/an")


# --- 5. Sauvegarder les CSV finaux propres ---
clean_const.to_csv("scenario_with_params_constant_clean.csv", index=False)
clean_minus.to_csv("scenario_with_params_minus1percent_clean.csv", index=False)
clean_plus.to_csv("scenario_with_params_plus1percent_clean.csv", index=False)

print("\n FICHIERS FINAUX GÉNÉRÉS :")
print("  ✔ scenario_with_params_constant_clean.csv")
print("  ✔ scenario_with_params_minus1percent_clean.csv")
print("  ✔ scenario_with_params_plus1percent_clean.csv")

