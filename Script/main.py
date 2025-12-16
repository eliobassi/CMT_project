import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
import geopandas as gpd
import rasterio
import rasterio.mask
import glob
import re
import matplotlib.pyplot as plt
import subprocess

# -------------------------
# FILE SETTINGS
# -------------------------
# Path to the shapefile containing boundary data for regions
shp_path = "CMT_project/data/swissBOUNDARIES3D_1_5_TLM_BEZIRKSGEBIET.shp"

# List all NDVI and NO2 raster files in the current directory, sorted by filename
ndvi_files = sorted(glob.glob("CMT_project/data/NDVI_*.tif"))
no2_files  = sorted(glob.glob("CMT_project/data/NO2_*.tif"))

# Regex pattern to extract the year from the filenames of NDVI raster files (e.g., NDVI_2010.tif)
year_pattern = re.compile(r".*_(\d{4})\.tif$")

# -------------------------
# STEP 1 — Load Shapefile
# -------------------------
# Load the shapefile into a GeoDataFrame. This shapefile contains regions for analysis.
regions = gpd.read_file(shp_path)

# -------------------------
# FUNCTION: extract mean raster value for a region
# -------------------------
def extract_mean_per_region(raster_path, regions):
     # Initialize an empty list to store results
    results = []

     # Open the raster file using rasterio
    with rasterio.open(raster_path) as src:
        # Reproject shapefile to match raster CRS
        regions_proj = regions.to_crs(src.crs)
        # Get the "no data" value from the raster (cells without data)
        nodata = src.nodata

        # Loop through each region in the shapefile
        for i, row in regions_proj.iterrows():
            # Extract the geometry (boundary) of the current region
            geom = [row.geometry]

            try:
                # Mask the raster using the region's geometry to extract values within the region
                masked, _ = rasterio.mask.mask(src, geom, crop=True)
            except ValueError:
                # If the region does not overlap with the raster, skip and append NaN
                results.append(np.nan)
                continue

            # Convert the masked raster data to float type for further calculations
            masked = masked.astype(float)
            # Replace 'no data' values with NaN
            if nodata is not None:
                masked[masked == nodata] = np.nan

            # Calculate the mean value for the region and append to results
            results.append(np.nanmean(masked))

    # Return the list of mean values for each region
    return results

# -------------------------
# STEP 2 — MASTER LIST TO STORE RESULTS
# -------------------------
# Initialize an empty list to store all the results (across years and regions)
all_rows = []

# -------------------------
# STEP 3 — LOOP THROUGH YEARS
# -------------------------
# Loop through pairs of NDVI and NO2 files (both should correspond to the same year)
for ndvi_path, no2_path in zip(ndvi_files, no2_files):

    # Extract the year from the NDVI file name using the regex pattern
    match = year_pattern.match(ndvi_path)
    if not match:
        # If the filename doesn't match the expected pattern (i.e., no year found), skip this file
        print(f"Skipping file with no year: {ndvi_path}")
        continue

    # Get the year from the matched group
    year = int(match.group(1))
    print(f"Processing year {year} ...")

    # ---- Extract NDVI for this year
    # Call the function to get the mean NDVI values for each region
    ndvi_vals = extract_mean_per_region(ndvi_path, regions)

    # ---- Extract NO2 for this year
    # Call the function to get the mean NO2 values for each region
    no2_vals = extract_mean_per_region(no2_path, regions)

    # ---- Append results for this year
    # For each region, store the region name, year, and the mean NDVI and NO2 values
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
# Convert the list of all results into a Pandas DataFrame
df = pd.DataFrame(all_rows)
# Sort the DataFrame by Region and Year
df = df.sort_values(["Region", "Year"])
# Save the results to a CSV file
df.to_csv("CMT_project/data/NDVI_NO2_timeseries.csv", index=False)

print("NDVI_NO2_timeseries.csv created successfully")

# Suppress optimization warnings to avoid clutter in the output
warnings.filterwarnings("ignore", category=OptimizeWarning)

# ------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------
# Load the NDVI and NO2 time series dataset from the CSV file
df = pd.read_csv("CMT_project/data/NDVI_NO2_timeseries.csv")

# Remove rows missing NDVI or NO₂
df = df.dropna(subset=["Mean_NDVI", "Mean_NO2"])

#Get a list of all unique regions in the dataset
regions = df["Region"].unique()

# ------------------------------------------------------
# Logistic model
# ------------------------------------------------------
# Define the logistic growth model, which describes the change in NDVI over time
def logistic(t, r, K, B0):
    return K / (1 + (K/B0 - 1)*np.exp(-r*t))


# ------------------------------------------------------
# 2. Fit logistic model for each region
# ------------------------------------------------------
# Loop through each region in the dataset
rows = []

# Loop through each region in the dataset
for region in regions:
    # Filter data for the current region and sort by year
    data = df[df.Region == region].sort_values("Year")

    # Extract years, normalized time (t), NDVI (B), and NO2 (P) values for this region
    years = data["Year"].values
    t = years - years.min()  # Normalize time to start at 0
    B = data["Mean_NDVI"].values
    P = data["Mean_NO2"].values

    # Remove invalid data points (where NDVI value is NaN)
    mask = ~np.isnan(B)
    t = t[mask]
    B = B[mask]
    P = P[mask]

    # Skip regions with fewer than 4 valid data points (logistic fitting requires more data)
    if len(B) < 4:
        continue

    # Initial guesses for the logistic model parameters
    B0_guess = B[0]
    K_guess = max(B) + 0.1
    r_guess = 0.1

    # Use bounded curve fitting with initial guesses and bounds for stability
    try:
        popt, _ = curve_fit(
            logistic, t, B,
            p0=[r_guess, K_guess, B0_guess],
            bounds=([0.0001, 0.1, 0.0], [2.0, 2.0, 2.0]),  # bounds for r, K, and B0
            maxfev=8000  # Maximum number of function evaluations
        )
    except:
        # If fitting fails (e.g., due to insufficient data), skip this region
        continue

    # Extract the optimized logistic model parameters (r, K, B0)
    r_est, K_est, B0_est = popt

    # Store the results for each year in the region
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

# Convert the fitted results into a DataFrame
results_df = pd.DataFrame(rows)

# ------------------------------------------------------
# 3. Fit pollution sensitivity
# ------------------------------------------------------
# Remove rows with missing or invalid values for the logistic growth rate (r_estimated) or NO2
clean = results_df.dropna(subset=["r_estimated", "Mean_NO2"])

# Keep only rows where the logistic growth rate (r) is positive (must be > 0)
clean = clean[clean["r_estimated"] > 0]

# Check if there are enough valid data points for fitting the pollution sensitivity
if len(clean) < 3:
    raise RuntimeError("Too few valid samples to fit pollution sensitivity α.")

# Extract NO2 and the logarithm of the logistic growth rate (r)
P = clean["Mean_NO2"].values
log_r = np.log(clean["r_estimated"].values)

# Fit a linear model to describe the relationship between log(r) and NO2 concentration
coef = np.polyfit(P, log_r, 1)
# Extract the pollution sensitivity parameter (alpha) and the initial value of r (r0)
alpha = -coef[0]
r0 = np.exp(coef[1])

# Print the fitted global parameters for r0 and alpha
print("-------------------------------------------------")
print("Global Fitted Parameters:")
print(f"  r0     = {r0:.6f}")
print(f"  alpha  = {alpha:.6f}")
print("-------------------------------------------------")


# ------------------------------------------------------
# 4. Save output
# ------------------------------------------------------
# Add the global fitted parameters (r0 and alpha) to the results DataFrame
results_df["r0_global"] = r0
results_df["alpha_global"] = alpha

# Save the results with the fitted parameters to a CSV file
results_df.to_csv("CMT_project/data/fitted_parameters.csv", index=False)
print("saved fitted_parameters.csv")


# Load fitted parameters to get last known NO2 per region
df = pd.read_csv("CMT_project/data/fitted_parameters.csv")

# Get a list of all regions
regions = df["Region"].unique()

# Determine last measured year in the dataset
last_year = df["Year"].max()

# Future prediction range 
future_years = range(last_year + 1, 2051)

# Helper function to build scenario rows for future NO2 predictions
def make_rows(modifier):
    rows = []
    # Loop through each region
    for region in regions:
        # Get the last observed NO2 value for the region
        last_no2 = df[df["Region"] == region]["Mean_NO2"].iloc[-1]

        # Generate future predictions based on the modifier (constant, decrease 1% per year, or increase 1% per year)
        for i, year in enumerate(future_years):
            if modifier == "constant":
                no2 = last_no2
            elif modifier == "minus1":
                no2 = last_no2 * ((0.99) ** (i + 1))
            elif modifier == "plus1":
                no2 = last_no2 * ((1.01) ** (i + 1))

            # Store the predicted NO2 values for each region and year
            rows.append({
                "Region": region,
                "Year": year,
                "NO2": no2
            })
    return rows

# Scenario A: NO2 stays constant
rows_const = make_rows("constant")
pd.DataFrame(rows_const).to_csv("CMT_project/data/future_NO2_constant.csv", index=False)

# Scenario B: NO2 decreases 1% per year
rows_minus = make_rows("minus1")
pd.DataFrame(rows_minus).to_csv("CMT_project/data/future_NO2_minus1percent.csv", index=False)

# Scenario C: NO2 increases 1% per year
rows_plus = make_rows("plus1")
pd.DataFrame(rows_plus).to_csv("CMT_project/data/future_NO2_plus1percent.csv", index=False)

# Print the generated CSV files for the different scenarios
print("Generated:")
print("  - future_NO2_constant.csv")
print("  - future_NO2_minus1percent.csv")
print("  - future_NO2_plus1percent.csv")


# ------------------------------------------------------
# --- 1. Load the fitted parameters ---
# ------------------------------------------------------
df_fitted = pd.read_csv("CMT_project/data/fitted_parameters.csv")

# List of columns that need to be removed from the fitted parameters (these will be merged later)
cols_to_drop = ["Mean_NO2", "Mean_NDVI", "Year"]

# Loop through and remove the columns if they exist in the DataFrame
for col in cols_to_drop:
    if col in df_fitted.columns:
        df_fitted = df_fitted.drop(columns=[col])
        print(f"Removed from fitted_parameters: {col}")


# --- 2. Load the NO2 scenario data ---
# ------------------------------------------------------
# Load the future NO2 scenarios generated earlier
scenario_const = pd.read_csv("CMT_project/data/future_NO2_constant.csv")
scenario_minus = pd.read_csv("CMT_project/data/future_NO2_minus1percent.csv")
scenario_plus  = pd.read_csv("CMT_project/data/future_NO2_plus1percent.csv")


# --- 3. Function to merge and clean the scenarios ---
# ------------------------------------------------------
def merge_and_clean(scenario_df, fitted_df, scenario_name):
    print(f"\n--- Processing scenario: {scenario_name} ---")

    # 3A. Remove columns from the fitted parameters that already exist in the scenario dataset (to avoid duplicates)
    overlapping = [c for c in fitted_df.columns if c in scenario_df.columns and c != "Region"]

    if overlapping:
        print(f"Columns removed to avoid duplication: {overlapping}")
        fitted_clean = fitted_df.drop(columns=overlapping)
    else:
        fitted_clean = fitted_df.copy()

    # 3B. Merge the fitted parameters with the scenario dataset on the "Region" column
    merged = scenario_df.merge(fitted_clean, on="Region", how="left")

    # 3C. Remove duplicates based on Region and Year to ensure no repeated rows
    before = len(merged)
    merged = merged.drop_duplicates(subset=["Region", "Year"], keep="first")
    after = len(merged)

    # Print how many duplicates were removed
    print(f"Duplicates removed: {before - after}")

    return merged



# --- 4. Apply the merge_and_clean function to each of the 3 scenarios ---
clean_const = merge_and_clean(scenario_const, df_fitted, "NO2 constant")
clean_minus = merge_and_clean(scenario_minus, df_fitted, "NO2 -1%/an")
clean_plus  = merge_and_clean(scenario_plus, df_fitted, "NO2 +1%/an")


# -# --- 5. Save the cleaned and merged scenarios to new CSV files ---
# Save the final cleaned datasets to CSV files for each scenario
clean_const.to_csv("CMT_project/data/scenario_with_params_constant_clean.csv", index=False)
clean_minus.to_csv("CMT_project/data/scenario_with_params_minus1percent_clean.csv", index=False)
clean_plus.to_csv("CMT_project/data/scenario_with_params_plus1percent_clean.csv", index=False)

print("\n Files generated :")
print(" scenario_with_params_constant_clean.csv created")
print(" scenario_with_params_minus1percent_clean.csv created")
print(" scenario_with_params_plus1percent_clean.csv created")

# -------------------------------
# Running the C program via subprocess 

# Compile the C program
subprocess.run(["gcc", "CMT_project/Script/growth_sim_30years.c", "-Wall", "-lm", "-o", "CMT_project/Bin/growth_sim_30years"], check=True)

# Run the compiled C program
subprocess.run(["./CMT_project/Bin/growth_sim_30years"], check=True)


# -------------------------------
# DATA VISUALIZATION

# choose region to plot
# -------------------------------
region_of_interest = "Ouest lausannois"   

# -------------------------------
# Load the three scenario outputs
# -------------------------------
df_constant = pd.read_csv("CMT_project/Results/NDVI_scenario_constant.csv")
df_minus = pd.read_csv("CMT_project/Results/NDVI_scenario_minus1percent.csv")
df_plus  = pd.read_csv("CMT_project/Results/NDVI_scenario_plus1percent.csv")

# -------------------------------
# Filter for the chosen region
# -------------------------------
constant_reg = df_constant[df_constant["Region"] == region_of_interest]
minus_reg = df_minus[df_minus["Region"] == region_of_interest]
plus_reg  = df_plus[df_plus["Region"] == region_of_interest]

if constant_reg.empty or minus_reg.empty or plus_reg.empty:
    print(f"Region '{region_of_interest}' not found in scenario files.")
    exit()

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(10, 6))

# Plot all three scenarios
plt.plot(
    constant_reg["Year"], constant_reg["B_predicted"],
    label="NO₂ Constant", linewidth=2
)

plt.plot(
    minus_reg["Year"], minus_reg["B_predicted"],
    label="NO₂ -1% per year", linewidth=2
)

plt.plot(
    plus_reg["Year"], plus_reg["B_predicted"],
    label="NO₂ +1% per year", linewidth=2
)

plt.xlabel("Year", fontsize=14)
plt.ylabel("Predicted NDVI", fontsize=14)
plt.title(f"Future NDVI Predictions for {region_of_interest}", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig("Results/Ouest_Lausanne_predictions.png")
plt.show()

print(" Plot 'Future NDVI Predictions for Ouest Lausanne' created ")
# ----------------------------------------------------
# Sensitivity visualization
# ----------------------------------------------------
# Load dataset
# ----------------------------------------------------
df = pd.read_csv("CMT_project/data/fitted_parameters.csv")

# Filter valid rows (positive r values only)
clean = df[df["r_estimated"] > 0].copy()

# Compute log(r)
clean["log_r"] = np.log(clean["r_estimated"])

# Extract NO2 and log_r
P = clean["Mean_NO2"].values
log_r = clean["log_r"].values

# -------------import numpy as np---------------------------------------
# Fit linear model log(r) = log(r0) - alpha * P
# ----------------------------------------------------
coef = np.polyfit(P, log_r, 1)
slope, intercept = coef

alpha = -slope
r0 = np.exp(intercept)

print("Fitted model:")
print(f"  log(r) = log(r0) - α * P")
print(f"  α  = {alpha:.6f}")
print(f"  r0 = {r0:.6f}")

# Regression line values
P_line = np.linspace(min(P), max(P), 200)
log_r_line = slope * P_line + intercept

# ----------------------------------------------------
# Plot
# ----------------------------------------------------
plt.figure(figsize=(10, 6))

# Scatter of data points
plt.scatter(P, log_r, alpha=0.6, label="Observed log(r)", color="blue")

# Regression line
plt.plot(P_line, log_r_line, label=f"Fitted Line  (α={alpha:.3f})", linewidth=2, color="red")

plt.xlabel("Mean NO₂", fontsize=14)
plt.ylabel("log(r)", fontsize=14)
plt.title("NDVI Sensitivity to NO₂\nlog(r) vs NO₂", fontsize=16)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig("Results/NDVI_sensitivity_to_NO2.png")
plt.show()
print(" Plot 'NDVI Sensitivity to NO₂\nlog(r) vs NO₂' created ")
