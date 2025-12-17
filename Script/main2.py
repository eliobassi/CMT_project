import os
import glob
import re
import numpy as np
import pandas as pd
import rasterio
import warnings
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, OptimizeWarning
import subprocess
import matplotlib.pyplot as plt
from main import treatment


treatment()
# -------------------------
# SETTINGS
# -------------------------
data_folder = "data"  # Define the folder where your input data (e.g., .tif and .csv files) is stored
pattern = os.path.join(data_folder, "*.tif")  # Search pattern for finding all .tif files in the 'data' folder

# Regex to extract the year from the filename (e.g., "pollutant_2010.tif" -> "2010")
year_regex = re.compile(r".*_(\d{4})\.tif$")
# Regex to extract the pollutant name from the filename (e.g., "NO2_2010.tif" -> "NO2")
pollutant_regex = re.compile(r"([^/\\]+)_\d{4}\.tif$")  # Captures the pollutant name before _YEAR.tif

# -------------------------pattern
# UTILS
# -------------------------
def extract_mean_country(raster_path):
    """Compute the mean value over the entire raster, ignoring NoData values."""
    with rasterio.open(raster_path) as src:
        arr = src.read(1).astype(float)  # Read the first band as a float array
        nodata = src.nodata  # Get the NoData value for the raster
        if nodata is not None:
            arr[arr == nodata] = np.nan  # Replace NoData values with NaN
        return np.nanmean(arr)  # Calculate the mean, ignoring NaNs

def read_csv_tolerant(path):
    """Attempt to read a CSV with automatic separator detection, return a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    try:
        # Try auto separator detection with the Python engine
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # If that fails, try with common separators (comma, semicolon)
        try:
            df = pd.read_csv(path, sep=",")
        except Exception:
            df = pd.read_csv(path, sep=";")
    # Clean up column names (strip leading/trailing spaces)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_year_col(df):
    """Look for the column containing the year (case-insensitive search for 'year', 'annee', 'yr')."""
    for c in df.columns:
        if c.lower() in ("year", "annee", "yr"):
            return c
    # Fallback: search for the first column that contains numeric data (likely the year)
    for c in df.columns:
        try:
            pd.to_numeric(df[c].dropna().iloc[:5], errors='raise')
            return c
        except Exception:
            continue
    raise ValueError("No Year column found")  # Raise an error if no year column is found

def find_value_col(df, key):
    """Search for a column containing a given key (case-insensitive)."""
    for c in df.columns:
        if key.lower() in c.lower() and c.lower() != "year":
            return c
    # Fallback: find the first numeric column that's not 'Year'
    for c in df.columns:
        if c != "Year" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError(f"No value column found for {key} in file")  # Raise an error if no value column is found

# -------------------------
# LOAD CH4 and CO2 from data/
# -------------------------
ch4_path = os.path.join(data_folder, "CH4_concentration.csv")  # Path to the CH4 concentration file
co2_path = os.path.join(data_folder, "CO2_concentration.csv")  # Path to the CO2 concentration file

ch4_df = None  # Initialize empty DataFrame for CH4
co2_df = None  # Initialize empty DataFrame for CO2

# Load CH4 data if file exists
if os.path.exists(ch4_path):
    try:
        tmp = read_csv_tolerant(ch4_path)  # Read the CH4 CSV file
        year_col = find_year_col(tmp)  # Find the 'Year' column
        val_col = find_value_col(tmp, "CH4")  # Find the column for 'CH4' concentration
        ch4_df = tmp[[year_col, val_col]].rename(columns={year_col: "Year", val_col: "CH4"})  # Rename columns to standard names
        ch4_df["Year"] = ch4_df["Year"].astype(int)  # Ensure the 'Year' column is of type int
        print(f"Loaded CH4 from {ch4_path} with columns Year and {val_col}")
    except Exception as e:
        print(f"Error loading CH4 file: {e}")
else:
    print(f"CH4 file not found at {ch4_path} — CH4 will be omitted")

# Load CO2 data if file exists
if os.path.exists(co2_path):
    try:
        tmp = read_csv_tolerant(co2_path)  # Read the CO2 CSV file
        year_col = find_year_col(tmp)  # Find the 'Year' column
        val_col = find_value_col(tmp, "CO2")  # Find the column for 'CO2' concentration
        co2_df = tmp[[year_col, val_col]].rename(columns={year_col: "Year", val_col: "CO2"})  # Rename columns to standard names
        co2_df["Year"] = co2_df["Year"].astype(int)  # Ensure the 'Year' column is of type int
        print(f"Loaded CO2 from {co2_path} with columns Year and {val_col}")
    except Exception as e:
        print(f"Error loading CO2 file: {e}")
else:
    print(f"CO2 file not found at {co2_path} — CO2 will be omitted")

# -------------------------
# PROCESS ALL TIF FILES IN data/
# -------------------------
tif_files = sorted(
    f for f in glob.glob(pattern)  # Get all .tif files in the 'data' folder except the NDVI ones 
    if not os.path.basename(f).startswith("NDVI_")   
) 
if len(tif_files) == 0:
    raise SystemExit("No .tif files found in data/")  # Exit if no .tif files are found

rows = []  # List to store data for final DataFrame
for tif in tif_files:
    m_year = year_regex.match(os.path.basename(tif))  # Extract the year from the filename
    m_poll = pollutant_regex.match(os.path.basename(tif))  # Extract the pollutant from the filename
    if not m_year or not m_poll:
        print(f"Skipping (bad name): {tif}")  # Skip files that don't match the expected pattern
        continue
    year = int(m_year.group(1))  # Convert the extracted year to an integer
    pollutant_full = m_poll.group(1)  # Extract the full pollutant name (e.g., "NO2")
    pollutant = pollutant_full.strip()  # Clean up the pollutant name
    print(f"Processing {pollutant} - {year}")
    mean_val = extract_mean_country(tif)  # Calculate the mean value for the raster
    rows.append({"Year": year, "Pollutant": pollutant, "Mean_Value": mean_val})  # Add the result to the rows list

df_long = pd.DataFrame(rows)  # Convert the rows list to a DataFrame

# -------------------------
# PIVOT to wide format
# -------------------------
df_wide = df_long.pivot_table(index="Year", columns="Pollutant", values="Mean_Value").reset_index()  # Pivot the DataFrame to wide format

# -------------------------
# MERGE CH4 and CO2 by Year (no interpolation)
# -------------------------
if ch4_df is not None:
    df_wide = df_wide.merge(ch4_df, on="Year", how="left")  # Merge CH4 data with the wide DataFrame
else:
    df_wide["CH4"] = np.nan  # If no CH4 data, set the column to NaN

if co2_df is not None:
    df_wide = df_wide.merge(co2_df, on="Year", how="left")  # Merge CO2 data with the wide DataFrame
else:
    df_wide["CO2"] = np.nan  # If no CO2 data, set the column to NaN

# Reorder columns to make 'Year' the first column
cols = ["Year"] + [c for c in df_wide.columns if c != "Year"]
df_wide = df_wide[cols]

# -------------------------
# SAVE final wide CSV
# -------------------------
out_path = "data/Switzerland_pollution_timeseries_COMPLETE.csv"
df_wide.to_csv(out_path, index=False)  # Save the DataFrame to a CSV file
print(f"\nSaved {out_path}")

# ---------------------------------------------------------
# ADD NATIONAL NDVI (2010–2018) FROM NDVI_NO2_timeseries.csv
# ---------------------------------------------------------
print("\nAdding national NDVI averages from NDVI_NO2_timeseries.csv ...")

# Define the path to the NDVI dataset
ndvi_path = os.path.join(data_folder, "NDVI_NO2_timeseries.csv")  
if not os.path.exists(ndvi_path):
    raise FileNotFoundError("NDVI_NO2_timeseries.csv not found in data/ folder.")

# Load the regional NDVI dataset
ndvi_df = pd.read_csv(ndvi_path)

# Filter the data to keep only years from 2010 to 2018
ndvi_df = ndvi_df[(ndvi_df["Year"] >= 2010) & (ndvi_df["Year"] <= 2018)]

# Compute national NDVI per year (mean NDVI per year)
ndvi_national = (
    ndvi_df.groupby("Year")["Mean_NDVI"]  # Group by 'Year' and calculate the mean for 'Mean_NDVI'
    .mean()  # Compute the mean of 'Mean_NDVI' per year
    .reset_index()  # Reset the index to make 'Year' a column again
    .rename(columns={"Mean_NDVI": "NDVI"})  # Rename 'Mean_NDVI' to 'NDVI'
)

# Print the computed national NDVI to verify
print("National NDVI computed:\n", ndvi_national)

# Merge national NDVI data into the main DataFrame (df_wide) by year
df_wide = df_wide.merge(ndvi_national, on="Year", how="left")

# ---------------------------------------------------------
# SAVE **ONLY** THE FINAL CSV (NO TEMPORARY FILES)
# ---------------------------------------------------------
final_path = "data/Switzerland_pollution_timeseries_COMPLETE.csv"  # Path to save the final DataFrame
df_wide.to_csv(final_path, index=False)  # Save the DataFrame to a CSV file without the index column

# Print confirmation that the final file has been created
print(f"\n✔ FINAL FILE CREATED: {final_path}")

# ---------------------------------------------------------
# REMOVE YEAR 2012
# ---------------------------------------------------------
# Remove the data for the year 2012 from the DataFrame
df_wide = df_wide[df_wide["Year"] != 2012]

# ---------------------------------------------------------
# SAVE FINAL CSV
# ---------------------------------------------------------
# Save the DataFrame after removing the year 2012
final_path = "data/Switzerland_pollution_timeseries_COMPLETE.csv"
df_wide.to_csv(final_path, index=False)  # Save the DataFrame without the index column

# Print confirmation that the final file (without 2012) has been created
print(f"\n✔ FINAL FILE CREATED (without 2012): {final_path}")


# ---------------------------------------------------------
# LOAD THE FINAL CSV AND PROCESS THE POLLUTION DATA
# ---------------------------------------------------------
# Load the final pollution dataset into a DataFrame
df = pd.read_csv("data/Switzerland_pollution_timeseries_COMPLETE.csv")

# Print available column names to verify correct loading
print("Available columns:", df.columns.tolist())

# Rename columns if necessary (e.g., "S02" to "SO2" for consistency)
df.rename(columns={"S02": "SO2"}, inplace=True)

# Define the list of pollutants to use in the model
pollutants = ["O3", "NO2", "PM10", "CO2", "CH4", "SO2"]

# Define X (features) and y (target) for the regression model
X = df[pollutants]  # Pollutants will be the feature columns
y = df["NDVI"]  # NDVI is the target variable

# ---------------------------------------------------------
# GLOBAL LINEAR REGRESSION MODEL
# ---------------------------------------------------------
# Create and train a linear regression model using the pollutants as features to predict NDVI
model = LinearRegression()
model.fit(X, y)

# Extract the coefficients from the model (i.e., the weight for each pollutant)
weights = pd.Series(model.coef_, index=X.columns)

# Save the weights in a CSV file
weights.to_csv("data/poids_globaux.csv", header=["Poids"], index_label="Pollutants")

# Print out the computed global weights for each pollutant
print("Global weights calculated and saved in 'poids_globaux.csv'")
print(weights)

# ---------------------------------------------------------
# CALCULATE THE GLOBAL POLLUTION FOR EACH YEAR
# ---------------------------------------------------------
# Load the updated pollution data
df = pd.read_csv("data/Switzerland_pollution_timeseries_COMPLETE.csv")

# Load the global weights from the CSV file
weights = pd.read_csv("data/poids_globaux.csv", index_col=0)["Poids"]

# Calculate the global pollution index for each row in the DataFrame using the linear model
df["PollutionGlobale"] = (
    df["O3"] * weights["O3"] +
    df["NO2"] * weights["NO2"] +
    df["PM10"] * weights["PM10"] +
    df["CO2"] * weights["CO2"] +
    df["CH4"] * weights["CH4"] +
    df["SO2"] * weights["SO2"]
)

# Aggregate the pollution data by year (compute the average)
pollution_each_year = df.groupby("Year")["PollutionGlobale"].mean().reset_index()

# Save the aggregated pollution data to a CSV file
pollution_each_year.to_csv("data/pollution_each_year.csv", index=False)

# Print confirmation that the pollution data for each year has been created
print("File 'pollution_each_year.csv' created successfully!")
print(pollution_each_year.head())

# ---------------------------------------------------------
# CALCULATE THE NATIONAL NDVI AND MERGE WITH POLLUTION DATA
# ---------------------------------------------------------
# Load the pollution data and the NDVI dataset
poll_path = "data/pollution_each_year.csv"
ndvi_path = "data/NDVI_NO2_timeseries.csv"

df_poll = pd.read_csv(poll_path)  # Pollution data by year
df_ndvi = pd.read_csv(ndvi_path)  # NDVI data

# ---------------------------------------------------------
# CALCULATE THE NATIONAL NDVI PER YEAR
# ---------------------------------------------------------
# Compute the national NDVI per year by calculating the mean of 'Mean_NDVI'
ndvi_national = (
    df_ndvi.groupby("Year")["Mean_NDVI"]  # Group by year and calculate mean NDVI
    .mean()
    .reset_index()  # Reset the index to make 'Year' a column again
    .rename(columns={"Mean_NDVI": "NDVI"})  # Rename column to 'NDVI'
)

# Print the computed national NDVI values
print("National NDVI calculated:")
print(ndvi_national)

# ---------------------------------------------------------
# MERGE POLLUTION DATA WITH NATIONAL NDVI
# ---------------------------------------------------------
# Merge the national NDVI data into the pollution DataFrame
df_final = df_poll.merge(ndvi_national, on="Year", how="left")

# ---------------------------------------------------------
# SAVE FINAL FILE WITH NDVI AND POLLUTION DATA
# ---------------------------------------------------------
# Save the final DataFrame that contains both pollution and national NDVI data
out_path = "data/pollution_each_year_WITH_NDVI.csv"
df_final.to_csv(out_path, index=False)

# Print confirmation that the final merged file has been created
print(f"\n✔ New file created: {out_path}")

# ---------------------------------------------------------
# LOGISTIC GROWTH MODEL FITTING
# ---------------------------------------------------------
# Suppress warnings from the optimization process (e.g., warnings from curve fitting)
warnings.filterwarnings("ignore", category=OptimizeWarning)

# ------------------------------------------------------
# 1. Load dataset with pollution and NDVI data
# ------------------------------------------------------
# Load the dataset that contains 'Year', 'NDVI', and 'PollutionGlobale' columns
df = pd.read_csv("data/pollution_each_year_WITH_NDVI.csv")
df = df.dropna(subset=["NDVI", "PollutionGlobale"])  # Drop rows with missing NDVI or pollution values

# ------------------------------------------------------
# LOGISTIC GROWTH MODEL FUNCTION
# ------------------------------------------------------
def logistic(t, r, K, B0):
    """Logistic growth model function."""
    return K / (1 + (K / B0 - 1) * np.exp(-r * t))  # Logistic function equation

# ------------------------------------------------------
# 2. Fit logistic model globally (using all years)
# ------------------------------------------------------
# Prepare the data: years, pollution (P), and NDVI (B)
years = df["Year"].values
t = years - years.min()  # Relative time (difference from the minimum year)
B = df["NDVI"].values  # NDVI values
P = df["PollutionGlobale"].values  # Pollution values

# Initial guesses for the logistic model parameters
B0_guess = B[0]  # Initial guess for B0 (initial NDVI)
K_guess = max(B) + 0.1  # Guess for the carrying capacity K (max NDVI)
r_guess = 0.1  # Guess for the growth rate r

try:
    # Fit the logistic model to the data using curve fitting
    popt, _ = curve_fit(
        logistic, t, B,
        p0=[r_guess, K_guess, B0_guess],  # Initial parameter guesses
        bounds=([0.0001, 0.1, 0.0], [2.0, 2.0, 2.0]),  # Bounds for the parameters
        maxfev=8000  # Maximum number of function evaluations
    )
except Exception as e:
    # Handle any errors during curve fitting
   raise RuntimeError(f"Logistic fit failed: {e}")

r_est, K_est, B0_est = popt

# ------------------------------------------------------
# 3. CALCULATE r(P) FOR EACH YEAR
# ------------------------------------------------------
# r(P) is defined as r_est multiplied by the exponential of P (Pollution).
r_values = r_est * np.exp(P)

# ------------------------------------------------------
# 4. BUILD THE FINAL RESULTS DATAFRAME
# ------------------------------------------------------
# Create a DataFrame with Year, Pollution, NDVI, r(P), and logistic model parameters.
rows = []
for yr, Pi, Bi, ri in zip(years, P, B, r_values):
    rows.append({
        "Year": yr,    # Year
        "P": Pi,       # Pollution value for the year
        "NDVI": Bi,    # NDVI value for the year
        "r0": ri,      # r(P) value computed for each year
        "B0": B0_est,  # Initial NDVI (B0) from logistic fit
        "K": K_est     # Carrying capacity (K) from logistic fit
    })
results_df = pd.DataFrame(rows)

# ------------------------------------------------------
# 5. SAVE THE OUTPUT DATAFRAME
# ------------------------------------------------------
# Save the DataFrame to CSV with fitted parameters.
results_df.to_csv("data/fitted_parameters.csv", index=False)
print("saved fitted_parameters.csv")
print(results_df.head())


# ------------------------------------------------------
# LOAD THE HISTORICAL FILE AND EXTRACT LAST ROW FOR FUTURE CONSTANTS
# ------------------------------------------------------
df = pd.read_csv('data/fitted_parameters.csv').sort_values('Year')
last_row = df.iloc[-1]  # Get last row for constants (latest year)
r0_const, K_const, B0_const, P_base = map(float, last_row[['r0', 'K', 'B0', 'P']])

# ------------------------------------------------------
# DEFINE THE RANGE OF YEARS FOR FUTURE PROJECTIONS
# ------------------------------------------------------
years = np.arange(2019, 2051)  # Future years from 2019 to 2050

# ------------------------------------------------------
# DEFINE SCENARIOS FOR FUTURE POLLUTION
# ------------------------------------------------------
# Define future pollution scenarios based on last known value of P:
P_const = np.full_like(years, P_base, dtype=float)  # Constant pollution
P_minus1 = P_base * (0.99 ** (years - int(last_row['Year'])))  # Decreasing pollution
P_plus1 = P_base * (1.01 ** (years - int(last_row['Year'])))  # Increasing pollution

# ------------------------------------------------------
# FUNCTION TO CREATE DATAFRAME FOR SCENARIOS
# ------------------------------------------------------
# Function to create a DataFrame with the projected pollution and parameters.
def build_df(P_series):
    return pd.DataFrame({
        'Year': years,
        'P': P_series,
        'r0': r0_const,
        'K': K_const,
        'B0': B0_const
    })

# ------------------------------------------------------
# CREATE AND SAVE THE SCENARIO FILES
# ------------------------------------------------------
# Create and save the scenario DataFrames for each pollution scenario.
build_df(P_const).to_csv('data/scenario_P_constant.csv', index=False)
build_df(P_minus1).to_csv('data/scenario_P_down.csv', index=False)
build_df(P_plus1).to_csv('data/scenario_P_up.csv', index=False)

print("Scenario files created: 'scenario_P_constant.csv', 'scenario_P_down.csv', 'scenario_P_up.csv'")

# -------------------------------
# SUBPROCESS RUNNING C PROGRAM
# -------------------------------
# Running the C program via subprocess 

# Compile the C program
subprocess.run(["gcc", "Script/simulate_ndvi.c", "-Wall", "-lm", "-o", "Bin/simulate_ndvi"], check=True)

# Run the compiled C program
subprocess.run(["./Bin/simulate_ndvi"], check=True)


# PLOT 
# ==================================================
# Load dataset (same folder as this script)
# ==================================================
df = pd.read_csv("Results/ndvi_futur_combined.csv")

# NDVI columns for the three scenarios
ndvi_cols = ["NDVI_up", "NDVI_down", "NDVI_cst"]

# ==================================================
# ----------- PLOT 1 : ADAPTIVE SCALE ---------------
# ==================================================

# Stack all NDVI values to analyze real variability
ndvi_values = df[ndvi_cols].values.flatten()
ndvi_values = ndvi_values[~np.isnan(ndvi_values)]

# Compute year-to-year absolute differences
diffs = []
for col in ndvi_cols:
    diffs.extend(np.abs(np.diff(df[col].values)))

diffs = np.array(diffs)
diffs = diffs[diffs > 0]

# Smallest meaningful NDVI variation
min_diff = diffs.min()

# Define adaptive limits
ndvi_min = ndvi_values.min()
ndvi_max = ndvi_values.max()
margin = 5 * min_diff

ymin = ndvi_min - margin
ymax = ndvi_max + margin

# ---- Plot 1
plt.figure(figsize=(10, 6))

plt.plot(df["Year"], df["NDVI_up"],
         label="Global pollution +1% / year", linewidth=2, color="green")

plt.plot(df["Year"], df["NDVI_down"],
         label="Global pollution -1% / year", linewidth=2, color="orange")

plt.plot(df["Year"], df["NDVI_cst"],
         label="Constant global pollution", linewidth=2, color="blue")

plt.xlabel("Year", fontsize=13)
plt.ylabel("Predicted NDVI", fontsize=13)
plt.title("Projected NDVI for Switzerland (2019–2050)\nAdaptive scale", fontsize=15)

plt.ylim(ymin, ymax)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig("Results/NDVI_predictions_Switzerland_adaptive_scale.png", dpi=300)
plt.show()

print("Adaptive-scale NDVI plot created.")

# ==================================================
# ----------- PLOT 2 : FIXED PRECISE SCALE ----------
# ==================================================

plt.figure(figsize=(10, 6))

plt.plot(
    df["Year"], df["NDVI_up"],
    label="Pollution +1% per year",
    linewidth=2,
    color="green"
)

plt.plot(
    df["Year"], df["NDVI_down"],
    label="Pollution -1% per year",
    linewidth=2,
    color="orange"
)

plt.plot(
    df["Year"], df["NDVI_cst"],
    label="Constant pollution",
    linewidth=2,
    color="blue"
)

plt.xlabel("Year", fontsize=14)
plt.ylabel("Predicted NDVI", fontsize=14)
plt.title("NDVI Predictions for Switzerland (2019–2050)\nFixed high-precision scale",
          fontsize=16)

# Very narrow NDVI range to highlight tiny differences
plt.ylim(0.49000, 0.49500)

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig("Results/NDVI_predictions_Switzerland_precise_scale.png", dpi=300)
plt.show()

print("Fixed-scale high-precision NDVI plot created.")

# ==================================================
# Plot 3 — Global pollution scenarios (P vs Year)
# ==================================================
# This plot shows the evolution of the global pollution
# index P over time for the three future scenarios.

plt.figure(figsize=(10, 6))

plt.plot(
    df["Year"], df["P_up"],
    label="Pollution +1% per year",
    linewidth=2,
    color="green"
)

plt.plot(
    df["Year"], df["P_down"],
    label="Pollution -1% per year",
    linewidth=2,
    color="orange"
)

plt.plot(
    df["Year"], df["P_cst"],
    label="Constant pollution",
    linewidth=2,
    color="blue"
)

plt.xlabel("Year")
plt.ylabel("Global pollution index (P)")
plt.title("Global Pollution Scenarios for Switzerland (2019–2050)")

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig("Results/Global_pollution_scenarios_Switzerland.png", dpi=300)
plt.show()

