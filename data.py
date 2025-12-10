import os
import glob
import re
import numpy as np
import pandas as pd
import rasterio

# -------------------------
# SETTINGS
# -------------------------
data_folder = "data"
pattern = os.path.join(data_folder, "*.tif")

year_regex = re.compile(r".*_(\d{4})\.tif$")
pollutant_regex = re.compile(r"([^/\\]+)_\d{4}\.tif$")  # capture pollutant before _YEAR.tif

# -------------------------
# UTILS
# -------------------------
def extract_mean_country(raster_path):
    """Compute mean over the entire raster (ignoring nodata)."""
    with rasterio.open(raster_path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return np.nanmean(arr)

def read_csv_tolerant(path):
    """Read CSV trying automatic sep detection, return DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    try:
        # try python engine auto-sep
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # fallback to comma then semicolon
        try:
            df = pd.read_csv(path, sep=",")
        except Exception:
            df = pd.read_csv(path, sep=";")
    # normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_year_col(df):
    for c in df.columns:
        if c.lower() in ("year", "annee", "yr"):
            return c
    # fallback: first integer-like column
    for c in df.columns:
        try:
            pd.to_numeric(df[c].dropna().iloc[:5], errors='raise')
            return c
        except Exception:
            continue
    raise ValueError("No Year column found")

def find_value_col(df, key):
    """Find a column name containing key (case-insensitive)."""
    for c in df.columns:
        if key.lower() in c.lower() and c.lower() != "year":
            return c
    # if not found, try to find any numeric column aside from Year
    for c in df.columns:
        if c != "Year":
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
    raise ValueError(f"No value column found for {key} in file")

# -------------------------
# LOAD CH4 and CO2 from data/
# -------------------------
ch4_path = os.path.join(data_folder, "CH4_concentration.csv")
co2_path = os.path.join(data_folder, "CO2_concentration.csv")

ch4_df = None
co2_df = None

if os.path.exists(ch4_path):
    try:
        tmp = read_csv_tolerant(ch4_path)
        year_col = find_year_col(tmp)
        val_col = find_value_col(tmp, "CH4")
        ch4_df = tmp[[year_col, val_col]].rename(columns={year_col: "Year", val_col: "CH4"})
        ch4_df["Year"] = ch4_df["Year"].astype(int)
        print(f"Loaded CH4 from {ch4_path} with columns Year and {val_col}")
    except Exception as e:
        print(f"Error loading CH4 file: {e}")

else:
    print(f"CH4 file not found at {ch4_path} — CH4 will be omitted")

if os.path.exists(co2_path):
    try:
        tmp = read_csv_tolerant(co2_path)
        year_col = find_year_col(tmp)
        val_col = find_value_col(tmp, "CO2")
        co2_df = tmp[[year_col, val_col]].rename(columns={year_col: "Year", val_col: "CO2"})
        co2_df["Year"] = co2_df["Year"].astype(int)
        print(f"Loaded CO2 from {co2_path} with columns Year and {val_col}")
    except Exception as e:
        print(f"Error loading CO2 file: {e}")
else:
    print(f"CO2 file not found at {co2_path} — CO2 will be omitted")

# -------------------------
# PROCESS ALL TIF FILES IN data/
# -------------------------
tif_files = sorted(glob.glob(pattern))
if len(tif_files) == 0:
    raise SystemExit("No .tif files found in data/")

rows = []
for tif in tif_files:
    m_year = year_regex.match(os.path.basename(tif))
    m_poll = pollutant_regex.match(os.path.basename(tif))
    if not m_year or not m_poll:
        print(f"Skipping (bad name): {tif}")
        continue
    year = int(m_year.group(1))
    pollutant_full = m_poll.group(1)  # e.g., NO2 or O3 or PM10
    # clean pollutant name (remove potential suffixes)
    pollutant = pollutant_full.strip()
    print(f"Processing {pollutant} - {year}")
    mean_val = extract_mean_country(tif)
    rows.append({"Year": year, "Pollutant": pollutant, "Mean_Value": mean_val})

df_long = pd.DataFrame(rows)

# -------------------------
# PIVOT to wide
# -------------------------
df_wide = df_long.pivot_table(index="Year", columns="Pollutant", values="Mean_Value").reset_index()

# -------------------------
# MERGE CH4 and CO2 directly by Year (no interpolation),
# BUT if CH4/CO2 are in data/ and years match, they will fill the columns
# If CH4/CO2 contain years that exactly match df_wide['Year'], merge works.
# -------------------------
if ch4_df is not None:
    df_wide = df_wide.merge(ch4_df, on="Year", how="left")
else:
    df_wide["CH4"] = np.nan

if co2_df is not None:
    df_wide = df_wide.merge(co2_df, on="Year", how="left")
else:
    df_wide["CO2"] = np.nan

# reorder columns (Year first)
cols = ["Year"] + [c for c in df_wide.columns if c != "Year"]
df_wide = df_wide[cols]

# -------------------------
# SAVE final wide CSV
# -------------------------
out_path = "Switzerland_pollution_timeseries_COMPLETE.csv"
df_wide.to_csv(out_path, index=False)
print(f"\nSaved {out_path}")
