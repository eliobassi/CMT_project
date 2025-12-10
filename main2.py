import os
import glob
import re
import numpy as np
import pandas as pd
import rasterio
import warnings
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, OptimizeWarning

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


# ---------------------------------------------------------
# ADD NATIONAL NDVI (2010–2018) FROM NDVI_NO2_timeseries.csv
# ---------------------------------------------------------

print("\nAdding national NDVI averages from NDVI_NO2_timeseries.csv ...")

ndvi_path = os.path.join(data_folder, "NDVI_NO2_timeseries.csv")
if not os.path.exists(ndvi_path):
    raise FileNotFoundError("NDVI_NO2_timeseries.csv not found in data/ folder.")

# Load regional NDVI dataset
ndvi_df = pd.read_csv(ndvi_path)

# Keep only years 2010–2018
ndvi_df = ndvi_df[(ndvi_df["Year"] >= 2010) & (ndvi_df["Year"] <= 2018)]

# Compute national NDVI per year
ndvi_national = (
    ndvi_df.groupby("Year")["Mean_NDVI"]
    .mean()
    .reset_index()
    .rename(columns={"Mean_NDVI": "NDVI"})
)

print("National NDVI computed:\n", ndvi_national)

# Merge into df_wide
df_wide = df_wide.merge(ndvi_national, on="Year", how="left")


# ---------------------------------------------------------
# SAVE **ONLY** THE FINAL CSV (NO TEMPORARY FILES)
# ---------------------------------------------------------
final_path = "Switzerland_pollution_timeseries_COMPLETE.csv"
df_wide.to_csv(final_path, index=False)

print(f"\n✔ FINAL FILE CREATED: {final_path}")


# ---------------------------------------------------------
# REMOVE YEAR 2012
# ---------------------------------------------------------

df_wide = df_wide[df_wide["Year"] != 2012]

# ---------------------------------------------------------
# SAVE FINAL CSV
# ---------------------------------------------------------
final_path = "Switzerland_pollution_timeseries_COMPLETE.csv"
df_wide.to_csv(final_path, index=False)

print(f"\n✔ FINAL FILE CREATED (without 2012): {final_path}")


# Charger ton fichier CSV
df = pd.read_csv("Switzerland_pollution_timeseries_COMPLETE.csv")

# Vérifier les noms de colonnes
print("Colonnes disponibles :", df.columns.tolist())

# Corriger si besoin (par exemple S02 -> SO2)
df.rename(columns={"S02": "SO2"}, inplace=True)

# Définir X et y
polluants = ["O3","NO2","PM10","CO2","CH4","SO2"]
X = df[polluants]
y = df["NDVI"]

# Créer et entraîner le modèle global
model = LinearRegression()
model.fit(X, y)

# Extraire les coefficients
poids = pd.Series(model.coef_, index=X.columns)

# Sauvegarder proprement dans un CSV AVEC nom de colonne d'index
poids.to_csv("poids_globaux.csv", header=["Poids"], index_label="Pollutants")

print("Poids globaux calculés et sauvegardés dans 'poids_globaux.csv'")
print(poids)

# Charger les données de pollution
df = pd.read_csv("Switzerland_pollution_timeseries_COMPLETE.csv")

# Charger les poids globaux
poids = pd.read_csv("poids_globaux.csv", index_col=0)["Poids"]

# Calculer la pollution globale pour chaque ligne
df["PollutionGlobale"] = (
    df["O3"] * poids["O3"] +
    df["NO2"] * poids["NO2"] +
    df["PM10"] * poids["PM10"] +
    df["CO2"] * poids["CO2"] +
    df["CH4"] * poids["CH4"] +
    df["SO2"] * poids["SO2"]
)

# Agréger par année (somme ou moyenne selon ton besoin)
polution_each_Year = df.groupby("Year")["PollutionGlobale"].mean().reset_index()

# Sauvegarder dans un CSV
polution_each_Year.to_csv("pollution_each_year.csv", index=False)

print("Fichier 'pollution_each_year.csv' créé avec succès !")
print(polution_each_Year.head())

# -----------------------------
# Charger les deux fichiers
# -----------------------------
poll_path = "pollution_each_year.csv"
ndvi_path = "data/NDVI_NO2_timeseries.csv"

df_poll = pd.read_csv(poll_path)
df_ndvi = pd.read_csv(ndvi_path)

# -----------------------------
# Calcul du NDVI national par année
# -----------------------------
ndvi_national = (
    df_ndvi.groupby("Year")["Mean_NDVI"]
    .mean()
    .reset_index()
    .rename(columns={"Mean_NDVI": "NDVI"})
)

print("NDVI national calculé :")
print(ndvi_national)

# -----------------------------
# Fusion avec le CSV pollution
# -----------------------------
df_final = df_poll.merge(ndvi_national, on="Year", how="left")

# -----------------------------
# Sauvegarde finale
# -----------------------------
out_path = "pollution_each_year_WITH_NDVI.csv"
df_final.to_csv(out_path, index=False)

print(f"\n✔ Nouveau fichier créé : {out_path}")

warnings.filterwarnings("ignore", category=OptimizeWarning)

# ------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------
df = pd.read_csv("pollution_each_year_WITH_NDVI.csv")  # doit contenir Year, NDVI, PollutionGlobale
df = df.dropna(subset=["NDVI", "PollutionGlobale"])

# ------------------------------------------------------
# Logistic model
# ------------------------------------------------------
def logistic(t, r, K, B0):
    return K / (1 + (K/B0 - 1)*np.exp(-r*t))

# ------------------------------------------------------
# 2. Fit logistic model globally (toutes années)
# ------------------------------------------------------
years = df["Year"].values
t = years - years.min()   # temps relatif
B = df["NDVI"].values
P = df["PollutionGlobale"].values

# Initial guesses
B0_guess = B[0]
K_guess = max(B) + 0.1
r_guess = 0.1

try:
    popt, _ = curve_fit(
        logistic, t, B,
        p0=[r_guess, K_guess, B0_guess],
        bounds=([0.0001, 0.1, 0.0], [2.0, 2.0, 2.0]),
        maxfev=8000
    )
except Exception as e:
    raise RuntimeError(f"Logistic fit failed: {e}")

r_est, K_est, B0_est = popt

# ------------------------------------------------------
# 3. Calculer r(P) pour chaque année
# ------------------------------------------------------
# Ici r_est est le paramètre global ajusté.
# On définit r(P) = r_est * exp(P)
r_values = r_est * np.exp(P)

# ------------------------------------------------------
# 4. Construire le tableau final
# ------------------------------------------------------
rows = []
for yr, Pi, Bi, ri in zip(years, P, B, r_values):
    rows.append({
        "Year": yr,
        "P": Pi,
        "NDVI": Bi,
        "r_estimated": ri,
        "B0_estimated": B0_est,
        "K_estimated": K_est
    })

results_df = pd.DataFrame(rows)

# ------------------------------------------------------
# 5. Save output
# ------------------------------------------------------
results_df.to_csv("fitted_parameters.csv", index=False)
print("saved fitted_parameters.csv")
print(results_df.head())


