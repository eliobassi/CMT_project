import warnings
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

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
        "r0": ri,
        "B0": B0_est,
        "K": K_est
    })

results_df = pd.DataFrame(rows)

# ------------------------------------------------------
# 5. Save output
# ------------------------------------------------------
results_df.to_csv("fitted_parameters.csv", index=False)
print("saved fitted_parameters.csv")
print(results_df.head())
