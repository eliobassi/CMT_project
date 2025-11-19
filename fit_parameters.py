import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

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



df = pd.read_csv("fitted_parameters.csv")

# Select relevant columns and ensure order
c_input = df[[
    "Region",
    "Year",
    "Mean_NDVI",
    "Mean_NO2",
    "r_estimated",
    "K_estimated",
    "B0_estimated",
    "r0_global",
    "alpha_global"
]]

c_input.to_csv("c_input.csv", index=False)

print("Generated c_input.csv")