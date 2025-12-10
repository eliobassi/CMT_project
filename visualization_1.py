import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------------------------------
# DATA VISUALIZATION

# USER: choose region to plot
# -------------------------------
region_of_interest = "Ouest lausannois"   

# -------------------------------
# Load the three scenario outputs
# -------------------------------
df_constant = pd.read_csv("NDVI_scenario_constant.csv")
df_minus = pd.read_csv("NDVI_scenario_minus1percent.csv")
df_plus  = pd.read_csv("NDVI_scenario_plus1percent.csv")

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
plt.show()

print(" Plot 'Future NDVI Predictions for Ouest Lausanne' created ")
# ----------------------------------------------------
# Sensitivity visualization
# ----------------------------------------------------
# Load dataset
# ----------------------------------------------------
df = pd.read_csv("fitted_parameters.csv")

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
plt.show()
print(" Plot 'NDVI Sensitivity to NO₂\nlog(r) vs NO₂' created ")