import pandas as pd
import numpy as np

# Charger ton fichier fitted_parameters.csv
df = pd.read_csv("fitted_parameters.csv")

# On prend la dernière année observée comme point de départ
last_year = df["Year"].max()
last_P = df.loc[df["Year"] == last_year, "P"].values[0]
K_est = df["K_estimated"].iloc[0]
B0_est = df["B0_estimated"].iloc[0]
r_est = df["r_estimated"].iloc[0]

# Créer les années futures
years_future = np.arange(last_year+1, 2051)

# Fonction pour générer un scénario
def generate_scenario(P0, rate, years, K, B0, r):
    rows = []
    P = P0
    for yr in years:
        # appliquer le changement de pollution
        P = P * (1 + rate)
        rows.append({
            "Year": yr,
            "P": P,
            "K_estimated": K,
            "B0_estimated": B0,
            "r_estimated": r
        })
    return pd.DataFrame(rows)

# Scénario 1 : pollution augmente de 1% par an
scenario_up = generate_scenario(last_P, 0.01, years_future, K_est, B0_est, r_est)
scenario_up.to_csv("scenario_pollution_up.csv", index=False)

# Scénario 2 : pollution reste constante (variation aléatoire ±0.5%)
np.random.seed(42)
rows = []
P = last_P
for yr in years_future:
    P = P * (1 + np.random.uniform(-0.005, 0.005))  # petite fluctuation
    rows.append({
        "Year": yr,
        "P": P,
        "K_estimated": K_est,
        "B0_estimated": B0_est,
        "r_estimated": r_est
    })
scenario_const = pd.DataFrame(rows)
scenario_const.to_csv("scenario_pollution_constant.csv", index=False)

# Scénario 3 : pollution diminue de 1% par an
scenario_down = generate_scenario(last_P, -0.01, years_future, K_est, B0_est, r_est)
scenario_down.to_csv("scenario_pollution_down.csv", index=False)

print("Trois fichiers créés : scenario_pollution_up.csv, scenario_pollution_constant.csv, scenario_pollution_down.csv")
