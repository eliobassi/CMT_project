import pandas as pd

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
