import pandas as pd
from sklearn.linear_model import LinearRegression

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

# Sauvegarder dans un CSV
poids.to_csv("poids_globaux.csv", header=["Poids"])

print("Poids globaux calculés et sauvegardés dans 'poids_globaux.csv'")
print(poids)
