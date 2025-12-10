# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load CSV file
df = pd.read_csv("Switzerland_pollution_timeseries_COMPLETE.csv")

# Define predictors (pollutants) and target (NDVI)
X = df[["O3", "NO2", "PM10", "CO2", "CH4", "SO2"]]
y = df["NDVI"]

# Create and train the global model
model = LinearRegression()
model.fit(X, y)

# Extract coefficients
poids = pd.Series(model.coef_, index=X.columns)

# Save to CSV
poids.to_csv("poids_globaux.csv", header=["Poids"])

print("Global weights calculated and saved in 'poids_globaux.csv'")
print(poids)
