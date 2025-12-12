
import pandas as pd
import numpy as np

# Charger le fichier historique
df = pd.read_csv('fitted_parameters.csv')
df = df.sort_values('Year')
last_row = df.iloc[-1]

# Constantes futures
r0_const = float(last_row['r0'])
K_const = float(last_row['K'])
B0_const = float(last_row['B0'])
P_base = float(last_row['P'])

# Plage des années
years = np.arange(2019, 2051)

# Scénarios pour P
P_const = np.full_like(years, P_base, dtype=float)
P_minus1 = P_base * (0.99 ** (years - int(last_row['Year'])))
P_plus1 = P_base * (1.01 ** (years - int(last_row['Year'])))

# Fonction pour créer le DataFrame
def build_df(P_series):
    return pd.DataFrame({
        'Year': years,
        'P': P_series,
        'r0': r0_const,
        'K': K_const,
        'B0': B0_const
    })

# Créer et sauvegarder les fichiers
build_df(P_const).to_csv('scenario_P_constant.csv', index=False)
build_df(P_minus1).to_csv('scenario_P_down.csv', index=False)
build_df(P_plus1).to_csv('scenario_P_up.csv', index=False)
