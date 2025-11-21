import pandas as pd
import numpy as np

# --- 1. Charger les paramètres fités ---
df_fitted = pd.read_csv("fitted_parameters.csv")

# Colonnes qui doivent disparaître des paramètres
cols_to_drop = ["Mean_NO2", "Mean_NDVI", "Year"]

for col in cols_to_drop:
    if col in df_fitted.columns:
        df_fitted = df_fitted.drop(columns=[col])
        print(f"Supprimé du fitted_parameters : {col}")

# --- 2. Charger les scénarios NO2 ---
scenario_const = pd.read_csv("future_NO2_constant.csv")
scenario_minus = pd.read_csv("future_NO2_minus1percent.csv")
scenario_plus  = pd.read_csv("future_NO2_plus1percent.csv")


# --- 3. Fonction pour merger proprement SANS créer de doublons ---
def merge_and_clean(scenario_df, fitted_df, scenario_name):

    print(f"\n--- Traitement du scénario : {scenario_name} ---")

    # 3A. Retirer du fitted_parameters les colonnes déjà présentes dans le scénario
    overlapping = [c for c in fitted_df.columns if c in scenario_df.columns and c != "Region"]

    if overlapping:
        print(f"Colonnes supprimées pour éviter doublons : {overlapping}")
        fitted_clean = fitted_df.drop(columns=overlapping)
    else:
        fitted_clean = fitted_df.copy()

    # 3B. Merge propre
    merged = scenario_df.merge(fitted_clean, on="Region", how="left")

    # 3C. Supprimer les doublons (le vrai problème !)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["Region", "Year"], keep="first")
    after = len(merged)

    print(f"Doublons supprimés : {before - after}")

    return merged


# --- 4. Appliquer la fonction aux 3 scénarios ---
clean_const = merge_and_clean(scenario_const, df_fitted, "NO2 constant")
clean_minus = merge_and_clean(scenario_minus, df_fitted, "NO2 -1%/an")
clean_plus  = merge_and_clean(scenario_plus, df_fitted, "NO2 +1%/an")


# --- 5. Sauvegarder les CSV finaux propres ---
clean_const.to_csv("scenario_with_params_constant_clean.csv", index=False)
clean_minus.to_csv("scenario_with_params_minus1percent_clean.csv", index=False)
clean_plus.to_csv("scenario_with_params_plus1percent_clean.csv", index=False)

print("\n🎉 FICHIERS FINAUX GÉNÉRÉS :")
print("  ✔ scenario_with_params_constant_clean.csv")
print("  ✔ scenario_with_params_minus1percent_clean.csv")
print("  ✔ scenario_with_params_plus1percent_clean.csv")
