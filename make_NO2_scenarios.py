import pandas as pd
import numpy as np

# Load fitted parameters to get last known NO2 per region
df = pd.read_csv("fitted_parameters.csv")

regions = df["Region"].unique()

# Determine last measured year in the dataset
last_year = df["Year"].max()

# Future prediction range (you can adjust here)
future_years = range(last_year + 1, 2051)

# Helper: build scenario rows
def make_rows(modifier):
    rows = []
    for region in regions:
        # Last observed NO2 for this region
        last_no2 = df[df["Region"] == region]["Mean_NO2"].iloc[-1]

        for i, year in enumerate(future_years):
            if modifier == "constant":
                no2 = last_no2
            
            elif modifier == "minus1":
                no2 = last_no2 * ((0.99) ** (i + 1))
            
            elif modifier == "plus1":
                no2 = last_no2 * ((1.01) ** (i + 1))

            rows.append({
                "Region": region,
                "Year": year,
                "NO2": no2
            })
    return rows

# Scenario A: NO2 stays constant
rows_const = make_rows("constant")
pd.DataFrame(rows_const).to_csv("future_NO2_constant.csv", index=False)

# Scenario B: NO2 decreases 1% per year
rows_minus = make_rows("minus1")
pd.DataFrame(rows_minus).to_csv("future_NO2_minus1percent.csv", index=False)

# Scenario C: NO2 increases 1% per year
rows_plus = make_rows("plus1")
pd.DataFrame(rows_plus).to_csv("future_NO2_plus1percent.csv", index=False)

print("Generated:")
print("  - future_NO2_constant.csv")
print("  - future_NO2_minus1percent.csv")
print("  - future_NO2_plus1percent.csv")
