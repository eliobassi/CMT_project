

## Project Description
This project investigates the relationship between air pollution and vegetation dynamics in Switzerland using annually mean NDVI (Normalized Difference Vegetation Index) as a proxy for vegetation health.

The core objective is to model and project future NDVI trajectories at the national scale under three global pollution scenarios (increase, decrease, constant).

The project combines :

- Remote sensing data (NDVI and air pollutants),

- Statistical parameter fitting,

- Numerical simulation of vegetation growth,

- Scenario-based projections, and

- Scientific visualization.


### Input files

The project relies on several categories of input data :

### Raster data (GeoTIFF)

Used to extract yearly mean pollution and NDVI values :

- NDVI_YYYY.tif
  These files are to heavy to be on Github, they have to be added manually to the data folder. They are accessible in the 'swissdatacube.org' website (see Data sources for the link).

- NO2_YYYY.tif, O3_YYYY.tif, SO2_YYYY.tif, PM10_YYYY.tif

These files contain spatially distributed yearly averages (2010–2018).

### Vector data

- swissBOUNDARIES3D_1_5_TLM_BEZIRKSGEBIET.shp
  
Administrative boundaries used for spatial aggregation.

### CSV files

- NDVI_NO2_timeseries.csv
  
NDVI and pollution dataframe per region (2010–2018).

- CH4_concentration.csv
- CO2_concentration.csv


### Output files

### CSV outputs

- NDVI_scenario_XXX.csv

Final dataset containg NDVI for each scenario in 3 csv file per region (2019-2050)

- ndvi_futur_combined.csv
  
Final dataset containing projected NDVI and pollution values for Switzerland (2019–2050).

### Figures

NDVI projections under three scenarios (different scale, region/Switzerland, NO2/P_global) :

- NDVI_predictions_Switzerland_adaptive_scale.png

- NDVI_predictions_Switzerland_precise_scale.png

- Ouest_Lausanne_prediction.png

Global pollution scenario evolution :

- Global_pollution_scenarios_Switzerland.png

High-precision NDVI sensitivity to NO2 :

- NDVI_sensitivity_to_NO2.png


### Report

The template for the report is as following :
  1. Deviations from project proposal

  3. Introduction
  4. Approach used
  5. Results
  6. Conclusion
  7. Autorship statement

## Running the program

### Dependencies

The project is designed to run on Linux (SIE VDI compatible).

Required software :

- Python 3.11.9 ('lte')

- C compiler (gcc)

All scripts were tested using a Micromamba/Conda environment.

Required Python packages:

- **pandas**: A library for data manipulation and analysis.
- **scipy**: A library for scientific and technical computing, including optimization.
- **geopandas**: A library that extends Pandas to handle spatial data.
- **rasterio**: A library for reading and writing geospatial raster data.
- **scikit-learn**: A library for machine learning and data mining, including linear regression.

### Build

Th C program is compiled using 'gcc file_name.c -Wall -lm' and run via subprocess in the python code.

### Execute

The program is executed by running the main2.py file wich, by automation runs all the other files.

## Contributors

Bassi Elio, Saissi Emilie, Strazza Maxime

## Acknowledgments

### Data sources

- NDVI datasets from 'swissdatacube.org' : https://doi.org/10.26037/yareta:kpmscrogqbdhvjeuev2ydrzk7y
- Atmospheric pollution datasets from 'map.geo.admin.ch' (NO₂, O₃, SO₂, PM10) 
- CH4 datasets from 'ourworldindata.org' : https://ourworldindata.org/grapher/global-methane-concentrations
- CO2 datsets from 'ourworldindata.org'

### Code

The scientific modeling choices, parameter definitions and standard scientific programming patterns (e.g. logistic growth formulation, linear regression usage, raster averaging) were implemented directly by us.

Some parts of the project benefited from assistance by ChatGPT (OpenAI), including :

Help with :

- Code structuring and refactoring,

- Debugging Python and C interactions,

- Improving numerical stability and plotting clarity,

- CSV merging and cleaning,

- Plot formatting and visualization improvements.

All LLM-assisted code was reviewed, adapted, tested, and validated by us before integration into the project.
