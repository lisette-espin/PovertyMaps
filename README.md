# Inferring High-resolution Poverty Maps using multimodal data
Repository with examples on how to infer wealth index in populated places using: satellite images, mobility networks, crowdsource annotations, etc.
````(work-in-progress)````

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-3710/)

## Poverty Maps
Inferring IWI score from the Web.

1. Run all from scratch: [notebooks/PovertyMap_WebMetadata.ipynb](https://github.com/lisette-espin/PovertyMaps/blob/main/notebooks/PovertyMap_WebMetadata.ipynb)
    1. Compute IWI (international wealth index) from survey data (DHS and MIS) [notebooks/PovertyMap_IWI.ipynb](https://github.com/lisette-espin/PovertyMaps/blob/main/notebooks/PovertyMap_IWI.ipynb)
    2. Collect Nightlight luminosity from populated places using GEE [notebooks/PovertyMap_GEE.ipynb](https://github.com/lisette-espin/PovertyMaps/blob/main/notebooks/PovertyMap_GEE.ipynb)
    3. Cell towers from populated places using OpenCellid [notebooks/PovertyMap_Cells.ipynb](https://github.com/lisette-espin/PovertyMaps/blob/main/notebooks/PovertyMap_Cells.ipynb)
    4. Population (# of people - FB) [notebooks/PovertyMap_Population.ipynb](https://github.com/lisette-espin/PovertyMaps/blob/main/notebooks/PovertyMap_Population.ipynb)
    5. Extract OSM features [notebooks/PovertyMap_OSM.ipynb](https://github.com/lisette-espin/PovertyMaps/blob/main/notebooks/PovertyMap_OSM.ipynb)
    6. Extract Marketing features (audience reach - FB) [notebooks/PovertyMap_Marketing.ipynb](https://github.com/lisette-espin/PovertyMaps/blob/main/notebooks/PovertyMap_Marketing.ipynb)
    7. Extract Movement data (3-week prior corona crisis - FB) 
    8. XGBoost Regression (feature-based on DHS clusters)  [notebooks/PovertyMap_XGBoost.ipynb](https://github.com/lisette-espin/PovertyMaps/blob/main/notebooks/PovertyMap_XGBoost.ipynb)
    9. Collect populated places [notebooks/PovertyMap_PopulatedPlaces.ipynb](https://github.com/lisette-espin/PovertyMaps/blob/main/notebooks/PovertyMap_PopulatedPlaces.ipynb)
