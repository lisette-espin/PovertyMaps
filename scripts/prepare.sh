#!/bin/bash
#$1 is country name

# folder structure for each country
cd ../data/"$1"

mkdir -p cache/OSM
mkdir -p cache/OSMPP
mkdir -p cache/FBM
mkdir -p cache/FBMPP
mkdir -p cache/VIIRS
mkdir -p cache/VIIRSPP
mkdir -p connectivity
mkdir -p movement/Facebook
mkdir -p other
mkdir -p population/Facebook
mkdir -p survey/DHS
mkdir -p results/plots
mkdir -p results/features/households
mkdir -p results/features/clusters
mkdir -p results/features/pplaces
mkdir -p results/samples

# cell data
cp ../OpenCellid/country/cell_towers_"$1".csv connectivity/

# install here all python libraries
pip install geopandas==0.9.0
pip install pyshp==2.1.3
pip install rtree==0.9.7
pip install earthengine-api==0.1.290
pip install OSMPythonTools==0.3.2
pip install facebook_business==11.0.0
pip install fast-pagerank==0.0.4
pip install seaborn==0.11.2
pip install xgboost==1.5.1
pip install scikit-learn==1.0.1
pip install geopy==2.2.0
pip install tensorflow-addons==0.15.0
pip install pqdm==0.1.0