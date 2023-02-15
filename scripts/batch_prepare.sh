#!/bin/bash
#$1 is country name

# folder structure for each country
folder="$1" #"../data/$1"
echo $folder
country=$(echo "$folder" | cut -d'/' -f3)
country="${country/\/" "}"   
echo $country

# default folders
mkdir -p "../logs"
mkdir -p "$folder/cache/OSM"
mkdir -p "$folder/cache/OSMPP"
mkdir -p "$folder/cache/FBM"
mkdir -p "$folder/cache/FBMPP"
mkdir -p "$folder/cache/VIIRS"
mkdir -p "$folder/cache/VIIRSPP"
mkdir -p "$folder/connectivity"
mkdir -p "$folder/movement/Facebook"
mkdir -p "$folder/other"
mkdir -p "$folder/population/Facebook"
mkdir -p "$folder/population/OCHA"
mkdir -p "$folder/survey/"
mkdir -p "$folder/results/plots"
mkdir -p "$folder/results/staticmaps/clusters"
mkdir -p "$folder/results/staticmaps/pplaces"
mkdir -p "$folder/results/staticmaps/augmented"
mkdir -p "$folder/results/features/households"
mkdir -p "$folder/results/features/clusters"
mkdir -p "$folder/results/features/pplaces"
mkdir -p "$folder/results/samples"

# cells, movement, population, survey
cp -r "../resources/survey/$country/"* "$folder/survey/"
cp "../resources/OpenCellid/country/cell_towers_$country".csv "$folder/connectivity/"
cp "../resources/Facebook/Movement/$country/"*_csv.zip "$folder/movement/Facebook/"
cp "../resources/Facebook/Population/$country/"*_general_2020_csv.zip "$folder/population/Facebook/"

# Google earth engine auth
echo "./auth_gee.sh"
./auth_gee.sh