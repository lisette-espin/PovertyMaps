#!/bin/bash

cd ../data/"$1"

### MIS
cd survey/MIS/2016
unzip *_2016*MIS_*.zip

cd ../2018
unzip *_2018*MIS_*.zip

### DHS

cd ../../DHS/2019
unzip *_2019_DHS_*.zip

cd ../2016
unzip *_2016_DHS_*.zip