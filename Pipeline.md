# Pipeline

---

## File system

### Resources
```bash
├── resources
│   ├── Facebook
│   ├── maps
│   ├── OpenCellid
│   ├── survey
```

### Facebook (data * API tokens)
```bash
├── resources
│   ├── Facebook
│   │   ├── MarketingAPI
│   │   ├── Movement
│   │   ├── Population
```

### Maps (api keys)
```bash
├── resources
│   ├── maps
│   │   ├── copernicus
│   │   ├── geopy
│   │   ├── googleearth
│   │   ├── staticmaps

```

### OpenCellId
```bash
├── resources
│   ├── OpenCellid
│   │   ├── fulldatabase
│   │   ├── <download_date>
│   │   │   ├── cell_towers.csv.gz
│   │   │   ├── country
│   │   │   │   ├── cell_towers_<country_name>.csv

```

### Survey (ground-truth)
```bash
├── resources
│   ├── survey
│   │   ├── <country_name>
│   │   │   ├── <source>
│   │   │   │   ├── <year>
│   │   │   │   │   ├── *GC*FL.zip
│   │   │   │   │   ├── *GE*FL.zip
│   │   │   │   │   ├── *HR*DT.zip
```



---

## Inferring Poverty Maps

### Ground-Truth
1. Download ground-truth GT (e.g., DHS data)
2. The GT must be under `resources/survey/<country>/<source>/<year>/<files>`
3. Add country metadata in `COUNTRY` constant under `resources/survey/available_metadata.json` (country name, country code, years, capital)
    - If DHS data is used, note the ccode used in the first two letters of the files.
4. Add country `code` and ground truth `source` in `resources/survey/available_source.json` 
5. Add country code `ccode` in:
    - `scripts/batch_init.sh`
    - `scripts/batch_preprocessing.sh`
    - `scripts/batch_xgb_train.sh`
6. (DHS data only) Add question ids to `WATER`, `TOILET` and `FLOOR` for the respective country.
    1. Open `*HR**FL.MAP` from the GT household data (`*HR*DT.zip`)
    2. For each question identify the `high`, `medium`, and `low` categories (hints in `libs/utils/constants.py`)
      * `hv201` for water supply: `../resources/survey/dhs_water_codes.json`
      * `hv205` for toilet facility: `../resources/survey/dhs_toilet_codes.json` 
      * `hv2013` for floor quality: `../resources/survey/dhs_floor_codes.json`

### Environment
1. Download conda [[guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)]
2. Go to project folder: `cd SES-Inference/`
3. Create enviroment with python 3.7: `conda create --name myenv python=3.7`
4. Install packages: `pip install -r requirements.txt`
    

### Features
1. Download population density (`*_general_2020_csv.zip`): [[doc](https://dataforgood.facebook.com/dfg/docs/methodology-high-resolution-population-density-maps)] [[data](https://data.humdata.org/organization/meta?q=population+density)]
    1. Locate it under `resources/Facebook/Population/<country>/`
1. Download populated places (for African countries only: `*_geojson.zip`): [[data](https://data.humdata.org/dataset/?dataseries_name=HOTOSM+-+Populated+Places)]
    1. Locate the files under `resources/OCHA/Population/<country>/`
2. Download mobility data:
    1. Locate it under: `resources/Facebook/Movement`
3. Download (again to update) the OpenCellID data under `resources/OpenCellid/full_database/YYYY-MM-DD/` (use the date of download) 
    1. From `scripts` run `python batch_pre_oci.py -fnzip ../resources/OpenCellid/full_database/{download_date}/cell_towers.csv.gz -njobs 20`
4. Reactivate Facebook Marketing API tokens
    1. Go to [[Meta for developers](https://developers.facebook.com/apps/)]
    2. Store them in `resources/Facebook/MarketingAPI/tokens/tokens-<FB_account>`
        1. Each `FB_account` can generate at most `15` tokens. Use as many as possible.
    3. Check tokens here: `notebooks/_FBM-check.ipynb`
    4. Run `resources/Facebook/MarketingAPI/copy_to_tokens.sh <num>`
        - Tokens will be distributed across `num` folders (in case you want to run the `batch_features.sh` script for `clusters` and `pplaces` (or different countries) at the same time. One folder for each instance. If so, pass the correct `path` under the argument `-t` in `batch_features.sh`.
5. Get API keys from Google Earth Engine
    1. Go to [[Obtaining an API Key](https://developers.google.com/earth-engine/guides/app_key)]
    2. You need to create these files, in each add the respective key:
        1. `api_key`
        2. `project_id`
        3. `service_account`
        4. `clientsecret.json`
            - This file is generated when adding a `key` [here](https://console.cloud.google.com/iam-admin/serviceaccounts/details/102400032297592831224;edit=true/keys)
6. Get API keys for Google Maps Static API
    1. Go to [[Use API Keys with Maps Static API](https://developers.google.com/maps/documentation/maps-static/get-api-key)]
    2. You need to create these files, in each add the respective key:
        1. `api_key`
        2. `secret`
        
### Running scripts
For example, `cd scripts/`

1. Run init: `./batch_init.sh -r ../data/Uganda -c UG -y 2016,2018 -n 10`
    - Note: It runs 3 scripts (i) prepares folder structure for the given country, (ii) prepares the GT data, (iii) prepares PPlaces
    - If you have your own PPLACES.csv, you should move it to `../data/<country>/features/pplaces/`
2. Run features GT: `./batch_features.sh -r ../data/Uganda -c UG -y 2016,2018 -n 10`
    - Note 1: pass argument `-z` to specify the size (width x height) of satellite images, e.g., -z 160x160 (when using 400x400m grid-cells) or 640x640 (when using OSM pplaces)
    - Note 2: pass argument `-m` to specify the bounding boxes for the VIIRS (comma separated meters), e.g., -m 400,800,1200,1600 (when using 400x400m grid-cells). If nothing is passed the default is: 1600,2000,5000,10000 (e.g., when using OSM pplaces)
    - Note 3: pass argument `-w` to specify the bounding box for OSM in meters, e.g., 1600 (when using OSM pplaces) or 400 (when using 400x400m grid-cells)
    - Note 4: pass correct path to API keys.
    - After running this step, you can move to step \#4
3. Run features PP: `./batch_features.sh -r ../data/Uganda -c UG -n 10`
    - Same as in step 2. Be aware of passing the correct values for `-z`, `-m` and `-w`
    - Note that this step is not needed to run steps \#4, 5, 6, 7, 8, and 9.
4. Run Pre-processing: `./batch_preprocessing.sh -r ../data/Uganda -c UG -y 2016,2018 -o none -t all -k 5 -e 3`
    - Note: `-t` is the argument to specify the recency of the ground truth data to use for training. The available values are: all, newest, oldest
5. Run CatBoost training: `./batch_xgb_train.sh -r ../data/Uganda -c UG -y 2016,2018 -l none -t all -a mean_wi,std_wi -f all -k 4 -v 1`
    - Weights: Run `notebooks/_CatBoost_Weights_Cat10.ipynb` then update weights in `resources/CB_weights.json` and `maxval` in `resources/survey/available_metadata.json` (@TODO: make it a script)
    - Weighted: `./batch_xgb_train.sh -r ../data/Uganda -c UG -y 2016,2018 -l none -t all -a mean_wi,std_wi -f all -k 4 -v 1 -w 1`
    - Note: `-f` is the argument to specify the source of features. Available: all, FBM, FBP, FBMV, NTLL, OCI, OSM, and any combination of the individual sources sorted ASC alphabetically, and separated with `_`, eg. FBM_FBMP_OCI
6. Run Augmentation: `python cnn_augmentation.py -r ../data/Uganda -years 2016,2018 -dhsloc none -probaug 1.0 -njobs 10 -imgwidth 640 -imgheight 640`
    - Note 1: Change `imgwidth` and `imgheight` accordingly, e.g., 640 when using OSM pplaces, and 160 when using 400x400m grid cells.
7. Run CNN training (& feature maps): Run scripts in `SLURM` (for `CNN` and `CNNa`)
    - Create folder: `slurm/dhsloc_none/logs-<ccode>`
    - see `../slurm/LB_train_r1_f12.sh` (this runs Liberia, run 1, fold 1 and 2)
    - Send job to schedule: `sbatch LB_train_r1_f12.sh`
    - Run CNN merging files: see `notebooks/_CNN_Merging_Tuning_Files.ipynb`
    - Run `../slurm/LB_train_r1_r2.sh` (this runs Liberia, run 1 and 2)
8. Run fmaps: Run scripts in `SLURM` (for `CNN` and `CNNa`)
    - see `../slurm/LB_fmaps_noaug_r12.sh` (this runs Liberia, run 1 and 2)
    - Send job to schedule: `sbatch LB_fmaps_noaug_r12.sh`
9. Run CNN+CatBoost training: `./batch_xgb_train.sh -r ../data/Uganda -c UG -y 2016,2018 -l none -t all -a mean_wi,std_wi -f all -k 4 -v 1 -n offaug_cnn_mp_dp_relu_sigmoid_adam_mean_std_regression -e 19 -w 1`
   - Note 1: change `-m` (CNN model with and without augmentation) and `-w` (weighted CB or not) accordingly.
10. Run Poverty map: `python batch_infer_poverty_maps.py -ccode UG -model CB`
    - First, make sure you have collected the features for PPLACES (step \#3)
    - Second, run models: `CB, CBw, CNN, CNNa` (passing each of them as `-model <model_name>`)
        - Note that, running `-model CNN*` will create the feature maps for the pplaces, required to run the combined models
    - Third, run the combined models: `CNN+CB, CNNa+CB, CNN+CBw, CNNa+CBw`
    - For countries with many `pplaces` e.g., `>10K` it is recommended to use a high-memory server.
    - `CNN` and `CNN_a` must be run on a GPU.
11. Run Cross-country testing: `python batch_cross_predictions.py` (on a GPU)
    - First, update list of countries `COUNTRIES` to include in the transfer (in the same script).

---


