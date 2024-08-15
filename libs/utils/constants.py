import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import AUC
from tensorflow.keras import losses
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE
from tensorflow.keras.metrics import MeanSquaredError as MSE
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import make_scorer

################################################
# SAMPLING
################################################
TTYPE_ALL = 'all'
TTYPE_OLDEST = 'oldest'
TTYPE_NEWEST = 'newest'
TTYPE_DOWNSAMPLE = 'all300'
TTYPE_BALANCE = 'balance'
TRAIN_YEAR_TYPES = [TTYPE_ALL, TTYPE_OLDEST, TTYPE_NEWEST, TTYPE_DOWNSAMPLE, TTYPE_BALANCE]
DOWNSAMPLE_SIZE = 300

################################################
# GEO
################################################
PROJ_DEG = "EPSG:4326" # lon,lat
PROJ_MET = "EPSG:3857" # meters
PROJ_EOV = "EPSG:23700" # the Uniform National Projection (EOV,
KM_TO_M = 1000
MILE_TO_M = 1609.34
DEGREE_TO_M = 111000
METERS = [MILE_TO_M, KM_TO_M*2, KM_TO_M*5, KM_TO_M*10]
MAX_DISTANCE_ANTENNA_METERS = 10

################################################
# GOOGLE STATIC MAPS
################################################
IMG_TYPE = 'png'
IMG_HEIGHT = 640
IMG_WIDTH = 640
SIZE = f"{IMG_WIDTH}x{IMG_HEIGHT}"
ZOOM = 16
SCALE = 1
MAPTYPE = "satellite"
BANDS = 3
GT_PLACE = 'clusters'
P_PLACE = 'pplaces'
GT_AUGM = 'augmented'

################################################
# CNN IMAGES
################################################
BATCH_SIZE = 32
BATCH_SIZE_DEFAULT = 32 # THIS IS THE TENSORFLOW DEFAULT
PIXELS_LOGO = 20
LOSS_REGRESSION = MeanSquaredError(name='loss')
METRICS_REGRESSION = ['mae',MSE(name='mse'),RMSE(name='rmse')]
LOSS_CLASSIFICATION = losses.categorical_crossentropy
METRICS_CLASSIFICATION = ['accuracy', AUC(name='auc')]
FEATURE_MAP_LAYERID = 23 #maybe 24? # for online augmentation only
FN_MODEL_CNN = 'model.h5'
CNN_TUNING_ITER = 200      # how many random tries
CNN_TUNING_OPTIMIZERS = ['adam', 'sgd', 'rmsprop']
CNN_TUNING_EPOCHS = 10    # epochs within each combination
CNN_TUNING_BATCHSIZES = [8,16,32,64] # remove 8?
CNN_REG_SCORING = "neg_mean_squared_error"
CNN_CLASS_SCORING = "roc_auc"
CNN_LOSS_NAN = 999999
CNN_TUNNING_SUMMARY_FILE = "tuning.csv"
CNN_BEST_PARAMS_FILE = 'best_params.json'
CNN_EVALUATION_FILE = 'evaluation.json'
CNN_LOG_FILE = 'log.json'
ASK_RETRAIN = 0
RETRAIN = 1
JUST_EVAL = 2
TRAIN_FROM_SCRATCH = 3
PROB_AUGMENTED = 0.5
N_AUGMENTATIONS = 5 #18
CNN_VALIDATION_SPLIT = 0.3

################################################
# FB MARKETING
################################################

FBM_API_VERSION = 'v20.0' #'v17.0' #'v13.0' #'v12'

FBM_NOT_YET = -1
FBM_LOADED_DONE = 0
FBM_QUERIED = 1
FBM_SKIP_LOC = 2
FBM_QUOTA = 3
FBM_NEEDS_QUERY = 4

FBM_ERR_CODE_WRONG_LOC = 100
FBM_ERR_SUBCODE_WRONG_LOC = 1487851

FBM_ERR_CODE_QUOTA = 80004
FBM_ERR_SUBCODE_QUOTA = 2446079

FBM_HQUOTA = 2 # Quota of tokens gets restarted every 2 hours
FBM_DEFAULT_WRONG_LOCATION = np.nan #-1


################################################
# FB MOVEMENT DISTRIBUTION
################################################

FBMD_DISTANCE_GROUPS = ['0', '(0, 10)', '[10, 100)', '100+']


################################################
# CatBoost METADATA
################################################
PRECISION = 2
WEIRD_RS = [2902046601,4261256967,3806232969]
TIMEVAR_VALIDATIONS = ['deltatime','gdp','gdpp','gdppg','gdppgp','gdpg','gdpgp','gni','gnip']

SOURCE_ANTENNA = 'OCI'
SOURCE_FBMARKETING = 'FBM'
SOURCE_NIGHTLIGHT = 'NTLL'
SOURCE_FBMOVEMENT = 'FBMV'
SOURCE_FBPOPULATION = 'FBP'
SOURCE_OPENSTREETMAP = 'OSM'
FEATURE_SOURCES = [SOURCE_ANTENNA, SOURCE_FBMARKETING, SOURCE_NIGHTLIGHT, SOURCE_FBMOVEMENT, SOURCE_FBPOPULATION, SOURCE_OPENSTREETMAP]
COL_POPULATION = 'population_closest_tile'
ALL_FEATURES = 'all'


################################################
# PLOTS
################################################
FONT_SCALE = 1.5

################################################
# TRANSFERABILITY
################################################
SIMILARITY_DISTANCES = ['D3', 'D4k5', 'D4k10', 'D4k20', 'D4k50', 'D4k100'] # IF any of these THEN metric: the lower the better

################################################
# UTILS
################################################
import os 
current_path = os.path.dirname(os.path.realpath(__file__))
print(current_path)

def load_json(fn):
  import json
  with open(fn, 'r') as f:
    return json.load(f)
      
AVAILABLE_GT_SOURCE_FN = os.path.join(current_path, '../../resources/survey/available_source.json')
GT_SOURCES = load_json(AVAILABLE_GT_SOURCE_FN)

AVAILABLE_COUNTRY_METADATA_FN = os.path.join(current_path, '../../resources/survey/available_metadata.json')
COUNTRIES = load_json(AVAILABLE_COUNTRY_METADATA_FN)

YES = ['y','yes','1',1,True,'True','true']
NO = ['n','no','0',0,False,'False','false']
NONE = [None, np.nan, '', 'none', ' ', 'nan', 'None']
PLOTEXT = 'png'
REGRESSION_VARS = ['mean_wi','std_wi']
LON, LAT = 'lon', 'lat'
RURAL = 'rural'
OSMID = 'OSMID'
PPLACE_RURAL_BY_TYPE = ['village', 'hamlet', 'isolated_dwelling']
YEAR = 'year'
WEALTH = 'wi'
CLUSTER = 'cluster'
YMAX_DHS = 100
YMAX_INGATLAN = 4


################################################
# DHS
################################################
SES_LABELS = ['poor','lower_middle','upper_middle','rich']
GTID = 'gtID'
DHS_RURAL = 2 # R
N_STD_DEV_OUTLIER = 40

################################################
# DHSLOC
################################################
DHSLOC_OPTIONS = {'none':'No change',
                  'rc':'Urban no change, Rural to closest rural PPlace progressively',
                  'ruc':'Both rural and urban changed to closest pplaces progresively',
                 }
                  # 'cc':'Change to closest PPlace',
                  # 'ccur':'Change to closest urban/rural PPlace',
                  # 'gc':'Group survey clusters to closest PPlace',
                  # 'gcur':'Group survey clusters to closest urban/rural PPlace',
DISPLACEMENT_M = {0:2000, 1:5000} #0: URBAN, 1:RURAL
EXTRA = {0:0, 1:10000}
EXTRAS_ALLOWED = {0:0, 1:1/100.}
LABEL = {0:'URBAN', 1:'RURAL'}
DISTCOLS = ['cluster_id','cluster_year','cluster_number','cluster_rural',f'mean_{WEALTH}',f'std_{WEALTH}','OSMID','pplace_cluster_distance','pplace_rural']
EXPENSIVE_UTENSILS_COLS = ['hv212', 'hv243e', 'hv211', 'hv243d', 'hv243a']

##############################################################################
# DHS IWI
# Betas: https://link.springer.com/article/10.1007/s11205-014-0683-x/tables/1
# https://globaldatalab.org/iwi/using/
# DHS file: <CCODE>HR**FL.MAP
##############################################################################

### Water supply (hv201)
### - high quality is bottled water or water piped into dwelling or premises;
### - middle quality is public tap, protected well, tanker truck, etc.;
### - low quality is unprotected well, spring, surface water, etc.
### 30 DUG WELL (OPEN/PROTECTED) , should it go to mid instead of low?
DHS_WATER_CODES_FN = os.path.join(current_path, '../../resources/survey/dhs_water_codes.json')
WATER = load_json(DHS_WATER_CODES_FN)

                
### Toilet facility (hv205)
### - high quality is any kind of private flush toilet; 
### - middle quality is public toilet, improved pit latrine, etc.;
### - low quality is traditional pit latrine, hanging toilet, or no toilet facility.
DHS_TOILET_CODES_FN = os.path.join(current_path, '../../resources/survey/dhs_toilet_codes.json')
TOILET = load_json(DHS_TOILET_CODES_FN)

                          
### Floor quality (hv213)
### - high quality is finished floor with parquet, carpet, tiles, ceramic etc.; 
### - middle quality is cement, concrete, raw wood, etc. 
### - low quality is none, earth, dung etc., 
DHS_FLOOR_CODES_FN = os.path.join(current_path, '../../resources/survey/dhs_floor_codes.json')
FLOOR = load_json(DHS_FLOOR_CODES_FN)
                          
DHS_BETAS_FN = os.path.join(current_path, '../../resources/survey/dhs_betas.json')
BETAS = load_json(DHS_BETAS_FN)

BETA_CHEAP_UTENSILS = BETAS['cheap']['beta']
BETA_EXPENSIVE_UTENSILS = BETAS['expensive']['beta']
CONSTANT = BETAS['constant']['beta']

COLS_SURVEY = ['country','year','survey','hhid','hv001','hv002','hv024','hv025','hv270','hv271','hv005','hv243e','hv211','hv243d','hv243a']
COLS_SURVEY.extend(BETAS.keys())
COLS_CLUSTER = ['DHSCC','DHSYEAR','DHSCLUST','URBAN_RURA','LATNUM','LONGNUM','SOURCE','ALT_GPS','ALT_DEM','DATUM']



# ################################################
# # XGBOOST
# # https://xgboost.readthedocs.io/en/latest/python/python_api.html
# # https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7
# # https://datascience.stackexchange.com/questions/12318/how-to-interpret-the-output-of-xgboost-importance
# ################################################
XGBOOST_IMPORTANCE_TYPE = 'gain'
BLUE = "#5975a4" 
XGB_TUNING_ITER = 200
