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

FBM_API_VERSION = 'v13.0' #'v12'

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
# UTILS
################################################
YES = ['y','yes','1',1,True,'True','true']
NO = ['n','no','0',0,False,'False','false']
NONE = [None, np.nan, '', 'none', ' ', 'nan', 'None']
PLOTEXT = 'png'
COUNTRIES = {'Sierra Leone':{'code':'SL', 'years':'2016,2019', 'tz':'Africa/Freetown'}, 
             'Uganda':{'code':'UG', 'years':"2016,2018", 'tz':'Africa/Kampala'},
             'Hungary':{'code':'HU', 'years':'2018', 'tz':'Europe/Budapest'},
             'Zimbabwe':{'code':'ZW', 'years':'2015', 'tz':'Africa/Harare'},
             'Ecuador':{'code':'EC', 'years':'2021', 'tz':'America/Guayaquil'}}

REGRESSION_VARS = ['mean_wi','std_wi'] #['dhs_mean_iwi','dhs_std_iwi']
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


##############################################################################
# DHS IWI
# Betas: https://link.springer.com/article/10.1007/s11205-014-0683-x/tables/1
# https://globaldatalab.org/iwi/using/
##############################################################################

### Water supply (hv201)
### - high quality is bottled water or water piped into dwelling or premises;
### - middle quality is public tap, protected well, tanker truck, etc.;
### - low quality is unprotected well, spring, surface water, etc.
### 30 DUG WELL (OPEN/PROTECTED) , should it go to mid instead of low?

WATER = {'UG':{2006:{'high':[11,12,71],'mid':[13,31,33,34,41,61],'low':[20,21,22,23,30,32,35,36,40,42,43,44,45,46,51,62,91]},
               2009:{'high':[10,11,12,71],'mid':[13,31,33,34,41,61],'low':[20,21,22,23,30,32,35,40,42,43,44,45,46,51,62]},
               2011:{'high':[10,11,12,71],'mid':[13,31,33,34,41,61,71,72],'low':[20,21,22,23,30,32,35,36,40,42,43,44,45,46,51,62]},
               2014:{'high':[11,12,91],'mid':[13,31,41,61,63,71],'low':[21,22,32,42,43,44,51,62,81]},
               2016:{'high':[11,12,13,91],'mid':[14,31,41,61,63,72,92],'low':[21,32,42,43,51,71,81]},
               2018:{'high':[11,12,13,91],'mid':[14,31,41,61,63,72,92],'low':[21,32,42,43,51,71,81]}},
         'SL':{2008:{'high':[10,11,12,71], 'mid':[13,31,41,61], 'low':[20,21,30,32,40,42,43,51,62]},
               2013:{'high':[10,11,12,91,71], 'mid':[13,31,41,61,92], 'low':[20,21,30,32,40,42,43,51,62]},
               2016:{'high':[10,11,12,13,71], 'mid':[14,31,41,61,72], 'low':[20,21,30,32,40,42,43,51,62]},
               2019:{'high':[10,11,12,13,71], 'mid':[14,31,41,61,72], 'low':[20,21,30,32,40,42,43,51,62,81]}},
         'ZW':{2015:{'high':[10,11,12,13,71], 'mid':[14,31,41,61], 'low':[20,21,30,32,40,42,43,51,62]}}}

                
### Toilet facility (hv205)
### - high quality is any kind of private flush toilet; 
### - middle quality is public toilet, improved pit latrine, etc.;
### - low quality is traditional pit latrine, hanging toilet, or no toilet facility.

TOILET = {'UG':{2006:{'high':[10,11],'mid':[21,23],'low':[20,22,24,25,30,31,41,42,43]},
               2009:{'high':[10,11],'mid':[21,23],'low':[20,22,24,25,30,31,41,42,43]},
               2011:{'high':[1,10,11],'mid':[2,4,21,23],'low':[3,5,6,7,8,9,20,22,24,25,30,31,41,43,44]},
               2014:{'high':[11,12,13,14,15],'mid':[21,22],'low':[23,24,25,31,41,42,43,51,61]},
               2016:{'high':[11,12,13,14,15],'mid':[21,22],'low':[23,31,41,42,43,51,61]},
               2018:{'high':[11,12,13,14,15],'mid':[21,22],'low':[23,31,41,42,43,51,61]}},
         'SL':{2008:{'high':[10,11,12,13,14,15], 'mid':[21,22], 'low':[20,23,30,31,41,42,43,71]},
               2013:{'high':[10,11,12,13,14,15], 'mid':[21,22], 'low':[20,23,30,31,41,42,43]},
               2016:{'high':[10,11,12,13,14,15], 'mid':[21,22], 'low':[20,23,30,31,41,42,43]},
               2019:{'high':[10,11,12,13,14,15], 'mid':[21,22], 'low':[20,23,30,31,41,42,43]}},
         'ZW':{2015:{'high':[10,11,12,13,14,15], 'mid':[21,22], 'low':[20,23,30,31,41,42,43]}}}

                          
### Floor quality (hv213)
### - high quality is finished floor with parquet, carpet, tiles, ceramic etc.; 
### - middle quality is cement, concrete, raw wood, etc. 
### - low quality is none, earth, dung etc., 
FLOOR = {'UG':{2006:{'high':[30,31,33],'mid':[34,35,36],'low':[10,11,12,20,]},
               2009:{'high':[30,31,33],'mid':[34,35,36],'low':[10,11,12,20]},
               2011:{'high':[30,31,33],'mid':[34,35,36],'low':[10,11,12,20]},
               2014:{'high':[30,31,32],'mid':[21,22,33,34,35],'low':[10,11,12,20]},
               2016:{'high':[30,31,33,35],'mid':[21,22,32,34,36,37],'low':[10,11,12,20,]},
               2018:{'high':[30,31,33,35],'mid':[21,22,32,34,36,37],'low':[10,11,12,20]}},
         'SL':{2008:{'high':[30,31,32,33,35], 'mid':[13,21,22,34], 'low':[10,11,12,20]},                          
               2013:{'high':[30,31,32,33,35], 'mid':[21,22,34], 'low':[10,11,12,20]},
               2016:{'high':[31,32,33,35], 'mid':[21,22,34], 'low':[11,12]},
               2019:{'high':[30,31,32,33,35], 'mid':[21,22,34], 'low':[10,11,12,20]}},
         'ZW':{2015:{'high':[30,31,32,33,35], 'mid':[21,22,34], 'low':[10,11,12,20]}}} # 22 remove?
                          
BETAS = {'hv208':8.612657,  # tv
         'hv209':8.429076,  # fridge
         'hv221':7.127699,  # telephone
         'hv212':4.651382,  # car
         'hv210':1.846860,  # bike
         'hv206':8.056664,  # electricity 
         'hv201':{1:-6.306477,2:-2.302023,3:7.952443},  # water
         'hv213':{1:-7.558471,2:1.227531, 3:6.107428},  # floor 
         'hv205':{1:-7.439841,2:-1.090393,3:8.140637},  # toilet
         'hv216':{1:-3.699681,2:0.384050, 3:3.445009}}  # sleeping rooms
BETA_CHEAP_UTENSILS = 4.118394     # cheap utensils
BETA_EXPENSIVE_UTENSILS = 6.507283 # expensive utensils 
CONSTANT = 25.004470

COLS_SURVEY = ['country','year','survey','hhid','hv001','hv002','hv024','hv025','hv270','hv271','hv005','hv243e','hv211','hv243d','hv243a']
COLS_SURVEY.extend(BETAS.keys())
COLS_CLUSTER = ['DHSCC','DHSYEAR','DHSCLUST','URBAN_RURA','LATNUM','LONGNUM','SOURCE','ALT_GPS','ALT_DEM','DATUM']



################################################
# XGBOOST (deprecated)
# https://xgboost.readthedocs.io/en/latest/python/python_api.html
# https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7
# https://datascience.stackexchange.com/questions/12318/how-to-interpret-the-output-of-xgboost-importance
################################################
XGBOOST_IMPORTANCE_TYPE = 'gain'
BLUE = "#5975a4" 
XGB_TUNING_ITER = 200
