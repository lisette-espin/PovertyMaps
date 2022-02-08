import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import AUC
# import tensorflow_addons as tfa
from tensorflow.keras import losses
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE
from tensorflow.keras.metrics import MeanSquaredError as MSE
from tensorflow.keras.losses import MeanSquaredError
# from ses.r2 import R2
from sklearn.metrics import make_scorer

################################################
# SAMPLING
################################################
TTYPE_ALL = 'all'
TTYPE_OLDEST = 'oldest'
TTYPE_NEWEST = 'newest'
TRAIN_YEAR_TYPES = [TTYPE_ALL, TTYPE_OLDEST, TTYPE_NEWEST]

def validate_traintype(traintype):
  if traintype not in TRAIN_YEAR_TYPES:
    raise Exception("train type does not exist.")

def validate_years_traintype(years, traintype):
  if traintype!=TTYPE_ALL and len(years.split(','))==1:
    print(f"ERROR: traintype:{traintype} | years:{years} | You need at least 2 years.")
    raise Exception("train type and years do not match.")

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
CNN_TUNING_EPOCHS = 20    # epochs within each combination
CNN_TUNING_BATCHSIZES = [8,16,32,64]
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
N_AUGMENTATIONS = 18
#COLUMN_TARGET = 'dhs_mean_iwi'

################################################
# CNN METADATA
################################################
PRECISION = 2

# ################################################
# # XGBOOST
# # https://xgboost.readthedocs.io/en/latest/python/python_api.html
# # https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7
# # https://datascience.stackexchange.com/questions/12318/how-to-interpret-the-output-of-xgboost-importance
# ################################################
# COLS_DHS_REMOVE = ['DHSCC','DHSYEAR','DHSCLUST','URBAN_RURA','LATNUM','LONGNUM','SOURCE','ALT_GPS','ALT_DEM','DATUM','mean_iwi','iwi_bin','iwi_cat','iwi_cat_id','lat','lon']
# COLS_PPLACE_REMOVE = ['id','lat','lon','loc_name','name','type','place','population','capital','is_capital','admin_level','GNS:dsg_code']
# PRECISION = 2
# EARLYSTOP = 50
# FN_MODEL_XGBOOST = 'model.pickle'
# DATASETS_KIND = ['train','val','test']
# VALID_TUNING_TYPES = ['fast','mid','high','robust']
# XGBOOST_IMPORTANCE_TYPE = 'gain'
# BLUE = "#5975a4" 
# XGB_TUNING_ITER = 100
# XGB_VERBOSE = 20
# XGB_SCORING = "neg_mean_squared_error"


################################################
# PLOTS
################################################
FONT_SCALE = 1.5

################################################
# UTILS
################################################
YES = ['y','yes','1',1,True,'True']
NO = ['n','no','0',0,False,'False']
NONE = [None,np.nan,'','none',' ','nan','None']
PLOTEXT = 'png'
COUNTRIES = {'Sierra Leone':'SL', 'Uganda':'UG'}

################################################
# DHS
################################################
SES_LABELS = ['poor','lower_middle','upper_middle','rich']

################################################
# DHSLOC
################################################
DHSLOC_OPTIONS = {'none':'No change',
                  'cc':'Change to closest PPlace',
                  'ccur':'Change to closest urban/rural PPlace',
                  'gc':'Group survey clusters to closest PPlace',
                  'gcur':'Group survey clusters to closest urban/rural PPlace',
                  'rc':'Urban no change, Rural to closest rural PPlace progressively'}
DISPLACEMENT_M = {0:2000, 1:5000} #0: URBAN, 1:RURAL
EXTRA = {0:0, 1:10000}
LABEL = {0:'URBAN', 1:'RURAL'}
DISTCOLS = ['dhs_id','dhs_year','dhs_cluster','dhs_rural','dhs_mean_iwi','dhs_std_iwi','OSMID','pplace_dhs_distance','pplace_rural']
PPLACE_RURAL = ['village', 'hamlet', 'isolated_dwelling']
DHS_RURAL = 2

