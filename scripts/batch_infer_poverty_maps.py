import pandas as pd
import numpy as np
import json
import glob
import os 
import argparse
import time

from numpy import savez_compressed
from catboost import CatBoostRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from itertools import permutations
from collections import OrderedDict
import gc

from keras.models import Model
import tensorflow as tf
from tensorflow.random import set_seed
from tensorflow.keras.models import load_model

from ses.data import Data
from ses.metadata import SESMetadata
from ses.images import SESImages
from utils import ios
from utils import constants
from utils.augmentation import RandomColorDistortion

ROOT = '../data'
COUNTRIES = {k:v for k,v in constants.COUNTRIES.items() if k in ['Sierra Leone','Uganda']}
MODELS = OrderedDict({'catboost':'CB', 
                      'weighted_catboost':'CB$_w$', 
                      'cnn':'CNN', 
                      'cnn_aug':r'CNN$_a$',
                      'cnn+catboost':'CNN+CB', 
                      'cnn+weighted_catboost':'CNN+CB$_w$', 
                      'cnn_aug+catboost':r'CNN$_a$+CB', 
                      'cnn_aug+weighted_catboost':r'CNN$_a$+CB$_w$'})
MODELS_CB = [MODELS['catboost'], MODELS['weighted_catboost']]
MODELS_CNN = [MODELS['cnn'], MODELS['cnn_aug']]
MODELS_FMAP = [MODELS['cnn+catboost'], MODELS['cnn+weighted_catboost'], MODELS['cnn_aug+catboost'], MODELS['cnn_aug+weighted_catboost']]
OUTPUT = '../paper/results/pplaces_inference'

#################################################################################
# Main
#################################################################################

def run(ccode, model, year=None):
  country = validate_country(ccode)
  validate_model(model)
  if already_exists(ccode, model, year):
    print("[INFO] Already exists. Nothing to do.")
  else:
    country_prediction(ccode, country, model, year)
  
def get_predictions_fn(ccode, model, year=None):
  postfix = "" if year is None else f"_{year}"
  fn = os.path.join(OUTPUT,f'{ccode}_{model}{postfix}.csv')
  return fn

def already_exists(ccode, model, year):
  fn = get_predictions_fn(ccode, model, year)
  ios.validate_path(os.path.dirname(fn))
  return ios.exists(fn)
  
def validate_country(ccode):
  country = [k for k,obj in COUNTRIES.items() if obj['code']==ccode]
  if len(country)==0:
    raise Exception("Country does not exist.")
  return country[0]

def validate_model(model):
  model = [k for k,v in MODELS.items() if model==v.replace("$","").replace("_","")]
  if len(model)==0:
    raise Exception("Model does not exist.")

def get_model_fn(country, model, year=None):
  # Country's models
  dhsloc = 'none'
  ttype = 'all'
  if year is None:
    prefix = ios.get_prefix_surveys(df=None, root=os.path.join(ROOT,country), years=COUNTRIES[country]['years'])
  else:
    prefix = ios.get_prefix_surveys(df=None, root=os.path.join(ROOT,country), years=[year])
    
  root_models = f"{prefix}_{ttype}_{dhsloc}"
  root = os.path.join(ROOT, country, 'results','samples',root_models) # DHS_MSI_all_none folder
  files_1 = glob.glob(os.path.join(root,'*','*','model.h5')) # cnn, cnn_a
  files_2 = glob.glob(os.path.join(root,'*','*','*','model.json')) # cnn+cb, cnna+cb, cbw-*
  files_3 = glob.glob(os.path.join(root,'*','*','*','*','model.json')) # cnn+cbw, cnna+cbw
  files_4  = glob.glob(os.path.join(root,'*','*','model.json')) # cb
  
  cnn = False
  if model == 'CB':
    files = [fn for fn in files_4 if 'xgb-all' in fn]
  elif model == 'CBw':
    files = [fn for fn in files_2 if 'cnn' not in fn and 'xgb-all' in fn]
  elif model == 'CNN':
    files = [fn for fn in files_1 if 'offaug' not in fn]
    cnn = True
  elif model == 'CNNa':
    files = [fn for fn in files_1 if 'noaug' not in fn]
    cnn = True
  elif model == 'CNN+CB':
    files = [fn for fn in files_2 if 'offaug' not in fn and 'weighted' not in fn]
  elif model == 'CNNa+CB':
    files = [fn for fn in files_2 if 'noaug' not in fn and 'weighted' not in fn]
  elif model == 'CNN+CBw':
    files = [fn for fn in files_3 if 'offaug' not in fn and 'weighted' in fn]
  elif model == 'CNNa+CBw':
    files = [fn for fn in files_3 if 'noaug' not in fn and 'weighted' in fn]

  # Out of 3 runs, search for the best
  # @TODO: Should we create a 4th model, where we use the entire gt?
  eval_values = []
  for fn in files:
    if cnn:
      fn_eval = fn.replace('model.h5','log.json')
      obj = ios.load_json(fn_eval)
      eval_values.append(obj['test_mse'])
    else:
      fn_eval = fn.replace('model.json','evaluation.json')
      obj = ios.load_json(fn_eval)
      eval_values.append(obj['mse'])
  fn = files[eval_values.index(min(eval_values))]
  
  print(' - '.join([f"run{f.split('-rs')[0].split('/')[-1].replace('epoch','')}: {v}" for v,f in zip(*(eval_values, files))]))
  print(f"[INFO] {fn} to be loaded.")
  return fn
  
def country_prediction(ccode, country, model, year=None):
  print("============================================================")
  print(f"Country: {country} ({ccode}) - Model: {model} - Year: {year}")
  print("============================================================")
  
  model_fn = get_model_fn(country, model, year)
  model_name = get_model_name(model_fn)
  
  # Pplaces
  root = os.path.join(ROOT,country)
  y_attribute = 'mean_wi,std_wi'
  
  offlineaug = False # because test set does not get augmented
  is_reg = True
  features_source = model_fn.split("/xgb-")[-1].split('/')[0] if 'xgb-' in model_fn else None
  if features_source is not None and features_source!='all':
    raise Exception('Feature source must be all')
  
  cv = 4
  include_fmaps = '_cnn_' in model_fn and 'xgb-' in model_fn
  weighted = 'weighted' in model_fn
  n_jobs = 10
  timevar = None
  viirsnorm = True
  layer_id = 19
  cnn_name = 'offaug_cnn_mp_dp_relu_sigmoid_adam_mean_std_regression'
  fmap = None
  
  print(f"************** {model_name} *****************")
  if model_name in MODELS_CNN:
    df_pred, fmap = p_cnn(model_fn, model_name, country, ccode,
                    root, y_attribute, 
                    features_source, layer_id)
  elif model_name in MODELS_CB:
    df_pred = p_cb(model_fn, model_name, country, ccode,
                    root, y_attribute, 
                    features_source)
  elif model_name in MODELS_FMAP:
    df_pred = p_cnn_cb(model_fn, model_name, country, ccode,
                      root, y_attribute, 
                      features_source, layer_id)
  else:
    print(f"[ERROR] {model_name} mistmatch {model_fn}")
  
  # Saving predictions
  fn_pred = get_predictions_fn(ccode, model, year)
  ios.save_csv(df_pred, fn_pred)
  print(f'[INFO] {fn_pred} saved!')
  
  # saving fmap
  if fmap is not None:
    fn_fmap = os.path.join(OUTPUT, 'fmaps', f"{ccode}_{model}.npz")
    ios.validate_path(os.path.dirname(fn_fmap))
    savez_compressed(fn_fmap, fmap) 

  
#################################################################################
# HAndlers
################################################################################# 
def get_model_name(fn):
  print(fn)
  model =  'cnn' if 'cnn' in fn and 'xgb' not in fn and 'noaug_' in fn else \
           'cnn_aug' if 'cnn' in fn and 'xgb' not in fn and 'noaug_' not in fn else \
           'catboost' if 'cnn' not in fn and 'xgb-' in fn and 'weighted' not in fn else \
           'weighted_catboost' if 'cnn' not in fn and 'xgb-' in fn and 'weighted' in fn else \
           'cnn+catboost' if 'cnn' in fn and 'noaug' in fn and 'xgb-' in fn and 'weighted' not in fn else \
           'cnn_aug+catboost' if 'cnn' in fn and 'noaug' not in fn and 'xgb-' in fn and 'weighted' not in fn else \
           'cnn+weighted_catboost' if 'cnn' and 'noaug' in fn in fn and 'xgb-' in fn and 'weighted' in fn else \
           'cnn_aug+weighted_catboost' if 'cnn' in fn and 'xgb-' in fn and 'noaug_' not in fn and 'weighted' in fn else None
  return MODELS[model]
  
def prepare_output(y_pred, df, y_attribute):
  tmp = df.loc[:,['OSMID']].copy()

  print('tmp', tmp.shape)
  print('y_pred',y_pred.shape)

  for ia, at in enumerate(y_attribute.split(',')):
    print(ia,at)
    tmp.loc[:,f'pred_{at}'] = y_pred[:,ia]
   
  return tmp

#################################################################################
# Prediction using CNN
#################################################################################

def p_cnn(model_fn, model_name, country, ccode, root, y_attribute, features_source, layer_id):
  
  # Setup
  rs = int(model_fn.split('-rs')[-1].split('/')[0])
  np.random.seed(rs)
  set_seed(rs)

  # Load model
  print("[INFO] loading model...")
  model = load_model(model_fn, custom_objects={'RandomColorDistortion':RandomColorDistortion()})

  # Loading data
  print("[INFO] loading X...")
  pplaces = Data.get_pplaces(root, metadata=False)
  X = Data.cnn_get_X(root=root, df=pplaces)
  
  # Prediction
  print("[INFO] inferring y_pred...")
  y_pred = model.predict(X) 
  del(model)

  # Fmaps
  print("[INFO] extracting fmaps...")
  fmap = get_fmap(model_fn, layer_id, X)
  
  # return results
  return prepare_output(y_pred, pplaces, y_attribute), fmap
  

def get_fmap(model_fn, layer_id, X):
  model = load_model(model_fn, custom_objects={'RandomColorDistortion':RandomColorDistortion()})
  remodel = Model(inputs=model.inputs, outputs=model.layers[layer_id].output)
  fmap = remodel.predict(X)
  return fmap

#################################################################################
# Prediction using CatBoost
#################################################################################

def p_cb(model_fn, model_name, country, ccode, root, y_attribute, features_source):
  
  # Setup
  rs = int(model_fn.split('-rs')[-1].split('/')[0])
  np.random.seed(rs)

  # Load model
  print(f"[INFO] loading model {model_name}...")
  model = CatBoostRegressor()
  model.load_model(model_fn, format='json')
 
  # Data
  print("[INFO] loading X...")
  pplaces = Data.get_pplaces(root)
  feature_names = ios.read_txt_to_list(model_fn.replace('model.json','features.txt'))
  X = Data.metadata_get_X(root=root, df=pplaces, feature_names=feature_names, features_source=features_source)

  # Predict
  print("[INFO] inferring y_pred...")
  y_pred = model.predict(X) 
  del(model)

  # Save features (not needed but just in case for vis)
  save_features(pplaces, country, ccode)
  
  # return results
  return prepare_output(y_pred, pplaces.reset_index(), y_attribute)

#################################################################################
# Prediction using CNN & CatBoost
#################################################################################


def p_cnn_cb(model_fn, model_name, country, ccode, root, y_attribute, features_source, layer_id):
  
  # Setup
  rs = int(model_fn.split('-rs')[-1].split('/')[0])
  np.random.seed(rs)

  # Load model
  print(f"[INFO] loading model {model_name}...")
  model = CatBoostRegressor()
  model.load_model(model_fn, format='json')
  
  # Data
  print("[INFO] loading fmap...")
  cnn_model_name = get_cnn_model_name_from_combined(model_fn)
  fmaps = load_fmap(ccode, cnn_model_name)

  print("[INFO] loading X...")
  pplaces = Data.get_pplaces(root)
  feature_names = [f for f in ios.read_txt_to_list(model_fn.replace('model.json','features.txt')) if not (f.startswith('cnn') and len(f)<=6)]
  X = Data.metadata_get_X(root=root, df=pplaces, feature_names=feature_names, features_source=features_source, fmaps=fmaps)
  
  # Predict
  print("[INFO] inferring y_pred...")
  y_pred = model.predict(X) 
  del(model)

  # return results
  return prepare_output(y_pred, pplaces.reset_index(), y_attribute)

def get_cnn_model_name_from_combined(model_fn):
  cnn_fn = model_fn.split('/xgb-')[0]
  cnn_model = 'CNN' if 'noaug_cnn_' in cnn_fn else 'CNNa' if 'offaug_cnn_' in cnn_fn else None
  print(f"[INFO] \n-CB:{model_fn}\n-CNN:{cnn_fn}\n-CNN MODEL:{cnn_model}")
  return cnn_model
  
def load_fmap(ccode, model_name):
  if model_name is None:
    raise Exception("model name is None.")
  files = [fn for fn in glob.glob(os.path.join(OUTPUT,'fmaps',f"{ccode}_{model_name}.npz"))]
  if len(files) == 0 or len(files) > 1:
    raise Exception("Something is wrong. There should be 1 file.")
  fmap = ios.load_array(files[0])['arr_0']
  print(f"[INFO] fmap size: {fmap.shape}")
  return fmap

#################################################################################
# Others
#################################################################################

def save_features(df, country, ccode):
  fn = os.path.join(OUTPUT,f'{ccode}_features.csv')
  if not ios.exists(fn):
    df.loc[:,'country'] = country
    df.loc[:,'ccode'] = ccode
    ios.save_csv(df, fn)
  
#################################################################################
#
#################################################################################

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-ccode", help="Country code (UG: Uganda, SL: Sierra Leone)", type=str, required=True)
  parser.add_argument("-model", help="Model to use for inference (CB, CBw, CNN, CNNa, CNN+CB, CNN+CBw, CNNa+CB, CNNa+CBw).", type=str, required=True, default='CB')
  parser.add_argument("-year", help="Year of survey", type=int, required=False, default=None)
  
  args = parser.parse_args()
  for arg in vars(args):
    print("{}: {}".format(arg, getattr(args, arg)))

  start_time = time.time()
  run(args.ccode, args.model, args.year)
  print("--- %s seconds ---" % (time.time() - start_time))
