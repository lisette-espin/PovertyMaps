import pandas as pd
import numpy as np
import json
import glob
import os 
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
OUTPUT = '../paper/results/cross_modeling_fmaps'

#################################################################################
# Main
#################################################################################

def run():
  df_performance = pd.DataFrame()
  df_residuals = pd.DataFrame()
  
  for source_country, target_country in list(permutations(COUNTRIES.keys(),2)):
    partial_performance, partial_residuals = cross_country_prediction(source_country, target_country)
    # append
    df_performance = pd.concat([df_performance, partial_performance], ignore_index=True)
    df_residuals = pd.concat([df_residuals, partial_residuals], ignore_index=True)
      
  # save to file
  fn = '../paper/results/residuals_cross_country_testing.csv'
  df_residuals.to_csv(fn, index=True)
  print(f"{fn} saved!")
  
  fn = '../paper/results/performance_cross_country_testing.csv'
  df_performance.to_csv(fn, index=True)
  print(f"{fn} saved!")
  
def cross_country_prediction(source_country, target_country):
  print("============================================================")
  print(f"Source: {source_country} - Target: {target_country}")
  print("============================================================")
  
  # Source country (model: CB_w)
  dhsloc = 'none'
  ttype = 'all'
  prefix = ios.get_prefix_surveys(df=None, root=os.path.join(ROOT,source_country), years=COUNTRIES[source_country]['years'])
  root_models = f"{prefix}_{ttype}_{dhsloc}"
  source_root = os.path.join(ROOT, source_country, 'results','samples',root_models) # DHS_MSI_all_none folder
  files_1 = glob.glob(os.path.join(source_root,'*','*','model.h5')) # cnn, cnn_a
  files_2 = glob.glob(os.path.join(source_root,'*','*','*','model.json')) # cnn+cb, cnna+cb, cbw
  files_3 = glob.glob(os.path.join(source_root,'*','*','*','*','model.json')) # cnn+cbw, cnna+cbw
  files_4  = glob.glob(os.path.join(source_root,'*','*','model.json')) # cb

  files = files_1 + files_2 + files_3 + files_4 
  print(f"[INFO] {len(files)} models to load from {root_models}.")

  # results
  df_performance = pd.DataFrame()
  df_residuals = pd.DataFrame()

  # Target country (test set)
  root = os.path.join(ROOT,target_country)
  y_attribute = 'mean_wi,std_wi'
  for model_fn in files:
    years = COUNTRIES[target_country]['years']
    model_name = get_model_name(model_fn)
    offlineaug = False # because test set does not get augmented
    is_reg = True
    features_source = model_fn.split("/xgb-")[-1].split('/')[0] if 'xgb-' in model_fn else None
    if features_source is not None and features_source!='all':
      continue
    cv = 4
    include_fmaps = '_cnn_' in model_fn and 'xgb-' in model_fn
    weighted = 'weighted' in model_fn
    n_jobs = 10
    timevar = None
    viirsnorm = True
    layer_id = 19
    cnn_name = 'offaug_cnn_mp_dp_relu_sigmoid_adam_mean_std_regression'

    print(f"************** {model_name} *****************")
    if model_name in MODELS_CNN:
      partial_performance, partial_residuals = ccp_cnn(model_fn, model_name, source_country, 
                                                      target_country, root, years, dhsloc, ttype,
                                                      is_reg, viirsnorm, y_attribute, 
                                                      features_source, timevar,offlineaug, 
                                                      layer_id, cnn_name)
    elif model_name in MODELS_FMAP:
      partial_performance, partial_residuals = ccp_cnn_cb(model_fn, model_name, source_country, 
                                                          target_country, root, years, dhsloc, ttype,
                                                          is_reg, viirsnorm, y_attribute, 
                                                          features_source, timevar,offlineaug, 
                                                          layer_id, cnn_name)
    elif model_name in MODELS_CB:
      partial_performance, partial_residuals = ccp_cb(model_fn, model_name, source_country, 
                                                      target_country, root, years, dhsloc, ttype,
                                                      is_reg, viirsnorm, y_attribute, 
                                                      features_source, timevar)
    else:
      print(f"[ERROR] {model_name} mistmatch {model_fn}")
    
    # append
    df_performance = pd.concat([df_performance, partial_performance], ignore_index=True)
    df_residuals = pd.concat([df_residuals, partial_residuals], ignore_index=True)
    gc.collect()
    
  return df_performance, df_residuals
  
#################################################################################
# HAndlers
################################################################################# 
def get_model_name(fn):
  model =  'cnn' if 'cnn' in fn and 'xgb' not in fn and 'noaug_' in fn else \
           'cnn_aug' if 'cnn' in fn and 'xgb' not in fn and 'noaug_' not in fn else \
           'catboost' if 'cnn' not in fn and 'xgb-' in fn and 'weighted' not in fn else \
           'weighted_catboost' if 'cnn' not in fn and 'xgb-' in fn and 'weighted' in fn else \
           'cnn+catboost' if 'cnn' in fn and 'noaug' in fn and 'xgb-' in fn and 'weighted' not in fn else \
           'cnn_aug+catboost' if 'cnn' in fn and 'noaug' not in fn and 'xgb-' in fn and 'weighted' not in fn else \
           'cnn+weighted_catboost' if 'cnn' and 'noaug' in fn in fn and 'xgb-' in fn and 'weighted' in fn else \
           'cnn_aug+weighted_catboost' if 'cnn' in fn and 'xgb-' in fn and 'noaug_' not in fn and 'weighted' in fn else None
  return MODELS[model]
  
def save_prediction(y_pred, y, df, fn_pred, y_attribute):
  tmp = df.loc[:,['cluster_id']].copy() # @TODO here it should be gtID?

  print('tmp', tmp.shape)
  print('y',y.shape)
  print('y_pred',y_pred.shape)

  for ia, at in enumerate(y_attribute.split(',')):
    print(ia,at)
    tmp.loc[:,f'true_{at}'] = y[:,ia]
    tmp.loc[:,f'pred_{at}'] = y_pred[:,ia]
    
  ios.save_csv(tmp, fn_pred)
  
#################################################################################
# Predictionc using CNN
#################################################################################
def cnn_predict(cb_model_fn, layer_id, df, X, y, model_name, source_country, target_country, source_runid, target_runid, y_attribute):
  b = 1
  cnn_model_fn = os.path.join('/'.join(cb_model_fn.split('/')[:-b]), 'model.h5') 
  print(cnn_model_fn)
  model = load_model(cnn_model_fn, custom_objects={'RandomColorDistortion':RandomColorDistortion()})
  y_pred = model.predict(X) 
  del(model)
  return y_pred

def ccp_cnn(model_fn, model_name, source_country, target_country, root, years, dhsloc, ttype, is_reg, viirsnorm, y_attribute, features_source, timevar, offlineaug, layer_id, cnn_name):
  df_performance = pd.DataFrame()
  df_residuals = pd.DataFrame()
  
  # Predictions
  # test set does not get augmented (that is only for the training set)
  data = Data(root=root, years=years, dhsloc=dhsloc, traintype=ttype, epoch=None, rs=None, fold=None, model_name=None, offaug=offlineaug, cnn_path=None, isregression=is_reg)
  for _, test, path, target_runid, target_rs in data.iterate_train_test():
    source_runid = int(model_fn.split('/epoch')[-1].split('-rs')[0])
    source_rs = int(model_fn.split('-rs')[-1].split('/')[0])
    np.random.seed(source_rs)
    set_seed(source_rs)
    
    path = os.path.join(OUTPUT, model_name.replace('$',''), source_country, target_country, f"s{source_runid}", f"t{target_runid}")
    ios.validate_path(path)
    fn_pred = os.path.join(path,'test_pred_cnn.csv')
    if ios.exists(fn_pred):
      tmp = ios.load_csv(fn_pred)
      y_pred = tmp.loc[:,[f"pred_{c}" for c in y_attribute.split(',')]].values
      y = tmp.loc[:,[f"true_{c}" for c in y_attribute.split(',')]].values
    else:
      # build fmaps with source model
      X, y = data.cnn_get_X_y(df=test, y_attribute=y_attribute)
      y_pred = cnn_predict(model_fn, layer_id, test, X, y, model_name, source_country, target_country, source_runid, target_runid, y_attribute)
      save_prediction(y_pred, y, test, fn_pred, y_attribute)

    ## residuals
    tm, ts = y[:,0], y[:,1]
    pm, ps = y_pred[:,0], y_pred[:,1]
    rm = tm - pm
    rs = ts - ps

    n = y_pred.shape[0]
    obj = {'source_country':[source_country]*n, 'source_model':[model_name]*n, 'source_runid':[source_runid]*n, 'source_rs':[source_rs]*n,
           'target_country':[target_country]*n, 'target_runid':[target_runid]*n, 'target_rs':[target_rs]*n, 
           'true_mean':tm, 'true_std':ts,
           'pred_mean':pm, 'pred_std':ps,
           'residual_mean':rm, 'residual_std':rs}
    df_residuals = pd.concat([df_residuals, pd.DataFrame(obj)], ignore_index=True)

    ## performance
    col=0
    r2_mean = r2_score(y[:,col],y_pred[:,col])
    mse_mean = mean_squared_error(y[:,col],y_pred[:,col])
    rmse_mean = mean_squared_error(y[:,col],y_pred[:,col], squared=False)
    nrmse_mean = rmse_mean / np.std(y[:,col])

    col=1
    r2_std = r2_score(y[:,col],y_pred[:,col])
    mse_std = mean_squared_error(y[:,col],y_pred[:,col])
    rmse_std = mean_squared_error(y[:,col],y_pred[:,col], squared=False)
    nrmse_std = rmse_std / np.std(y[:,col])

    r2 = r2_score(y,y_pred)
    mse = mean_squared_error(y,y_pred)
    rmse = mean_squared_error(y,y_pred, squared=False)
    nrmse = (nrmse_mean+nrmse_std)/2

    obj = {'source_country':source_country, 'source_model':model_name, 'source_runid':source_runid, 'source_rs':source_rs,
           'target_country':target_country, 'target_runid':target_runid, 'target_rs':target_rs, 
           'r2':r2, 'mse':mse, 'rmse':rmse, 'nrmse':nrmse,
           'r2_mean':r2_mean, 'mse_mean':mse_mean, 'rmse_mean':rmse_mean, 'nrmse_mean':nrmse_mean,
           'r2_std':r2_std, 'mse_std':mse_std, 'rmse_std':rmse_std, 'nrmse_std':nrmse_std}
    df_performance = pd.concat([df_performance, pd.DataFrame(obj, index=[1])], ignore_index=True)

  return df_performance, df_residuals

#################################################################################
# Predictionc using CNN
#################################################################################
def get_fmap(cb_model_fn, layer_id, X, model_name, source_country, target_country, source_runid, target_runid):
  path = os.path.join(OUTPUT, model_name.replace('$',''), source_country, target_country, f"s{source_runid}", f"t{target_runid}")
  ios.validate_path(path)
  
  fn_fmap = os.path.join(path,'fmap_test.npz')
  if ios.exists(fn_fmap):
    fmap = ios.load_array(fn_fmap)['arr_0']
  else:
    b = 2 if model_name in [MODELS['cnn+catboost'],MODELS['cnn_aug+catboost']] else 3 if model_name in [MODELS['cnn+weighted_catboost'],MODELS['cnn_aug+weighted_catboost']] else None
    if b is None:
      raise Exception("Something went wrong with b in get_fmap")
    cnn_model_fn = os.path.join('/'.join(cb_model_fn.split('/')[:-b]), 'model.h5') 
    print(cnn_model_fn)
    model = load_model(cnn_model_fn, custom_objects={'RandomColorDistortion':RandomColorDistortion()})
    remodel = Model(inputs=model.inputs, outputs=model.layers[layer_id].output)
    fmap = remodel.predict(X)
    savez_compressed(fn_fmap, fmap) 
  return fmap
    
def ccp_cnn_cb(model_fn, model_name, source_country, target_country, root, years, dhsloc, ttype, is_reg, viirsnorm, y_attribute, features_source, timevar, offlineaug, layer_id, cnn_name):
  df_performance = pd.DataFrame()
  df_residuals = pd.DataFrame()
  
  # Predictions
  # test set does not get augmented (that is only for the training set)
  data = Data(root=root, years=years, dhsloc=dhsloc, traintype=ttype, epoch=None, rs=None, fold=None, model_name=None, offaug=offlineaug, cnn_path=None, isregression=is_reg)
  data.load_metadata(viirsnorm=viirsnorm)
  
  for _, test, path, target_runid, target_rs in data.iterate_train_test():
    source_runid = int(model_fn.split('/epoch')[-1].split('-rs')[0])
    source_rs = int(model_fn.split('-rs')[-1].split('/')[0])
    
    path = os.path.join(OUTPUT, model_name.replace('$',''), source_country, target_country, f"s{source_runid}", f"t{target_runid}")
    ios.validate_path(path)
    fn_pred = os.path.join(path,'test_pred_cnn_cb.csv')

    if ios.exists(fn_pred):
      tmp = ios.load_csv(fn_pred)
      y_pred = tmp.loc[:,[f"pred_{c}" for c in y_attribute.split(',')]].values
      y = tmp.loc[:,[f"true_{c}" for c in y_attribute.split(',')]].values
    else:
      np.random.seed(source_rs)
      set_seed(source_rs)

      # build fmaps with source cnn model
      X, y = data.cnn_get_X_y(df=test, y_attribute=y_attribute)
      fmaps = get_fmap(model_fn, layer_id, X, model_name, source_country, target_country, source_runid, target_runid)
      del(X)
      del(y)

      # building features X's for catboost
      X, y, feature_names = data.metadata_get_X_y(df=test, y_attribute=y_attribute, fmaps=fmaps, offlineaug=offlineaug, features_source=features_source, timevar=timevar)
      print(type(X), X.shape, type(y), y.shape, feature_names)

      # model
      model = CatBoostRegressor()
      model.load_model(model_fn, format='json')
      y_pred = model.predict(X)
      save_prediction(y_pred, y, test, fn_pred, y_attribute)
      del(model)
      
    ## residuals
    tm, ts = y[:,0], y[:,1]
    pm, ps = y_pred[:,0], y_pred[:,1]
    rm = tm - pm
    rs = ts - ps

    n = y_pred.shape[0]
    obj = {'source_country':[source_country]*n, 'source_model':[model_name]*n, 'source_runid':[source_runid]*n, 'source_rs':[source_rs]*n,
           'target_country':[target_country]*n, 'target_runid':[target_runid]*n, 'target_rs':[target_rs]*n, 
           'true_mean':tm, 'true_std':ts,
           'pred_mean':pm, 'pred_std':ps,
           'residual_mean':rm, 'residual_std':rs}
    df_residuals = pd.concat([df_residuals, pd.DataFrame(obj)], ignore_index=True)

    ## performance
    col=0
    r2_mean = r2_score(y[:,col],y_pred[:,col])
    mse_mean = mean_squared_error(y[:,col],y_pred[:,col])
    rmse_mean = mean_squared_error(y[:,col],y_pred[:,col], squared=False)
    nrmse_mean = rmse_mean / np.std(y[:,col])

    col=1
    r2_std = r2_score(y[:,col],y_pred[:,col])
    mse_std = mean_squared_error(y[:,col],y_pred[:,col])
    rmse_std = mean_squared_error(y[:,col],y_pred[:,col], squared=False)
    nrmse_std = rmse_std / np.std(y[:,col])

    r2 = r2_score(y,y_pred)
    mse = mean_squared_error(y,y_pred)
    rmse = mean_squared_error(y,y_pred, squared=False)
    nrmse = (nrmse_mean+nrmse_std)/2

    obj = {'source_country':source_country, 'source_model':model_name, 'source_runid':source_runid, 'source_rs':source_rs,
           'target_country':target_country, 'target_runid':target_runid, 'target_rs':target_rs, 
           'r2':r2, 'mse':mse, 'rmse':rmse, 'nrmse':nrmse,
           'r2_mean':r2_mean, 'mse_mean':mse_mean, 'rmse_mean':rmse_mean, 'nrmse_mean':nrmse_mean,
           'r2_std':r2_std, 'mse_std':mse_std, 'rmse_std':rmse_std, 'nrmse_std':nrmse_std}
    df_performance = pd.concat([df_performance, pd.DataFrame(obj, index=[1])], ignore_index=True)
    
  return df_performance, df_residuals


################################################################################# 
# PRedictions using CatBoost
#################################################################################
def ccp_cb(model_fn, model_name, source_country, target_country, root, years, dhsloc, ttype, is_reg, viirsnorm, y_attribute, features_source, timevar):
  offlineaug = False
  layer_id = None
  cnn_name = None
  fmaps = None
  
  df_performance = pd.DataFrame()
  df_residuals = pd.DataFrame()
  
  # Predictions
  data = Data(root=root, years=years, dhsloc=dhsloc, traintype=ttype, epoch=None, rs=None, fold=None, model_name=None, offaug=offlineaug, cnn_path=None, isregression=is_reg)
  data.load_metadata(viirsnorm=viirsnorm)
  for _, test, path, target_runid, target_rs in data.iterate_train_test():
    source_runid = int(model_fn.split('/epoch')[-1].split('-rs')[0])
    source_rs = int(model_fn.split('-rs')[-1].split('/')[0])
      
    path = os.path.join(OUTPUT, model_name.replace('$',''), source_country, target_country, f"s{source_runid}", f"t{target_runid}")
    ios.validate_path(path)
    fn_pred = os.path.join(path,'test_pred_cb.csv')

    if ios.exists(fn_pred):
      tmp = ios.load_csv(fn_pred)
      y_pred = tmp.loc[:,[f"pred_{c}" for c in y_attribute.split(',')]].values
      y = tmp.loc[:,[f"true_{c}" for c in y_attribute.split(',')]].values
    else:
      # building features X's for catboost
      X, y, feature_names = data.metadata_get_X_y(df=test, y_attribute=y_attribute, fmaps=fmaps, offlineaug=offlineaug, features_source=features_source, timevar=timevar)
      print(type(X), X.shape, type(y), y.shape, feature_names)

      np.random.seed(source_rs)
      model = CatBoostRegressor()
      model.load_model(model_fn, format='json')
      y_pred = model.predict(X) 
      save_prediction(y_pred, y, test, fn_pred, y_attribute)
      del(model)
      
    ## residuals
    tm, ts = y[:,0], y[:,1]
    pm, ps = y_pred[:,0], y_pred[:,1]
    rm = tm - pm
    rs = ts - ps

    n = y_pred.shape[0]
    obj = {'source_country':[source_country]*n, 'source_model':[model_name]*n, 'source_runid':[source_runid]*n, 'source_rs':[source_rs]*n,
           'target_country':[target_country]*n, 'target_runid':[target_runid]*n, 'target_rs':[target_rs]*n, 
           'true_mean':tm, 'true_std':ts,
           'pred_mean':pm, 'pred_std':ps,
           'residual_mean':rm, 'residual_std':rs}
    df_residuals = pd.concat([df_residuals, pd.DataFrame(obj)], ignore_index=True)

    ## performance
    col=0
    r2_mean = r2_score(y[:,col],y_pred[:,col])
    mse_mean = mean_squared_error(y[:,col],y_pred[:,col])
    rmse_mean = mean_squared_error(y[:,col],y_pred[:,col], squared=False)
    nrmse_mean = rmse_mean / np.std(y[:,col])

    col=1
    r2_std = r2_score(y[:,col],y_pred[:,col])
    mse_std = mean_squared_error(y[:,col],y_pred[:,col])
    rmse_std = mean_squared_error(y[:,col],y_pred[:,col], squared=False)
    nrmse_std = rmse_std / np.std(y[:,col])

    r2 = r2_score(y,y_pred)
    mse = mean_squared_error(y,y_pred)
    rmse = mean_squared_error(y,y_pred, squared=False)
    nrmse = (nrmse_mean+nrmse_std)/2

    obj = {'source_country':source_country, 'source_model':model_name, 'source_runid':source_runid, 'source_rs':source_rs,
           'target_country':target_country, 'target_runid':target_runid, 'target_rs':target_rs, 
           'r2':r2, 'mse':mse, 'rmse':rmse, 'nrmse':nrmse,
           'r2_mean':r2_mean, 'mse_mean':mse_mean, 'rmse_mean':rmse_mean, 'nrmse_mean':nrmse_mean,
           'r2_std':r2_std, 'mse_std':mse_std, 'rmse_std':rmse_std, 'nrmse_std':nrmse_std}
    df_performance = pd.concat([df_performance, pd.DataFrame(obj, index=[1])], ignore_index=True)
    
  return df_performance, df_residuals


#################################################################################
#
#################################################################################

if __name__ == '__main__':
  run()