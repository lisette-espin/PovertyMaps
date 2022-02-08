#####################################################################################################
# DEPENDENCIES
#####################################################################################################
import os
import gc
import glob
import time
import json

import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import norm, skew
from scipy import stats
from numpy import savez_compressed
from scipy.stats import pearsonr
import xgboost as xgb

import numpy as np
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras import regularizers
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from census.dhsmis import DHSMIS
from utils import viz

#####################################################################################################
# FUNCTIONS
#####################################################################################################
include_fmaps = False

def run(root, years, dhsloc, traintype, isregression, viirsnorm):
  run(root, years, dhsloc, traintype, isregression, viirsnorm)
  validate(root, years, dhsloc, traintype, isregression)
  
def train(root, years, dhsloc, traintype, isregression, viirsnorm):
  data = Data(root=root, years=years, dhsloc=dhsloc, traintype=traintype, isregression=isregression)
  data.load_metadata(viirsnorm)
  
  for train, val, test, path, runid, rs, fold in data.iterate_train_val_test(tune_img=False):

    ### Some large rs values throw an exception.
    ### so, we reduce it by /10.
    while True:
      try:
        _ = xgb.XGBRegressor(random_state=rs)
        break
      except:
        rs = int(rs/10)

    ### init
    print("==========================================")
    print(f"1. LOADING: {runid}-{fold} ({path})")

    ### tunning
    df_evaluation = get_evaluation_file(path, rs, runid, kfolds, include_fmaps, features_source)
    tmp = df_evaluation.query(f"loss_fold{fold} in @NONE")
    if tmp.shape[0]==0:
      continue

    ### 1. Train, val, test sets
    X_train, y_train, feature_names = data.metadata_get_X_y(train, y_attributes, None, offlineaug, features_source)
    X_val, y_val, _ = data.metadata_get_X_y(val, y_attributes, None, False, features_source)

    print(X_train.shape, y_train.shape, len(feature_names))
    print(X_val.shape, y_val.shape)

    for i,row in tqdm(tmp.iterrows(), total=tmp.shape[0]):
      params = json.loads(row.params.replace("'",'"'))

      # fit
      start = time.time()
      model = MultiOutputRegressor(xgb.XGBRegressor(objective=params['objective'], 
                                                    eval_metric=params['eval_metric'],
                                                    n_estimators=params['n_estimators'],
                                                    max_depth=params['max_depth'],
                                                    learning_rate=params['learning_rate'],
                                                    booster=params['booster'],
                                                    gamma=params['gamma'],
                                                    min_child_weight=params['min_child_weight'],
                                                    max_delta_step=params['max_delta_step'],
                                                    subsample=params['subsample'],
                                                    colsample_bytree=params['colsample_bytree'],
                                                    colsample_bylevel=params['colsample_bylevel'],
                                                    colsample_bynode=params['colsample_bynode'],
                                                    reg_lambda=params['reg_lambda'],
                                                    reg_alpha=params['reg_alpha'],
                                                    gpu_id=params['gpu_id'],
                                                    tree_method=params['tree_method'],
                                                    random_state=params['random_state'],
                                                    )).fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
      fit_time = time.time() - start

      #eval
      start = time.time()
      y_pred = model.predict(X_val) 
      if np.isnan(y_pred).sum() > 0:
        y_pred = np.nan_to_num(y_pred, nan=-100)
        print(y_pred)
      mse = mean_squared_error(y_val, y_pred)
      eval_time = time.time() - start

      # update
      update_evaluation(df_evaluation, i, mse, fit_time, eval_time, path, include_fmaps, features_source)
      
def validation(root, years, dhsloc, traintype, isregression):
  data = Data(root=root, years=years, dhsloc=dhsloc, traintype=traintype, isregression=isregression)
  data.load_metadata(viirsnorm)
  
  for train, test, path, runid, rs in data.iterate_train_test():
    print("==========================================")
    print(f"1. LOADING: {runid} ({path})")

    ### 1. Train, val, test sets
    print('train:')
    X_train, y_train, feature_names = data.metadata_get_X_y(train, y_attributes, None, offlineaug, features_source)
    print('test:')
    X_test, y_test, _ = data.metadata_get_X_y(test, y_attributes, None, False, features_source)

    ### hyper-params
    # summary
    df_evaluation = get_evaluation_file(path, rs, runid, kfolds, include_fmaps, features_source)
    if df_evaluation.mean_loss.isnull().values.any():
      sumarize_evaluation(df_evaluation, path,  include_fmaps, features_source)
    
    # best
    best = df_evaluation.query("rank==1").iloc[0]
    params = json.loads(best.params.replace("'",'"'))
    print(params)

    model = MultiOutputRegressor(xgb.XGBRegressor(objective=params['objective'], 
                                                  eval_metric=params['eval_metric'],
                                                  n_estimators=params['n_estimators'],
                                                  max_depth=params['max_depth'],
                                                  learning_rate=params['learning_rate'],
                                                  booster=params['booster'],
                                                  gamma=params['gamma'],
                                                  min_child_weight=params['min_child_weight'],
                                                  max_delta_step=params['max_delta_step'],
                                                  subsample=params['subsample'],
                                                  colsample_bytree=params['colsample_bytree'],
                                                  colsample_bylevel=params['colsample_bylevel'],
                                                  colsample_bynode=params['colsample_bynode'],
                                                  reg_lambda=params['reg_lambda'],
                                                  reg_alpha=params['reg_alpha'],
                                                  gpu_id=params['gpu_id'],
                                                  tree_method=params['tree_method'],
                                                  random_state=params['random_state']
                                                  )).fit(X_train, y_train, verbose=False)
    # predict
    y_pred = model.predict(X_test)

    # save evaluation
    path_results = os.path.dirname(get_evaluation_fn(path, include_fmaps, features_source))
    prefix = f'layer{fmap_layer_id}-' if fmap_layer_id is not None else ''

    fn = os.path.join(path_results, f'{prefix}evaluation.json')
    save_evaluation(model, X_test, y_test, y_pred, fn)

    # plot true vs pred
    fnimg = os.path.join(path_results, f'{prefix}prediction_test.png')
    plot_pred(y_test, y_pred, f'test - run{runid}', include_fmaps, fnimg)

    # plot confusion
    tmp_confusion = get_confusion(test, y_pred)
    
    for norm in [True,False]:
      fnimg = os.path.join(path_results, f'{prefix}confusion{'_norm' if norm else ''}_test.png')
      viz.plot_confusion_matrix(tmp_confusion.ses_true, tmp_confusion.ses_pred, labels=None, norm=norm, fn=fnimg)
    
    # intersectional 
    for rural in [0,1]:
      tmp = tmp_confusion.query("dhs_rural==@rural").copy()
      for norm in [True,False]:
        fnimg = os.path.join(path_results, f'{prefix}confusion_{'rural' if rural else 'urban'}{'_norm' if norm else ''}_test.png')
        viz.plot_confusion_matrix(tmp.ses_true, tmp.ses_pred, labels=None, norm=norm, fn=fnimg)
    
def get_confusion(true, pred):
    tmp_prediction = true.loc[:,['dhs_mean_iwi','dhs_std_iwi','dhs_rural']]
    tmp_prediction.loc[:,'y_pred_mean'] = pred[:,0]
    tmp_prediction.loc[:,'y_pred_std'] = pred[:,1]
    _tmp = DHSMIS.add_ses_categories(tmp_prediction, 'dhs_mean_iwi')
    tmp_prediction.loc[:,'ses_true'] = _tmp.loc[:,'iwi_cat']
    _tmp = DHSMIS.add_ses_categories(tmp_prediction, 'y_pred_mean')
    tmp_prediction.loc[:,'ses_pred'] = _tmp.loc[:,'iwi_cat']
    return tmp_prediction
    
def get_all_hparams(rs=None):
  hparams = {'objective':['reg:squarederror'],
            'eval_metric':["rmse"],
            'n_estimators': [50,100,200,300,500],
            'max_depth': [3,4,5,6,7,8,9,10,12,15],
            'learning_rate': [0.01,0.05,0.1,0.15,0.2,0.25,0.3], #learning rate
            'booster': ['gbtree','dart'],
            'gamma': [0,0.1,0.3,0.5,1.0,2.0,3.0], #min_split_loss
            'min_child_weight':[0,1,3,5,7,10],
            'max_delta_step': [1,2,3,4,5,6,7,8,9,10],
            'subsample': [0.5,1],
            'colsample_bytree': [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95],
            'colsample_bylevel': [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95],
            'colsample_bynode': [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95],
            'reg_lambda': [0,1,1.5,2,2.5,3,3.5], # L2 reg
            'reg_alpha': [0,1,1.5,2,2.5,3,3.5], # L1 reg 
            'gpu_id':[0],
            'tree_method':['gpu_hist'],
            'random_state':[int(rs)]}

  combinations = np.prod([len(v) for v in hparams.values()])
  print("- All possible combinations hyper-params: ", combinations)
  n_iter = min(CNN_TUNING_ITER,int(round(combinations/2)))
  print("- # random candidates: ", n_iter)
  return hparams, n_iter, combinations

def get_evaluation_fn(path, include_fmaps=False, features_source='all'):
  fn = os.path.join(path, f'xgboost-{features_source}', 'tuning.csv')
  os.makedirs(os.path.dirname(fn), exist_ok=True)
  return fn

def get_evaluation_file(path, rs, runid, kfolds, include_fmaps, features_source):
  fn = get_evaluation_fn(path, include_fmaps, features_source)
  if ios.exists(fn):
    df = ios.load_csv(fn)
  else:
    # hparams
    hparams, n_iter, combinations = get_all_hparams(rs)
    # populate
    cols = ['rs','runid','mean_fit_time','std_fit_time','mean_eval_time','std_eval_time','params']
    cols.extend([f'param_{k}' for k,v in hparams.items()])
    cols.extend([f'loss_fold{k}' for k in np.arange(1,kfolds+1, 1)])
    cols.extend(['mean_loss','std_loss','rank'])
    df = pd.DataFrame(columns=cols)
    strhp = ['objective','eval_metric','booster','tree_method']
    
    while True:
      if df.shape[0] == n_iter:
        break
      hp = get_random_hyper_params_combination(hparams)
      q = " and ".join([f"param_{k}=='{v}'" if k in strhp else f"param_{k}=={v}" for k,v in hp.items()])
      if df.query(q).shape[0] == 0:
        obj = {f'param_{k}':f'{v}' if k in strhp else v for k,v in hp.items()}
        obj.update({"rs":rs})
        obj.update({"runid":runid})
        obj.update({"params":str(hp)})
        df = df.append(pd.DataFrame(obj, index=[0], columns=cols), ignore_index=True)
    ios.save_csv(df, fn)  
  return df

def update_evaluation(df, index, mse, fit_time, eval_time, path,include_fmaps, features_source):
  df.loc[index,f'loss_fold{fold}'] = mse
  df.loc[index,'mean_fit_time'] = f"{df.loc[index,'mean_fit_time']},{fit_time}" if df.loc[index,'mean_fit_time'] not in NONE else f"{fit_time}"
  df.loc[index,'mean_eval_time'] = f"{df.loc[index,'mean_eval_time']},{eval_time}" if df.loc[index,'mean_eval_time'] not in NONE else f"{eval_time}"
  ios.save_csv(df, get_evaluation_fn(path,include_fmaps, features_source), verbose=False)

def sumarize_evaluation(df, path, include_fmaps, features_source):
  if df.mean_loss.isnull().values.any():
    loss_cols = [c for c in df.columns if c.startswith('loss_fold')]
    df.loc[:,'mean_loss'] = df.apply(lambda row: np.mean([row[c] for c in loss_cols]), axis=1)
    df.loc[:,'std_loss'] = df.apply(lambda row: np.std([row[c] for c in loss_cols]), axis=1)
    df.loc[:,'rank'] = df.mean_loss.rank(na_option='bottom',ascending=True,pct=False)
    df.loc[:,'std_fit_time'] = df.mean_fit_time.apply(lambda c: np.std([float(t) for t in c.split(',') if t not in NONE]))
    df.loc[:,'mean_fit_time'] = df.mean_fit_time.apply(lambda c: np.mean([float(t) for t in c.split(',') if t not in NONE]))
    df.loc[:,'std_eval_time'] = df.mean_eval_time.apply(lambda c: np.std([float(t) for t in c.split(',') if t not in NONE]))
    df.loc[:,'mean_eval_time'] = df.mean_eval_time.apply(lambda c: np.mean([float(t) for t in c.split(',') if t not in NONE]))
    ios.save_csv(df, get_evaluation_fn(path,include_fmaps, features_source), verbose=False)

def get_random_hyper_params_combination(hparams):
  hp = {}
  for k,v in hparams.items():
    hp[k] = np.random.choice(v)
  return hp

  
#####################################################################################################
# main
#####################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
    parser.add_argument("-years", help="Years (comma-separated): 2016,2019", type=str, required=True)
    parser.add_argument("-dhsloc", help="Type of DHS location: none, cc, ccur, gc, gcur, rc", type=str, default='none')
    parser.add_argument("-traintype", help="Years to include in training: all, newest, oldest", type=str, default='none')
    parser.add_argument("-isregression", help="Is regression (boolean)", action='store_true')
    parser.add_argument("-viirsnorm", help="# Epochs (repeats)", type=int, action='store_true')

    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))
      
    start_time = time.time()
    run(args.r, args.years, args.dhsloc, args.traintype, args.isregression, args.viirsnorm)
    print("--- %s seconds ---" % (time.time() - start_time))