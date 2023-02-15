import os
import re
import sys
import glob
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns 
from tqdm import tqdm 
import geopandas as gpd
from scipy import stats
from matplotlib import rc
from itertools import product
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import shapiro
from scipy.stats import anderson
from scipy.stats import normaltest
from mycolorpy import colorlist as mcp
import matplotlib.image as mpimg
from collections import OrderedDict
import matplotlib.gridspec as gridspec
from numpy.polynomial import Polynomial
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from pandas.api.types import CategoricalDtype
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append("../libs")
from utils import ios
from utils import validations
from maps import geo
from ses.data import Data
from utils.constants import COUNTRIES

RUNS = 3
SIZE = 6
RECENCY_TEST = ['N300','O300']
POLYNOMIAL_DEGREE = 2
N_QUANTILES = 5
FEATURE_MAP = OrderedDict({'FBM':'De', 'FBMV':'Mo', 'FBP':'Po', 'NTLL':'Nl', 'OCI':'An', 'OSM':'In', 'GMSA':'Im'})

MODELS = OrderedDict({'catboost':'CB', 
                      'weighted_catboost':'CB$_w$', 
                      'cnn':'CNN', 
                      'cnn_aug':r'CNN$_a$',
                      'cnn+catboost':'CNN+CB', 
                      'cnn+weighted_catboost':'CNN+CB$_w$', 
                      'cnn_aug+catboost':r'CNN$_a$+CB', 
                      'cnn_aug+weighted_catboost':r'CNN$_a$+CB$_w$'})
CNN_CLASS = [v for k,v in MODELS.items() if k in ['cnn','cnn_aug']]
MODELS_BEST = [MODELS['weighted_catboost'], MODELS['cnn_aug'], MODELS['cnn_aug+weighted_catboost']]
MODELS_CB = [MODELS['catboost'], MODELS['weighted_catboost']]
MODEL_CNNA = MODELS['cnn_aug']

RECENCY = {'old-old':'O-O','old300':'O300','new-new':'N-N','new300':'N300','old-new':'O-N','combined':'ON'}
RECENCY_CLASS = [v for k,v in RECENCY.items()]
RECENCY_BEST = RECENCY['combined']
RELOCATION_BEST = 'none'
MODEL_BEST = MODELS['cnn_aug+weighted_catboost']

FEATURES = ['images','metadata','images+metadata','FBM','FBMV','FBP','NTLL','OCI','OSM']
BASIC_FEATURES = ['images','metadata']
MAIN_FEATURES = ['images','metadata','images+metadata']
IMAGE_FEATURE = ['images']
METADATA_FEATURES = ['metadata']

REC_TYPE = CategoricalDtype(categories=RECENCY_CLASS, ordered=True)
REL_TYPE = CategoricalDtype(categories=['none','rc','ruc'], ordered=True)
AUG_TYPE = CategoricalDtype(categories=['no','yes'], ordered=True)
MOD_TYPE = CategoricalDtype(categories=MODELS.values(), ordered=True)
FEA_TYPE = CategoricalDtype(categories=FEATURES, ordered=True)

SL_BEST_MODEL = 'CB$_w$'
UG_BEST_MODEL = 'CB'

# https://data.worldbank.org/indicator
WB_INDICATORS = {'Sierra Leone':{'gini':'35.7 (2018)', 'gdp_per_capita':'501, 522', 'gdp_growth':'6.1, 5.3'},
                 'Uganda':      {'gini':'42.7 (2019)', 'gdp_per_capita':'733, 771', 'gdp_growth':'4.8, 6.3'}}


####################################################################################################
# LOAD DATA
####################################################################################################

def load_gt_data(root, countries):
  df = pd.DataFrame()
  for country in countries:
    files = glob.glob(os.path.join(root,country,'results','features','clusters','*_cluster.csv'))
    if len(files)==0:
      print(f"[ERROR] supmat.py | load_gt_data | no file found for {country}")
      raise Exception("No files found!")
    
    for fn in files:
      if len(fn.split("_")) == 4:
        print(f"[INFO] {fn} loaded.")
        tmp = ios.load_csv(fn)
        tmp.loc[:,'countryname'] = country
        df = pd.concat([df,tmp], ignore_index=True)
        break
        
  print(f"[INFO] {df.shape[0]} records, {df.shape[1]} columns.")
  return df

def get_performance_from_predictions(df_rs, output=None):
  from sklearn.metrics import mean_squared_error
  
  # load data
  fn = os.path.join(output, 'performance.csv') if output is not None else None
  if fn is not None and ios.exists(fn):
    df_dr = ios.load_csv(fn)
  else:
    # create
    df_dr = pd.DataFrame()

    for group, df in df_rs.groupby(['country','model','relocation','recency','weighted',
                                    'augmented','features','epoch']):
        n = df.shape[0]

        if n > 0:
            tmp = df.drop(columns=[c for c in df.columns if c in ['rural','year','cluster_id'] or c.startswith('pred') or c.startswith('true') or c.startswith('residual') or c.startswith('population')]).copy()
            tmp.drop_duplicates(inplace=True)
            
            # for each output
            for var in ['mean','std']:
                sum_squared_residuals = df.apply(lambda row: (row[f'true_{var}']-row[f"pred_{var}"])**2, 
                                                 axis=1).sum()
                true_mean = df.loc[:,f'true_{var}'].mean()
                true_std = df.loc[:,f'true_{var}'].std()

                # MSE
                metric = 'mse'
                MSE = sum_squared_residuals / n
                tmp.loc[:,f"{metric}_{var}_wi"] = MSE

                # RMSE
                metric = 'rmse'
                RMSE = np.sqrt(MSE)
                tmp.loc[:,f"{metric}_{var}_wi"] = RMSE
                
                mse = mean_squared_error(df.loc[:,f'true_{var}'], df.loc[:,f'pred_{var}'], squared=True)
                rmse = mean_squared_error(df.loc[:,f'true_{var}'], df.loc[:,f'pred_{var}'], squared=False)
                
                # NRMSE
                metric = 'nrmse'
                NRMSE = RMSE / true_std
                tmp.loc[:,f"{metric}_{var}_wi"] = NRMSE

                # R2
                metric = 'r2'
                sum_squared_mean_diff = df.apply(lambda row: (row[f'true_{var}']-true_mean)**2, axis=1).sum()
                R2 = 1 - ((sum_squared_residuals) / (sum_squared_mean_diff))
                tmp.loc[:,f"{metric}_{var}_wi"] = R2

                
            # for all outputs at once (average)
            for metric in ['mse','rmse','nrmse','r2']:
                tmp.loc[:,metric] = tmp.apply(lambda row: (row[f"{metric}_mean_wi"]+row[f"{metric}_std_wi"])/2,axis=1)
            df_dr = pd.concat([df_dr, tmp], ignore_index=True)

    # set column order
    cols = ['gt_config', 'relocation', 'recency', 'weighted', 'augmented',
            'features', 'epoch', 'rs', 'model', 'fn', 
            'r2', 'r2_mean_wi','r2_std_wi', 
            'rmse', 'rmse_mean_wi', 'rmse_std_wi', 
            'nrmse', 'nrmse_mean_wi', 'nrmse_std_wi', 
            'mse','mse_mean_wi', 'mse_std_wi', 'country']
    df_dr = df_dr.loc[:,cols]
    
    if fn is not None:
      ios.save_csv(df_dr, fn)
      
  return df_dr

def get_performance(root=None, countries=None, output=None):
  
  # load data
  fn = os.path.join(output, 'performance.csv') if output is not None else None
  if fn is not None and ios.exists(fn):
    df = ios.load_csv(fn)
  else:
    df = pd.DataFrame()
    for r,country in enumerate(countries):
      tmp = summary_from_logs(root, country)
      if tmp is None or tmp.shape[0]==0:
        continue
      tmp.loc[:,'country'] = country
      df = pd.concat([df,tmp])
  
    if fn is not None:
      ios.save_csv(df, fn)
    
  # format
  df.loc[:,'recency'] = df.loc[:,'recency'].astype(REC_TYPE)    
  df.loc[:,'relocation'] = df.loc[:,'relocation'].astype(REL_TYPE)
  df.loc[:,'augmented'] = df.loc[:,'augmented'].astype(AUG_TYPE)
  df.loc[:,'features'] = df.loc[:,'features'].astype(FEA_TYPE)
  df.loc[:,'model'] = df.loc[:,'model'].astype(MOD_TYPE)
    
  # Missing results:
  print(f"[INFO] {df.shape[0]} records, {df.shape[1]} columns")
  g = df.groupby(['country','model','gt_config'])
  tmp = g.filter(lambda x: len(x) < RUNS)[['country','model','gt_config','epoch']]
  if tmp.shape[0] > 0:
    print("[WARNING] Missing values.")
    print(tmp)
        
  return df

def get_summary_performance(df, only_best=True, metric='mse', output=None):
  
  if metric not in ['r2','mse','rmse','nrmse']:
    raise Exception("Metric not valid.")
  
  ascending = 'mse' in metric
  
  fn = os.path.join(output, f"summary_performance_{metric}_{'best' if only_best else 'all'}.csv") if output is not None else None
  if fn is not None and ios.exists(fn):
    df_baselines = ios.load_csv(fn)
  else:
    data = df.query("recency not in  @RECENCY_TEST").copy()
    print(f"[INFO] {data.shape[0]} valid records, {data.shape[1]} columns.")
    
    #recency
    group_rec = ['country','model','relocation','augmented','weighted','recency']
    df_rec = data.query("augmented=='no' and weighted=='no' and features in @BASIC_FEATURES").copy()
    df_rec = df_rec.query("relocation==@RELOCATION_BEST") if only_best else df_rec
    df_rec = df_rec.groupby(group_rec).agg({metric:['mean','std'], 
                                            f"{metric}_mean_wi":['mean','std'], 
                                            f"{metric}_std_wi":['mean','std'],}).reset_index()
    df_rec.columns = [col[0] if col[1]=='' else '_'.join(col).strip() for col in df_rec.columns.values]
    
    # relocation
    group_rel = ['country','model','recency','augmented','weighted','relocation']
    df_rel = data.query("augmented=='no' and weighted=='no' and features in @BASIC_FEATURES").copy()
    df_rel = df_rel.query("recency==@RECENCY_BEST") if only_best else df_rel
    df_rel = df_rel.groupby(group_rel).agg({metric:['mean','std'], 
                                            f"{metric}_mean_wi":['mean','std'], 
                                            f"{metric}_std_wi":['mean','std'],}).reset_index()
    df_rel.columns = [col[0] if col[1]=='' else '_'.join(col).strip() for col in df_rel.columns.values]
    
    # augmented
    group_aug = ['country','model','recency','relocation','weighted','augmented']
    df_aug = data.query("model in @CNN_CLASS and weighted=='no' and features in @IMAGE_FEATURE").copy()
    df_aug = df_aug.query("recency==@RECENCY_BEST and relocation==@RELOCATION_BEST") if only_best else df_aug
    df_aug = df_aug.groupby(group_aug).agg({metric:['mean','std'], 
                                            f"{metric}_mean_wi":['mean','std'], 
                                            f"{metric}_std_wi":['mean','std'],}).reset_index()
    df_aug.columns = [col[0] if col[1]=='' else '_'.join(col).strip() for col in df_aug.columns.values]
    
    # weighted samples
    group_wei = ['country','model','recency','relocation','augmented','weighted']
    df_wei = data.query("model in @MODELS_CB and features in @METADATA_FEATURES").copy()
    df_wei = df_wei.query("recency==@RECENCY_BEST and relocation==@RELOCATION_BEST") if only_best else df_wei
    df_wei = df_wei.groupby(group_wei).agg({metric:['mean','std'], 
                                            f"{metric}_mean_wi":['mean','std'], 
                                            f"{metric}_std_wi":['mean','std'],}).reset_index()
    df_wei.columns = [col[0] if col[1]=='' else '_'.join(col).strip() for col in df_wei.columns.values]
    
    # condensed aggregated
    tmp = df_rec.groupby(['country',group_rec[-1]]).mean().round(2).reset_index().sort_values(['country',f'{metric}_mean'],ascending=[True,ascending])
    tmp.loc[:,'kind'] = 'recency'
    tmp.rename(columns={'recency':'configuration'}, inplace=True)
    df_baselines = tmp.copy()
    
    tmp = df_rel.groupby(['country',group_rel[-1]]).mean().round(2).reset_index().sort_values(['country',f'{metric}_mean'],ascending=[True,ascending])
    tmp.loc[:,'kind'] = 'relocation'
    tmp.rename(columns={'relocation':'configuration'}, inplace=True)
    df_baselines = pd.concat([df_baselines,tmp])
    
    tmp = df_aug.groupby(['country',group_aug[-1]]).mean().round(2).reset_index().sort_values(['country',f'{metric}_mean'],ascending=[True,ascending])
    tmp.loc[:,'kind'] = 'augmented'
    tmp.rename(columns={'augmented':'configuration'}, inplace=True)
    df_baselines = pd.concat([df_baselines,tmp])
    
    tmp = df_wei.groupby(['country',group_wei[-1]]).mean().round(2).reset_index().sort_values(['country',f'{metric}_mean'],ascending=[True,ascending])
    tmp.loc[:,'kind'] = 'weighted'
    tmp.rename(columns={'weighted':'configuration'}, inplace=True)
    df_baselines = pd.concat([df_baselines,tmp])
    
    df_baselines.dropna(inplace=True)
    
    if fn is not None:
      ios.save_csv(df_baselines, fn)
      
  return df_baselines
      
  
def get_residuals(root, countries, output=None):
  
  fn = os.path.join(output, f"residuals.csv") if output is not None else None
  if fn is not None and ios.exists(fn):
    df = ios.load_csv(fn)
  else:
    df = pd.DataFrame()
    for country in countries:
      tmp = summary_from_predictions(root, country)
      tmp.loc[:,'country'] = country
      df = pd.concat([df,tmp], ignore_index=True)
    if fn is not None:
      ios.save_csv(df, fn)
    
  # format
  df.loc[:,'recency'] = df.loc[:,'recency'].astype(REC_TYPE)    
  df.loc[:,'relocation'] = df.loc[:,'relocation'].astype(REL_TYPE)
  df.loc[:,'augmented'] = df.loc[:,'augmented'].astype(AUG_TYPE)
  df.loc[:,'features'] = df.loc[:,'features'].astype(FEA_TYPE)
  df.loc[:,'model'] = df.loc[:,'model'].astype(MOD_TYPE)
  df.loc[:,'country'] = df.loc[:,'country'].astype(CategoricalDtype(categories=sorted(df.country.unique()), ordered=True))
  
  
  return df
      
def get_ground_truth(root, countries, output=None):
  fn = os.path.join(output, f"ground_truth.csv") if output is not None else None
  if fn is not None and ios.exists(fn):
    df = ios.load_csv(fn)
  else:
    df = pd.DataFrame()
    for country in countries:
      years = COUNTRIES[country]['years']
      metadata = Data(root=os.path.join(root,country), years=years, dhsloc=RELOCATION_BEST, traintype='all')
      metadata.load_metadata(viirsnorm=True, dropyear=False)
      tmp = metadata.df_clusters.copy()
      tmp.loc[:,'country'] = country
      df = pd.concat([df, tmp], ignore_index=True)
    ios.save_csv(df, fn)
      
  return df
  
  
####################################################################################################
# HANDLERS
####################################################################################################

def gini(array):
  """Calculate the Gini coefficient of a numpy array."""
  # based on bottom eq:
  # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
  # from:
  # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
  # All values are treated equally, arrays must be 1d:
  # https://github.com/oliviaguest/gini/blob/master/gini.py
  array = array.flatten()
  if np.amin(array) < 0:
    # Values cannot be negative:
    array -= np.amin(array)
  # Values cannot be 0:
  array += 0.0000001
  # Values must be sorted:
  array = np.sort(array)
  # Index per array element:
  index = np.arange(1,array.shape[0]+1)
  # Number of array elements:
  n = array.shape[0]
  # Gini coefficient:
  return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def get_recency_code(gt_config):
  
  if len(gt_config.split("_"))==4 and "_oldest_" in gt_config:
    r = RECENCY['old-new']
  elif len(gt_config.split("_"))==4 and "_all_" in gt_config:
    r = RECENCY['combined']
  elif '2016' in re.findall(r'\d+', gt_config) and '_all300_' in gt_config:
    r = RECENCY['old300']
  elif '_all300_' in gt_config:
    r = RECENCY['new300']
  elif '2016' in re.findall(r'\d+', gt_config) and '_all_' in gt_config:
    r = RECENCY['old-old']
  elif '_all_' in gt_config:
    r = RECENCY['new-new']
  else:
    r = None
  return r

def polynomial_regression(X, y, order=1, confidence=95, num=100):
    # https://stackoverflow.com/questions/22852244/how-to-get-the-numerical-fitting-results-when-plotting-a-regression-in-seaborn
    # https://stackoverflow.com/questions/49179463/python-quadratic-correlation
    confidence = 1 - ((1 - (confidence / 100)) / 2)
    pfit = np.polyfit(X, y, order)
    # print(pfit) # coefficients
    y_model = np.polyval(pfit, X)
    rmse = mean_squared_error(y, y_model, squared=False)
    r2 = r2_score(y, y_model)
    residual = y - y_model
    n = X.size                     
    m = 2                          
    dof = n - m  
    t = stats.t.ppf(confidence, dof) 
    std_error = (np.sum(residual**2) / dof)**.5
    X_line = np.linspace(np.min(X), np.max(X), num)
    y_line = np.polyval(np.polyfit(X, y, order), X_line)
    ci = t * std_error * (1/n + (X_line - np.mean(X))**2 / np.sum((X - np.mean(X))**2))**.5
    return X_line, y_line, ci, rmse, r2
  
def get_model_name(fn):
#   cnnxgb = '_cnn_' in fn and 'xgb' in fn
#   model = 'cnn+catboost' if cnnxgb and 'noaug_' in fn else \
#           'cnn_aug+weighted_catboost' if cnnxgb and 'offaug_' in fn else \
#           'cnn' if 'noaug_' in fn else \
#           'cnn_aug' if 'offaug_' in fn else \
#           'weighted_catboost' if 'weighted' in fn else \
#           'catboost'
  
  model =  'cnn' if 'cnn' in fn and 'xgb-' not in fn and 'noaug_' in fn else \
           'cnn+catboost' if 'cnn' in fn and 'xgb-' in fn and 'noaug_' in fn and 'weighted' not in fn else \
           'cnn+weighted_catboost' if 'cnn' in fn and 'xgb-' in fn and 'noaug_' in fn and 'weighted' in fn else \
           'catboost' if 'cnn' not in fn and 'xgb-' in fn and 'weighted' not in fn else \
           'weighted_catboost' if 'cnn' not in fn and 'xgb-' in fn and 'weighted' in fn else \
           'cnn_aug' if 'cnn' in fn and 'xgb-' not in fn and 'noaug_' not in fn else \
           'cnn_aug+catboost' if 'cnn' in fn and 'xgb-' in fn and 'noaug_' not in fn and 'weighted' not in fn else \
           'cnn_aug+weighted_catboost' if 'cnn' in fn and 'xgb-' in fn and 'noaug_' not in fn and 'weighted' in fn else None
  
  return MODELS[model]
  
def get_augmented_flag(fn):
  flag = 'aug_' in fn and 'noaug_' not in fn
  return 'yes' if flag else 'no'

def get_weighted_flag(fn):
  flag = 'weighted' in fn
  return 'yes' if flag else 'no'

def get_features_type(fn):
  return 'images' if 'cnn' in fn and 'xgb' not in fn else \
         'metadata' if 'cnn' not in fn and 'xgb-all' in fn else \
         'images+metadata' if 'cnn' in fn and 'xgb-all' in fn else \
         fn.split('xgb-')[-1].split('/')[0] if 'xgb-' in fn else None


def summary_from_logs(root, country):
  files1 = glob.glob(os.path.join(root,country,'results','samples','*','epoch*-rs*','*','*.json')) # cnn or catboost
  files2 = glob.glob(os.path.join(root,country,'results','samples','*','epoch*-rs*','*','*','*.json')) # cnn + catboost or weighted_catboost
  files3 = glob.glob(os.path.join(root,country,'results','samples','*','epoch*-rs*','*','*','*','*.json')) # cnn + weighted_catboost
  
  files = files1.copy()
  files.extend(files2)
  files.extend(files3)
  
  files = set([fn for fn in files if fn.endswith('evaluation.json') or fn.endswith('log.json')])
  print(f"[INFO] {len(files)} files.")
  
  df = pd.DataFrame()
  for fn in files:
    try:
      
      data = ios.load_json(fn)
      
      gt_config = fn.split('/epoch')[0].split('/')[-1] 
      relocation = gt_config.split("_")[-1]
      recency = get_recency_code(gt_config)
      epoch = int(fn.split('/epoch')[-1].split('-rs')[0]) 
      rs = int(fn.split('-rs')[-1].split('/')[0]) 
      cnn = '_cnn_' in fn and 'xgb' not in fn and fn.endswith('log.json') 
      model = get_model_name(fn)
      augmented = get_augmented_flag(fn)
      weighted = get_weighted_flag(fn)
      features = get_features_type(fn)
      
      obj = {'gt_config':gt_config,
             'relocation':relocation,
             'recency':recency,
             'weighted':weighted,
             'augmented':augmented,
             'features':features,
             'epoch':epoch,
             'rs':rs,
             'model':model,
             'fn':fn.split('/samples/')[-1]}
      if cnn:
        #print("read cnn")
        obj.update({'r2':data['test_r2'], 'r2_mean_wi':float(data['test_y0_r2']), 'r2_std_wi':float(data['test_y1_r2']),
                    'rmse':data['test_rmse'], 'rmse_mean_wi':float(data['test_y0_rmse']), 'rmse_std_wi':float(data['test_y1_rmse']),
                    'mse':data['test_mse'], 'mse_mean_wi':float(data['test_y0_mse']), 'mse_std_wi':float(data['test_y1_mse']),
                   })
        
        
      else:
        #print("read xgb")
        obj.update({'r2':data['r2'], 'r2_mean_wi':float(data['r2_mean']), 'r2_std_wi':float(data['r2_std']),
                    'rmse':data['rmse'], 'rmse_mean_wi':float(data['rmse_mean']), 'rmse_std_wi':float(data['rmse_std']),
                    'mse':data['mse'], 'mse_mean_wi':float(data['mse_mean']), 'mse_std_wi':float(data['mse_std'])
                   })

       

      tmp = pd.DataFrame(obj, index=[1])
      df = pd.concat([df,tmp], ignore_index=True)
    except Exception as ex:
      pass
  
  return df

def summary_gt(root, country):
  files = glob.glob(os.path.join(root,country,'results','features','clusters','cluster.csv'))
  
def summary_from_predictions(root, country):
  # n_classes = validate_nclasses(n_classes)
  files1 = glob.glob(os.path.join(root,country,'results','samples','*','epoch*-rs*','*','test_pred_*.csv')) # cnn or catboost
  files2 = glob.glob(os.path.join(root,country,'results','samples','*','epoch*-rs*','*','*','test_pred_*.csv')) # cnn + catboost or weighted_catboost
  files3 = glob.glob(os.path.join(root,country,'results','samples','*','epoch*-rs*','*','*','*','test_pred_*.csv')) # cnn + weighted_catboost
  
  files = files1.copy()
  files.extend(files2)
  files.extend(files3)
  files = set(files)
  print(f"[INFO] {len(files)} files.")
  
  df = pd.DataFrame()
  for fn in files:
    try:
      
      gt_config = fn.split('/epoch')[0].split('/')[-1] #fn.split("/")[-4 -(1 if cnnxgb else 0)]
      relocation = gt_config.split("_")[-1]
      recency = get_recency_code(gt_config)
      epoch = int(fn.split('/epoch')[-1].split('-rs')[0]) #fn.split("/")[-3-(1 if cnnxgb else 0)].split("-rs")[0].replace("epoch","")
      rs = int(fn.split('-rs')[-1].split('/')[0]) #fn.split("/")[-3-(1 if cnnxgb else 0)].split("-rs")[-1]
      cnn = '_cnn_' in fn and 'xgb' not in fn and fn.endswith('log.json') 
      model = get_model_name(fn)
      augmented = get_augmented_flag(fn)
      weighted = get_weighted_flag(fn)
      features = get_features_type(fn)
      
      data = ios.load_csv(fn)
      
      # adding cluster metadata
      fn_data = glob.glob(os.path.join(fn.split('/epoch')[0],'*','data.csv'))[0]
      tmp = ios.load_csv(fn_data) 
      data = data.set_index('cluster_id').join(tmp[['cluster_id','cluster_year','cluster_number',
                                                    'cluster_rural','pplace_cluster_distance']].set_index('cluster_id'))
      
      # adding population
      fn_pop = glob.glob(os.path.join(fn.split('/samples')[0],'features',
                                      'clusters',f"{'_'.join(gt_config.split('_')[:-2])}*_population.csv"))[0]
      tmp = ios.load_csv(fn_pop) 
      data = data.join(tmp[['gtID','population_closest_tile','population_in_1.61km']].set_index('gtID'))
      
      tm = data.true_mean_wi.values
      pm = data.pred_mean_wi.values
      ts = data.true_std_wi.values
      ps = data.pred_std_wi.values
      year = data.cluster_year.apply(lambda v: '2016' if str(v)=='2016' else '2018-19')
      rural = data.cluster_rural.apply(lambda v:'rural' if v in [1,'1'] else 'urban')
      mr = data.apply(lambda row: row.true_mean_wi - row.pred_mean_wi, axis=1).values
      sr = data.apply(lambda row: row.true_std_wi - row.pred_std_wi, axis=1).values
      pop_closest = data.population_closest_tile.values
      pop_mile = data.loc[:,'population_in_1.61km'].values
      cluster_id = data.index.values
      # tmclass = data.apply(lambda row: int(row.true_mean_wi*n_classes/100), axis=1).values
      # pmclass = data.apply(lambda row: int(row.pred_mean_wi*n_classes/100), axis=1).values
      # tsclass = data.apply(lambda row: int(row.true_std_wi*n_classes/30), axis=1).values
      # psclass = data.apply(lambda row: int(row.pred_std_wi*n_classes/30), axis=1).values
      # tmclass = None
      # pmclass = None
      # tsclass = None
      # psclass = None
      s = mr.shape[0]
      
      obj = {'gt_config':[gt_config]*s,
            'relocation':[relocation]*s,
            'recency':[recency]*s,
            'epoch':[epoch]*s,
            'rural':rural,
            'year':year,
            'rs':[rs]*s,
            'features':[features]*s,
            'model':[model]*s,
            'weighted':[weighted]*s,
            'augmented':[augmented]*s,
            'cluster_id':cluster_id,
            'true_mean':tm,
            'pred_mean':pm,
            'true_std':ts,
            'pred_std':ps,
            'residual_mean':mr,
            'residual_std':sr,
            'population_closest_tile':pop_closest,
            'population_in_1mile':pop_mile,
            'fn':fn.split('/samples/')[-1]}
      
      tmp = pd.DataFrame(obj)
      df = pd.concat([df,tmp], ignore_index=True)
    except Exception as ex:
      print(f"[ERROR] summary_from_predictions | {ex} | {fn}")
      #pass
  
  return df


####################################################################################################
# Normal test
####################################################################################################

def print_normal_test_ground_truth(root, country_years):
  
  for country, years in country_years.items():
    print(f"==== {country},{years} =====")
    path = os.path.join(root, country)
    
    # data
    fn_clusters = ios.get_places_file(path, years=years, verbose=False)
    df_clusters = ios.load_csv(fn_clusters)
    fn_households = fn_clusters.replace('/clusters/','/households/').replace('_cluster.csv','_household.csv')
    df_households = ios.load_csv(fn_households)
    df_households.loc[:,'gtID'] = df_households.apply(lambda row:f"{row.ccode}{row.year:4d}{row.hv001:010}", axis=1)
    
    # quintiles
    labels = [f"Q{q}" for q in np.arange(1,N_QUANTILES+1,1)]
    df_clusters.loc[:,f'quintile'] = pd.qcut(df_clusters.mean_wi, N_QUANTILES, labels=labels, precision=0, retbins=False)
  
    is_gaussian = {}
    for q in labels:
      is_gaussian[q] = []
      qhh = df_clusters.query("quintile==@q").set_index('gtID').join(df_households[['gtID','hv001','wi']].set_index('gtID'))
      
      for (c,y), df in qhh.groupby(['hv001','year']):  
        x = df.wi
        if x.shape[0]<8:
          print(f"Cluster {c} ({y}) is too small: {x.shape[0]} households only (at least 8 are required)")
        else:
          is_gaussian[q].append(normal_test(x, verbose=False))
      is_gaussian[q] = np.array(is_gaussian[q])
      
    all_clusters = np.array([])
    for q,data in is_gaussian.items():
      print(f"{q}: {np.count_nonzero(data == 1)} of {data.shape[0]} ({np.count_nonzero(data == 1)*100/data.shape[0]:.2f}%)")
      all_clusters = np.append(all_clusters, data)
      
    print(f"All clusters: {np.count_nonzero(all_clusters == 1)} of {all_clusters.shape[0]} ({np.count_nonzero(all_clusters == 1)*100/all_clusters.shape[0]:.2f}%)")
    
  return

def normal_test(x, alpha=0.05, test_name='k2', verbose=True):
    ### https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
    valid_tests = ['k2','normaltest','shapiro']
    if test_name not in valid_tests:
        raise Exception('Not valid test_name. Try: k2, normaltest, shapiro')
        
    m = np.mean(x)
    std = np.std(x)
    if verbose:
        print(f"mean = {m:g}, std.dev. = {std:g}")
    
    if verbose:
        print(f"H0: x comes from a normal distribution ($\alpha$={alpha})")
    
    fnc = normaltest if test_name in ['k2','normaltest'] else shapiro
    name = fnc.__name__
    k, p = fnc(x)
    if verbose:
      print(f"k = {k:g}, p = {p:g}")
    if p <= alpha:
        if verbose:
            print('Data does NOT look Gaussian (reject H0)')
        return False
    else:
        if verbose:
            print('Data looks Gaussian (fail to reject H0)')
        return True
    
def summary_normal_test(x, alpha=0.05):
    ### https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
    valid_tests = [normaltest,shapiro,anderson]
    
    m = np.mean(x)
    std = np.std(x)
    print(f"mean = {m:g}, std.dev. = {std:g}")
    
    print(f"H0: x comes from a normal distribution ($\alpha$={alpha})")
    
    for fnc in valid_tests:
        name = fnc.__name__
        print(f"====== {name} ======")
        
        if name in ['normaltest','shapiro']:
            if normal_test(x, test_name=name, verbose=False):
                print('Data looks Gaussian (fail to reject H0)')
            else:
                print('Data does NOT look Gaussian (reject H0)')
        else:
            result = anderson(x)
            print(f"statistic: {result.statistic:g}")
     
            for i in range(len(result.critical_values)):
                sl, cv = result.significance_level[i], result.critical_values[i]
                if result.statistic < result.critical_values[i]:
                    print(f'{sl:g} {cv:g}, data looks Gaussian (fail to reject H0)')
                else:
                    print(f'{sl:g}: {cv:g}, data does NOT look Gaussian (reject H0)')
                    
                    
                    
####################################################################################################
# Validations
####################################################################################################

def validate_nclasses(n_classes):
  if n_classes < 2 and n_classes > 100:
    print("[WARNING] there can be minimun 2 classes and maximun 100. Set to 10.")
    n_classes = 10
  return n_classes
  
def validate_var(var):
  if var not in ['mean','std']:
    print('[WARNING] var must be mean or std. Set to mean.')
    var = 'mean'
  return var

def validate_norm(norm):
  if norm not in ['true','pred','all']:
    print("[WARNING] only true, pred, all are norm values allowed. Set to true.")
    norm = 'true'
  return norm

def validate_metric(metric):
  if metric not in ['nrmse','rmse','mse','r2']:
    print('[WARNING] metric must be nrmse, rmse, mse, or r2. Set to rmse.')
    metric = 'rmse'
  return metric

def validate_evaluation(evaluation):
  if evaluation not in ['recall','precision','f1','nrmse','rmse','mse','r2']:
    print('[WARNING] evaluation must be recall, precision, f1, nrmse, rmse, mse, or r2. Set to recall.')
    evaluation = 'recall'
  return evaluation

def validate_class_type(class_type):
  if class_type not in ['class','quantiles','discrete']:
    print('[WARNING] class_type must be class or quantiles or discrete. Set to class.')
    class_type = 'class'
  return class_type
  
  
####################################################################################################
# VISUAL HANDLERS
####################################################################################################

def sns_reset():
  sns.reset_orig()
  #matplotlib.use('agg')

def sns_paper_style():
  sns.set_context("paper", font_scale=1.5) #rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5}) 
  rc('font', family = 'serif')
  
def set_style(font_scale=1.5):
    sns.reset_orig()
    sns.set_context("poster", font_scale=font_scale) #rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5}) 
    rc('font', family = 'serif')
    
def set_latex():
  # rc('text', usetex=True)
  matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

class SeabornFig2Grid(object):

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())
  

def cm_heatmap(**kwargs):
    data = kwargs.pop('data')
    true = data.loc[:,kwargs.pop('true')]
    pred = data.loc[:,kwargs.pop('pred')]
    norm = kwargs.pop('norm')
    mi,ma = kwargs.pop('axisminmax')
    vmin,vmax = (0,1) if norm else (None,None)
    
    cm = confusion_matrix(true,pred,normalize=norm)
    ax = sns.heatmap(cm,vmin=vmin,vmax=vmax,**kwargs)
    for c in np.arange(mi,ma+1):
        ax.add_patch(Rectangle((c, c), 0.95, 0.95, fill=False, edgecolor='black', lw=0.9))

def get_statistical_significance_symbol_from_pvalue(p):
  '''
  https://www.graphpad.com/support/faq/what-is-the-meaning-of--or--or--in-reports-of-statistical-significance-from-prism-or-instat/
  ns   P > 0.05
  *    P ≤ 0.05
  **   P ≤ 0.01
  ***  P ≤ 0.001
  **** P ≤ 0.0001
  '''
  return '****' if p<=0.0001 else '***' if p<=0.001 else '**' if p<=0.01 else '*' if p<=0.05 else 'ns' if p>0.05 else '-'


####################################################################################################
# PLOTS ON PPLACES
####################################################################################################

def plot_poverty_maps(query, output=None):
  
  nr = int(len(query.values()))
  nc = 1
  sw = 9.
  sh = 9.
  
  fig,axes = plt.subplots(nr,nc,figsize=(nc*sw,nr*sh))
  
  data = pd.DataFrame()
  for ccode, model in query.items():
    fn = f'results/pplaces_inference/{ccode}_{model}.csv'
    df1 = ios.load_csv(fn)
    fn = f'results/pplaces_inference/{ccode}_features.csv'
    df2 = ios.load_csv(fn, index_col=None)
    tmp = df1.set_index('OSMID').join(df2.set_index('OSMID'))
    data = pd.concat([data, tmp], ignore_index=True)
    del(df1)
    del(df2)
    del(tmp)
  
  data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lon, data.lat))
  letters = list('abcdefghijklmnopqrstuvwxyz')
  
  mmin, mmax = data.pred_mean_wi.min(), data.pred_mean_wi.max()
  smin, smax = data.pred_std_wi.min(), data.pred_std_wi.max()
  
  for ax, letter, (country, gdf) in zip(*[axes, letters[:nr], data.groupby("country")]):
    ccode = COUNTRIES[country]['code']
    
    # plot mean
    n = gdf.shape[0]
    vmin, vcenter, vmax, vstd = gdf.pred_mean_wi.min(), gdf.pred_mean_wi.mean(), gdf.pred_mean_wi.max(), gdf.pred_mean_wi.std()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = 'BrBG'
    gdf.plot(column='pred_mean_wi', 
             ax=ax,
             # cax=cax,
             markersize=6., 
             vmin=mmin, vmax=mmax,
             cmap=cmap,
             norm=norm,
             legend=True,
             legend_kwds={'shrink':0.20, 
                          'location':'bottom',  # 'left', 'right', 'top', 'bottom'
                          'orientation':'horizontal', 
                          # # 'label':'IWI mean',
                          #'pad':0.01, #-0.2 if ccode=='UG' else -0.1, 
                          'anchor':(0.30 if ccode=='SL' else 0.55, 
                                    2.55 if ccode=='SL' else 2.6)}
            )
    ax.text(s=f"({letter}) {country}", x=0.0, y=0.95, va='top', ha='left', transform=ax.transAxes)
    # ax.text(s=f"{country}", x=1.0, y=0.74, va='top', ha='center', transform=ax.transAxes)
    ax.text(s=f"IWI mean", x=0.22 if ccode=='SL' else 0.46, 
                           y=0.11 if ccode=='SL' else 0.12, 
                           va='bottom', ha='left', transform=ax.transAxes)
    ax.set_axis_off()
    # ax.collections[0].set_rasterized(True)
    
    # plot std
    ax_in = inset_axes(ax, width="100%", height="100%",
                    bbox_to_anchor=(.66, .04, 1.11, 0.49),
                    bbox_transform=ax.transAxes)
    n = gdf.shape[0]
    vmin, vcenter, vmax, vstd = gdf.pred_std_wi.min(), gdf.pred_std_wi.mean(), gdf.pred_std_wi.max(), gdf.pred_std_wi.std()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = 'PuOr'
    gdf.plot(column='pred_std_wi', 
             ax=ax_in,
             # cax=cax,
             markersize=1., 
             vmin=smin, vmax=smax,
             cmap=cmap,
             norm=norm,
             legend=True,
             legend_kwds={'shrink':0.30 if ccode=='SL' else 0.33, 
                          'location':'bottom', 
                          'orientation':'horizontal', 
                          # 'label':'IWI std.dev.',
                          # 'pad':0.01, #-0.2 if ccode=='UG' else -0.1, 
                          'anchor':(1.32 if ccode=='SL' else 1.152, #2.15 
                                    -1.6 if ccode=='SL' else 2.035)}
            )
    ax_in.text(s=f"IWI STD", x=1.0 if ccode=='SL' else 0.91, 
                             y=0.11 if ccode=='SL' else 0.12, 
                             va='bottom', ha='right', transform=ax.transAxes)
    ax_in.set_axis_off()
    # ax_in.collections[0].set_rasterized(True)
    
  fig.subplots_adjust(hspace=-0.35)
  
  if output is not None:
    fn = os.path.join(output,f"hr_poverty_maps.pdf")
    fig.savefig(fn, dpi=300, bbox_inches='tight')
    print(f"{fn} saved!")
    
  plt.show()
  plt.close()
  
def plot_poverty_map(ccode, model, output=None):
  fn = f'results/pplaces_inference/{ccode}_{model}.csv'
  df1 = ios.load_csv(fn)
  fn = f'results/pplaces_inference/{ccode}_features.csv'
  df2 = ios.load_csv(fn, index_col=None)
  df = df1.set_index('OSMID').join(df2.set_index('OSMID'))
  del(df1)
  del(df2)
  gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
  
  fig,ax = plt.subplots(1,1,figsize=(7,7))
  
  # moving the middle color to the mean
  n = df.shape[0]
  vmin, vcenter, vmax, vstd = df.pred_mean_wi.min(), df.pred_mean_wi.mean(), df.pred_mean_wi.max(), df.pred_mean_wi.std()
  norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
  cmap = 'BrBG'
  #print(ccode, vmin, vcenter, vmax)
  
  # plot
  gdf.plot(column='pred_mean_wi', 
           ax=ax,
           # cax=cax,
           markersize=5, 
           vmin=0, vmax=100,
           cmap=cmap,
           norm=norm,
           legend=True,
           legend_kwds={'shrink':0.25, 'location':'bottom', 
                        'orientation':'horizontal', 
                        #'label':'IWI mean',
                        'pad':-0.2 if ccode=='UG' else -0.1, 
                        'anchor':(0.8 if ccode=='UG' else 0.15, 1.0)}
          )
  fig.axes[1].tick_params(labelsize=11)
  ax.set_axis_off()
  title = f"({'a' if ccode=='SL' else 'b'}) {[c for c,obj in COUNTRIES.items() if obj['code']==ccode][0]}"
  fig.axes[1].set_title(title, y=-8)
  
  # rasterize scatter dots
  ax.collections[0].set_rasterized(True)
    
  if output is not None:
    fn = os.path.join(output,f"sm_hr_poverty_map_{ccode}_{model}.pdf")
    fig.savefig(fn, dpi=300, bbox_inches='tight')
    print(f"{fn} saved!")
    
  plt.show()
  plt.close()


def plot_pplace_mean_std(query, output=None):
  # annotations
  def annotations(data, **kws):
    ax = plt.gca()
    y = 0.21
    
    # model
    model = data.model.unique()[0]
    ax.text(s=model.replace("CBw","CB$_w$").replace("CNNa","CNN$_a$"), x=1.0, y=0.21, 
            ha='right', va='bottom', transform=ax.transAxes, c='grey')
    
    # corr
    for color,settlement in zip(*(['tab:blue','tab:orange'],['urban','rural'])):
      df = data.query("settlement==@settlement").copy()
      y-=0.1
      n = df.shape[0]
      r,p = pearsonr(df.pred_mean_wi, df.pred_std_wi)
      pv = get_statistical_significance_symbol_from_pvalue(p)
      ax.text(s=f"n={n} r={r:.2f} {pv}", x=1.0, y=y, ha='right', va='bottom', transform=ax.transAxes, c=color)
    
  # data
  data = pd.DataFrame()
  for ccode,model in query.items():
    fn = f'results/pplaces_inference/{ccode}_{model}.csv'
    df1 = ios.load_csv(fn)
    fn = f'results/pplaces_inference/{ccode}_features.csv'
    df2 = ios.load_csv(fn, index_col=None)
    df = df1.set_index('OSMID').join(df2.set_index('OSMID'))
    df.loc[:,'model'] = model
    del(df1)
    del(df2)
    data = pd.concat([data,df], ignore_index=True)
  data.loc[:,'settlement'] = data.rural.apply(lambda v: 'rural' if v in [1,'1'] else 'urban')
  data.rename(columns={'population_in_1.61km':'population'}, inplace=True)
  
  # figure
  fg = sns.relplot(data=data.sort_values('settlement', ascending=True), 
                   col='country', col_order=data.country.unique(),
                   x='pred_mean_wi', y='pred_std_wi', 
                   hue='settlement', hue_order=data.settlement.unique(),
                   size='population', sizes=(1,200), palette=['tab:blue','tab:orange'],
                   alpha=0.3,
                   height=3.8, aspect=0.68)

  fg.set_xlabels("Pred. IWI_mean")
  fg.set_ylabels("Pred. IWI_std")
  fg.map_dataframe(annotations)
  fg.set_titles(row_template = '{row_name}', col_template = '{col_name}')
  
  add_panel_labels(fg)
  plt.tight_layout()
  plt.subplots_adjust(wspace=0.2, hspace=0.25)
  sns.move_legend(fg, "upper left", bbox_to_anchor=(0.935, 0.855), handletextpad=0.1)
  #h,l = fg.axes.flatten()[0].get_legend_handles_labels()
  #fg.legend(h[0:7],l[0:7], bbox_to_anchor=(0.935, 0.855), loc='upper left', handletextpad=0.1)

#   for t in fg._legend.texts:
#     text = t.get_text()
#     t.set_text(text if text=='0' or text in ['settlement','population','urban','rural'] else  f"{int(float(text)/1000)}K")
    
  # rasterize scatter dots
  for ax in fg.axes.flatten():
    ax.collections[0].set_rasterized(True)

  if output is not None:
    fn = os.path.join(output,f"pplaces_mean_vs_std.pdf")
    fg.savefig(fn, dpi=300, bbox_inches='tight') 
    print(f"{fn} saved!")
    
  plt.show()
  plt.close() 


def plot_pplace_variability(query, output=None):
  ### Annotate start
  def annotate_pplace_corr(data, **kws):
    n = data.shape[0]
    is_rural = data.settlement.unique()[0]=='rural'
    r,p = pearsonr(data.pred_mean_wi, data.pred_std_wi)
    pv = get_statistical_significance_symbol_from_pvalue(p)
    model = ','.join(data.model.unique())
    ax = plt.gca()

    if is_rural:
      s = model
      x = 0.5
      y = 0.9
      ha = 'center'
      va = 'center'
      ax.text(x=x, y=y, s=s, color='grey', va=va, ha=ha, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='none',pad=0.0))

    n = int(round(n))
    n = (r"n=<n>, ").replace('<n>',str(n))
    r = (r"r=<r>").replace("<r>",f"{r:.2f}")
    s = f"{n} {r} ({pv})"
    x = 1.0
    y = 0.01 if is_rural else 0.11
    print(data.settlement.unique(),is_rural, y)
    ha = 'right'
    va = 'bottom'
    fs = 12
    ax.text(x=x, y=y, s=s, color=kws['color'], va=va, ha=ha, transform=ax.transAxes, size=fs, bbox=dict(facecolor='white', edgecolor='none',pad=0.0))
  ### Annotate end 

  data = pd.DataFrame()
  for ccode,model in query.items():
    fn = f'results/pplaces_inference/{ccode}_{model}.csv'
    df1 = ios.load_csv(fn)
    fn = f'results/pplaces_inference/{ccode}_features.csv'
    df2 = ios.load_csv(fn, index_col=None)
    df = df1.set_index('OSMID').join(df2.set_index('OSMID'))
    df.loc[:,'model'] = model
    del(df1)
    del(df2)
    data = pd.concat([data,df], ignore_index=True)

  data.loc[:,'settlement'] = data.rural.apply(lambda v: 'rural' if v in [1,'1'] else 'urban')
  fg = sns.FacetGrid(data=data, col='country', hue='settlement', hue_order=['urban','rural'], 
                     height=4.0, aspect=0.9, margin_titles=True, legend_out=True)
  
  xlabel='pred_mean_wi'
  ylabel='pred_std_wi'
  fg.map_dataframe(annotate_pplace_corr)
  fg.map_dataframe(sns.regplot, x=xlabel, y=ylabel, order=POLYNOMIAL_DEGREE, x_ci='sd', scatter=False)
  fg.add_legend()
  fg.map_dataframe(sns.scatterplot, x=xlabel, y=ylabel, alpha=0.05)
  fg.set_titles(row_template = '{row_name}', col_template = '{col_name}')
  
  
  if output is not None:
    fn = os.path.join(output,f"pplaces_mean_vs_std.pdf")
    fg.savefig(fn, dpi=300, bbox_inches='tight') 
    print(f"{fn} saved!")
    
  plt.show()
  plt.close() 

####################################################################################################
# FIGURES
####################################################################################################

def plot_map(df, kind='wealth', output=None):
  
  if kind not in ['year','settlement','wealth','year_settlement']:
    raise Exception("kind not implemented.")
  
  if type(df) == pd.DataFrame:
     data = geo.get_GeoDataFrame(df.copy(), lat='lat', lon='lon')
  else:
    data = df.copy()
  
  if kind == 'year_settlement':
    data.loc[:,'year_settlement'] = data.apply(lambda row:f"{'r' if row.rural in [1,'1'] else 'u'}-{'2018-19' if row.year!=2016 else row.year}", axis=1)
    data.loc[:,'year_settlement'] = pd.Categorical(data.loc[:,'year_settlement'], ["u-2016", "u-2018-19", "r-2016", "r-2018-19"])
  elif kind == 'settlement':
    data.loc[:,'rural'] = data.loc[:,'rural'].apply(lambda v:'rural' if v in ['1',1] else 'urban')
    data.loc[:,'rural'] = pd.Categorical(data['rural'], ["urban", "rural"])
    
  column = kind if kind in ['year','year_settlement'] else 'rural' if kind=='settlement' else 'mean_wi'
  categorical = column!='mean_wi'
  
  if categorical:
    if kind=='year':
      cmap = 'Set2'
      m = 2
    else:
      m = 20 if kind=='year_settlement' else 10
      cmap = f"tab{m}"
    
    colors = mcp.gen_color(cmap=cmap,n=m)
  
  nc = data.ccode.nunique()
  countries = data.ccode.unique().tolist()
  nr = 1
  size = SIZE
  marker_size = 10
  
  # main fig
  fig,axes = plt.subplots(nr,nc,figsize=(nc*size,nr*size), sharex=False, sharey=False)
  c=0
  for group, tmp in data.groupby('ccode'):
    ax = axes[c]
    
    if categorical:
      g=0
      for group2, tmp2 in tmp.groupby(column):
        group2 = group2 if kind!='year_settlement' else f"{'r' if tmp2.rural.unique()[0] in [1,'1'] else 'u'}-{tmp2.year.unique()[0]}"
        ax.scatter(tmp2.lon, tmp2.lat, color=colors[g], label=group2)
        g+=1
      ax.legend()
    else:
      divider = make_axes_locatable(ax)
      cax = None if categorical else divider.append_axes("right", size="5%", pad=0.1)
      tmp.plot(ax=ax, column=column, cmap='RdYlBu', markersize=marker_size, legend=True, cax=cax)
    
    ax.set_title(tmp.countryname.unique()[0])
    if not categorical:
      cax.set_title("IWI")
    ax.set_xlabel('longitude')
    c+=1 
    
  axes[0].set_ylabel("latitute")
  
  if output is not None:
    fn = os.path.join(output,f"sm_gt_maps_{kind}.pdf")
    fig.savefig(fn, dpi=300, bbox_inches='tight')
    print(f"{fn} saved!")
    
  plt.show()
  plt.close()


def plot_correlation(df, output=None):
  if type(df) == pd.DataFrame:
     data = geo.get_GeoDataFrame(df.copy(), lat='lat', lon='lon')
  else:
    data = df.copy()
    
  data.loc[:,'settlement'] = data.rural.apply(lambda v: 'rural' if v==1 else 'urban')
  settlement = CategoricalDtype(categories=['urban','rural'], ordered=True)
  data.loc[:,'settlement'] = data.settlement.astype(settlement)
  
  hue_order = settlement.categories.values
  data.rename(columns={'mean_wi':'mean_iwi', 'std_wi':'std_iwi'}, inplace=True)
  
  groups = ['settlement','ccode']
  nc = data.ccode.nunique()
  nr = len(groups)
  size = SIZE-1
  marker = 10
  
  fig = plt.figure(figsize=(nc*size,nr*size))
  gs = gridspec.GridSpec(nr, nc)
  
  i=0
  for (ccode,settlement), tmp in data.groupby(groups):
    r,c=int(i/nc),i%nc
    g = sns.jointplot(data=tmp, x="mean_iwi", y="std_iwi", hue='settlement', hue_order=hue_order, space=0, legend=i==0)
      
    if r==0:
      g.ax_marg_x.set_title(tmp.countryname.unique()[0])
      g.ax_joint.set_xlabel("")
    if c!=0:
      g.ax_joint.set_ylabel("")
    
    g.ax_joint.set_xlim(0,100)
    g.ax_joint.set_ylim(0,25)
    
    xx=65
    yy=24
    ro,p = pearsonr(tmp.mean_iwi, tmp.std_iwi)
    s = get_statistical_significance_symbol_from_pvalue(p)
    txt = f"r={ro:.2f} ({s})\n{tmp.shape[0]} clusters"
    g.ax_joint.text(s=txt, x=xx, y=yy, va='top', ha='left', size=12)
    
    SeabornFig2Grid(g, fig, gs[i])
    i+=1
    
  gs.tight_layout(fig)

  if output is not None:
    fn = os.path.join(output,'sm_gt_corr_mean_std.pdf')
    fig.savefig(fn, dpi=300)
    print(f"{fn} saved!")
    
  plt.show()
  plt.close()
  
def plot_cells_reg(root, countries, output=None):

  features = ['distance_closest_cell','cells_in_2.0km','towers_in_2.0km']
  rows=len(countries)
  cols=len(features)
  size=3.0
  xval='mean_iwi'
  
  fig,axes = plt.subplots(rows,cols,figsize=(cols*size, rows*size))

  for r,country in enumerate(countries):
      # cells
      fn_cells = glob.glob(os.path.join(root,f"{country}/results/features/clusters/*_*_*_cluster_cells.csv"))[0]
      df_cells = pd.read_csv(fn_cells, index_col=0)
      
      # clusters
      fn_clusters = fn_cells.replace("_cells","") #glob.glob(os.path.join(root,f"{country}/results/features/clusters/*_*_*_cluster.csv"))[0]
      df_clusters = pd.read_csv(fn_clusters, index_col=0)
      df_clusters.rename(columns={'mean_wi':'mean_iwi', 'std_wi':'std_iwi'}, inplace=True)
      gdf_clusters = gpd.GeoDataFrame(df_clusters, geometry=gpd.points_from_xy(df_clusters.lon, df_clusters.lat))
      
      # join
      gdf = gdf_clusters.set_index('gtID').join(df_cells.set_index('gtID'), how='inner')
      print(f"[INFO] shapes: cells {df_cells.shape[0]}  | clusters {gdf_clusters.shape[0]} | join {gdf.shape[0]}")
      
      for c,col in enumerate(features):
          ax = axes[r,c]
          sns.regplot(ax=ax, data=gdf, x=xval, y=col)
          ax.set_ylabel('')
          ax.set_title('' if r>0 else col)
          ax.set_xlabel('' if r<(rows-1) else xval)
          
          # right y-label
          ax2 = ax.twinx()
          ax2.set_yticks([])
          ro,p = pearsonr(gdf[xval].values, gdf[col].values)
          s = get_statistical_significance_symbol_from_pvalue(p)
          ax2.text(s=f"r={ro:.2f}{s}", x=gdf[xval].max() if col.startswith('distance') else gdf[xval].min(), y=0.9, ha='right' if col.startswith('distance') else 'left', va='top')
          if c==cols-1:
              ax2.set_ylabel(country, rotation=270, labelpad=15)
        
  plt.tight_layout()
  if output is not None:
    fn = os.path.join(output,'sm_gt_corr_cells_mean.pdf')
    fig.savefig(fn, dpi=300)
    print(f"{fn} saved!")
    
  plt.show()
  plt.close()

def plot_data_augmentation(root, country, prefix=None, osmid=None, output=None):
  prefix = prefix if prefix is not None else '' if osmid is None else f'OSMID{osmid}'
  fn = np.random.choice(glob.glob(os.path.join(root,country,'results','staticmaps','augmented',f"{prefix}*.png")),1)[0]
  fn = '-'.join(fn.split('-')[:-1])
  files2 = sorted(glob.glob(f"{fn}*"))
  
  fn = fn.replace("augmented","pplaces" if osmid is not None else 'clusters')
  files1 = glob.glob(f"{fn}*")
  print(f"[INFO] {fn}")
  
  files = files1
  files.extend(files2)
  
  cols = 3
  rows = int(round((len(files)+1)/cols))
  size = 3.0
  
  fig,axes = plt.subplots(rows,cols,figsize=(cols*size, rows*size))
  plt.gca().clear()
  
  for cell in np.arange(cols*rows):
    r = int(cell/cols)
    c = cell%cols
    ax = axes[r,c] if rows>1 and cols>1 else axes[r] if cols==1 else axes[c] if rows==1 else axes[0,0]
    
    if cell < len(files):
      fn = files[cell]
      augmentation = 'original' if cell==0 else fn.split("-")[-1].split('.')[0]
      image = mpimg.imread(fn) 
      ax.imshow(image)
      ax.set_title(augmentation)
    ax.axis("off")
    
    
  if output is not None:
    fn = os.path.join(output,'sm_augmentation.pdf')
    fig.savefig(fn, dpi=300, bbox_inches='tight')
    print(f"{fn} saved!")
    
  plt.subplots_adjust(wspace=0.2, hspace=0.2)
  plt.show()
  plt.close()

  
def plot_distance_distribution_location_change(root, countries, output=None):

  df = pd.DataFrame()
  cols = ['country','cluster_rural','pplace_rural','settlement','mean_wi','std_wi','pplace_cluster_distance']
  
  for country in countries:
    fn = np.random.choice(glob.glob(os.path.join(root,country,'results','samples',f'*_all_ruc','*','data.csv')),1)[0]
    tmp = ios.load_csv(fn)
    ccode, _ = re.match(r"([a-z]+)([0-9]+)", tmp.loc[0].cluster_id, re.I).groups()
    tmp.loc[:,'country'] = fn.split("/results")[0].split("/")[-1] # ccode
    tmp.loc[:,'pplace_cluster_distance'] = tmp.pplace_cluster_distance / 1000
    tmp.loc[:,'settlement'] = tmp.cluster_rural.apply(lambda v: 'rural' if v in [1,'1'] else 'urban')
    settlement = CategoricalDtype(categories=['urban','rural'], ordered=True)
    tmp.loc[:,'settlement'] = tmp.settlement.astype(settlement)
    df = pd.concat([df,tmp[cols]], ignore_index=True)
    
  size = 4 #SIZE-1
  fg = sns.displot(data=df, x='pplace_cluster_distance', col='country', hue='settlement', height=size, aspect=1)
  fg.set_titles(row_template = '{row_name}', col_template = '{col_name}')
  
  for ax in fg.axes.flatten():
    ax.set_xlabel("Distance (Km)")
    ax.set_yscale('log')
    # ax.set_xscale('log')
    
  fg.fig.tight_layout()
  
  
  if output is not None:
    fn = os.path.join(output,'sm_distance_reallocation.pdf')
    fg.fig.savefig(fn, dpi=300, bbox_inches='tight')
    print(f"{fn} saved!")
    
  #plt.subplots_adjust(wspace=0.2, hspace=0.2)
  plt.show()
  plt.close()
   
  
def plot_performance(df, metric='r2', kind='swarm', output=None):
  wide_kinds = ['box','boxen','violin']
  
  data = df.query("features in @MAIN_FEATURES").copy()
  palette = 'tab20'
  
  if kind=='line':
    fg = sns.FacetGrid(data=data, 
                       row='country', col='relocation',
                       hue='model', 
                       palette=palette,
                       margin_titles=True, height=int(SIZE/2.), aspect=1.0)

    fg.map(sns.lineplot, 'recency', metric, estimator=np.mean, ci='sd', sort=True)
    fg.add_legend()

  else:
      fg = sns.catplot(data=data, 
                   x='recency', y=metric, 
                   row='country', col='relocation',
                   hue='model', 
                   kind=kind,
                   palette=palette,
                   margin_titles=True, height=int(SIZE/2.), aspect=1.0 if kind not in wide_kinds else 2.0)
  

  fg.set_xticklabels(rotation=90 if kind not in wide_kinds else 0)
  fg.set_ylabels(metric.upper())
  for ax in fg.axes.flatten():
    ax.grid(axis='y')
    
  if output is not None:
    fn = os.path.join(output,f'sm_performance_{metric}.pdf')
    fg.savefig(fn, dpi=300, bbox_inches='tight')
    print(f"{fn} saved!")
    
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  plt.show()
  plt.close()
  
def plot_wealth_distribution_per_year(root, countries, output=None):
  
  def annotate_gini(data, **kws):
    ax = plt.gca()
    
    fs = 11
    y = 0.8
    ax.text(1., y, "year: (    $n$,   $\mu$,  $\sigma$,  Gini)", transform=ax.transAxes, color='grey', fontsize=fs, ha='right')
    
    for year,tmp in data.groupby('year'):
      values = tmp.mean_iwi
      n = values.shape[0]
      m = values.mean()
      s = values.std()
      g = gini(values.values)
      y -= 0.08
      ax.text(1., y, f"{year}: ({n:d}, {m:.0f}, {s:.0f}, {g:.2f})", transform=ax.transAxes, color='grey', fontsize=fs, ha='right')
      
    
  df = pd.DataFrame()
  for country in countries:
    files = glob.glob(os.path.join(root,country,'results','features','clusters','*_cluster.csv'))
    fn = [fn for fn in files if len(fn.split("_"))>4][0]
    print(f"[INFO] {fn} loaded.")
    df = pd.concat([df, ios.load_csv(fn)], ignore_index=True)
    
  df.rename(columns={'mean_wi':'mean_iwi'}, inplace=True)
  df.loc[:,'country'] = df.loc[:,'ccode'].apply(lambda v: 'Sierra Leone' if v=='SL' else 'Uganda' if v=='UG' else 'NN')
  fg = sns.displot(data=df, 
                   hue='year', col='country', x='mean_iwi', 
                   kind='kde',common_norm=False,
                   height=4, aspect=1.0, palette='viridis')
    
  fg.map_dataframe(annotate_gini)
  fg.set_titles(row_template = '{row_name}', col_template = '{col_name}')
  
  if output is not None:
    fn = os.path.join(output,f'sm_wealth_distribution_per_year.pdf')
    fg.savefig(fn, dpi=300, bbox_inches='tight')
    print(f"{fn} saved!")
    
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  plt.show()
  plt.close()
  
  
def plot_wealth_distribution_per_year_and_settlement(root, countries, output=None):
  
  # data
  df = pd.DataFrame()
  for country in countries:
    files = glob.glob(os.path.join(root,country,'results','features','clusters','*_cluster.csv'))
    fn = [fn for fn in files if len(fn.split("_"))>4][0]
    print(f"[INFO] {fn} loaded.")
    df = pd.concat([df, ios.load_csv(fn)], ignore_index=True)
    
  df.rename(columns={'mean_wi':'mean_iwi'}, inplace=True)
  df.loc[:,'country'] = df.loc[:,'ccode'].apply(lambda v: 'Sierra Leone' if v=='SL' else 'Uganda' if v=='UG' else 'NN')
  
  nc = 3
  nr = df.country.nunique()
  fig, axes = plt.subplots(nr, nc, figsize=(nc*SIZE/2, nr*SIZE/2), sharex=True, sharey=False)
  years = df.year.unique()
  
  i = 0
  for country in df.country.unique():
    for settlement in ['urban','rural','all']:
      query = f"country=='{country}'"
      if settlement != 'all':
        query = f"{query} and rural=={1 if settlement=='rural' else 0}"
        
      tmp = df.query(query)
      
      r = int(i/nc)
      c = i%nc
      
      ax = axes[r,c]
      sns.kdeplot(data=tmp, x='mean_iwi', hue='year', common_norm=False, ax=ax, palette='viridis', hue_order=years, legend=c==1)
      ax2 = ax.twinx()
      ax2.set_yticks([])
      ax2.set_ylabel(country if c==nc-1 else '', rotation=270, labelpad=15)
      
      ax.set_title(settlement.title() if r==0 else '')
      ax.set_ylabel('Density' if c==0 else '')
      ax.set_xlabel('mean_iwi' if r==nr-1 else '')
      i+=1
    
  if output is not None:
    fn = os.path.join(output,f'sm_wealth_distribution_per_year_settlement.pdf')
    fig.savefig(fn, dpi=300, bbox_inches='tight')
    print(f"{fn} saved!")
    
  plt.tight_layout()
  # plt.subplots_adjust(wspace=0.1, hspace=0.1)
  plt.show()
  plt.close()
  

def add_panel_labels(fg):
  panels = fg.axes.flatten()
  npanels = panels.shape[0]
  letters = list('abcdefghijklmnopqrstuvwxyz')
  
  for ax, label in zip(*[panels,letters[:npanels]]):
    ax.text(s=f'({label})', x=-0.15, y=1.12, va='top', ha='left', transform=ax.transAxes)
    
  
def plot_summary_performance(df, metric='nrmse', output=None):

  def annot(x, y, **kwargs):
    
    ax = plt.gca()
    data = kwargs.pop("data")
    fs = 10 # fontsize
    
    metric = data['metric'].unique()[0]
    for kind, tmp in data.query("outputn=='mean'").groupby("kind",sort=False):
      confs = tmp.configuration.unique()
      
      for outputn in data.outputn.unique():
      
        df = data.query("outputn==@outputn and kind==@kind").sort_values('value', ascending=True).copy()

        delta_y = 0.55 if metric=='rmse' else 0.08 if metric=='nrmse' else 0
        delta_x = -0.23 if outputn=='mean' else 0.23 if outputn=='std' else 0
    
        for xx, conf in enumerate(confs):
          row = df.query("configuration==@conf").iloc[0]

          yy = row[y]+delta_y
          ax.text(s=str(row[y]), x=xx+delta_x, y=yy, ha='center', va='top', 
                  fontsize=fs, color='grey')
      
  # metric
  metrics = [c.split('_')[0] for c in df.columns if c not in ['country','configuration','kind']]
  if metric not in metric:
    raise Exception('Metric does not exist.')
  
  message = f"The {'lower' if 'mse' in metric else 'larger'} {metric}, the better"
  print(message)
  
  # data
  cols = ['kind','country','performance','configuration','value','outputn']
  
  gen = '_wi_mean'
  hue_order = []
          
  key = f'{metric}_mean{gen}'
  data1 = df.copy()
  data1.rename(columns={key:'value'}, inplace=True)
  data1.loc[:,'outputn'] = 'mean'
  data1.loc[:,'performance'] = r'$\epsilon_\mu$'
  data1 = data1[cols].copy()
  hue_order.append(data1.performance.unique()[0])
          
  key = f'{metric}_std{gen}'
  data2 = df.copy()
  data2.rename(columns={key:'value'}, inplace=True)
  data2.loc[:,'outputn'] = 'std'
  data2.loc[:,'performance'] = r'$\epsilon_\sigma$'
  data2 = data2[cols].copy()
  hue_order.append(data2.performance.unique()[0])
  
  data = pd.concat([data1,data2], ignore_index=True)
  data.loc[:,'metric'] = metric
  del(data1)
  del(data2)
  
  # plot
  # indexes = df.query("kind=='model'").index  
  fg = sns.catplot(data=data.sort_values('value', ascending=True), 
                   kind='bar',
                   col="kind", row="country", 
                   col_order=['recency','relocation','augmented','weighted'],
                   x="configuration", y='value', 
                   hue='performance', hue_order=hue_order,
                   sharex=False,sharey=True,
                   height=2.6, aspect=1.0,
                   legend_out=True,
                   palette='Paired',
                   margin_titles=True)
  fg.map_dataframe(annot, x="configuration", y='value')

  sns.move_legend(fg, "upper left", bbox_to_anchor=(0.98, 0.6))
  add_panel_labels(fg)
  
  fg.set_titles(row_template = '{row_name}', col_template = '{col_name}')
  # fg.set_xticklabels(rotation=90)
  fg.set_ylabels(metric.upper())
  fg.set_xlabels('')
  

  plt.tight_layout()
  plt.subplots_adjust(hspace=0.25)
  
  if output is not None:
    fn = os.path.join(output,f'baselines_{metric}.pdf')
    fg.savefig(fn, dpi=300, bbox_inches='tight')
    print(f"{fn} saved!")
  
  plt.show()
  plt.close()
  
  
def plot_prediction_errors(df, var='mean', relocation='none', recency='ON', output=None):
  
  var = validate_var(var)
    
  data_rs = df.query("relocation==@relocation and recency==@recency and features in @MAIN_FEATURES").copy()
  data_rs.loc[:,'model'] = data_rs.model.apply(lambda v: "cnn" if v=='cnn_noaug' else v)
  
  fg = sns.relplot(data=data_rs, row='rural', col='country', 
                   y=f'error_{var}', x=f'true_{var}_class', 
                   hue='model', style='year',
                   row_order=['rural','urban'],
                   kind='line',
                   facet_kws={"margin_titles": True},
                   height=int(SIZE/2.), aspect=1.45)

  fg.refline(y=0)
  
  [plt.setp(ax.texts, text="") for ax in fg.axes.flat]
  fg.set_titles(row_template = '{row_name}', col_template = '{col_name}')
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  
  if output is not None:
    fn = os.path.join(output,f'sm_prediction_errors_{var}.pdf')
    fg.savefig(fn, dpi=300, bbox_inches='tight')
    print(f"{fn} saved!")
  
  plt.show()
  plt.close()
  

def plot_confusion_matrices(df, country='Sierra Leone', var='mean', relocation='none', recency='ON', norm='true', annot=False, output=None):
  
  norm = validate_norm(norm)
  var = validate_var(var)
  
  data_rs = df.query("country==@country and relocation==@relocation and recency==@recency").copy()
  
  fg = sns.FacetGrid(data=data_rs, col='model', row='rural', 
                     row_order=['rural','urban'],
                     height=int(SIZE/2.), aspect=1.0,
                     margin_titles=True, sharex=True, sharey=True)
  y = f'true_{var}_class'
  x = f'pred_{var}_class'
  minmax = min(data_rs[x].min(),data_rs[y].min()),max(data_rs[x].max(),data_rs[y].max())
  
  cbar_ax = fg.fig.add_axes([.94, .3, .02, .4])  # <-- Create a colorbar axes (https://stackoverflow.com/questions/34552770/getting-a-legend-in-a-seaborn-facetgrid-heatmap-plot)
  fg.map_dataframe(cm_heatmap, true=y, pred=x, axisminmax=minmax, norm=norm, cmap='Blues', annot=annot, square=True, annot_kws={"fontsize":10}, cbar_ax=cbar_ax)

  fg.set_titles(row_template = '{row_name}', col_template = '{col_name}')

  true_label = f'True {var}'
  fg.axes[0,0].set_ylabel(true_label);
  fg.axes[1,0].set_ylabel(true_label);
  
  pred_label = f'Predicted {var}'
  if data_rs.model.nunique()%2==0:
    for c in np.arange(data_rs.country.nunique()):
      fg.axes[-1,c].set_xlabel(pred_label);
  else:
    fg.axes[-1,int(data_rs.model.nunique()/2)].set_xlabel(pred_label);
  
  fg.figure.suptitle(country, y=1.0) #, size=16)
  fg.fig.subplots_adjust(right=.9)  # <-- Add space so the colorbar doesn't overlap the plot
  
  if output is not None:
    fn = os.path.join(output,f"sm_confusion_{var}_{COUNTRIES[country]['code']}.pdf")
    fg.savefig(fn, dpi=300)
    print(f"{fn} saved!")
  
  plt.show()
  plt.close()
  
  
def plot_confusion_matrices_final(df, var='mean', relocation='none', recency='ON', model=MODELS['cnn_aug+weighted_catboost'], norm='true', annot=False, output=None):
  
  norm = validate_norm(norm)
  var = validate_var(var)
  
  data_rs = df.query("relocation==@relocation and recency==@recency and model==@model").copy()
  
  fg = sns.FacetGrid(data=data_rs, col='country', row='rural', 
                     row_order=['rural','urban'],
                     height=int(SIZE/2.), aspect=1.0,
                     margin_titles=True, sharex=True, sharey=True)
  y = f'true_{var}_class'
  x = f'pred_{var}_class'
  minmax = min(data_rs[x].min(),data_rs[y].min()),max(data_rs[x].max(),data_rs[y].max())
  
  cbar_ax = fg.fig.add_axes([.94, .3, .02, .4])  # <-- Create a colorbar axes (https://stackoverflow.com/questions/34552770/getting-a-legend-in-a-seaborn-facetgrid-heatmap-plot)
  fg.map_dataframe(cm_heatmap, true=y, pred=x, axisminmax=minmax, norm=norm, cmap='Blues', annot=annot, square=True, annot_kws={"fontsize":10}, cbar_ax=cbar_ax)

  # [plt.setp(ax.texts, text="") for ax in fg.axes.flat]
  fg.set_titles(row_template = '{row_name}', col_template = '{col_name}')

  true_label = f'True {var}'
  fg.axes[0,0].set_ylabel(true_label);
  fg.axes[1,0].set_ylabel(true_label);
  
  pred_label = f'Predicted {var}'
  if data_rs.country.nunique()%2==0:
    for c in np.arange(data_rs.country.nunique()):
      fg.axes[-1,c].set_xlabel(pred_label);
  else:
    fg.axes[-1,int(data_rs.country.nunique()/2)].set_xlabel(pred_label);
  
  fg.fig.subplots_adjust(right=.85)  # <-- Add space so the colorbar doesn't overlap the plot
  
  if output is not None:
    fn = os.path.join(output,f"sm_confusion_final_{var}.pdf")
    fg.savefig(fn, dpi=300)
    print(f"{fn} saved!")
  
  plt.show()
  plt.close()
  
  
def plot_variability(df_rs, df_gt, output=None):
  ### Annotate start
  def annotate_corr(data, **kws):
    is_rural = data.settlement.unique()[0]=='rural'
    is_gt = data.kind.unique()[0]=='ground-truth'
    group = data.groupby('epoch')
    n = group.size().mean()
    r,p = pearsonr(data.IWI_mean, data.IWI_std)
    pv = get_statistical_significance_symbol_from_pvalue(p)
    model = ','.join(data.model.unique())
    ax = plt.gca()
    
    if not is_gt and is_rural:
      s = model
      x = 0.5
      y = 0.75
      ha = 'center'
      va = 'center'
      ax.text(x=x, y=y, s=s, color='grey', va=va, ha=ha, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='none',pad=0.0))

    n = int(round(n))
    n = (r"$\overline{n}$=<n>, " if is_gt else '').replace('<n>',str(n))
    r = (r"$\overline{r}$=<r>").replace("<r>",f"{r:.2f}")
    s = f"{n} {r} ({pv})"
    x = 1.0
    y = 0.01 if is_rural else 0.11
    ha = 'right'
    va = 'bottom'
    fs = 12
    ax.text(x=x, y=y, s=s, color=kws['color'], va=va, ha=ha, transform=ax.transAxes, size=fs, 
            bbox=dict(facecolor='white', edgecolor='none',pad=0.0))
    ### Annotate end  
  

  
  # Only necesary data (best models)
  tmp = df_rs.query("relocation==@RELOCATION_BEST and recency==@RECENCY_BEST and features in @MAIN_FEATURES").copy()
  tmp = tmp.query("(model==@SL_BEST_MODEL and country=='Sierra Leone') or (model==@UG_BEST_MODEL and country=='Uganda')").copy()
  tmp.rename(columns={'rural':'settlement'}, inplace=True)
  tmp1 = tmp.loc[:,['country','model','features','epoch','settlement','cluster_id','population_closest_tile','population_in_1mile','true_mean','true_std']].copy()
  tmp1.rename(columns={'true_mean':'IWI_mean', 'true_std':'IWI_std'}, inplace=True)
  tmp1.loc[:,'kind'] = 'ground-truth'
  tmp2 = tmp.loc[:,['country','model','features','epoch','settlement','cluster_id','population_closest_tile','population_in_1mile','pred_mean','pred_std']].copy()
  tmp2.rename(columns={'pred_mean':'IWI_mean', 'pred_std':'IWI_std'}, inplace=True)
  tmp2.loc[:,'kind'] = 'prediction'
  data = pd.concat([tmp1,tmp2],axis=0,ignore_index=True)
  del(tmp)
  del(tmp1)
  del(tmp2)

  # figure
  linestyles = ['solid','dotted','dashed']
  fg = sns.FacetGrid(data=data, col='country', row='kind', hue='settlement', hue_order=['urban','rural'], 
                     height=2.6, aspect=1., margin_titles=True, legend_out=True)
  
  # titles
  fg.set_titles(row_template = '{row_name}', col_template = '{col_name}')
  
  # true IWI mean quantiles
  xlabel = 'IWI_mean'
  ylabel = 'IWI_std'
  qold = 0
  for c in np.arange(data.country.nunique()):
      country = fg.axes[0,c].get_title()
      _,quantiles = pd.qcut(df_gt.query("country==@country").mean_wi, q=N_QUANTILES, retbins=True, precision=0)
      for i,q in enumerate(quantiles):
        print(f"[INFO] {country} Q-{i}: {q:.2f}")
        for r, kind in enumerate(data.kind.unique()):
          ax=fg.axes[r,c]
          ax.axvline(q, ls='dotted', c='lightgrey') #(0, (1, 10))
          if i>0 and ((i!=2 and country=='Sierra Leone') or (i not in [2,3] and country=='Uganda')) :
            ax.text(s=f'Q{i}', x=(q+qold)/2, y=1.0, transform=ax.get_xaxis_transform(), 
                    ha='center', c='lightgrey',va='top',size=12, bbox=dict(facecolor='white', edgecolor='none', pad=2.5)) 
        qold = q.copy()
          
  # mean vs std
  fg.map_dataframe(annotate_corr)
  fg.map_dataframe(sns.regplot, x=xlabel, y=ylabel, order=POLYNOMIAL_DEGREE, x_ci='sd', scatter=False)
  fg.add_legend()
  fg.map_dataframe(sns.scatterplot, x=xlabel, y=ylabel, alpha=0.05)
  #fg.axes[0,1].legend(loc='upper right', title='settlement')
  
  # titles (again), lables, and final touches
  fg.set_titles(row_template = '{row_name}', col_template = '{col_name}')
  add_panel_labels(fg)
  plt.tight_layout()
  plt.subplots_adjust(hspace=0.25)
  sns.move_legend(fg, "upper left", bbox_to_anchor=(0.98, 0.6))
  
  # goodnes of fit (polynomial)
  for country, df1 in data.groupby('country'):
    for kind, df2 in df1.groupby('kind'):
      for settlement, df3 in df2.groupby('settlement'): 
          _, _, _, rmse, r2 = polynomial_regression(df3.loc[:,xlabel].values, df3.loc[:,ylabel].values, confidence=95, order=POLYNOMIAL_DEGREE)
          print(f"Goodness-of-fit (deg={POLYNOMIAL_DEGREE}): rmse={rmse:.2f}, r2={r2:.2f} | {country} - {kind} - {settlement}")
        
  # savefig
  if output is not None:
    fn = os.path.join(output,f"mean_vs_std.pdf")
    fg.savefig(fn, dpi=300)
    print(f"{fn} saved!")
  
  plt.show()
  plt.close()

  
def plot_variability_full(df_rs, output=None):
  ### Annotate start
  def annotate_corr_full(data, **kws):
    is_rural = data.settlement.unique()[0]=='rural'
    is_gt = data.kind.unique()[0]=='ground-truth'
    group = data.groupby('epoch')
    n = group.size().mean()
    r,p = pearsonr(data.IWI_mean, data.IWI_std)
    pv = get_statistical_significance_symbol_from_pvalue(p)
    model = ','.join(data.model.unique())
    ax = plt.gca()
    
    n = int(round(n))
    n = (r"$\overline{n}$=<n>, " if is_gt else '').replace('<n>',str(n))
    r = (r"$\overline{r}$=<r>").replace("<r>",f"{r:.2f}")
    s = f"{n} {r} ({pv})"
    x = 1.0
    y = 0.01 if is_rural else 0.11
    ha = 'right'
    va = 'bottom'
    fs = 12
    ax.text(x=x, y=y, s=s, color=kws['color'], va=va, ha=ha, transform=ax.transAxes, 
            size=fs, bbox=dict(facecolor='white', edgecolor='none',pad=0.0))
    ### Annotate end  
    
  # Only necesary data (best models)
  tmp = df_rs.query("relocation==@RELOCATION_BEST and recency==@RECENCY_BEST and features in @MAIN_FEATURES").copy()
  tmp.rename(columns={'rural':'settlement'}, inplace=True)
  tmp1 = tmp.loc[:,['country','model','features','epoch','settlement','cluster_id','population_closest_tile','population_in_1mile','true_mean','true_std']].copy()
  tmp1.rename(columns={'true_mean':'IWI_mean', 'true_std':'IWI_std'}, inplace=True)
  tmp1.loc[:,'kind'] = 'ground-truth'
  tmp2 = tmp.loc[:,['country','model','features','epoch','settlement','cluster_id','population_closest_tile','population_in_1mile','pred_mean','pred_std']].copy()
  tmp2.rename(columns={'pred_mean':'IWI_mean', 'pred_std':'IWI_std'}, inplace=True)
  tmp2.loc[:,'kind'] = tmp2.loc[:,'model']
  data = pd.concat([tmp1,tmp2],axis=0,ignore_index=True)
  cs = CategoricalDtype(categories=['ground-truth']+list(MODELS.values()), ordered=True)
  data.loc[:,'model'] = data.loc[:,'model'].astype(cs)
  data.loc[:,'kind'] = data.loc[:,'kind'].astype(cs)
  del(tmp)
  del(tmp1)
  del(tmp2)

  # figure
  linestyles = ['solid','dotted','dashed']
  countries = data.country.unique()
  fg = sns.FacetGrid(data=data, col='country', col_order=countries, 
                     row='kind', 
                     hue='settlement', hue_order=['urban','rural'], 
                     height=3.0, aspect=1.2, margin_titles=True, legend_out=False)
  
  # titles
  fg.set_titles(row_template = '{row_name}', col_template = '{col_name}')
  
  # true IWI mean quantiles
  xlabel = 'IWI_mean'
  ylabel = 'IWI_std'
  for c, country in enumerate(countries):
    for i,q in enumerate([0.2,0.4,0.6,0.8,1.0]):
      v = np.quantile(data.query("country==@country and kind=='ground-truth'").drop_duplicates(subset=['cluster_id']).IWI_mean, q)
      print(f"[INFO] {country} Q-{q}: {v:.2f}")
      for r, kind in enumerate(data.kind.unique()):
        ax=fg.axes[r,c]
        ax.axvline(v, ls='dotted', c='lightgrey') #(0, (1, 10))
        #if r>0 and q !=0.4:
        if i+1 != 2:
          ax.text(s=f'Q{i+1}', x=v, y=25, ha='center', c='lightgrey',va='top',size=12,bbox=dict(facecolor='white', edgecolor='none',pad=3.0)) 
  
  # mean vs std
  fg.map_dataframe(annotate_corr_full)
  fg.map_dataframe(sns.regplot, x=xlabel, y=ylabel, order=POLYNOMIAL_DEGREE, scatter_kws={"alpha": 0.1},x_ci='sd',x_estimator=np.mean,scatter=False)
  fg.axes[0,1].legend(loc='upper right', title='settlement')
  
  # titles (again), lables, and final touches
  fg.set_titles(row_template = '{row_name}', col_template = '{col_name}')
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  plt.tight_layout()
  
  # goodnes of fit (polynomial)
  for country, df1 in data.groupby('country'):
    for kind, df2 in df1.groupby('kind'):
      for settlement, df3 in df2.groupby('settlement'): 
          _, _, _, rmse, r2 = polynomial_regression(df3.loc[:,xlabel].values, df3.loc[:,ylabel].values, confidence=95, order=POLYNOMIAL_DEGREE)
          print(f"Goodness-of-fit (deg={POLYNOMIAL_DEGREE}): rmse={rmse:.2f}, r2={r2:.2f} | {country} - {kind} - {settlement}")
        
  # savefig
  if output is not None:
    fn = os.path.join(output,f"sm_mean_vs_std.pdf")
    fg.savefig(fn, dpi=300)
    print(f"{fn} saved!")
  
  plt.show()
  plt.close()
  
  
def plot_cross_country_performance(df_dr, metric, results, overall=True, output=None, only={'country':None,'kind':None}):

  ### create summary first
  df_summary = pd.DataFrame(columns=['model','country',f'{metric}_within',f'{metric}_across','kind'])
  data = tbl_cross_testing(df_dr, metric=metric, output=results, save=False).reset_index().copy()
  
  kinds = ['overall', 'mean', 'std']
  if not overall:
    _ = kinds.pop(0)
    
  for kind in kinds:
      tmp = data.loc[:,['source_model','source_country']].copy()
      tmp.rename(columns={'source_model':'model', 'source_country':'country'}, inplace=True)
      tmp.loc[:,'kind'] = kind
      df_summary = pd.concat([df_summary, tmp], ignore_index=True)

      for id, row in data.iterrows():
          model = row.source_model
          country = row.source_country
          country_within_code = 'SL' if country == 'Sierra Leone' else 'UG'
          country_across_code = 'UG' if country_within_code == 'SL' else 'SL'
          var_col = "" if kind =='overall' else f'_{kind}_wi'
          within = row[f"{country_within_code}_{metric}{var_col}"]
          across = row[f"{country_across_code}_{metric}{var_col}"]

          index = df_summary.query("model==@model and country==@country and kind==@kind").index
          df_summary.loc[index,f"{metric}_within"] = within
          df_summary.loc[index,f"{metric}_across"] = across
          df_summary.loc[index,'kind'] = f"IWI_{kind}"

  palette = 'tab20' #tab10
  if only['country'] is not None and only['kind'] is not None:
    # only 1 country
    onlyc = only['country']
    onlyk = only['kind'] 
    only['kind'] = f'IWI_{onlyk}' if not onlyk.startswith('IWI_') else onlyk
    onlyk = only['kind']
    df_summary = df_summary.query("country==@onlyc and kind==@onlyk")
    markers = ['X']
    fg = sns.FacetGrid(data=df_summary.round(2), height=6, aspect=1.2, legend_out=True, sharex=False, sharey=False)
    fg.map_dataframe(sns.scatterplot, x=f'{metric}_within', y=f"{metric}_across", 
                     hue='model', palette=palette, markers=markers)
  else:
    # all countries
    fg = sns.FacetGrid(data=df_summary.round(2), col='kind',height=3.8, aspect=0.62, 
                       legend_out=True, sharex=False, sharey=False)
    markers = ['X','o']
    fg.map_dataframe(sns.scatterplot, x=f'{metric}_within', y=f"{metric}_across", hue='model', 
                     style='country',  palette=palette, markers=markers, s=60)
  fg.add_legend()
  fg.set_titles(row_template = '{row_name}', col_template = '{col_name}')
  
  ### diagonal
  num_cols = [c for c in df_summary.columns if 'within' in c or 'across' in c]
  mins = {}
  for ax in fg.axes.flatten():
    kind = only['kind'] if only['kind'] is not None else ax.get_title()
    tmp = df_summary.query("kind==@kind").copy()
    mi, ma = tmp[num_cols].min().min(), tmp[num_cols].max().max()
    ax.plot([mi,ma],[mi,ma],ls='solid',c='grey',lw=1,zorder=0)
    mins[kind] = mi

  ### best metadata-single
  for ax in fg.axes.flatten():
    kind = only['kind'] if only['kind'] is not None else ax.get_title()
    mi = df_summary.query("kind==@kind")[num_cols].min().min()
    for country,marker in zip(*[df_dr.country.unique(),markers]):
      tmp = df_dr.query("country==@country and recency==@RECENCY_BEST \
                         and relocation==@RELOCATION_BEST and features not in @MAIN_FEATURES").groupby(['model','features']).mean().reset_index()
      
      for m in ['max','min']:
        tmp2 = tmp[tmp[metric]==tmp[metric].min()].reset_index(drop=True) if m=='min' else tmp[tmp[metric]==tmp[metric].max()].reset_index(drop=True)
        var_col = metric if kind=='overall' else f'{metric}_{kind}_wi'
        val = tmp2.iloc[0][var_col.replace('IWI_','')]
        
        subi = FEATURE_MAP[tmp2.iloc[0].features]
        c = 'grey'
        
        ax.plot([mins[kind],val], [val,val], ls='dotted', color=c, zorder=0, lw=1)
        ax.plot([val,val], [mins[kind],val], ls='dotted', color=c, zorder=0, lw=1)
        ax.plot([val],[mins[kind]], marker=marker, markersize=7, markerfacecolor='black', markeredgecolor='white')
        
        y = 0.085
        if m == 'max':
          ha = 'right' if COUNTRIES[country]['code']=='SL' else 'left' if COUNTRIES[country]['code']=='UG' else 'center'
        else:
          special = COUNTRIES[country]['code']=='SL' and m=='min' and kind=='IWI_mean'
          ha = 'center' if special else 'left'
          y = -0.06 if special else y
        ax.text(s=subi, x=val, y=y, ha=ha, va='center', c=c, transform=ax.get_xaxis_transform())
  
  fg.set_xlabels(f"{metric.upper()} within-country")
  fg.set_ylabels(f"{metric.upper()} cross-country")
  
  add_panel_labels(fg)
  plt.tight_layout()
  plt.subplots_adjust(wspace=0.25, hspace=0.25)
  sns.move_legend(fg, "upper left", bbox_to_anchor=(0.935, 0.99), handletextpad=0.1)

  #plt.subplots_adjust(wspace=0.2, hspace=0.2)
  
  ### savefig
  if output is not None:
    fn = os.path.join(output,f"cross_country_performance_{metric}.pdf")
    fg.savefig(fn, dpi=300)
    print(f"{fn} saved!")
  
  plt.show()
  plt.close()
  
  
def plot_sample_weights_performance(metric, nclasses, output=None):

  metric = validate_metric(metric)
  
  if nclasses not in [4,10]:
    raise Exception("nclasses must be either 4 or 10")
    
  fn = f'results/sample_weights/summary_results_ses{nclasses}.csv'
  df = ios.load_csv(fn)
  
  df.rename(columns={metric:metric.upper()}, inplace=True)
  metric = metric.upper()
  
  def refline(data, **kws):
      n = len(data)
      ax = plt.gca()
      k = '(wENS 0.9)'
      metric = kws['metric']
      tmp = data.query("weight_kind==@k")
      x, = np.where(data.weight_kind.unique() == k)
      y = tmp[metric].mean()
      ax.axhline(y, lw=1, ls='--', c='red')
      ax.axvline(x, lw=1, ls='--', c='red')

  fg = sns.catplot(data=df, 
                   kind='point',
                   col='metric', row='country', 
                   x='weight_kind', y=metric,
                   margin_titles=True, height=2.5, aspect=1.7,
                   sharey=False,
                  )
  fg.map_dataframe(refline, metric=metric)
  fg.set_titles(row_template = '{row_name}', col_template = '{col_name}')
  fg.set_xticklabels(rotation=90)
  plt.tight_layout()
  plt.subplots_adjust(wspace=0.16, hspace=0.16)
  
  ### savefig
  if output is not None:
    fn = os.path.join(output, f'sm_performance_{metric}_sample_weights_{nclasses}classes.pdf')
    fg.savefig(fn, dpi=300)
    print(f"{fn} saved!")
  
  plt.show()
  plt.close()
  
####################################################################################################
# TABLES
####################################################################################################

def record_per_year_and_country(df):
  for group,tmp in df.groupby(['ccode','year']):
      ccode,year = group
      miny = df.query("ccode==@ccode").year.min()
      df.loc[tmp.index,'syear'] = 'Oldest' if year==miny else 'Newest'
  tmp = df.groupby(['ccode','syear']).size().reset_index(name='records')
  tmp = tmp.pivot_table(index="syear", columns='ccode')
  return tmp
  
def compute_gini_per_country(df, column, year=False):
  if year:
    cols = ['Country','Year','Gini']
    tmp = pd.DataFrame(columns=cols)
    for (country,year),data in df.groupby(['ccode','year']):
      g = gini(data[column].values)
      tmp = pd.concat([tmp,pd.DataFrame({'Country':[country], 'Year':year, 'Gini':[g]}, index=[0])], ignore_index=True)
  else:
    cols = ['Country','Gini']
    tmp = pd.DataFrame(columns=cols)
    for country,data in df.groupby(['ccode']):
      g = gini(data[column].values)
      tmp = pd.concat([tmp,pd.DataFrame({'Country':[country], 'Gini':[g]}, index=[0])], ignore_index=True)
  return tmp

def tbl_features_summary(root, countries, years, output=None):
  df_summary = pd.DataFrame()
  for country in countries:
    files = [fn for fn in glob.glob(os.path.join(root,country,'results','features','clusters',"*.csv")) if validations.valid_file_year(fn, years[country]) \
             and not fn.endswith('_cluster.csv')]
    
    for fn in files:
      source = fn.split('_')[-1].replace('.csv','')
      tmp = ios.load_csv(fn)
      obj = {'country':country, 'source':source, 'n_features':tmp.columns.shape[0]-1}
      df_summary = pd.concat([df_summary, pd.DataFrame(obj, index=[1])], ignore_index=True)
    
    obj = {'country':[country]*2, 'source':['GMSA','settlement'], 'n_features':[784,1]}
    df_summary = pd.concat([df_summary, pd.DataFrame(obj, index=[1,2])], ignore_index=True)
    
  df_summary = df_summary.pivot_table(index='source', columns='country', values='n_features')
  
  if output is not None:
    fn = os.path.join(output, f'features.tex')
    tex = df_summary.to_latex()
    ios.write_txt(tex, fn)
    
  print(df_summary.sum(axis=0))
  return df_summary

def tbl_data_summary(root, countries, years, output=None):
  df_summary = pd.DataFrame()
  for country in countries:
    fn_households = [fn for fn in glob.glob(os.path.join(root,country,'results','features','households',"*_household.csv")) if validations.valid_file_year(fn, years[country])][0]
    fn_clusters = [fn for fn in glob.glob(os.path.join(root,country,'results','features','clusters',"*_cluster.csv")) if validations.valid_file_year(fn, years[country])][0]
    fn_pplaces = glob.glob(os.path.join(root,country,'results','features','pplaces',"PPLACES.csv"))[0]
    fn_relocation = [fn for fn in glob.glob(os.path.join(root,country,'results','features',"*ruc_cluster_pplace_ids.csv")) if validations.valid_file_year(fn, years[country])][0]
    
    df_households = ios.load_csv(fn_households)
    print(f"[INFO] {fn_households} loaded!")
    df_clusters = ios.load_csv(fn_clusters)
    print(f"[INFO] {fn_clusters} loaded!")
    df_pplaces = ios.load_csv(fn_pplaces)
    print(f"[INFO] {fn_pplaces} loaded!")
    df_relocation = ios.load_csv(fn_relocation)
    print(f"[INFO] {fn_relocation} loaded!")
    
    c = df_clusters.shape[0]
    cu = df_clusters.query('rural==0').shape[0]
    cr = df_clusters.query('rural==1').shape[0]
    r,p = pearsonr(df_clusters.mean_wi, df_clusters.std_wi)
    pv = get_statistical_significance_symbol_from_pvalue(p)
    g = gini(df_clusters.mean_wi.values)
    
    pp = df_pplaces.shape[0]
    ppu = df_pplaces.query("rural==0").shape[0]
    ppr = df_pplaces.query("rural==1").shape[0]
    
    rl = df_relocation.query("pplace_cluster_distance>0").shape[0]
    rlu = df_relocation.query("pplace_cluster_distance>0 and cluster_rural==0").shape[0]
    rlr = df_relocation.query("pplace_cluster_distance>0 and cluster_rural==1").shape[0]
    
    obj = {'country':country,
           'years':','.join(df_clusters.year.astype(str).unique()),
           'households':df_households.shape[0],
           'clusters':c,
           'urban':f"{cu} ({cu*100/c:.0f}%)",
           'rural':f"{cr} ({cr*100/c:.0f}%)",
           'IWI mean min':f"{df_clusters.mean_wi.min():.1f}",
           'IWI mean max':f"{df_clusters.mean_wi.max():.1f}",
           'IWI mean mean':f"{df_clusters.mean_wi.mean():.1f}",
           'IWI mean std.dev.':f"{df_clusters.mean_wi.std():.1f}",
           'IWI std min':f"{df_clusters.std_wi.min():.1f}",
           'IWI std max':f"{df_clusters.std_wi.max():.1f}",
           'IWI std mean':f"{df_clusters.std_wi.mean():.1f}",
           'IWI std std.dev.':f"{df_clusters.std_wi.std():.1f}",
           'pearson':f"{r:.2f} ({pv})",
           'gini':f"{g:.2f}",
           'WB gini index':WB_INDICATORS[country]['gini'],
           'WB gdp per capita (US $)':WB_INDICATORS[country]['gdp_per_capita'],
           'WB growth anual (%)':WB_INDICATORS[country]['gdp_growth'],
           'populated places':pp,
           'pplaces urban':f"{ppu} ({ppu*100/pp:.0f}%)",
           'pplaces rural':f"{ppr} ({ppr*100/pp:.0f}%)",
           'relocation':f"{rl} ({rl*100/c:.0f}%)",
           'relocation urban':f"{rlu} ({rlu*100/cu:.0f}%)",
           'relocation rural':f"{rlr} ({rlr*100/cr:.0f}%)",
          }
    df_summary = pd.concat([df_summary,pd.DataFrame(obj,index=[1])], ignore_index=True)
    
  df_summary.set_index('country', inplace=True)
  df_summary = df_summary.T
  
  if output is not None:
    fn = os.path.join(output, f'datasets.tex')
    tex = df_summary.to_latex()
    ios.write_txt(tex, fn)
    
  return df_summary

def tbl_performance_by_model(df, metric='rmse', include_baselines=False, output=None):
  
  metric = validate_metric(metric)
  q = "" if include_baselines else "and (weighted=='yes' or augmented=='yes')"
  data = df.query(f"recency==@RECENCY_BEST and relocation==@RELOCATION_BEST and features in @MAIN_FEATURES {q}").copy()
  index = data.loc[:,['model','features']].drop_duplicates()
  data = data.groupby(['country','model','features']).mean().reset_index()

  columns = ['model','features']
  columns.extend([f"{metric}{var}{COUNTRIES[c]['code']}" for c in data.country.unique() for var in ['_','_mean_','_std_']])
  df_results = pd.DataFrame(columns=columns)
  
  df_results = pd.concat([df_results, index])
  df_results.set_index('model', inplace=True)
  
  for c, df in data.groupby('country'):
    tmp = df.rename(columns={f"{metric}{'' if var=='_' else f'{var}wi'}":f"{metric}{var}{COUNTRIES[c]['code']}" for var in ['_','_mean_','_std_']}).copy()
    tmp.drop(columns=[c for c in tmp.columns if not c.startswith(metric) and c not in ['model','features']], axis=1, inplace=True)
    tmp.dropna(inplace=True)
    tmp.set_index('model', inplace=True)
    df_results.loc[tmp.index,tmp.columns] = tmp

  df_results = df_results.sort_index()
  if output is not None:
    fn = os.path.join(output, f'performance_by_model_{metric}.tex')
    tex = df_results.to_latex(float_format=lambda x: '%.2f' % x)
    ios.write_txt(tex, fn)
    
  return df_results

def tbl_performance_by_datasource(df, metric='rmse', include_images=False, output=None):
  
  metric = validate_metric(metric)
  q = "or model==@MODEL_CNNA" if include_images else ''
  data = df.query(f"recency==@RECENCY_BEST and relocation==@RELOCATION_BEST and (features not in @MAIN_FEATURES {q})").copy()
  index = data.loc[:,['model','features']].drop_duplicates()
  data = data.groupby(['country','model','features']).mean().reset_index()

  columns = ['model','features']
  columns.extend([f"{metric}{var}{COUNTRIES[c]['code']}" for c in data.country.unique() for var in ['_','_mean_','_std_']])
  df_results = pd.DataFrame(columns=columns)
  
  df_results = pd.concat([df_results, index])
  df_results.set_index('features', inplace=True)
  
  for c, df in data.groupby('country'):
    tmp = df.rename(columns={f"{metric}{'' if var=='_' else f'{var}wi'}":f"{metric}{var}{COUNTRIES[c]['code']}" for var in ['_','_mean_','_std_']}).copy()
    tmp.drop(columns=[c for c in tmp.columns if not c.startswith(metric) and c not in ['model','features']], axis=1, inplace=True)
    tmp.dropna(inplace=True)
    tmp.set_index('features', inplace=True)
    df_results.loc[tmp.index,tmp.columns] = tmp

  df_results = df_results.sort_index()
  if output is not None:
    fn = os.path.join(output, f'performance_by_datasource_{metric}.tex')
    tex = df_results.to_latex(float_format=lambda x: '%.2f' % x)
    ios.write_txt(tex, fn)
    
  return df_results

def tbl_performance_by_model_and_datasource(df, metric='rmse', overall=True, output=None):
  df_results_model = tbl_performance_by_model(df=df, metric=metric, include_baselines=True, output=output).reset_index()
  df_results_source = tbl_performance_by_datasource(df=df, metric=metric, include_images=False, output=output).reset_index()
  
  
  for k,v in FEATURE_MAP.items():
    df_results_model.loc[:,v] = df_results_model.features.apply(lambda f:"y" if (f=='images' and v=='Im') or 
                                                                                (f=='metadata' and v!='Im') or 
                                                                                (f=='images+metadata') else '-')
    df_results_source.loc[:,v] = df_results_source.features.apply(lambda f:"y" if (k==f) else '-')
    
  df_results = df_results_model.copy()
  df_results = pd.concat([df_results, df_results_source.loc[:,df_results.columns]], ignore_index=True)
  feature_cols = [v for k,v in FEATURE_MAP.items()]
  df_results.set_index(['model'] + feature_cols , inplace=True)
  df_results.drop(columns=['features'], inplace=True)
  
  if not overall:
    df_results.drop(columns=[c for c in df_results.columns if len(c.split('_'))==2 and c.split('_')[0] in ['nrmse','rmse','mse','r2']], inplace=True)

  if output is not None:
    fn = os.path.join(output, f'performance_by_model_and_datasource_{metric}.tex')
    tex = df_results.to_latex(float_format=lambda x: '%.2f' % x)
    ios.write_txt(tex, fn)
  
  return df_results

def assign_classes(df, df_gt, n_classes=3, class_type='class', var='mean',show_warning=False):
  class_type = validate_class_type(class_type)
  var = validate_var(var)
  labels = np.arange(1,n_classes+1,1)
  country = ', '.join(df.country.unique())
  
  tmp = df.drop_duplicates(subset=['cluster_id'])
  
  if class_type in ['discrete','class']:
    if class_type == 'discrete':
      _,bins = pd.cut(pd.concat([tmp.true_mean, tmp.pred_mean], axis=0, ignore_index=True),n_classes,retbins=True,precision=0,include_lowest=True)
    elif class_type == 'class':
      _,bins = pd.cut(np.arange(0,1+(100 if var=='mean' else 30)),n_classes,retbins=True,precision=0,include_lowest=True)
    df.loc[:,f'true_{var}_class'] = pd.cut(df.true_mean, bins=bins, labels=labels)
    df.loc[:,f'pred_{var}_class'] = pd.cut(df.pred_mean, bins=bins, labels=labels)
    print(f'[INFO] {country} bins: {np.round(bins,2)}')
  
  elif class_type == 'quantiles':
    if var!='mean':
      print(f'[INFO] quantiles are only measured on the mean wealth index. Using true_mean instead of true_{var} to create the quantiles.')
    
    _,bins = pd.qcut(df_gt.query("country==@country").mean_wi, n_classes, labels=labels, precision=0, retbins=True)
    df.loc[:,f'true_{var}_class'] = pd.cut(df.true_mean, labels=labels, bins=bins)
    df.loc[:,f'pred_{var}_class'] = pd.cut(df.pred_mean, labels=labels, bins=bins)
    
    if show_warning:
      print("[WARNING] Bins (or discretization) is not the same for true and predicted classes.")
      print(f'[INFO] {country} bins (true&pred class): {np.round(bins,2)}')

  return df, labels, class_type
  
def tbl_intersectionality(df, df_gt, n_classes=3, class_type='class', var='mean', evaluation='recall', output=None):
  evaluation = validate_evaluation(evaluation)
  models = MODELS.values()
  
  data = df.query("recency==@RECENCY_BEST and relocation==@RELOCATION_BEST and model in @models").copy()
  data.rename(columns={'rural':'settlement'}, inplace=True)
  
  # quantiles per country
  new_cols = [f'true_{var}_class',f'pred_{var}_class']
  for country, df in data.groupby('country'):
    df, labels, class_type = assign_classes(df, df_gt, n_classes, class_type, var, evaluation in ['recall','precision','f1'])
    if new_cols[0] not in data.columns:
      data = data.merge(df.loc[:,new_cols],left_index=True, right_index=True, how='left')
    else:
      data.loc[df.index,new_cols] = df.loc[df.index,new_cols]
    
  # aggregated metrics
  if evaluation in ['recall','precision','f1']:
    fnc = recall_score if evaluation=='recall' else precision_score if evaluation=='precision' else f1_score
    data = data.groupby(['country','settlement','model']).apply(lambda group: fnc(group[f'true_{var}_class'], group[f'pred_{var}_class'], 
                                                                                  labels=labels, average=None)).reset_index(name=evaluation)
  elif evaluation in ['rmse','mse']:
    fnc = mean_squared_error
    data = data.groupby(['country','settlement','model']).apply(lambda g1: [0 if g2.shape[0]==0 else fnc(g2[f'true_{var}'], g2[f'pred_{var}'], squared=evaluation=='mse') 
                                                                for _,g2 in g1.groupby(f'true_{var}_class')]).reset_index(name=evaluation)
  elif evaluation in ['nrmse']:
    fnc = mean_squared_error
    data = data.groupby(['country','settlement','model']).apply(lambda g1: [0 if g2.shape[0]==0 else fnc(g2[f'true_{var}'], g2[f'pred_{var}'], squared=False)/g2[f'true_{var}'].std()
                                                                for _,g2 in g1.groupby(f'true_{var}_class')]).reset_index(name=evaluation)
  else:
    fnc = r2_score
    data = data.groupby(['country','settlement','model']).apply(lambda g1: [0 if g2.shape[0]==0 else fnc(g2[f'true_{var}'], g2[f'pred_{var}']) 
                                                                  for _,g2 in g1.groupby(f'true_{var}_class')]).reset_index(name=evaluation)
  data.loc[:,labels] = data.loc[:,evaluation].tolist()
  data.drop(columns=[evaluation], inplace=True)

  columns = ['model','settlement']
  columns.extend([f"Q{l}_{COUNTRIES[c]['code']}" for c in data.country.unique() for l in labels])
  df_results = pd.DataFrame(columns=columns)
  df_results = pd.concat([df_results, pd.DataFrame(list(product(data.settlement.unique(),data.model.unique())), columns=['settlement','model'])])
  df_results.set_index(['settlement','model'], inplace=True)

  for group, df in data.groupby(['country','settlement']):
    c = group[0]
    s = group[1]
    colsr = {l:f"Q{l}_{COUNTRIES[c]['code']}" for l in labels}
    tmp = df.rename(columns=colsr).copy()
    tmp.drop(columns=['country'], axis=1, inplace=True)
    tmp.set_index(['settlement','model'], inplace=True)
    df_results.loc[tmp.index,tmp.columns] = tmp

  df_results = df_results.loc[:, (df_results != 0).any(axis=0)]
  df_results = df_results.replace(0, '-')
  df_results = df_results.fillna('')
  
  if output is not None:
    fn = os.path.join(output, f'intersectionality_{evaluation}_{var}_{n_classes}{class_type}.tex')
    tex = df_results.to_latex(float_format=lambda x: '%.2f' % x)
    ios.write_txt(tex, fn)
    
  return df_results


def tbl_cross_testing(df_dr, metric='r2', output=None, save=True):
  ''' summaries generated in:
      scripts/batch_cross_predictions.py (for all possibilities)
      notebooks/_Prediction_Across_Countries.ipynb (for a couple of examples)
  '''
  # param validation
  if output is None:
    raise Exception('output (path) must be set.')  
  metric = validate_metric(metric)
  
  # loading cros-country-test results
  fn_cc = os.path.join(output, 'performance_cross_country_testing.csv')
  df_cr = pd.DataFrame()
  if ios.exists(fn_cc):
    df_cr = ios.load_csv(fn_cc)
  else:
    raise Exception("Run first: scripts/batch_cross_predictions.py")
      
  # merging with same-country-test results
  data = df_cr.copy()
  data.rename(columns={c:f"{c}_wi" for c in data.columns if "_mean" in c or '_std' in c}, inplace=True)
  for (source_country, source_model), _ in df_cr.groupby(['source_country','source_model']):
    tmp = df_dr.query("country==@source_country and model==@source_model and recency==@RECENCY_BEST and relocation==@RELOCATION_BEST and features in @MAIN_FEATURES").copy()
    tmp.loc[:,'target_country'] = source_country
    tmp.loc[:,'target_runid'] = tmp.epoch
    tmp.loc[:,'target_rs'] = tmp.rs

    tmp.rename(columns={'country':'source_country',
                        'model':'source_model',
                        'epoch':'source_runid',
                        'rs':'source_rs',}, inplace=True)
    cols = data.columns
    data = pd.concat([data, tmp.loc[:,cols]], ignore_index=True)

  df_results = pd.DataFrame()
  for var in [None,'mean','std']:
    column = f'{metric}_{var}_wi' if var is not None else metric
    tmp = data.pivot_table(index=['source_country','source_model'], columns='target_country', values=column, aggfunc=np.mean).reset_index()
    tmp.rename(columns={'Sierra Leone':f'SL_{column}', 'Uganda':f'UG_{column}'}, inplace=True)
    tmp.columns.name = ''

    new_cols = [f'SL_{column}',f'UG_{column}']
    if df_results.shape[0]==0:
      df_results = tmp.copy()
    elif new_cols[0] not in df_results.columns:
      df_results = df_results.merge(tmp.loc[:,new_cols],left_index=True, right_index=True, how='left')
    else:
      df_results.loc[tmp.index,new_cols] = tmp.loc[tmp.index,new_cols]

  df_results.loc[:,'source_model'] = df_results.loc[:,'source_model'].astype(MOD_TYPE)
  df_results.sort_values('source_model', inplace=True)
  df_results.set_index('source_model', inplace=True)
  if output is not None and save:
    fn = os.path.join(output, f'performance_same_and_cross_country_{metric}.tex')
    tex = df_results.to_latex(float_format=lambda x: '%.2f' % x)
    ios.write_txt(tex, fn)
    
  return df_results


def get_feature_counts_from_gt(df_gt):
  metadata_features = [c for c in df_gt.columns if c not in ['year','lon','lat','mean_wi','std_wi','country']]
  feature_counts = {}
  for f in ['FBP','FBM','FBMV','NTLL','OCI','SETTL','OSM','GMSA']: # do not change the order of the last 2
      if f=='FBP':
          fn = sum([1 for c in df_gt.columns if c.startswith('population') or c in ['distance_closest_tile']])
      elif f=='FBM':
          fn = sum([1 for c in df_gt.columns if c.startswith('FBM_')])
      elif f=='FBMV':
          fn = sum([1 for c in df_gt.columns if c.startswith('FBMV_')])
      elif f=='NTLL':
          fn = sum([1 for c in df_gt.columns if c.startswith('NTLL_')])
      elif f=='OCI':
          fn = sum([1 for c in df_gt.columns if 'cells_' in c or 'towers_' in c or c in ['distance_closest_cell']])
      elif f=='GMSA':
          fn = 784
      elif f=='SETTL':
          fn = 1
      elif f=='OSM':
          fn = len(metadata_features) - sum(feature_counts.values())
      print(f, fn)
      feature_counts[f] = fn
  return feature_counts

