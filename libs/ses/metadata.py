################################################################################
# Dependencies
################################################################################
import os
import glob
import time
import json
import shap
import numpy as np 
import pandas as pd

import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from catboost import Pool

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.multioutput import MultiOutputRegressor

from utils import ios
from utils import viz
from utils import system
from utils import validations

################################################################################
# Constants
################################################################################

from utils.constants import *

################################################################################
# Class
################################################################################
class SESMetadata(object):

    def __init__(self, path, traintype='all', features_source='all', timevar=False, runid=1, cv=5, 
                 fold=1, include_fmaps=False, random_state=None, weighted=False, n_jobs=1):      
      self.path = path
      self.ttype = traintype
      self.features_source = features_source
      self.timevar = timevar
      self.runid = runid
      self.cv = cv
      self.fold = fold
      self.include_fmaps = include_fmaps
      self.random_state = np.random.default_rng().integers(999999999) if random_state is None else int(random_state)
      self.weighted = weighted
      self.n_jobs = n_jobs
      
      self.df_tuning = None
      self.df_best = None

      if self.path:
        self.output_folder = os.path.join(self.path, 
                                          f'xgb-{self.features_source}{f"-{self.timevar}" if self.timevar is not None else ""}', 
                                          'weighted' if weighted else '')
        ios.validate_path(self.output_folder)

      # if not self.pplaces:
      validations.validate_traintype(self.ttype)
      np.random.seed(self.random_state)
  
    ############################################################################
    # HYPER-PARAM TUNING: TRAIN + VAL
    ############################################################################
    
    def tuning(self, X_train, y_train, X_val, y_val, feature_names, weights_t=None):
      ### get pending combinations
      df_tuning = self.load_tuning_file()
      
      tmp = df_tuning.query(f"loss_fold{self.fold} in @NONE")
      if tmp.shape[0]==0:
        return # all done

      print('Train:', X_train.shape, y_train.shape, len(feature_names))
      print('Val:', X_val.shape, y_val.shape)
      print('Njobs:', self.n_jobs)
      
      for i,row in tqdm(tmp.iterrows(), total=tmp.shape[0]):
        hparams = json.loads(row.params.replace("'",'"'))
        
        # fit
        start = time.time()
        
        cat_train = Pool(X_train, label = y_train, weight=weights_t)
        cat_val = Pool(X_val, label = y_val)
        model = CatBoostRegressor(**hparams)
        model.fit(cat_train, eval_set=cat_val, use_best_model=True, verbose=False)
        model.set_feature_names(feature_names)
        
        fit_time = time.time() - start

        #eval
        start = time.time()
        y_pred = model.predict(X_val) # why there can be nans?
        if np.isnan(y_pred).sum() > 0:
          print("[WARNING] NANs in prediction")
          y_pred = np.nan_to_num(y_pred, nan=-100)
          print(y_pred)
        mse = mean_squared_error(y_val, y_pred)
        eval_time = time.time() - start

        # update
        self.update_tuning(df_tuning, i, mse, fit_time, eval_time)
        
    def get_all_hparams(self):
      
      # CATBoost
      hparams = {'loss_function': ['MultiRMSE'], 
                 'eval_metric': ['MultiRMSE'],
                 'bootstrap_type':['Bernoulli'],
                 'boosting_type':['Plain'],
                 'grow_policy':['Lossguide'],
                 'score_function':['Cosine','L2'],
                 'iterations': [10,50,100,200,300,500],
                 'learning_rate': [0.01,0.05,0.1,0.15,0.2,0.25,0.3], #learning rate
                 'random_seed':[int(self.random_state)],
                 'l2_leaf_reg': [0.,0.5,1,1.5,2,2.5,3,3.5], # L2 reg weights
                 'subsample': [0.5,0.6,0.8,0.9,1.0],
                 'random_strength':[0.9,1.0],
                 'best_model_min_trees':[2,4],
                 'depth': [0,3,4,5,6,7,8,9,10,12,15],
                 'min_data_in_leaf':[5,10,15,20,25,30],
                 'max_leaves': [3,5,10,20,31,40],
                 'rsm': [0.1,0.3,0.5,0.7,0.9,1.0],
                 'early_stopping_rounds':[20],}
      
      combinations = np.prod([len(v) for v in hparams.values()])
      print("- All possible combinations hyper-params: ", combinations)
      n_iter = min(XGB_TUNING_ITER,int(round(combinations/2)))
      print("- # random candidates: ", n_iter)
      return hparams, n_iter, combinations

    def get_tuning_fn(self):
      fn = os.path.join(self.output_folder, 'tuning.csv')
      os.makedirs(os.path.dirname(fn), exist_ok=True)
      return fn

    def load_tuning_file(self):
      
      class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
              return 'true' if obj else 'false'
            return json.JSONEncoder.default(self, obj)

      fn = self.get_tuning_fn()
      if ios.exists(fn):
        df = ios.load_csv(fn)
      else:
        # hparams
        hparams, n_iter, combinations = self.get_all_hparams()
        # populate
        cols = ['rs','runid','mean_fit_time','std_fit_time','mean_eval_time','std_eval_time','params']
        cols.extend([f'param_{k}' for k,v in hparams.items()])
        cols.extend([f'loss_fold{k}' for k in np.arange(1, self.cv+1, 1)])
        cols.extend(['mean_loss','std_loss','rank'])
        df = pd.DataFrame(columns=cols)
        # strhp = ['objective','eval_metric','booster','tree_method'] #xgbregressor
        # strhp = ['boosting_type','objective','importance_type'] #lgbm
        strhp = ['loss_function','eval_metric','boosting_type','score_function','bootstrap_type','grow_policy'] # catboost
        
        while True:
          if df.shape[0] == n_iter:
            break
          hp = SESMetadata.get_random_hyper_params_combination(hparams)
          q = " and ".join([f"param_{k}=='{v}'" if k in strhp else f"param_{k}=={v}" for k,v in hp.items()])
          if df.query(q).shape[0] == 0:
            obj = {f'param_{k}':f'{v}' if k in strhp else v for k,v in hp.items()}
            obj.update({"rs":self.random_state})
            obj.update({"runid":self.runid})
            obj.update({"params":json.dumps(hp, cls=NpEncoder)})
            df = df.append(pd.DataFrame(obj, index=[0], columns=cols), ignore_index=True)
        ios.save_csv(df, fn)  
      return df

                                                                             
    @staticmethod
    def get_random_hyper_params_combination(hparams):
        hp = {}
        for k,v in hparams.items():
          hp[k] = np.random.choice(v)
        return hp

    @staticmethod
    def get_fmaps(path, setname, fmap_layer_id=None, fold=None):
      try:
        postfix = f"_{fold}" if fold is not None else ""
        path = os.path.join(path,f"layer-{fmap_layer_id}") if fmap_layer_id is not None else path
        fn = os.path.join(path,f'fmap_{setname}{postfix}.npz')
        print(f"{fn} loading...")
        fmap = ios.load_array(fn)['arr_0']
      except Exception as ex:
        print(ex)
        fmap = None
      return fmap
      
    @staticmethod
    def get_weights(X_train, y_train, feature_names, y_attributes):
      beta = 0.9
      n_classes = 10
      train_class = pd.DataFrame(X_train, columns=feature_names)
      train_class.loc[:,y_attributes] = y_train.copy()
      train_class.loc[:,'ses'] = train_class.apply(lambda row: int(row.mean_wi*n_classes/100), axis=1)
      weights_t = train_class.apply(lambda row: SESMetadata.set_weights_wENS(train_class,beta,row,n_classes), axis=1) 
      return weights_t
    
    @staticmethod
    def set_weights_wENS(df, beta, row, n_classes):
      tmp = df.groupby('ses').size()
      ses= int(row.mean_wi*n_classes/100)
      nc = tmp[ses]
      return (1-beta) / (1-(beta**nc))

    def update_tuning(self, df, index, mse, fit_time, eval_time):
      df.loc[index,f'loss_fold{self.fold}'] = mse
      df.loc[index,'mean_fit_time'] = f"{df.loc[index,'mean_fit_time']},{fit_time}" if df.loc[index,'mean_fit_time'] not in NONE else f"{fit_time}"
      df.loc[index,'mean_eval_time'] = f"{df.loc[index,'mean_eval_time']},{eval_time}" if df.loc[index,'mean_eval_time'] not in NONE else f"{eval_time}"
      ios.save_csv(df, self.get_tuning_fn(), verbose=False)

    def sumarize_tuning(self, df):
      if df.mean_loss.isnull().values.any():
        loss_cols = [c for c in df.columns if c.startswith('loss_fold')]
        df.loc[:,'mean_loss'] = df.apply(lambda row: np.mean([row[c] for c in loss_cols]), axis=1)
        df.loc[:,'std_loss'] = df.apply(lambda row: np.std([row[c] for c in loss_cols]), axis=1)
        df.loc[:,'rank'] = df.mean_loss.rank(na_option='bottom',ascending=True,pct=False)
        df.loc[:,'std_fit_time'] = df.mean_fit_time.apply(lambda c: np.std([float(t) for t in c.split(',') if t not in NONE]))
        df.loc[:,'mean_fit_time'] = df.mean_fit_time.apply(lambda c: np.mean([float(t) for t in c.split(',') if t not in NONE]))
        df.loc[:,'std_eval_time'] = df.mean_eval_time.apply(lambda c: np.std([float(t) for t in c.split(',') if t not in NONE]))
        df.loc[:,'mean_eval_time'] = df.mean_eval_time.apply(lambda c: np.mean([float(t) for t in c.split(',') if t not in NONE]))
        ios.save_csv(df, self.get_tuning_fn(), verbose=False)

    ############################################################################
    # EVALUATION: TRAIN + TEST
    ############################################################################

    def oos_evaluation(self, X_train, y_train, X_test, y_test, feature_names, weights_t=None):
      # summary
      fn = self.get_tuning_fn()
      df_tuning = ios.load_csv(fn)
      if df_tuning.mean_loss.isnull().values.any():
        self.sumarize_tuning(df_tuning)

      # best
      self.df_best = df_tuning.query("rank==1").iloc[0]
      hparams = json.loads(self.df_best.params.replace("'",'"'))
      print(hparams)

      # CatBoost model
      hparams['early_stopping_rounds'] = 100
      cat_train = Pool(X_train, label = y_train, weight=weights_t)
      self.model = CatBoostRegressor(**hparams)
      self.model.fit(cat_train, verbose=False)
      self.model.set_feature_names(feature_names)
      
      # save model
      fn,f = self.get_model_fn()
      self.model.save_model(fn, format=f)
      #ios.save_h5(model, fn)

      # predict
      y_pred = self.model.predict(X_test) 
      return y_pred

    def get_evaluation_fn(self):
      return os.path.join(self.output_folder, 'evaluation.json')
    
    def save_evaluation(self, y_true, y_pred):
      self.metrics = {'mae':None, 'mse':None, 'rmse':None, 'r2':None, 
                  'mae_mean':None, 'mse_mean':None, 'rmse_mean':None, 'r2_mean':None, 
                  'mae_std':None, 'mse_std':None, 'rmse_std':None, 'r2_std':None, 
                  'corr_true':(None,None), 'corr_pred':(None,None)}

      # overall
      mae = mean_absolute_error(y_true, y_pred)
      rmse = mean_squared_error(y_true, y_pred, squared=False)
      mse = mean_squared_error(y_true, y_pred, squared=True)
      r2 = r2_score(y_true, y_pred)
      self.metrics['mae'] = mae
      self.metrics['mse'] = mse
      self.metrics['rmse'] = rmse
      self.metrics['r2'] = r2

      # For each output variable
      for i,name in enumerate(['mean','std']):
        yt = y_true[:,i]
        yp = y_pred[:,i]
        mae = mean_absolute_error(yt, yp)
        rmse = mean_squared_error(yt, yp, squared=False)
        mse = mean_squared_error(yt, yp, squared=True)
        r2 = r2_score(yt, yp)
        self.metrics[f'mae_{name}'] = mae
        self.metrics[f'mse_{name}'] = mse
        self.metrics[f'rmse_{name}'] = rmse
        self.metrics[f'r2_{name}'] = r2
        
      self.metrics['corr_true'] = pearsonr(y_true[:,0],y_true[:,1])
      self.metrics['corr_pred'] = pearsonr(y_pred[:,0],y_pred[:,1])

      fn = self.get_evaluation_fn() #os.path.join(self.output_folder, 'evaluation.json')
      ios.save_json(self.metrics, fn)

    def plot_evaluation(self, y_true, y_pred):
      fn = os.path.join(self.output_folder, 'plot_pred_true.png')
      viz.plot_pred_true(y_pred, y_true, {k:v for k,v in self.metrics.items() if "_" not in k}, fn=fn)

    def load_evaluation(self):
      fn = self.get_evaluation_fn()
      return ios.load_json(fn)
      
    def load_model(self):
      fn,f = self.get_model_fn()
      model = CatBoostRegressor()
      model.load_model(fn, format=f)
      return model
      
    def get_model_fn(self):
      return os.path.join(self.output_folder, 'model.json'), 'json'
      
    ############################################################################
    # EVALUATION: TRAIN + TEST
    ############################################################################
    
    def save_predictions(self, test, y_test, y_pred, y_attributes):
      fn = os.path.join(self.output_folder, 'test_pred_xgb.csv')
      df = test.loc[:,['cluster_id']] # @TODO here it should be gtID
      for ia, at in enumerate(y_attributes):
        df.loc[:,f'true_{at}'] = y_test[:,ia]
        df.loc[:,f'pred_{at}'] = y_pred[:,ia]
      ios.save_csv(df, fn)
      
    def save_feature_importance(self, X_test, y_test, y_pred, y_attributes, feature_names):
      maxf = 30
      
      try:
        fn = os.path.join(self.output_folder, 'features.txt')
        ios.write_list_to_txt(feature_names, fn)
      except Exception as ex:
        print(f"[ERROR] metadata.py | save_feature_importance | features | {ex}")
        
      try:
        self.feature_importance = pd.DataFrame({'feature_name':feature_names, 
                                                'importance':self.model.feature_importances_})
        fn = os.path.join(self.output_folder, 'feature_importance.csv')
        ios.save_csv(self.feature_importance, fn)
      except Exception as ex:
        print(f"[ERROR] metadata.py | save_feature_importance | csv | {ex}")
           
      try:
        fn = os.path.join(self.output_folder, 'feature_importance.pdf')
        viz.plot_feature_importance(self.feature_importance, figsize=(10, 10), max_num_features=30, fn=fn)
      except Exception as ex:
        print(f"[ERROR] metadata.py | save_feature_importance | plot | {ex}")
        
      try:
        explainer = shap.Explainer(self.model)
        shap_values = explainer(pd.DataFrame(X_test, columns=feature_names))
      except Exception as ex:
        print(f"[ERROR] metadata.py | save_feature_importance | shap values | {ex}")
      
      try:  
        df_shap = pd.DataFrame()
        for y_index,yat in enumerate(y_attributes):
          # values
          shap_importance = shap_values.abs.mean(0).values[:,y_index]
          sorted_idx = shap_importance.argsort()
          tmp = pd.DataFrame({'shap_value':shap_importance[sorted_idx], 'feature':np.array(feature_names)[sorted_idx], 'output':yat})
          df_shap = pd.concat([df_shap, tmp])

          # plot
          shap.plots.bar(shap_values[:,:,y_index], max_display=maxf, show=False)
          fn = os.path.join(self.output_folder, f'shap_{yat}.pdf')
          plt.tight_layout()
          plt.savefig(fn)
          plt.close()
          
        # save shap values
        fn = os.path.join(self.output_folder, f'shap_values.csv')
        ios.save_csv(df_shap, fn)
      except Exception as ex:
        print(f"[ERROR] metadata.py | save_feature_importance | shap plot | {ex}")

      return 

