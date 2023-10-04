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
# from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from catboost import Pool

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputRegressor

from utils import ios
from utils import viz
from utils import system
from utils import validations
# from utils import predictions as utils

################################################################################
# Constants
################################################################################

from utils.constants import *

################################################################################
# Class
################################################################################
class SESMetadata(object):

    def __init__(self, path, traintype='all', features_source='all', timevar=False, runid=1, cv=5, fold=1, include_fmaps=False, random_state=None, weighted=False, n_jobs=1):      
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
        self.output_folder = os.path.join(self.path, f'xgb-{self.features_source.replace(",","_")}{f"-{self.timevar}" if self.timevar is not None else ""}', 'weighted' if weighted else '')
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
        # model = MultiOutputRegressor(XGBRegressor(objective=params['objective'], 
        #                                             eval_metric=params['eval_metric'],
        #                                             n_estimators=params['n_estimators'],
        #                                             max_depth=params['max_depth'],
        #                                             learning_rate=params['learning_rate'],
        #                                             booster=params['booster'],
        #                                             gamma=params['gamma'],
        #                                             min_child_weight=params['min_child_weight'],
        #                                             max_delta_step=params['max_delta_step'],
        #                                             subsample=params['subsample'],
        #                                             colsample_bytree=params['colsample_bytree'],
        #                                             colsample_bylevel=params['colsample_bylevel'],
        #                                             colsample_bynode=params['colsample_bynode'],
        #                                             reg_lambda=params['reg_lambda'],
        #                                             reg_alpha=params['reg_alpha'],
        #                                             #gpu_id=params['gpu_id'],
        #                                             tree_method=params['tree_method'],
        #                                             random_state=params['random_state']
        #                                            ), n_jobs=njobs)
        # model.fit(X_train, y_train, verbose=False)
        
        # model = MultiOutputRegressor(LGBMRegressor(boosting_type=hparams['boosting_type'], 
        #                                             num_leaves=hparams['num_leaves'],
        #                                             max_depth=hparams['max_depth'],
        #                                             learning_rate=hparams['learning_rate'],
        #                                             n_estimators=hparams['n_estimators'],
        #                                             subsample_for_bin=hparams['subsample_for_bin'],
        #                                             objective=hparams['objective'],
        #                                             min_split_gain=hparams['min_split_gain'],
        #                                             min_child_weight=hparams['min_child_weight'],
        #                                             min_child_samples=hparams['min_child_samples'],
        #                                             colsample_bytree=hparams['colsample_bytree'],
        #                                             reg_alpha=hparams['reg_alpha'],
        #                                             reg_lambda=hparams['reg_lambda'],
        #                                             importance_type=hparams['importance_type'],
        #                                             random_state=hparams['random_state'],
        #                                             n_jobs=hparams['n_jobs']), n_jobs=self.n_jobs)
        # model.fit(X_train, y_train, verbose=False, eval_set=[(X_val, y_val)], eval_metric='rmse', feature_name=feature_names)
        
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
      # XGBOOST REGRESSOR
      # hparams = {'objective':['reg:squarederror'],
      #           'eval_metric':["rmse"],
      #           'n_estimators': [50,100,200,300,500],
      #           'max_depth': [3,4,5,6,7,8,9,10,12,15],
      #           'learning_rate': [0.01,0.05,0.1,0.15,0.2,0.25,0.3], #learning rate
      #           'booster': ['gbtree','dart','gblinear'],
      #           'gamma': [0,0.1,0.3,0.5,1.0,2.0,3.0], #min_split_loss
      #           'min_child_weight':[0,1,3,5,7,10],
      #           'max_delta_step': [1,2,3,4,5,6,7,8,9,10],
      #           'subsample': [0.5,1],
      #           'colsample_bytree': [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95],
      #           'colsample_bylevel': [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95],
      #           'colsample_bynode': [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95],
      #           'reg_lambda': [0,1,1.5,2,2.5,3,3.5], # L2 reg
      #           'reg_alpha': [0,1,1.5,2,2.5,3,3.5], # L1 reg 
      #           #'gpu_id':[0],
      #           'tree_method':['auto'],
      #           'random_state':[int(self.random_state)]}
      
      # # LIGHTGBM
      # hparams = {'boosting_type': ['gbdt','dart','goss','rf'],
      #            'num_leaves': [3,5,10,20,31,40],
      #            'max_depth': [0,3,4,5,6,7,8,9,10,12,15],
      #            'learning_rate': [0.01,0.05,0.1,0.15,0.2,0.25,0.3], #learning rate
      #            'n_estimators': [50,100,200,300,500],
      #            'subsample_for_bin': [10,50,100,200,300,500,600,700],
      #            'objective':['regression'],
      #            'min_split_gain': [0.,0.1,0.05,0.03,0.01,0.001],
      #            'min_child_weight':[0.,0.1,0.01,0.001,0.0001],
      #            'min_child_samples':[5,10,15,20,25,30],
      #            'colsample_bytree': [0.1,0.3,0.5,0.7,0.9,1.0],
      #            'reg_alpha': [0.,0.5,1,1.5,2,2.5,3,3.5], # L1 reg weights
      #            'reg_lambda': [0.,0.5,1,1.5,2,2.5,3,3.5], # L2 reg weights
      #            'importance_type':['gain'],
      #            'n_jobs':[self.n_jobs],
      #            'random_state':[int(self.random_state)]}

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
    def get_weights(X_train, y_train, feature_names, y_attributes, n_classes=10, beta=0.9, maxval=100):
      #beta = 0.9
      #n_classes = 10
      train_class = pd.DataFrame(X_train, columns=feature_names)
      train_class.loc[:,y_attributes] = y_train.copy()
      train_class.loc[:,'ses'] = train_class.apply(lambda row: int(row.mean_wi*n_classes/maxval), axis=1)
      weights_t = train_class.apply(lambda row: SESMetadata.set_weights_wENS(train_class,beta,row,n_classes,maxval), axis=1) 
      return weights_t
    
    @staticmethod
    def set_weights_wENS(df, beta, row, n_classes, maxval=100):
      tmp = df.groupby('ses').size()
      ses= int(row.mean_wi*n_classes/maxval)
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

      # model XGBRegressor
      # model = MultiOutputRegressor(XGBRegressor(objective=params['objective'], 
      #                                             eval_metric=params['eval_metric'],
      #                                             n_estimators=params['n_estimators'],
      #                                             max_depth=params['max_depth'],
      #                                             learning_rate=params['learning_rate'],
      #                                             booster=params['booster'],
      #                                             gamma=params['gamma'],
      #                                             min_child_weight=params['min_child_weight'],
      #                                             max_delta_step=params['max_delta_step'],
      #                                             subsample=params['subsample'],
      #                                             colsample_bytree=params['colsample_bytree'],
      #                                             colsample_bylevel=params['colsample_bylevel'],
      #                                             colsample_bynode=params['colsample_bynode'],
      #                                             reg_lambda=params['reg_lambda'],
      #                                             reg_alpha=params['reg_alpha'],
      #                                             #gpu_id=params['gpu_id'],
      #                                             tree_method=params['tree_method'],
      #                                             random_state=params['random_state']
      #                                             )).fit(X_train, y_train, verbose=False)
      
      # # LGBM model
      # model = MultiOutputRegressor(LGBMRegressor(boosting_type=hparams['boosting_type'], 
      #                                               num_leaves=hparams['num_leaves'],
      #                                               max_depth=hparams['max_depth'],
      #                                               learning_rate=hparams['learning_rate'],
      #                                               n_estimators=hparams['n_estimators'],
      #                                               subsample_for_bin=hparams['subsample_for_bin'],
      #                                               objective=hparams['objective'],
      #                                               min_split_gain=hparams['min_split_gain'],
      #                                               min_child_weight=hparams['min_child_weight'],
      #                                               min_child_samples=hparams['min_child_samples'],
      #                                               colsample_bytree=hparams['colsample_bytree'],
      #                                               reg_alpha=hparams['reg_alpha'],
      #                                               reg_lambda=hparams['reg_lambda'],
      #                                               importance_type=hparams['importance_type'],
      #                                               random_state=hparams['random_state'],
      #                                               n_jobs=hparams['n_jobs']), n_jobs=self.n_jobs).fit(X_train, y_train, verbose=False)
      
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

#       # general
#       feature_importance = model.feature_importances_
#       sorted_idx = np.argsort(feature_importance)
#       fig = plt.figure(figsize=(12, 6))
#       plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
#       plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
#       plt.title('Feature Importance')
      
#       # for each output
#       y_index = 1
#       explainer = shap.Explainer(model)
#       shap_values = explainer(X_test)
#       shap_importance = shap_values.abs.mean(0).values[:,y_index]
#       sorted_idx = shap_importance.argsort()
#       fig = plt.figure(figsize=(12, 6))
#       plt.barh(range(len(sorted_idx)), shap_importance[sorted_idx], align='center')
#       plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
#       plt.title('SHAP Importance')
      
#       # or
#       shap.plots.bar(shap_values[:,:,y_index], max_display=20, show=False)
#       fn = f"catboost_info/tmp/shap_o{y_index}.png"
#       plt.savefig(fn)
      return 



      #self.df_data = None
      #self.xgb_model = None
      #self.xgb_best_params = None
      #self.xgb_model = None
      # self.pplaces = traintype in NONE
      # self.cv_results = None
      # self.best_params = None
      # self.best_score = None
      # self.df_evaluation = None
      # self.feature_importance = None
      # self.X = {'train':None, 'val':None, 'test':None}
      # self.y_true = {'train':None, 'val':None, 'test':None}
      # self.y_pred = {'train':None, 'val':None, 'test':None}

    # def set_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
    #   self.X['train'] = X_train
    #   self.X['val'] = X_val
    #   self.X['test'] = X_test
    #   self.y_true['train'] = y_train 
    #   self.y_true['val'] = y_val
    #   self.y_true['test'] = y_test

    # def hyper_parameter_tuning(self, cv, njobs, gpu, verbose, tuning):
    #   if tuning == 'robust':
    #     param_tuning = {
    #         'learning_rate': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.03, 0.05, 0.1],
    #         'max_depth': [1, 3, 5, 7, 9, 10],
    #         'min_child_weight': [1, 3, 5, 7, 9, 10],
    #         'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         'n_estimators' : [10, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000],
    #         'gamma' : [0,0.1,0.3,0.5,0.7,0.9,1,5,10,15,20,25,30],
    #         'objective': ['reg:squarederror']
    #     }
    #   elif tuning == 'high':
    #     param_tuning = {
    #       'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.3],
    #       'max_depth': [1, 3, 5],
    #       'min_child_weight': [1, 3, 5], 
    #       'subsample': [0.3, 0.5, 0.7],
    #       'colsample_bytree': [0.3, 0.5, 0.7],
    #       'n_estimators' : [5,10,50,100],
    #       'gamma' : [0,1,5,10], 
    #       'objective': ['reg:squarederror']
    #     }
    #   elif tuning == 'mid':
    #     param_tuning = {
    #       'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    #       'max_depth': [3, 5],
    #       'min_child_weight': [1, 3, 5], 
    #       'subsample': [0.5, 0.7],
    #       'colsample_bytree': [0.5, 0.7],
    #       'n_estimators' : [5,10,50,100],
    #       'gamma' : [0,1,10], 
    #       'objective': ['reg:squarederror']
    #     }
    #   elif tuning == 'fast':
    #     param_tuning = {
    #       'learning_rate': [0.0001, 0.1],
    #       'max_depth': [3, 5],
    #       'min_child_weight': [1, 3], 
    #       'subsample': [0.5, 0.7],
    #       'colsample_bytree': [0.5, 0.7],
    #       'n_estimators' : [10,100],
    #       'gamma' : [0,5], 
    #       'objective': ['reg:squarederror']
    #     }


    #   self.gpu = gpu
    #   if self.gpu:
    #     system.check_gpu()
    #     param_tuning['tree_method'] = ["gpu_hist"]
    #     #param_tuning['predictor'] = ['gpu_predictor']
    #     #param_tuning['gpu_id'] = [0]

    #   xgb_model = XGBRegressor()
    #   # gsearch = GridSearchCV(estimator = xgb_model,
    #   #                       param_grid = param_tuning,
    #   #                       cv = cv,
    #   #                       n_jobs = njobs,
    #   #                       verbose = verbose)
      
    #   rsearch = RandomizedSearchCV(estimator=xgb_model, 
    #                                scoring = XGB_SCORING,
    #                                param_distributions = param_tuning, 
    #                                cv = cv,
    #                                n_iter = XGB_TUNING_ITER,
    #                                n_jobs = njobs,
    #                                verbose = verbose)
      
    #   X = np.append(self.X['train'], self.X['val'], axis=0)
    #   y = np.append(self.y_true['train'], self.y_true['val'], axis=0)
    #   print("Hyper-param tuning:", tuning)
    #   print("X:",X.shape)
    #   print("y:",y.shape)
    #   results = rsearch.fit(X, y)
    #   self.xgb_best_params = rsearch.best_params_
      
    #   # best hyper-params
    #   self.cv_results = pd.DataFrame.from_dict(results.cv_results_)
    #   self.best_params = rsearch.best_params_
    #   self.best_score = rsearch.best_score_
      
      
    # def fit(self, early_stoping=10):
    #   self.xgb_model = XGBRegressor(
    #     objective = self.xgb_best_params['objective'],
    #     colsample_bytree = self.xgb_best_params['colsample_bytree'],
    #     learning_rate = self.xgb_best_params['learning_rate'],
    #     max_depth = self.xgb_best_params['max_depth'],
    #     min_child_weight = self.xgb_best_params['min_child_weight'],
    #     n_estimators = self.xgb_best_params['n_estimators'],
    #     subsample = self.xgb_best_params['subsample'],
    #     importance_type = XGBOOST_IMPORTANCE_TYPE
    #     )

    #   print("train:",self.X['train'].shape,self.y_true['train'].shape)
    #   print("val:",self.X['val'].shape,self.y_true['val'].shape)
    #   self.xgb_model.fit(self.X['train'], self.y_true['train'], early_stopping_rounds=early_stoping, eval_set=[(self.X['val'], self.y_true['val'])], verbose=False)

    # def save_model(self):
    #   fn = self.get_model_fn()
    #   ios.write_pickle(self.xgb_model, fn)

    # def get_model_fn(self):
    #   return os.path.join(self.output_folder, FN_MODEL_XGBOOST)

    # def model_exists(self):
    #   fn = self.get_model_fn()
    #   return ios.exists(fn)

    # def load_model(self, fn=None):
    #   if fn is None:
    #     fn = self.get_model_fn()
    #   self.xgb_model = ios.read_pickle(fn)

    # def set_feature_importance(self, feature_names):
    #   self.feature_importance = pd.DataFrame({'feature_name':feature_names,
    #                                           'importance':self.xgb_model.feature_importances_})

    # def predict(self, X=None):
    #   for kind in DATASETS_KIND:
    #     if self.X[kind] is not None:
    #       self.y_pred[kind] = self.xgb_model.predict(self.X[kind])

    # def evaluate(self):
    #   #score=cross_val_score(classifier,X,y,cv=10)
    #   for kind in DATASETS_KIND:
    #     self.evaluation_metric[kind]['mae'] = mean_absolute_error(self.y_true[kind], self.y_pred[kind])
    #     self.evaluation_metric[kind]['rmse'] = mean_squared_error(self.y_true[kind], self.y_pred[kind], squared=False)
    #     self.evaluation_metric[kind]['r2'] = r2_score(self.y_true[kind], self.y_pred[kind])

    # def save_predictions(self, df_complement):
    #   for kind in DATASETS_KIND:
    #       df = pd.DataFrame({'y_pred':self.y_pred[kind]}, columns=['y_pred'])
    #       df_complement[kind].reset_index(drop=True, inplace=True)
    #       df = pd.concat([df_complement[kind], df], axis=1)
    #       fn = os.path.join(self.output_folder, f'pred_{kind}.csv')
    #       ios.save_csv(df, fn)

    # def tunning_summary(self):
    #   if self.output_folder:
    #     ios.save_csv(self.cv_results, fn=os.path.join(self.output_folder,'hp_summary.csv'))
    #     ios.save_json(self.best_params, fn=os.path.join(self.output_folder,'hp_bestparams.json'))
      
    # def save_summary(self, duration):
    #   fn = os.path.join(self.output_folder, 'summary.csv')
    #   df = pd.DataFrame(columns=['source','name','value'])
    #   df.loc[:,'name'] = self.feature_importance.feature_name.values
    #   df.loc[:,'value'] = self.feature_importance.importance.values
    #   df.loc[:,'source'] = 'feature_importance'

    #   for kind in DATASETS_KIND:
    #     for k,v in self.evaluation_metric[kind].items():
    #       df = df.append(pd.DataFrame({'source':[kind],
    #                                   'name':[k],
    #                                   'value':[v]}), ignore_index=True)

    #   df = df.append(pd.DataFrame({'source':['training'],
    #                                'name':['duration'],
    #                                'value':[duration]}), ignore_index=True)
      
    #   df = df.append(pd.DataFrame({'source':['training'],
    #                                'name':['best_score'],
    #                                'value':[self.best_score]}), ignore_index=True)
      
    #   df = df.append(pd.DataFrame({'source':['training'],
    #                                'name':['gpu'],
    #                                'value':[self.gpu==1]}), ignore_index=True)
      
    #   df = df.append(pd.DataFrame({'source':['param','param'],
    #                                'name':['ttype','random_state'],
    #                                'value':[self.ttype, self.random_state]}), ignore_index=True)

    #   xbp_t, xbp_k, xbp_v = zip(*[('hyperparam',k,v) for k,v in self.xgb_best_params.items()])
    #   df = df.append(pd.DataFrame({'source':xbp_t,
    #                                'name':xbp_k,
    #                                'value':xbp_v}), ignore_index=True)

    #   ios.save_csv(df, fn)

    # ###########################
    # # PPLACES
    # ###########################
    
    # def set_pplace(self, X):
    #   self.X = X

    # def predict_pplace(self):
    #   self.y =  self.xgb_model.predict(self.X)
      
    # def save_predictions_pplace(self, df_complement):
    #   df = pd.DataFrame({'y_pred':self.y}, columns=['y_pred'])
    #   df_complement.reset_index(drop=True, inplace=True)
    #   df = pd.concat([df_complement, df], axis=1)
    #   fn = os.path.join(self.output_folder, f'pred_pplaces.csv')
    #   ios.save_csv(df, fn)

    # ###########################
    # # PLOTS
    # ###########################

    # def plot_feature_importance(self):
    #   fn = os.path.join(self.output_folder, 'feature_importance_builtin.png')
    #   viz.plot_xgboost_feature_importance(self.xgb_model, figsize=(10, 10), max_num_features=30, height=0.7, fn=fn)

    #   fn = os.path.join(self.output_folder, 'feature_importance.png')
    #   viz.plot_feature_importance(self.feature_importance, figsize=(10, 10), max_num_features=30, fn=fn)


    # def plot_true_pred(self):
    #   for kind in DATASETS_KIND:
    #     fn = os.path.join(self.output_folder, f'true_pred_{kind}.png')
    #     viz.plot_pred_true(pred=self.y_pred[kind], true=self.y_true[kind], metrics=self.evaluation_metric[kind], fn=fn)
        



######
   

    # def load_pplaces_data(self, path):
       
    #   ### CSV files
    #   files = [os.path.join(path,fn) for fn in os.listdir(path) if fn.endswith(".csv") and fn.startswith("PPLACES_")]
    #   fn = [os.path.join(path,fn) for fn in os.listdir(path) if fn=="PPLACES.csv"][0]
    #   self.df_data = ios.load_csv(fn, index_col=0)
    #   for fn in files:
    #     tmp = ios.load_csv(fn, index_col=0)    
    #     self.df_data = pd.concat([self.df_data, tmp], axis=1)
    #   print("{} total CSV features".format(self.df_data.shape))

    #   ### CNN features
    #   tmp_features = pd.DataFrame()
    #   try:
    #     # collect data from all chunks
    #     nbatches = len(glob.glob(os.path.join(path,'PPLACES_FMaps*.npz')))
    #     for batch in np.arange(1,nbatches+1,1):
    #       # making sure it goes from 1 to nbatches (to keep same order of records)
    #       fn = os.path.join(path,"PPLACES_FMaps_{}-{}.npz".format(batch, nbatches))
    #       tmp = np.load(fn)['arr_0']
    #       print("{} features from NPZ file {}-{}".format(tmp.shape, batch, nbatches))
    #       tmp = pd.DataFrame(tmp)
    #       tmp.rename(columns={c:"cnn{}".format(c) for c in tmp.columns}, inplace=True)
    #       tmp_features = tmp_features.append(tmp, ignore_index=True)

    #     # put it together
    #     print("{} total CNN features".format(tmp_features.shape))
    #     self.df_data = self.df_data.join(tmp_features)
    #     print("{} total features".format(self.df_data.shape))
    #   except Exception as ex:
    #     print(ex)
    #     raise Exception("Error at loading CNN features file")

    # def pplaces_predict(self):
    #   cols = self.xgb_model.get_booster().feature_names
    #   print("- {} features: ".format(len(cols)), cols)
    #   self.X_pp = self.df_data[cols]
    #   self.y_pred_pp = self.xgb_model.predict(self.X_pp)
    
    # def save_pplaces_predictions(self, fname):
    #   fn = os.path.join(self.output_folder, fname)
    #   np.save(fn, self.y_pred_pp)

    # def save_pplaces_predictions_with_features(self, fname):
    #   fn = os.path.join(self.output_folder, fname)
    #   self.df_data.loc[:,IND_VAR_PRED] = self.y_pred_pp
    #   if self.logout:
    #     self.df_data.loc[:,'{}_float'.format(IND_VAR_PRED)] = np.exp(self.y_pred_pp).round(PRECISION)
    #   ios.save_csv(self.df_data, fn)








    # def fillna(self, value):
    #   self.df_data = self.df_data.fillna(0)
    
    # def encode(self):
    #   if not self.pplaces:
    #     # binary:
    #     self.df_data.loc[:,'rural'] = self.df_data.URBAN_RURA.apply(lambda c: int(c==2)) # orginal: 1urban, 2 rural -->  encoding: 0urban, 1rural 
      
    #     self.df_data.loc[:,IND_VAR_TRUE] = self.df_data.loc[:,IND_VAR].round(PRECISION)
    #     if self.logout:
    #       # convert output to log(output)
    #       self.df_data.loc[:,IND_VAR_TRUE] = np.log(self.df_data.loc[:,IND_VAR_TRUE])
    #   else:
    #     #Urban: city, town 
    #     #Rural: village, hamlet, isolated_dweling
    #     #for i,c in enumerate(self.df_data.columns):
    #     #  print(i, c)
    #     self.df_data.loc[:,'rural'] = self.df_data.place.apply(lambda c: int(c in ['village', 'hamlet', 'isolated_dwelling']))
        

    # def load_training_data(self, root, validate=False, location_kind=None,):

    #   ### CSV files
    #   files = [os.path.join(root,fn) for fn in os.listdir(root) if fn.endswith(".csv") and "_iwi_cluster_" in fn and '_cnn_' not in fn and '_model_' not in fn]
    #   #fn = [os.path.join(root,fn) for fn in os.listdir(root) if fn.endswith("_iwi_cluster.csv")][0]
      
    #   if location_kind is None:
    #     files = glob.glob(os.path.join(root,'*_iwi_cluster.csv'))
    #     options = ['cc','ccur','gc','gcur']
    #     for fn in files:
    #       if len([1 for o in options if o in fn]) == 0:
    #        break
    #   else:
    #     fn = glob.glob(os.path.join(root,'*_{}_iwi_cluster.csv'.format(location_kind)))[0]

    #   self.df_data = ios.load_csv(fn, index_col=0) # ground truth
    #   for fn in files:
    #     tmp = ios.load_csv(fn, index_col=0)    
    #     self.df_data =  self.df_data.join(tmp) #pd.concat([self.df_data, tmp], axis=1)
    #   print("{} features from CSV files".format(self.df_data.shape))

    #   ### CNN features
    #   tmp_features = pd.DataFrame()
    #   try:
    #     # collect data from all chunks
    #     prefix = location_kind if location_kind else ''
    #     nbatches = len(glob.glob(os.path.join(root,'*{}_iwi_cluster_FMaps_*.npz'.format(prefix))))
    #     for batch in np.arange(1,nbatches+1,1):
    #       # making sure it goes from 1 to nbatches (to keep same order of records)
    #       fn = glob.glob(os.path.join(root,'*{}_iwi_cluster_FMaps_{}-{}.npz'.format(prefix,batch, nbatches)))[0]
    #       tmp = np.load(fn)['arr_0']
    #       print("{} features from NPZ file {}-{}".format(tmp.shape, batch, nbatches))
    #       tmp = pd.DataFrame(tmp)
    #       tmp.rename(columns={c:"cnn{}".format(c) for c in tmp.columns}, inplace=True)
    #       tmp_features = tmp_features.append(tmp, ignore_index=True)

    #     # put it together
    #     print("{} total CNN features".format(tmp_features.shape))
    #     self.df_data = self.df_data.join(tmp_features)
    #     print("{} total features".format(self.df_data.shape))
    #   except Exception as ex:
    #     print(ex)
    #     raise Exception("Error at loading CNN features file")


    #   # try:
    #   #   ### TODO: change code to load FMaps batches (see pplaces below) - maybe?
    #   #   files = glob.glob(os.path.join(root,'*_iwi_cluster_FMaps_*.npz'))
    #   #   tmp = np.load(fn)['arr_0']
    #   #   print("{} features from NPZ file".format(tmp.shape))
    #   #   tmp = pd.DataFrame(tmp)
    #   #   tmp.rename(columns={c:"cnn{}".format(c) for c in tmp.columns}, inplace=True)
    #   #   self.df_data = self.df_data.join(tmp)
    #   #   print("{} total features".format(self.df_data.shape))
    #   # except Exception as ex:
    #   #   print(ex)
    #   #   raise Exception("Error at loading CNN features file")

    #   # if fn_img is not None:
    #   #   self.fn_img = fn_img
    #   #   tmp = ios.load_csv(self.fn_img, index_col=0)
    #   #   self.df_data = self.df_data.join(tmp)

    #   if validate:
    #     # VIIRS: mean0 to avoid big differences across years
    #     viirscols = [c for c in self.df_data.columns if c.startswith("NTLL")]
    #     tmp = self.df_data.groupby("DHSYEAR")[viirscols].transform(lambda x: (x - x.mean()) / x.std())
    #     self.df_data.loc[tmp.index,tmp.columns] = tmp

    
    # def split_train_test(self, frac_train):

    #   if IND_VAR_STRATIFY not in self.df_data.columns:
    #     ### @todo: this should not be done here... restructure folder/files
    #     NCLASSES = 4
    #     LABELS = ['poor','lower_middle','upper_middle','rich']
    #     self.df_data.loc[:,IND_VAR_STRATIFY] = pd.cut(self.df_data[IND_VAR], bins=NCLASSES, labels=LABELS, include_lowest=True, precision=0, right=False)
            
    #   # train and test
    #   if self.ttype == TTYPE_ALL:
    #     # all years
    #     self.train, self.test = utils.stratify_sampling(self.df_data, IND_VAR_STRATIFY, frac_train, self.random_state)
    #     #self.train = self.df_data.sample(frac = frac_train, random_state=self.random_state)
    #     #self.test = self.df_data.drop(self.train.index)
    #   elif self.ttype == TTYPE_PAST:
    #     # train on earlier, test on later
    #     y = self.df_data.DHSYEAR.min()
    #     print("year: {}".format(y))
    #     self.train = self.df_data.query("DHSYEAR==@y").copy()
    #     self.test = self.df_data.query("DHSYEAR!=@y").copy()
    #   elif self.ttype.startswith('only'):
    #     # train and test on 1 year
    #     y = self.df_data.DHSYEAR.min() if self.ttype == TTYPE_FORMER else self.df_data.DHSYEAR.max()
    #     print("year: {}".format(y))
    #     tmp = self.df_data.query("DHSYEAR==@y").copy()
    #     self.train, self.test = utils.stratify_sampling(tmp, IND_VAR_STRATIFY, frac_train, self.random_state)
    #     #self.train = tmp.sample(frac = frac_train, random_state=self.random_state)
    #     #self.test = tmp.drop(self.train.index)
      
    # def split_train_val(self, frac_val):
    #   # train and validation:
    #   #X = self.train.drop(COLS_NOT_TO_INCLUDE, axis=1)
    #   #y = self.train[IND_VAR_TRUE]
    #   #self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X,y,test_size=frac_val,random_state=self.random_state)

    #   # train and validation: stratified
    #   train, val = utils.stratify_sampling(self.train, IND_VAR_STRATIFY, 1-frac_val, self.random_state)
    #   self.y_train = train[IND_VAR_TRUE]
    #   self.y_val = val[IND_VAR_TRUE]

    #   cols = []
    #   for c in COLS_NOT_TO_INCLUDE:
    #     if c in train.columns:
    #       cols.append(c)
    #   self.X_train = train.drop(cols, axis=1)
      
    #   cols = []
    #   for c in COLS_NOT_TO_INCLUDE:
    #     if c in val.columns:
    #       cols.append(c)
    #   self.X_val = val.drop(cols, axis=1)
        
    #   print("- number of total features: ", self.X_train.shape)
    #   print("- features: ", self.X_train.columns)


 # def test_predict(self):
    #   cols = []
    #   for c in COLS_NOT_TO_INCLUDE:
    #     if c in self.test.columns:
    #       cols.append(c)
    #   self.X_test = self.test.drop(cols, axis=1)
    #   self.y_test = self.test[IND_VAR_TRUE]

    #   self.y_pred_test = self.xgb_model.predict(self.X_test)
    #   self.mae_test = mean_absolute_error(self.y_test, self.y_pred_test)
    #   self.r2_test = r2_score(self.y_test, self.y_pred_test)

    # def save_train(self, fname):
    #   fn = os.path.join(self.output_folder, fname)
    #   if fn is not None:
    #     ios.save_csv(self.train,fn)

    # def save_test(self, fname):
    #   fn = os.path.join(self.output_folder, fname)
    #   df = self.X_test.copy()
    #   df.loc[:,'y_true'] = self.y_test
    #   df.loc[:,'y_pred'] = self.y_pred_test
    #   if fn is not None:
    #     ios.save_csv(df,fn)

    

    # def plot_true_pred(self, fname):
    #   fn = os.path.join(self.output_folder, fname)
    #   viz.plot_pred_true(pred=self.y_pred, true=self.y_val, metric={'MAE':self.xgb_mae, 'R2':self.xgb_r2}, fn=fn.replace("<kind>","val"))
    #   viz.plot_pred_true(pred=self.y_pred_test, true=self.y_test, metric={'MAE':self.mae_test, 'R2':self.r2_test}, fn=fn.replace("<kind>","test"))
    #   if self.logout:
    #     viz.plot_pred_true(pred=np.exp(self.y_pred_test).round(PRECISION), true=np.exp(self.y_test).round(PRECISION), metric={'MAE':np.exp(self.mae_test).round(PRECISION), 'R2':np.exp(self.r2_test).round(PRECISION)}, fn=fn.replace("<kind>","test_float"))

    