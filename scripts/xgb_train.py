###############################################################################
# Dependencies
###############################################################################
import os
import gc
import json
import time
import glob
import argparse
import pandas as pd
from tqdm import tqdm 
from datetime import datetime
import multiprocessing

from ses.data import Data
from ses.metadata import SESMetadata
from utils import ios
from utils import system
from utils import validations

###############################################################################
# Constants
###############################################################################

from utils.constants import *

###############################################################################
# Functions
###############################################################################
    
def run(root, years, dhsloc, traintype, y_attributes, features_source, timevar, cv, viirsnorm, cnn_name=None, layer_id=None, weighted=None):
  # validation
  validations.validate_not_empty(root,'root')
  
  # data
  ### 0. Pre-processing
  y_attributes = validations.get_valid_output_names(y_attributes)
  isregression = validations.is_regression(y_attributes)
  years = validations.validate_years(years)
  timevar = validations.validate_timevar(timevar)
  cnn_name = validations.validate_none(cnn_name)
  layer_id = validations.validate_none(layer_id)
  validations.validate_traintype(traintype)
  validations.validate_years_traintype(years, traintype)
  include_fmaps = cnn_name is not None and layer_id is not None
  offlineaug = str(cnn_name).startswith("offaug_")
  n_jobs = multiprocessing.cpu_count()
  print(f"- Multiprocessing: {n_jobs} jobs")
  
  ### 1. K-FOLD
  print("========== HYPER-PARAM TUNING BEGIN ==========")
  data = Data(root=root, years=years, dhsloc=dhsloc, traintype=traintype, isregression=isregression)
  data.load_metadata(viirsnorm, only_training=True)
  for train, val, path, runid, rs, fold in data.iterate_train_val(tune_img=False):

    print("********************************")
    rs = int(rs/10) if rs in WEIRD_RS else rs
    path = os.path.join(path, cnn_name) if cnn_name else path
    print(f"1. LOADING: {runid}-{fold} ({path}) {rs}")

    ### Variables
    fmaps = {'train':None, 'val':None}
    if include_fmaps:
      print('[INFO] Including fmaps.')
      fmaps['train'] = SESMetadata.get_fmaps(path=path, setname='train', fmap_layer_id=layer_id, fold=fold)
      fmaps['val'] = SESMetadata.get_fmaps(path=path, setname='val', fmap_layer_id=layer_id, fold=fold)
    X_train, y_train, feature_names = data.metadata_get_X_y(train, y_attributes, fmaps['train'], offlineaug, features_source, timevar)
    X_val, y_val, _ = data.metadata_get_X_y(val, y_attributes, fmaps['val'], False, features_source, timevar)

    ### Sample weights (based on column ses)
    weights_t = None
    if weighted:
      # HU: maxval = 2.12, beta=0.495 
      # SL: maxval = 100, beta=0.9
      # UG: maxval = 100, beta=0.9
      
      beta=0.495 if 'Hungary' in root else 0.9
      maxval=2.12 if 'Hungary' in root else 100
      n_classes = 10 
      print(f"[INFO] CLASS WEIGHTS -> maxval:{maxval}, beta:{beta}, n_classes:{n_classes}")
      
      weights_t =  SESMetadata.get_weights(X_train, y_train, feature_names, y_attributes, n_classes=n_classes, beta=beta, maxval=maxval)
      print(f"[INFO] Sample weights added (k-fold {runid}-{fold}).")
      
    ### SESMetadata
    sesmeta = SESMetadata(path, traintype, features_source, timevar, runid, cv, fold, include_fmaps, rs, weighted, n_jobs)
    sesmeta.tuning(X_train, y_train, X_val, y_val, feature_names, weights_t)
    
    ### flush
    del(sesmeta)
    del(X_train)
    del(y_train)
    del(X_val)
    del(y_val)
    del(feature_names)
    gc.collect()

  ### 3. FINAL CV
  print("========== FINAL TRAINING ==========")
  data = Data(root=root, years=years, dhsloc=dhsloc, traintype=traintype, isregression=isregression)
  data.load_metadata(viirsnorm, only_training=True)
  for train, test, path, runid, rs in data.iterate_train_test():

    print("********************************")
    path = os.path.join(path, cnn_name) if cnn_name else path
    print(f"1. LOADING: {runid} ({path})")

    ### Variables
    fmaps = {'train':None, 'test':None}
    if include_fmaps:
      print('[INFO] Including fmaps.')
      fmaps['train'] = SESMetadata.get_fmaps(path=path, setname='train', fmap_layer_id=layer_id)
      fmaps['test'] = SESMetadata.get_fmaps(path=path, setname='test', fmap_layer_id=layer_id)
    X_train, y_train, feature_names = data.metadata_get_X_y(train, y_attributes, fmaps['train'], offlineaug, features_source, timevar)
    X_test, y_test, _ = data.metadata_get_X_y(test, y_attributes, fmaps['test'], False, features_source, timevar)

    ### Sample weights (based on column ses)
    weights_t = None
    if weighted:
      weights_t =  SESMetadata.get_weights(X_train, y_train, feature_names, y_attributes, n_classes=n_classes, beta=beta, maxval=maxval)
      print(f"[INFO] Sample weights added (final {runid}).")
      
    ### SESMetadata
    sesmeta = SESMetadata(path, traintype, features_source, timevar, runid, cv, fold, include_fmaps, rs, weighted)
    y_pred = sesmeta.oos_evaluation(X_train, y_train, X_test, y_test, feature_names, weights_t)
    sesmeta.save_evaluation(y_test, y_pred)
    sesmeta.plot_evaluation(y_test, y_pred)
    sesmeta.save_predictions(test, y_test, y_pred, y_attributes)
    sesmeta.save_feature_importance(X_test, y_test, y_pred, y_attributes, feature_names)
    
###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-root", help="Path to country folder", type=str, required=True)
    parser.add_argument("-years", help="Comma separated list of years to analyze: e.g., 2016,2019", type=str, required=True)
    parser.add_argument("-dhsloc", help="DHS location re-arrangement method: rc", type=str, default=None, required=False)
    parser.add_argument("-ttype", help="Training type: oldest, newest, all.", type=str, required=True)
    parser.add_argument("-yatt", help="Comma separated dependent variables y", type=str, default=None, required=True)

    parser.add_argument("-fsource", help="Data source of features Xs: all, OCI, FBM, FBP, FBMV, NTLL, OSM", type=str, default='all', required=False)
    parser.add_argument("-timevar", help="How to handle time variant data sources: deltatime, gdp, gdpp, gdppg, gdppgp, gdpg, gdpgp, gni, gnip", type=str, default=None, required=False)

    parser.add_argument("-cv", help="Cross-validation number of folds (stratified-kfold)", type=int, default=4, required=True)
    parser.add_argument("-viirs", help="Normalize VIIRS values: if 1 normalizes VIIRS data otherwise (0) VIIRS as it is (raw).",  action='store_true')
    
    parser.add_argument("-cnn_name", help="CNN (folder) name, eg., offaug_cnn_mp_dp_relu_sigmoid_adam_mean_std_regression", type=str, default=None, required=False)
    parser.add_argument("-layer_id", help="Layer id where to extract features: 19, 18, 17.", type=int, default=None, required=False)
    parser.add_argument("-weighted", help="Sample weights or not (default wENS 0.9 for UG and SL, and 0.495 for HU).",  action='store_true')
    parser.add_argument("-shutdown", help="Python script that shutsdown the server after training.", type=str, default=None, required=False)
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))

    start_time = time.time()
    run(args.root, args.years, args.dhsloc, args.ttype, args.yatt, args.fsource, args.timevar, args.cv, args.viirs, args.cnn_name, args.layer_id, args.weighted)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    if args.shutdown:
      system.google_cloud_shutdown(args.shutdown)





# beta = 0.9
# n_classes = 10
# train_class = pd.DataFrame(X_train, columns=feature_names)
# train_class.loc[:,y_attributes] = y_train.copy()
# train_class.loc[:,'ses'] = train_class.apply(lambda row: int(row.mean_wi*n_classes/100), axis=1)
# weights_t = train_class.apply(lambda row: SESMetadata.set_weights_wENS(train_class,beta,row,n_classes), axis=1) 

# beta = 0.9
# n_classes = 10
# train_class = pd.DataFrame(X_train, columns=feature_names)
# train_class.loc[:,y_attributes] = y_train.copy()
# train_class.loc[:,'ses'] = train_class.apply(lambda row: int(row.mean_wi*n_classes/100), axis=1)
# weights_t = train_class.apply(lambda row: SESMetadata.set_weights_wENS(train_class,beta,row,n_classes), axis=1) 


      
  # if skip(epoch, fold):
  #   print(f"epoch {epoch}, fold {fold}: skip")
  #   return
  # else:
  #   print(f"epoch {epoch}, fold {fold}: running...")
    
  # ### 3. For each epoch and fold 
  # data = Data(root, dhsloc=dhsloc, epoch=epoch, rs=rs, fold=fold, cnn_path=cnn_path)
  # data.load_original_features(viirs)
  
  # for train, val, test, path, epoch, rs, fold in data.iterate_train_val_test():
    
  #   ### X and y
  #   X_train, y_train, feature_names = data.xgboost_get_X_y(train, y_attribute)
  #   print("train:", train.shape, X_train.shape, y_train.shape)
    
  #   X_val, y_val, _ = data.xgboost_get_X_y(val, y_attribute)
  #   print("val:", val.shape, X_val.shape, y_val.shape)

  #   X_test, y_test, _ = data.xgboost_get_X_y(test, y_attribute)
  #   print("test:", test.shape, X_test.shape, y_test.shape)

  #   ### Model
  #   ses = SESFeatures(None, output_folder=output, random_state=rs)
  #   ses.set_data(X_train, y_train, X_val, y_val, X_test, y_test)

  #   ### Training
  #   start_train = time.time()
  #   ses.hyper_parameter_tuning(cv=cv, njobs=njobs, gpu=gpu, verbose=XGB_VERBOSE, tuning=tuning)
  #   ses.fit(early_stoping=EARLYSTOP)
  #   duration = time.time() - start_train
  #   ses.save_model()
  #   ses.tunning_summary()
    
  #   ### Feature importance
  #   ses.set_feature_importance(feature_names)
    
  #   ### Validation
  #   ses.predict()
  #   ses.evaluate()
  #   for kind in DATASETS_KIND:
  #     print('---')
  #     print(f"{kind}-MAE: ", ses.evaluation_metric[kind]['mae'])
  #     print(f"{kind}-RMSE: ", ses.evaluation_metric[kind]['rmse'])
  #     print(f"{kind}-R2: ", ses.evaluation_metric[kind]['r2'])
  #   print('---')

  #   ### Save results
  #   cols = ['dhs_id','dhs_rural',y_attribute]
  #   df_complement = {'train':train.loc[:,cols], 'val':val.loc[:,cols], 'test':test.loc[:,cols]}
  #   ses.save_predictions(df_complement)
  #   ses.save_summary(duration)
  #   ses.plot_true_pred()
  #   ses.plot_feature_importance()