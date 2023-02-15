#export PYTHONPATH=/env/python:/home/leespinn/code/SES-Inference/libs/

###############################################################################
# Dependencies
###############################################################################
import os
import gc
import glob
import time
import argparse
import multiprocessing
from collections import defaultdict

from ses.data import Data
from ses.images import SESImages
from utils import system
from utils import validations

###############################################################################
# Functions
###############################################################################

from utils.constants import *

###############################################################################
# Functions
###############################################################################

def run(root, years,  model_name, y_attributes, dhsloc, traintype, kfold=4, epochs=100, patience=50, class_weight=False, n_jobs=1, retrain=False, offaug=False, gpus='0', specific_run=None, specific_fold=None):
  # validation
  validations.validate_not_empty(root,'root')
  
  ### 0. Pre-validation
  isregression = model_name.endswith("_regression")
  y_attributes = validations.get_valid_output_names(y_attributes)
  print("INFO: {} | {} augmentation | predict: {}".format('regression' if isregression else 'classification', 'offline' if offaug else 'online' if model_name.startswith('aug_') else 'no', y_attributes))
  
  ### 1. Hyper-param tunning
  data = Data(root, years, dhsloc, traintype, model_name=model_name, offaug=offaug, isregression=isregression)
  for train, val, path, runid, rs, fold in data.iterate_train_val(specific_run=specific_run, specific_fold=specific_fold):
      
    print("==========================================")
    print(f"1. LOADING: {runid}-{fold}")
    print(f"   FOCUS: runid ({specific_run}), foldid ({specific_fold})")
    
    ### 1. Train, val sets
    X_train, y_train = data.cnn_get_X_y(train, y_attributes, offaug)
    X_val, y_val = data.cnn_get_X_y(val, y_attributes)
    pixels = X_train.shape[1]
    bands  = X_train.shape[3] 

    print("2. TUNING")
    ### 2. model  
    ses = SESImages(root=root, rs=rs, runid=runid, fold=fold, n_classes=data.n_classes, model_name=model_name, 
                    pixels=pixels, bands=bands, 
                    epochs=epochs, patience=patience, class_weight=class_weight, 
                    isregression=data.isregression, retrain=retrain, offaug=offaug, gpus=gpus,
                    specific_fold=specific_fold)
    
    ses.init(path)
    ses.load_hyper_params_tuning_eval(kfold)

    ### 3. tunning
    while ses.needs_tuning():
      ses.hyper_parameter_tuning(X_train, y_train, X_val, y_val, n_jobs)
      
    ### 4. flush memory
    del(ses.model)
    del(ses)
    del(X_train)
    del(y_train)
    del(X_val)
    del(y_val)
    gc.collect()
  
  ### 2. Best hyper-params
  print("3. BEST HYPER-PARAMS")
  for path, runid, rs in data.iterate_runid():
    _ = SESImages.best_hyper_params(path, model_name, offaug)

  ### 3. Training
  print("4. MODEL")
  data = Data(root, years, dhsloc, traintype, model_name=model_name, isregression=isregression)
  for train, test, path, runid, rs in data.iterate_train_test():
    
    ### 0. skip runids if specific_run is set, otherwise do all
    if specific_run is None:
      pass
    elif specific_run != runid:
      continue
      
    print("==========================================")
    print(f"5. LOADING: {runid}")
    print(f"   FOCUS: runid ({specific_run})")
    
    ### 4. Train, test sets 
    X_train, y_train = data.cnn_get_X_y(train, y_attributes, offaug)
    X_test, y_test = data.cnn_get_X_y(test, y_attributes)
    pixels = X_train.shape[1]
    bands  = X_train.shape[3] 

    print("6. TRAINING")
    ### 5. model  
    ses = SESImages(root=root, rs=rs, runid=runid, n_classes=data.n_classes, model_name=model_name, 
                    pixels=pixels, bands=bands, 
                    epochs=epochs, patience=patience, class_weight=class_weight, 
                    isregression=data.isregression, retrain=retrain, offaug=offaug, gpus=gpus)
    ses.init(path)
    ses.load_best_hyper_params()
    ses.define_model()
    ses.define_callbacks()
    ses.define_class_weights(y_train)

    ### 6. training
    n_jobs = multiprocessing.cpu_count()
    ses.fit(X_train, y_train, n_jobs)
    
    ### 7. results
    y_pred = ses.evaluate(X_test, y_test)
    ses.log()
    ses.model_summary()
    ses.plot_learning_curves()
    ses.save_predictions(test, y_test, y_pred, y_attributes)
    
    ### 8. flush memory
    del(ses.model)
    del(ses)
    del(X_train)
    del(y_train)
    del(X_test)
    del(y_test)
    gc.collect()
  
  gc.collect()
  print('.....')
      

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
  parser.add_argument("-years", help="Years (comma-separated): 2016,2019", type=str, required=True)
  parser.add_argument("-model", help="Model name", type=str, default='aug_cnn_mp_dp_relu_sigmoid_adam_reg', required=True)
  parser.add_argument("-yatt", help="Attributes (column names) for dependent variable y (comma separated)", type=str, default='mean_wi', required=True)
  parser.add_argument("-dhsloc", help="DHS cluster option (None, cc, ccur, gc, gcur, rc).", type=str, default=None, required=False)
  parser.add_argument("-traintype", help="Years to include in training: all, newest, oldest", type=str, default='none')
  parser.add_argument("-kfold", help="K-fold cross-validation", type=int, default=4)
  parser.add_argument("-epochs", help="Epochs.", type=int, required=True)
  parser.add_argument("-patience", help="Patience for early stop.", type=int, default=None, required=False)
  parser.add_argument("-cw", help="To add class weights in the fitting or not", action='store_true')
  parser.add_argument("-njobs", help="Parallel processes.", type=int, default=1)
  parser.add_argument("-retrain", help="0: if model exists ask what to do. 1: if model exists load it and continue training. 2: if model exists just eval. 3: train from scratch", type=int, default=0)
  parser.add_argument("-offaug", help="Whether or not to include offline augmented images.", action='store_true')
  parser.add_argument("-gpus", help="Visible GPUs: 0 or 0,1 or 1", type=str, default=None)
  parser.add_argument("-runid", help="Runid to focus on (so this script can be run in parallel)", type=int, default=None)
  parser.add_argument("-foldid", help="Foldid to focus on (so this script can be run in parallel) - Manual merge tuning.csv ", type=int, default=None)
  parser.add_argument("-shutdown", help="Python script that shutsdown the server after training.", type=str, default=None, required=False)
    
  args = parser.parse_args()
  for arg in vars(args):
    print("{}: {}".format(arg, getattr(args, arg)))

  start_time = time.time()
  try:
    run(args.r, args.years, args.model, args.yatt, args.dhsloc, args.traintype, args.kfold, args.epochs, args.patience, args.cw, args.njobs, args.retrain, args.offaug, args.gpus, args.runid, args.foldid)
  except Exception as ex:
    print(ex)
  print("--- %s seconds ---" % (time.time() - start_time))

  if args.shutdown:
    system.google_cloud_shutdown(args.shutdown)

