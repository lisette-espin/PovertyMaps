################################################################################
# Dependencies
################################################################################
import os
import gc
import time
import glob
import numpy as np
import pandas as pd

from keras.models import Model
import tensorflow as tf
#import tensorflow_hub as hub
from tensorflow.random import set_seed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from keras import backend as K

from utils import ios
from utils import viz
from utils import system
from ses import cnn_models as models
from google.staticmaps import StaticMaps

################################################################################
# Constants
################################################################################

from utils.constants import *

################################################################################
# Class
################################################################################
class SESImages(object):

  def __init__(self, root, runid, n_classes, model_name, pixels, bands, fold=None, epochs=None, patience=None, class_weight=None, rs=None, isregression=None, retrain=0, offaug=False, gpus=None):
    # data
    self.root = root
    self.runid = runid
    self.fold = fold
    
    # general
    self.n_classes = n_classes
    self.model_name = model_name
    self.pixels = pixels
    self.bands = bands
    self.isregression = isregression
    self.rs = rs
    self.session = None
    self.config = None
    self.offaug = offaug
    self.gpus = gpus 
    
    # for training
    self.patience = patience
    self.epochs = epochs
    self.retrain = retrain
    self.class_weight = class_weight
    self.results_path = None
    self.best_params = None
    self.duration_training = None
    
    # for tuning
    self.df_evaluation = None
    
    # final
    self.model = None
    self.fn_model = None
    self.callbacks = None
    self.weights = None
    self.history = None
    self.metrics = {'loss':None, 'accuracy':None, 'auc':None, 'mae':None, 'mse':None, 'rmse':None, 'r2':None}
    
    
  def init(self, path):
    '''
    Initializes random random_state (rs)
    '''
    print("===================================================")
    print("TF version:", tf.__version__)
    #print("Hub version:", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
    print("===================================================")
    np.random.seed(self.rs)
    set_seed(self.rs)
    system.check_gpu()
    K.clear_session()    
    self.config = tf.compat.v1.ConfigProto()
    self.config.gpu_options.allow_growth = True
    self.config.gpu_options.per_process_gpu_memory_fraction = 0.9
    self.session = tf.compat.v1.Session(config=self.config)
    
    if self.gpus is not None:
      os.environ["CUDA_VISIBLE_DEVICES"]=self.gpus
      
    #   physical_devices = tf.config.list_physical_devices('GPU')
    #   try:
    #     # Disable first GPU
    #     visible = [physical_devices[int(i)] for i in self.gpus.split(",")]
    #     tf.config.set_visible_devices(visible, 'GPU')
    #     logical_devices = tf.config.list_logical_devices('GPU')
    #     assert len(logical_devices) == len(physical_devices) - 1
    #   except Exception as ex:
    #     # Invalid device or cannot modify virtual devices once initialized.
    #     print(f"[ERROR] images.py | init | {ex}")
    #     import sys
    #     sys.exit(0)
        
    os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    print("TF_GPU_ALLOCATOR:", os.environ["TF_GPU_ALLOCATOR"])
    # print("LOGICAL_DEVICES (GPUS)", logical_devices)
    print("===================================================")
    self.create_results_path(path)


  def create_results_path(self, path):
    '''
    Creates the output folder
    '''
    #kind = 'on' if not self.offaug and self.model_name.startswith('aug_') else 'offaug_' if self.offaug else 'noaug_'
    #folder = "{}{}".format(kind, self.model_name)
    #self.results_path = os.path.join(path,folder)
    self.results_path = SESImages.get_results_path(path, self.model_name, self.offaug)
    ios.validate_path(self.results_path)
    print("CNN folder: {}".format(self.results_path))

  @staticmethod
  def get_results_path(path, model_name, offaug):
    kind = 'on' if not offaug and model_name.startswith('aug_') else 'offaug_' if offaug else 'noaug_'
    folder = "{}{}".format(kind, model_name)
    results_path = os.path.join(path,folder)
    return results_path
  
  ##############################################################################
  # Tuning
  ##############################################################################
  
  def load_hyper_params_tuning_eval(self, kfold):
    self.fn_eval = os.path.join(self.results_path,CNN_TUNNING_SUMMARY_FILE)
    if ios.exists(self.fn_eval):
      print('loading hparam file.')
      self.df_evaluation = ios.load_csv(self.fn_eval)
    else:
      print('creating hparam file.')
      self.df_evaluation = pd.DataFrame()
      hparams, n_iter, combinations = SESImages.get_all_hparams(self.model_name, self.n_classes, self.pixels, self.bands)
      strhp = ['model_name','optimizer_name']
      for i in np.arange(n_iter):
        newone = True
        while newone:
          # pick one combination that hasn't been picked already
          hp = SESImages.get_random_hyper_params_combination(hparams)
          if self.df_evaluation.shape[0] == 0:
            break

          q = " and ".join([f"param_{k}=='{v}'" if k in strhp else f"param_{k}=={v}" for k,v in hp.items()])
          newone = self.df_evaluation.query(q).shape[0] != 0

        obj = {'rs':self.rs,'runid':self.runid}
        obj.update({"mean_fit_time":None})
        obj.update({"std_fit_time":None})
        obj.update({"mean_eval_time":None})
        obj.update({"std_eval_time":None})

        obj.update({f"param_{k}":str(v) if type(v)==list else v for k,v in hp.items()})
        obj.update({'params':str(hp)})
        
        for k in np.arange(1,kfold,1):
          obj.update({f"loss_fold{k}":None})
        obj.update({"mean_loss":None})
        obj.update({"std_loss":None})
        obj.update({"rank":None})
        
        self.df_evaluation = self.df_evaluation.append(pd.DataFrame(obj, index=[0]), ignore_index=True)

      self.df_evaluation.drop_duplicates(inplace=True)
      if self.df_evaluation.shape[0] != n_iter:
        raise Exception("The number of hyper-param random combinations does not match the expected one.")
      ios.save_csv(self.df_evaluation, self.fn_eval)

  def needs_tuning(self):
    tmp = self.get_hyper_params_to_be_tuned()
    return tmp.shape[0] > 0

  def get_hyper_params_to_be_tuned(self):
    null = [None,np.nan]
    rs = self.rs
    ri = self.runid
    return self.df_evaluation.query("rs==@rs and runid==@ri and loss_fold" + str(self.fold) + " in @null")

  def hyper_parameter_tuning(self, X_train, y_train, X_val, y_val, njobs=1):
    import json
    import time
    
    K.clear_session()
    tmp = self.get_hyper_params_to_be_tuned().sample(1)
    
    params = json.loads(tmp.iloc[0].params.replace("'",'"'))
    print(params)
    
    model = models.define_model(params) 
    
    try:
      start = time.time()
      _ = model.fit(X_train,y_train,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_data=(X_val,y_val),
                    verbose = 1, 
                    workers = njobs,
                    use_multiprocessing=False,
                    shuffle=True)
      duration_fit = time.time() - start
      
      start = time.time()
      results = model.evaluate(X_val, y_val, batch_size=params['batch_size'], workers=njobs, use_multiprocessing=False, return_dict=True, verbose=1)
      duration_eval = time.time() - start
      
      loss = results['loss']
      self._update_hyper_params_tuning_eval(tmp.index[0], loss, duration_fit, duration_eval)
      
    except Exception as ex:
      print(ex)

    del(model)
    gc.collect()

  def _update_hyper_params_tuning_eval(self, id, loss, duration_fit, duration_eval):
    self.df_evaluation.loc[id,'mean_fit_time'] = str(duration_fit) if self.df_evaluation.loc[id,'mean_fit_time'] in NONE else "{},{}".format(self.df_evaluation.loc[id,'mean_fit_time'],duration_fit)
    self.df_evaluation.loc[id,'mean_eval_time'] = str(duration_eval) if self.df_evaluation.loc[id,'mean_eval_time'] in NONE else "{},{}".format(self.df_evaluation.loc[id,'mean_eval_time'],duration_eval)
    self.df_evaluation.loc[id,f'loss_fold{self.fold}'] = CNN_LOSS_NAN * (-1 if not self.isregression else 1) if np.isnan(loss) else loss
    ios.save_csv(self.df_evaluation, self.fn_eval)

    
  @staticmethod
  def get_all_hparams(model_name, n_classes, pixels, bands):
    # hyper-params
    lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5] 
    optimizer_names = CNN_TUNING_OPTIMIZERS
    dropouts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    batch_size = CNN_TUNING_BATCHSIZES
    epochs = [CNN_TUNING_EPOCHS]

    if model_name.startswith("aug_"):
      # online augmentation
      rotations = [0.1,0.2,0.3,0.4,0.5]
      contrasts = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
      translations = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
      zooms = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
      contrast_ranges = [[0.5, 0.6], [0.6,0.7], [0.7,0.8], [0.8,0.9],[0.9,1.0],[1.0,2.0]]
      brightness_deltas = [[-0.3,-0.2],[-0.2,-0.1],[-0.1,0.0],[0.0,0.1],[0.1,0.2]]

      hparams = dict(model_name=[model_name],
                    n_classes=[n_classes],
                    pixels=[pixels],
                    bands=[bands],
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lrs, 
                    dropout=dropouts,
                    optimizer_name=optimizer_names,
                    rotation=rotations,
                    contrast=contrasts,
                    translation=translations,
                    zoom=zooms,
                    contrast_range=contrast_ranges,
                    brightness_delta=brightness_deltas)
    else:
      print("Hypter-param does not include online augmentation.")
      hparams = dict(model_name=[model_name],
                    n_classes=[n_classes],
                    pixels=[pixels],
                    bands=[bands],
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lrs, 
                    dropout=dropouts,
                    optimizer_name=optimizer_names)
    
    combinations = np.prod([len(v) for v in hparams.values()])
    print("- All possible combinations hyper-params: ", combinations)
    n_iter = min(CNN_TUNING_ITER,int(round(combinations/2)))
    print("- # random candidates: ", n_iter)
    return hparams, n_iter, combinations

  @staticmethod
  def get_random_hyper_params_combination(hparams):
    hp = {}
    for k,v in hparams.items():
      if type(v[0]) == list or type(v[0]) == tuple:
        hp[k] = v[np.random.choice(np.arange(len(v)))]
      else:
        hp[k] = np.random.choice(v)
    return hp

  @staticmethod
  def needs_tuning_runid_fold(path, model_name, runid, rs, fold):
    fn = os.path.join(path,model_name,CNN_TUNNING_SUMMARY_FILE)
    null = [None,np.nan]
    if ios.exists(fn):
      df = ios.load_csv(fn)
      return df.query("rs==@rs and runid==@runid and loss_fold" + str(fold) + " in @null").shape[0] > 0
    return True

  @staticmethod
  def best_hyper_params(path, model_name, offaug):
    import json
    path = SESImages.get_results_path(path, model_name, offaug)
    fn_eval = os.path.join(path,CNN_TUNNING_SUMMARY_FILE)
    fn_best = os.path.join(path,CNN_BEST_PARAMS_FILE)

    if ios.exists(fn_best):
      return ios.load_json(fn_best)

    if ios.exists(fn_eval):
      df = ios.load_csv(fn_eval)
      loss_cols = [c for c in df.columns if c.startswith('loss_fold')]
      # check that all values exist
      if df[loss_cols].isnull().values.any():
        print('[WARNING] images.py | best_hyper_params | The hyper-parameter-tuning is not complete yet.')
        return None
      # update mean/std values
      df.loc[:,'mean_loss'] = df.apply(lambda row: np.mean([row[c] for c in loss_cols]), axis=1)
      df.loc[:,'std_loss'] = df.apply(lambda row: np.std([row[c] for c in loss_cols]), axis=1)
      df.loc[:,'rank'] = df.mean_loss.rank(na_option='bottom',ascending=True,pct=False)
      df.loc[:,'std_fit_time'] = df.mean_fit_time.apply(lambda c: np.std([float(t) for t in c.split(',') if t not in NONE]))
      df.loc[:,'mean_fit_time'] = df.mean_fit_time.apply(lambda c: np.mean([float(t) for t in c.split(',') if t not in NONE]))
      df.loc[:,'std_eval_time'] = df.mean_eval_time.apply(lambda c: np.std([float(t) for t in c.split(',') if t not in NONE]))
      df.loc[:,'mean_eval_time'] = df.mean_eval_time.apply(lambda c: np.mean([float(t) for t in c.split(',') if t not in NONE]))
      ios.save_csv(df, fn_eval)
      # best params
      best_params = df.query("rank==1").iloc[0].params
      best_params = json.loads(best_params.replace("'",'"'))
      ios.save_json(best_params, fn_best)
      return best_params
    raise Exception("Evaluation file does not exist.")


  
  ##############################################################################
  # Training
  ##############################################################################
  
  def load_best_hyper_params(self):
    fn_best = os.path.join(self.results_path,CNN_BEST_PARAMS_FILE)
    if ios.exists(fn_best):
      self.best_params = ios.load_json(fn_best)
    else:
      print(fn_best)
      raise Exception("Best hyper-params file does not exist.")

  def load_model(self, fn_model=None):
    # model and results paths
    if fn_model is not None:
      self.fn_model = fn_model
      self.results_path = os.path.dirname(self.fn_model)
    else:
      self.fn_model = os.path.join(self.results_path,FN_MODEL_CNN)
    
    # load model
    if ios.exists(self.fn_model):
      self.model = models.load_existing_model(self.fn_model)

  def define_model(self):
    '''
    Loads or creates a new CNN model
    '''
    self.fn_model = os.path.join(self.results_path,FN_MODEL_CNN)
    
    # load model because retrain
    if self.retrain in [RETRAIN, JUST_EVAL] and ios.exists(self.fn_model):
      print("Loading a pre-trained model: {}...".format(self.fn_model))
      self.model = models.load_existing_model(self.fn_model)
      return 
    
    # ask whether retrain or just_eval
    if self.retrain in [ASK_RETRAIN]:
      load = ''
      if ios.exists(self.fn_model):
        load = input('There is a pre-trained version. Would you like to load it? (y/n): ')

      if load.lower() in YES:
        retrain = input('Would you like to continue training? (y/n): ')
        print("Loading a pre-trained model: {}...".format(self.fn_model))
        self.model = models.load_existing_model(self.fn_model)
        self.retrain = RETRAIN if retrain.lower() in YES else JUST_EVAL # so it continues training or not
        return 
    
    # train from scratch
    if self.model is None:
      self.retrain = TRAIN_FROM_SCRATCH # so it continues training because the model is new
      print("Defining new model from best hyper-params...")
      self.model = models.define_model(self.best_params)
  
  def define_callbacks(self):
    '''
    Callbacks during training.
    '''
    monitor = 'val_loss' if self.isregression else 'val_accuracy'
    mode = 'min' if self.isregression else 'max'

    ### Callbacks: early stop
    self.callbacks = []
    if self.patience:
      early_stop = EarlyStopping(monitor = monitor,
                              mode = mode,
                              min_delta = 0,
                              patience = self.patience,
                              restore_best_weights = True)
      self.callbacks.append(early_stop)

    ### Callbacks: checkpoints
    checkpoint = ModelCheckpoint(filepath = os.path.join(self.results_path, FN_MODEL_CNN), 
                                 monitor = monitor, 
                                 mode = mode, 
                                 save_best_only = True)
    self.callbacks.append(checkpoint)
    
  def define_class_weights(self, y):
    '''
    Class weights when there is class imabalance
    '''
    ### Class weight
    if self.class_weight and not self.isregression:
      vals = np.argmax(y,axis=1)
      labels = np.unique(vals)
      self.weights = compute_class_weight('balanced',classes=labels, y=vals)
      self.weights = {l:w for l,w in zip(*[labels, self.weights])}
    print("class weights: ", self.weights)


  def fit(self, X_train, y_train, X_test, y_test, njobs=1):
    '''
    Fit model: training
    '''
    if self.retrain in [RETRAIN, TRAIN_FROM_SCRATCH]:
      start_time_training = time.time()
      self.history = self.model.fit(X_train,y_train,
                                    epochs=self.epochs,
                                    batch_size=self.best_params['batch_size'],
                                    validation_data=(X_test,y_test),
                                    verbose = 1, 
                                    workers = njobs,
                                    use_multiprocessing=True,
                                    shuffle=True,
                                    class_weight=self.weights,
                                    callbacks=self.callbacks)
      self.duration_training = time.time() - start_time_training

  ##############################################################################
  # Evalutaion
  ##############################################################################
  
  @staticmethod
  def get_ymax(path):
    if 'DHS' in path or 'MIS' in path or 'LSMS' in path:
      return YMAX_DHS
    if 'INGATLAN' in path:
      return YMAX_INGATLAN
    raise Exception("[ERROR] images.py | get_ymax | Not possible to infer ymax")
    
  def evaluate(self, X, y):
    '''
    Evaluating model on data and plotting confusion matrix if classification or true vs. pred if regression.
    '''
    from utils.constants import SES_LABELS
    from census.wealth import WI
    
    # get evaluation metrics
    metric_values = self.model.evaluate(X, y, verbose=1)
    self.metrics['loss'] = metric_values[0]
    all_metrics = METRICS_REGRESSION if self.isregression else METRICS_CLASSIFICATION

    for i, m in enumerate(all_metrics):
      try:
        m = m.name
      except:
        pass
      self.metrics[m] = metric_values[i+1]
    
    # prediction
    ypred = self.model.predict(X)
    
    # plot evaluation
    if self.isregression:
      # regression
      if self.metrics['r2'] is None:
        r2 = r2_score(y,ypred)
        self.metrics['r2'] = r2
        
      viz.plot_pred_true(ypred, y, self.metrics, os.path.join(self.results_path,f'plot_pred_true.{PLOTEXT}')) 
      
      ymax = SESImages.get_ymax(self.results_path)
      
      # n classes mean IWI (just to know)
      bins, minv, maxv = 10,0,ymax
      sestrue = WI.discretize_in_n_bins(y[:,0], bins, minv, maxv)
      sespred =  WI.discretize_in_n_bins(ypred[:,0], bins, minv, maxv)
      labels = [str(c) for c in sestrue.categories]
      sestrue = sestrue.astype(str)
      sespred = sespred.astype(str)
      viz.plot_confusion_matrix(sestrue, sespred, fn=os.path.join(self.results_path,f'plot_confusion_10ses_mean.{PLOTEXT}'), labels=labels, norm=False)
      viz.plot_confusion_matrix(sestrue, sespred, fn=os.path.join(self.results_path,f'plot_confusion_10ses_mean_norm.{PLOTEXT}'),  labels=labels, norm=True) 
      
      # n classes std IWI (just to know)
      bins, minv, maxv = 10,0,np.ceil(max(max(y[:,1]),max(ypred[:,1])))
      sestrue = WI.discretize_in_n_bins(y[:,1], bins, minv, maxv)
      sespred =  WI.discretize_in_n_bins(ypred[:,1], bins, minv, maxv)
      labels = [str(c) for c in sestrue.categories]
      sestrue = sestrue.astype(str)
      sespred = sespred.astype(str)
      viz.plot_confusion_matrix(sestrue, sespred, fn=os.path.join(self.results_path,f'plot_confusion_10ses_std.{PLOTEXT}'), labels=labels,  norm=False)
      viz.plot_confusion_matrix(sestrue, sespred, fn=os.path.join(self.results_path,f'plot_confusion_10ses_std_norm.{PLOTEXT}'), labels=labels,  norm=True) 
    
      # 4 classes (universal)
      col = '_mwi'
      bins, minv, maxv = 4,0,ymax
      sestrue = WI.add_ses_categories(pd.DataFrame({col:y[:,0]}),col,bins,minv,maxv)[f'{col}_cat']
      sespred = WI.add_ses_categories(pd.DataFrame({col:ypred[:,0]}),col,bins,minv,maxv)[f'{col}_cat']
      
    else:
      # classification
      ypred = np.argmax(ypred, axis=1)
      sespred = ypred.copy()
      sestrue = np.argmax(y, axis=1)
      
    # plot confusion matrix (SES values)
    viz.plot_confusion_matrix(sestrue, sespred, fn=os.path.join(self.results_path,f'plot_confusion.{PLOTEXT}'), labels=SES_LABELS, norm=False)
    viz.plot_confusion_matrix(sestrue, sespred, fn=os.path.join(self.results_path,f'plot_confusion_norm.{PLOTEXT}'), labels=SES_LABELS, norm=True) 
  
    return ypred
    
  ##############################################################################
  # RESULTS
  ##############################################################################

  def log(self):
    '''
    Saving all hyper-parameters and eval metrics
    '''
    
    if self.retrain == JUST_EVAL:
      print("- log just eval")
      return 
    
    fn = os.path.join(self.results_path,CNN_LOG_FILE)
    
    if ios.exists(fn):
      log = ios.load_json(fn)
      duration = log['training_duration_secs']
      n_retrained = log['n_retrained']
      print("- log loaded")
    else:
      log = {'rs':str(self.rs), 'epochs':self.epochs, 'patience':self.patience}
      log.update(self.best_params)
      duration = 0
      n_retrained = 0
      print("- log new")
    
    if self.retrain == TRAIN_FROM_SCRATCH:
      duration = 0
      n_retrained = 0
      print("- log train from scratch")
      
    if self.retrain in [RETRAIN, TRAIN_FROM_SCRATCH]:
      log.update({f'test_{k}':v for k,v in self.metrics.items()})
      log.update({'training_duration_secs':duration + self.duration_training})
      log.update({'n_retrained':n_retrained + 1})
      print("- log save")
      #ios.save_json(log, fn) 

  def model_summary(self):
    '''
    CNN architecture / structure
    '''
    stringlist = []
    self.model.summary(print_fn=lambda x: stringlist.append(x))
    ios.write_txt('\n'.join(stringlist), os.path.join(self.results_path,'model.txt')) 

  def plot_learning_curves(self):
    '''
    Plot learning curves. Metric values vs. epoch
    '''
    if self.history:
      fn = os.path.join(self.results_path,f'learning_curves.{PLOTEXT}')
      viz.plot_learning_curves(self.history, fn)
    
  ##############################################################################
  # Prediction
  ##############################################################################
  
  def predict(self, X):
    '''
    Predict y_pred
    '''
    pred = self.model.predict(X)
    return pred

  def extract_features_last_layer(self, X, layer_id):
    '''
    Extracts features from last layer
    '''
    remodel = Model(inputs=self.model.inputs, outputs=self.model.layers[layer_id].output)
    #remodel.summary()
    feature_maps = remodel.predict(X)
    
    print("=====")
    for i, layer in enumerate(self.model.layers):
      print(i, layer.name, layer.output.shape, "<-----" if i == layer_id else '')
    print("====")
    
    return feature_maps
  
  def save_predictions(self, test, y_test, y_pred, y_attributes):
      fn = os.path.join(self.results_path, 'test_pred_cnn.csv')
      df = test.loc[:,['cluster_id']]
      for ia, at in enumerate(y_attributes):
        df.loc[:,f'true_{at}'] = y_test[:,ia]
        df.loc[:,f'pred_{at}'] = y_pred[:,ia]
      ios.save_csv(df, fn)

################################################################################
################################################################################  
################################################################################
# Class AUgmentation
################################################################################
################################################################################
################################################################################
# https://www.analyticsvidhya.com/blog/2021/06/offline-data-augmentation-for-multiple-images/
# https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
from tqdm import tqdm 
from pqdm.threads import pqdm
import matplotlib.pyplot as plt
#from tensorflow.python.ops.numpy_ops import np_config
import tensorflow_addons as tfa
import skimage.transform
import random 

class Augmentation(object):

  def __init__(self, root, years, dhsloc, probability_aug=None):
    self.root = root
    self.dhsloc = dhsloc
    self.prefix = ios.get_prefix_surveys(root=root, years=years)
    self.df_locations = None
    self.probability_augmentation = PROB_AUGMENTED if probability_aug is None else probability_aug
    
    print("1. Init")
    self.fn_locations = os.path.join(self.root, 'results', 'features', f"{self.prefix}_{dhsloc}_cluster_pplace_ids.csv")
    self.path_augmented_imgs = os.path.join(self.root, 'results', 'staticmaps', 'augmented')
    self.path_clusters_imgs = os.path.join(self.root, 'results', 'staticmaps', 'clusters')
    self.path_pplaces_imgs = os.path.join(self.root, 'results', 'staticmaps', 'pplaces')
    

  def load_data(self):
    print("2. Loading data...")
    self.df_locations = ios.load_csv(self.fn_locations)
    self.df_locations.loc[:,'OSMID'] = self.df_locations.OSMID.astype(pd.Int64Dtype())
    print('locations:',self.df_locations.shape)


  def generate(self, njobs=1):
    # np_config.enable_numpy_behavior()
    print("3. Augmenting...")
    results = pqdm(self.df_locations.iterrows(), self.augment_image, n_jobs=njobs)
    self.save_summary(results)


  def save_summary(self, results):
    print("----------")
    print(f"Summary: {len(results)} results | {results[0] if len(results)>0 else '-'}")
    
    cluster_ids, augmented = zip(*results)
    df = pd.DataFrame({'cluster_id':cluster_ids, 'augmented':augmented})
    fn = os.path.join(self.path_augmented_imgs, "_summary.csv")
    ios.save_csv(df, fn)

    total = df.shape[0]
    print(f"{total} images processed.")
    
    augmented = df.query("augmented == True").shape[0]
    print(f"Augmentation applied on {augmented} ({round(augmented*100/total,1)}%) images.")


  def augment_image(self, obj):
    id, row = obj[0], obj[1]
    flag = False

    rv = np.random.rand()
    #print('rv:',rv,'prob:',(1-self.probability_augmentation))
    
    if  rv >= (1-self.probability_augmentation):
      
      ### 1. getting image
      path = self.path_pplaces_imgs if not pd.isna(row.OSMID) else self.path_clusters_imgs
      prefix = StaticMaps.get_prefix(row)
      fn = StaticMaps.get_satellite_img_filename(prefix, ZOOM, SCALE, MAPTYPE, IMG_TYPE, path=path, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)
      
      # if not pd.isna(row.OSMID):
      #   path = self.path_pplaces_imgs
      #   prefix = f"OSMID{int(row.OSMID)}-"
      #   #fn = f"OSMID{int(row.OSMID)}-LA*-LO*-ZO{ZOOM}-SC{SCALE}-{IMG_WIDTH}x{IMG_HEIGHT}-{MAPTYPE}.{IMG_TYPE}"
      # else:
      #   path = self.path_clusters_imgs
      #   prefix = f"Y{row.cluster_year}-C{row.cluster_number}-U{row.cluster_rural}-"
      #   #fn = f"Y{row.dhs_year}-C{row.dhs_cluster}-U{row.dhs_rural+1}-LA*-LO*-ZO{ZOOM}-SC{SCALE}-{IMG_WIDTH}x{IMG_HEIGHT}-{MAPTYPE}.{IMG_TYPE}"
      
      fn_img = glob.glob(fn)
      
      if len(fn_img) < 1:
        print('[ERROR] images.py | augment_image | ', fn_img)
        raise Exception("No image was found.")
        
      fn = os.path.basename(fn_img[0])

      ### 2. Apply <N_AUGMENTATIONS> Augmentations
      if not self.exists_ntimes(fn, N_AUGMENTATIONS):
        flag = True

        image = load_img(fn_img[0], target_size=(IMG_WIDTH, IMG_HEIGHT))
        image = img_to_array(image, dtype='uint8')
        image = image[0:-PIXELS_LOGO,0:-PIXELS_LOGO,:]
        image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT], method='nearest')
        
        # flip up/down
        tmp = tf.image.flip_up_down(image)
        self.save_augmented_image(tmp, fn, 'flipUD')
        
        # flip left/rigth
        tmp = tf.image.flip_left_right(image)
        self.save_augmented_image(tmp, fn, 'flipLR')
        
        # rotation
        tmp = tfa.image.rotate(image, angles=0.785398, fill_mode = 'wrap', interpolation='nearest')
        self.save_augmented_image(tmp, fn, 'rot45')
        
        tmp = tfa.image.rotate(image, angles=1.5708, fill_mode = 'wrap', interpolation='nearest')
        self.save_augmented_image(tmp, fn, 'rot90')
        
        tmp = tfa.image.rotate(image, angles=3.14159, fill_mode = 'wrap', interpolation='nearest')
        self.save_augmented_image(tmp, fn, 'rot180')
        
        tmp = tfa.image.rotate(image, angles=4.71239, fill_mode = 'wrap', interpolation='nearest')
        self.save_augmented_image(tmp, fn, 'rot270')
        
        angle = random.uniform(0.174533,6.10865) # 10 - 350
        tmp = tfa.image.rotate(image, angles=angle, fill_mode = 'wrap', interpolation='nearest')
        self.save_augmented_image(tmp, fn, 'rotrnd')
        
        # scale
        s = 2
        tmp = skimage.transform.rescale(image, scale=s, mode='wrap',  anti_aliasing=True, clip=True)
        tmp = tf.image.random_crop(tmp, size = [IMG_WIDTH, IMG_HEIGHT, BANDS])
        self.save_augmented_image(tmp, fn, f'scale{s}')
        
        # central crop
        p = 20
        tmp = tf.image.central_crop(image, p/100)
        tmp = tf.image.resize(tmp, size = [IMG_WIDTH, IMG_HEIGHT], method='nearest', antialias=True, preserve_aspect_ratio=True)
        self.save_augmented_image(tmp, fn, f'ccrop{p}')
        
        p = 50
        tmp = tf.image.central_crop(image, p/100)
        tmp = tf.image.resize(tmp, size = [IMG_WIDTH, IMG_HEIGHT], method='nearest', antialias=True, preserve_aspect_ratio=True)
        self.save_augmented_image(tmp, fn, f'ccrop{p}')
        
        # random crop
        p = 20
        tmp = tf.image.random_crop(image, size = [int(round(IMG_WIDTH * p/100,0)), int(round(IMG_HEIGHT * p/100,0)), BANDS])
        tmp = tf.image.resize(tmp, size = [IMG_WIDTH, IMG_HEIGHT])
        self.save_augmented_image(tmp, fn, f'rcrop{p}')
        
        p = 50
        tmp = tf.image.random_crop(image, size = [int(round(IMG_WIDTH * p/100,0)), int(round(IMG_HEIGHT * p/100,0)), BANDS])
        tmp = tf.image.resize(tmp, size = [IMG_WIDTH, IMG_HEIGHT])
        self.save_augmented_image(tmp, fn, f'rcrop{p}')
        
        p = 80
        tmp = tf.image.random_crop(image, size = [int(round(IMG_WIDTH * p/100,0)), int(round(IMG_HEIGHT * p/100,0)), BANDS])
        tmp = tf.image.resize(tmp, size = [IMG_WIDTH, IMG_HEIGHT])
        self.save_augmented_image(tmp, fn, f'rcrop{p}')
        
        # Adding Gaussian noise
        gnoise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.1, dtype=tf.float32)
        tmp = tf.image.convert_image_dtype(image, dtype=tf.float32, saturate=False)
        tmp = tf.add(tmp, gnoise)
        self.save_augmented_image(tmp, fn, 'noise')
        
        # Brightness
        p = 20
        tmp = tf.image.adjust_brightness(image, delta=p/100)
        self.save_augmented_image(tmp, fn, f'bright{p}')
        
        p = 20
        tmp = tf.image.adjust_brightness(image, delta=-p/100)
        self.save_augmented_image(tmp, fn, f'dark{p}')
        
        # Random erase
        tmp = Augmentation.random_erasing(image, 1)
        self.save_augmented_image(tmp, fn, f'rerase')
        
        # Mixed: rcrop20 + rfliplr + rflipud
        p = 20
        tmp = tf.image.random_crop(image, size = [int(round(IMG_WIDTH * p/100,0)), int(round(IMG_HEIGHT * p/100,0)), BANDS])
        tmp = tf.image.resize(tmp, size = [IMG_WIDTH, IMG_HEIGHT])
        tmp = tf.image.flip_up_down(image) if np.random.rand() > 0.5 else tmp
        tmp = tf.image.flip_left_right(image) if np.random.rand() > 0.5 else tmp
        self.save_augmented_image(tmp, fn, f'rcrop{p}flip')
        
    return (row.cluster_id, flag)

  @staticmethod
  def random_erasing(img, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3):
    '''
    img is a 3-D variable (ex: tf.Variable(image, validate_shape=False) ) and  HWC order
    '''
    img = tf.Variable(img, validate_shape=False)

    # HWC order
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    channel = tf.shape(img)[2]
    area = tf.cast(width*height, tf.float32)
    
    erase_area_low_bound = tf.cast(tf.round(tf.sqrt(sl * area * r1)), tf.int32)
    erase_area_up_bound = tf.cast(tf.round(tf.sqrt((sh * area) / r1)), tf.int32)
    h_upper_bound = tf.minimum(erase_area_up_bound, int(np.ceil(height/2)))
    w_upper_bound = tf.minimum(erase_area_up_bound, int(np.ceil(width/2)))

    h = tf.random.uniform([], erase_area_low_bound, h_upper_bound, tf.int32)
    w = tf.random.uniform([], erase_area_low_bound, w_upper_bound, tf.int32)

    x1 = tf.random.uniform([], 0, height+1 - h, tf.int32)
    y1 = tf.random.uniform([], 0, width+1 - w, tf.int32)

    erase_area = tf.cast(tf.random.uniform([h, w, channel], 0, 255, tf.int32), tf.uint8)

    erasing_img = img[x1:x1+h, y1:y1+w, :].assign(erase_area)

    return tf.cond(tf.random.uniform([], 0, 1) > probability, lambda: img, lambda: erasing_img)

  def exists_ntimes(self, fn_template, n_augmentations):
    fn = self.get_image_fn(fn_template, "*")
    nfiles = len(glob.glob(os.path.join(self.path_augmented_imgs,fn)))
    return nfiles == n_augmentations

  def get_image_fn(self, fn_template, name):
    fn = fn_template.replace(f".{IMG_TYPE}",f"-{name}.{IMG_TYPE}")
    return fn

  def save_augmented_image(self, image, fn_template, name):
    fn = self.get_image_fn(fn_template, name)
    if ios.exists(fn):
      return
    
    if type(image) != np.ndarray:
      image = image.numpy()
    
    if type(image[0,0,0]) != np.uint8 and name in ['scale2','noise']:
      image = image*255

    plt.imsave(os.path.join(self.path_augmented_imgs,f"{fn}"),image.reshape(IMG_WIDTH, IMG_HEIGHT, BANDS).astype('uint8'))
    plt.close()

    
        # def tunning_summary(self, duration):
  #   if self.results_path:
  #     print("--- %s secs tuning ---" % (duration))
  #     self.best_params.update({'duration':duration,'best_score':self.best_score})
  #     ios.save_csv(self.cv_results, fn=os.path.join(self.results_path,'hp_summary.csv'))
  #     ios.save_json(self.best_params, fn=self.get_best_params_summary_fn())
  
  
  # def flush(self):
  #   del(self.session)
  #   del(self.config)
  #   del(self.model)
  #   del(self.best_params)
  #   del(self.best_score)
  #   del(self.metrics)
  #   del(self.history)
  #   gc.collect()

 # def needs_tuning(self):
  #   self.best_params = ios.load_json(self.get_best_params_summary_fn(), verbose=True)
  #   if self.best_params:
  #     print("Best hyper-params loaded.")
  #     self.lr = self.best_params['lr']
  #     self.dropout = self.best_params['dropout']
  #     self.optimizer_name = self.best_params['optimizer_name']
      
  #     try:
  #       self.rotation = self.best_params['rotation']
  #       self.contrast = self.best_params['contrast']
  #       self.translation = self.best_params['translation']
  #       self.zoom = self.best_params['zoom']
  #       self.contrast_range = self.best_params['contrast_range']
  #       self.brightness_delta = self.best_params['brightness_delta']
  #     except:
  #       self.rotation = None
  #       self.contrast = None
  #       self.translation = None
  #       self.zoom = None
  #       self.contrast_range = None
  #       self.brightness_delta = None
      
  #     return False
    
  #   if self.lr is None or self.dropout is None or self.optimizer_name is None:
  #     print("This run needs hyper-param tuning.")
  #     return True
  #   return False

  # def progress_hyper_params_tuning_eval(self):
  #   progress = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))
  #   if self.df_evaluation.shape[0] == 0:
  #     return progress
  #   else:
  #     tmp = self.df_evaluation.groupby(['rs','run','fold']).size().reset_index(name='counts')
  #     for id, row in tmp.iterrows():
  #       progress[row.rs][row.run][row.fold] = row.counts
  #   return progress

  # def update_hyper_params_tuning_eval(results, epoch, fold, rs, hparams):
  #   cols = None
  #   if self.df_evaluation.shape[0] > 0:
  #     cols = self.df_evaluation.columns

  #   tmp = self.df_evaluation.query("run==@epoch and fold==@fold and rs==@rs")
  #   if tmp.shape[0] > 0:
  #     # update
  #     for hyperparam,value in hparams.items():
  #       self.df_evaluation.loc[tmp.index,hyperparam] = value
  #     for metric,value in results.items()
  #       self.df_evaluation.loc[tmp.index,metric] = value
  #   else:
  #     # new
  #     obj = {'run':epoch, 'fold':fold, 'rs':rs}
  #     obj.update(hparams)
  #     obj.update(results)
  #     if cols is None:
  #       cols = ['run','fold','rs'] + list(hparams.keys()) + list(results.keys())

  #     self.df_evaluation = self.df_evaluation.append(pd.DataFrame(obj, columns=cols), ignore_index=True)


  # def __hyper_parameter_tuning(self, X_train, y_train, X_val, y_val, kfold=5):
  #   # model: modelname, nclasses, pixels, bands, lr, dropout, rotation=0.2, contrast=0.9, translation=0.2, zoom=0.6

  #   # hyper-params
  #   lrs = [1e-2, 1e-3, 1e-4, 1e-5] #1e-1, 3e-1, 5e-2, 3e-2, 
  #   optimizer_names = CNN_TUNING_OPTIMIZERS
  #   dropouts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
  #   if self.model_name.startswith("aug_"):
  #     # online augmentation
  #     rotations = [0.1,0.2,0.3,0.4,0.5]
  #     contrasts = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
  #     translations = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
  #     zooms = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
  #     contrast_ranges = [[0.5, 0.6], [0.6,0.7], [0.7,0.8], [0.8,0.9],[0.9,1.0],[1.0,2.0]]
  #     brightness_deltas = [[-0.3,-0.2],[-0.2,-0.1],[-0.1,0.0],[0.0,0.1],[0.1,0.2]]

  #     params = dict(modelname=[self.model_name],
  #                   nclasses=[self.n_classes],
  #                   pixels=[self.pixels],
  #                   bands=[self.bands],
  #                   lr=lrs, 
  #                   dropout=dropouts,
  #                   optimizer_name=optimizer_names,
  #                   rotation=rotations,
  #                   contrast=contrasts,
  #                   translation=translations,
  #                   zoom=zooms,
  #                   contrast_range=contrast_ranges,
  #                   brightness_delta=brightness_deltas)
  #   else:
  #     print("Hypter-param does not include augmentation.")
  #     params = dict(modelname=[self.model_name],
  #                   nclasses=[self.n_classes],
  #                   pixels=[self.pixels],
  #                   bands=[self.bands],
  #                   lr=lrs, 
  #                   dropout=dropouts,
  #                   optimizer_name=optimizer_names)
    
  #   combinations = np.prod([len(v) for v in params.values()])
  #   print("- All possible combinations hyper-params: ", combinations)
  #   n_iter = min(CNN_TUNING_ITER,int(round(combinations/2)))
  #   print("- # random candidates: ", n_iter)
    
  #   # model
  #   model = models.get_Keras_model(self.model_name)(build_fn=models.define_model, epochs=CNN_TUNING_EPOCHS, batch_size=self.batch_size, verbose=1)
    
  #   # randomsearch
  #   randm_src = RandomizedSearchCV(estimator=model, 
  #                                  scoring = CNN_REG_SCORING if self.isregression else CNN_CLASS_SCORING,
  #                                  param_distributions = params, 
  #                                  cv = KFold(kfold),
  #                                  n_iter = n_iter,
  #                                  pre_dispatch=1,
  #                                  n_jobs=1,
  #                                  refit=False,
  #                                  verbose = 20)
    
  #   # Fit
  #   X = np.append(X_train, X_val, axis=0)
  #   y = np.append(y_train, y_val, axis=0)
  #   result = randm_src.fit(X, y)
  #   K.clear_session()
    
  #   print(" Results from Random Search " )
  #   #print("\n The best estimator across ALL searched params:\n", randm_src.best_estimator_) # commented when refit is False
  #   print("\n The best score across ALL searched params:\n", randm_src.best_score_)
  #   print("\n The best parameters across ALL searched params:\n", randm_src.best_params_)
  #   self.cv_results = pd.DataFrame.from_dict(result.cv_results_)
    
  #   # best hyper-params
  #   self.best_params = randm_src.best_params_
  #   self.best_score = randm_src.best_score_
  #   self.lr = self.best_params['lr']
  #   self.dropout = self.best_params['dropout']
  #   self.optimizer_name = self.best_params['optimizer_name']
    
  #   try:
  #     self.rotation = self.best_params['rotation']
  #     self.contrast = self.best_params['contrast']
  #     self.translation = self.best_params['translation']
  #     self.zoom = self.best_params['zoom']
  #     self.contrast_range = self.best_params['contrast_range']
  #     self.brightness_delta = self.best_params['brightness_delta']
  #   except:
  #     self.rotation = None
  #     self.contrast = None
  #     self.translation = None
  #     self.zoom = None
  #     self.contrast_range = None
  #     self.brightness_delta = None
      
  #   # flush
  #   del(X)
  #   del(y)
  #   del(result.cv_results_)
  #   del(result)
  #   del(model)
  #   del(randm_src.best_params_)
  #   del(randm_src.best_score_)
  #   # del(randm_src.best_estimator_)
  #   del(params)
  #   gc.collect()
  #   time.sleep(1)
