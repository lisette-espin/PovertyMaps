#export PYTHONPATH=/env/python:/home/leespinn/code/SES-Inference/libs/

###############################################################################
# Dependencies
###############################################################################
import os
import gc
import glob
import time
import argparse
import numpy as np
import pandas as pd
from numpy import savez_compressed
from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from ses.data import Data
from ses.images import SESImages
from utils import system
from utils import ios
from utils import validations

###############################################################################
# Constants
###############################################################################

from utils.constants import *

###############################################################################
# Functions
###############################################################################

def run(root, years,  model_name, y_attribute, dhsloc, traintype, fm_layer=20, njobs=1, offaug=False, specific_run=None):
  # validation
  validations.validate_not_empty(root,'root')
  
  # data
  isregression = model_name.endswith("_regression")

  if dhsloc not in DHSLOC_OPTIONS:
    print(f"DHSLOC: {dhsloc} does not exist.")
    raise Exception("DHSLOC does not exist.")
  
  ### 1. data
  data = Data(root=root, years=years, dhsloc=dhsloc, traintype=traintype, model_name=model_name, offaug=offaug, isregression=isregression)
  data.set_nclasses(y_attribute)
  
  ### 2. for each independent run and fold extract feaure map from train and val
  print("********** TRAIN & VAL SETS **********")
  for train, val, test, path, runid, rs, fold in data.iterate_train_val_test(tune_img=False, specific_run=specific_run):
    print("==========================================")
    print(f"1. LOADING: {runid}-{fold} ({path})")
    del(test)
    gc.collect()
    
    # 2.1. Train and val
    X_train, y_train = data.cnn_get_X_y(train, y_attribute, offaug)
    X_val, y_val = data.cnn_get_X_y(val, y_attribute)
    pixels = X_train.shape[1]
    bands  = X_train.shape[3]
    
    # 2.2. load CNN model
    ses = SESImages(root=root, runid=runid, n_classes=data.n_classes, model_name=model_name,
                    pixels=pixels, bands=bands, isregression=isregression, rs=rs, offaug=offaug)
    ses.init(path)
    ses.load_model()
    print("SESpath:",ses.results_path)
    
    # 2.3. extract feature map
    for setname, df, X, y_true in [('train',train,X_train,y_train),('val',val,X_val,y_val)]:
      
      # 2.3.1 feature maps
      feature_map = ses.extract_features_last_layer(X, fm_layer)
      save_feature_map(feature_map, fm_layer, setname, ses.results_path, fold)
      
      # 2.3.2 memory flush
      del(X)
      del(y_true)
      del(df)
      del(feature_map)
      gc.collect()
    
    # 2.4. general memory flush
    del(ses.model)
    del(ses)
    del(X_train)
    del(y_train)
    del(X_val)
    del(y_val)
    gc.collect()
  
  ### 3. for each independent run, predict and extract feaure map for test set
  print("********** TRAIN & TEST SETS **********")
  for train, test, path, runid, rs in data.iterate_train_test(specific_run=specific_run):
    print("==========================================")
    print(f"1. LOADING: {runid} ({path})")
    
    # 3.1. Train, Test
    X_train, y_train = data.cnn_get_X_y(train, y_attribute, offaug)
    X_test, y_test = data.cnn_get_X_y(test, y_attribute)
    pixels = X_train.shape[1]
    bands  = X_train.shape[3]
    
    # 3.2. load CNN model
    ses = SESImages(root=root, runid=runid, n_classes=data.n_classes, model_name=model_name,
                    pixels=pixels, bands=bands, isregression=isregression, rs=rs, offaug=offaug)
    ses.init(path)
    ses.load_model()
    
    # 3.3 train feature maps
    # if offaug, then it is stored with data offline data aug
    setname = 'train'
    feature_map = ses.extract_features_last_layer(X_train, fm_layer)
    save_feature_map(feature_map, fm_layer, setname, ses.results_path)
    
    # 3.4 if offaug, then it is stored without data offline data aug
    if offaug:
      del(X_train)
      del(y_train)
      gc.collect()
      setname = 'train_noaug'
      X_train, y_train = data.cnn_get_X_y(train, y_attribute)
    feature_map = ses.extract_features_last_layer(X_train, fm_layer)
    save_feature_map(feature_map, fm_layer, setname, ses.results_path)
    y_pred = ses.predict(X_train)
    #save_prediction(train, y_pred, setname, y_attribute, ses.results_path)
    
    # 3.5 test feature maps, prediction and evaluation
    setname = 'test'
    feature_map = ses.extract_features_last_layer(X_test, fm_layer)
    save_feature_map(feature_map, fm_layer, setname, ses.results_path)
    y_pred = ses.predict(X_test)
    #save_prediction(test, y_pred, setname, y_attribute, ses.results_path)
    #save_evaluation(y_test, y_pred, setname, y_attribute, ses.results_path, isregression=isregression)
    
    # 3.5 memory flush
    del(X_train)
    del(y_train)
    del(X_test)
    del(y_test)
    del(y_pred)
    del(feature_map)
    gc.collect()
    del(ses.model)
    del(ses)
    gc.collect()

  
def save_feature_map(fmap, fm_layer, name, path, fold=None):
  folder = os.path.join(path, f"layer-{fm_layer}")
  ios.validate_path(folder)
  
  postfix = f"_{fold}" if fold is not None else ""
  fn = os.path.join(folder, f'fmap_{name}{postfix}.npz')
  savez_compressed(fn, fmap) 
  print("Feature vectors: {}".format(fmap.shape))
  print("{} saved!".format(fn))
  

  
###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
  parser.add_argument("-years", help="Years (comma-separated): 2016,2019", type=str, required=True)
  parser.add_argument("-model", help="Model name", type=str, default='aug_cnn_mp_dp_relu_sigmoid_adam_reg', required=True)
  parser.add_argument("-yatt", help="Attributes (column names) for dependent variable y (comma separated)", type=str, default='dhs_mean_iwi', required=True)
  parser.add_argument("-dhsloc", help="DHS cluster option (None, cc, ccur, gc, gcur, rc).", type=str, default=None, required=False)
  parser.add_argument("-traintype", help="Years to include in training: all, newest, oldest", type=str, default='none')
  parser.add_argument("-fmlayer", help="Layer to extract feature map.", type=int, required=True)
  parser.add_argument("-njobs", help="Parallel processes.", type=int, default=1)
  parser.add_argument("-offaug", help="Whether or not to use model with offline augmented images.", action='store_true')
  parser.add_argument("-runid", help="Runid to focus on (so this script can be run in parallel)", type=int, default=None)
  parser.add_argument("-shutdown", help="Python script that shutsdown the server after training.", type=str, default=None, required=False)
  
  args = parser.parse_args()
  for arg in vars(args):
    print("{}: {}".format(arg, getattr(args, arg)))

  start_time = time.time()
  try:
    run(args.r, args.years, args.model, args.yatt, args.dhsloc, args.traintype, args.fmlayer, args.njobs, args.offaug, args.runid)
  except Exception as ex:
    print(ex)
  
  print("--- %s seconds ---" % (time.time() - start_time))
    
  if args.shutdown:
    system.google_cloud_shutdown(args.shutdown)


    
    
# def save_prediction(df, y_pred, name, y_attribute, path):
#   y_attribute = Data.get_valid_output_names(y_attribute)
#   y_attribute = f'pred_{y_attribute}' if type(y_attribute)==str else [f'pred_{a}'for a in y_attribute]

#   df.loc[:,y_attribute] = y_pred
#   fn = os.path.join(path,f'pred_{name}.csv')
#   ios.save_csv(df, fn) 
#   print("Predictions ({}): {}, {}".format(name, type(y_pred), y_pred.shape))
  
# def save_evaluation(y_true, y_pred, name, y_attribute, path, isregression=True):
#   if not isregression:
#     raise Exception("Not implemented for classification yet.")
#   if y_pred.shape[1] > 2:
#     raise Exception("More than 2 outputs is not suppoted yet.")

#   results = {'mse':None, 'r2':None, 'mse_mean':None, 'r2_mean':None, 'mse_std':None, 'r2_std':None, 'corr_true':(None,None), 'corr_pred':(None,None)}
#   y_attribute = Data.get_valid_output_names(y_attribute)

#   # overall
#   mse = mean_squared_error(y_true, y_pred)
#   r2 = r2_score(y_true, y_pred)
#   results['mse'] = float(mse)
#   results['r2'] = float(r2)
  
#   # individual outputs
#   for i,name in enumerate(y_attribute):
#     yt = y_true[:,i]
#     yp = y_pred[:,i]
#     mse = mean_squared_error(yt, yp)
#     r2 = r2_score(yt, yp)
#     name = name.replace('dhs','').replace('iwi','').replace('_','')
#     results[f'mse_{name}'] = float(mse)
#     results[f'r2_{name}'] = float(r2)
    
#   results['corr_true'] = list(pearsonr(y_true[:,0],y_true[:,1]))
#   results['corr_pred'] = list(pearsonr(y_pred[:,0],y_pred[:,1]))

#   print(results)
  
#   fn = os.path.join(path, CNN_EVALUATION_FILE)
#   ios.save_json(results, fn)
    
    
    
# def run_pplaces(root, dhsloc, epoch, rs, model_fn, y_attribute, verbose):

#   ### 1. data
#   data = Data(root, dhsloc, epoch, rs)
#   pplaces = data.iterate_pplaces(y_attribute)
#   X = data.cnn_get_X(pplaces)
#   pixels = X.shape[1]
#   bands  = X.shape[3] 

#   ### 2. model  
#   ses = SESImages(root, data.n_classes, None, pixels, bands, seed=rs)
#   ses.init()
#   ses.load_model(model_fn)

#   ### 3. predict
#   pred = ses.predict(X)
#   save_prediction(pred, pplaces[['id','lat','lon','rural']], 'pplaces', ses.results_path)
#   save_correlation(pred, 'pplaces', ses.results_path)
  
#   ### 4. extract features
#   feature_map = ses.extract_features_last_layer(X)
#   save_feature_map(feature_map, 'pplaces', ses.results_path)




    # init(seed)
    # root = os.path.join(root,'results')
    # print("Results folder: {}".format(root))
    
    # print("==========\n{} SET\n==========".format(dataset))
    
    # ### 0. validation
    # if dataset not in VALID_DATASETS:
    #     raise Exception("Invalid dataset.")
    
    # ### 1. model
    # if verbose:
    #     print('Loading model...')
    # model = load_model(model_fn)
    # nclasses = model.layers[-1].output.shape[1]
    # pixels = model.layers[0].output.shape[1]
    
    # if verbose:
    #     print(model.summary())
    #     print('- nclasses: {}'.format(nclasses))
    #     print('- input image size: {}x{}'.format(pixels,pixels))
    
    # for batch in np.arange(1,nbatches+1,1):
        
    #     ### 2. data: X
    #     if verbose:
    #         print('Loading data: batch {} of {}...'.format(batch, nbatches))
    #     X, path_cnn = load_data(root, nclasses, pixels, dataset, location_kind, batch, nbatches, seed, verbose)

        
    #     ### 3. predict
    #     if verbose:
    #         print('Predicting...')
    #     pred = predict(model, X)
    #     posfix = '' if dataset != 'pplaces' else '-{}-{}'.format(batch, nbatches)
    #     fn_pred = os.path.join(path_cnn,'{}_pred{}.npz'.format(dataset,posfix))
    #     savez_compressed(fn_pred, pred)
    #     if verbose:
    #         print("Predictions: {}".format(pred.shape))
    #         print("{} saved!".format(fn_pred))
                
    #     ### 4. feature extraction
    #     if verbose:
    #         print('Extracting features (last layer)...')

    #     feature_maps = extract_features_last_layer(model, X)
    #     fn_fmaps = fn_pred.replace("_pred","_fmaps")
    #     savez_compressed(fn_fmaps, feature_maps) 
    #     if verbose:
    #         print("Feature vectors: {}".format(feature_maps.shape))
    #         print("{} saved!".format(fn_fmaps))        
        
###############################################################################
# Handlers
###############################################################################

# def init(seed):
#     import tensorflow as tf
#     import tensorflow_hub as hub
#     print("===================================================")
#     print("TF version:", tf.__version__)
#     print("Hub version:", hub.__version__)
#     print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
#     print("===================================================")
#     np.random.seed(seed)
#     set_seed(seed)

# def load_data(root, nclasses, pixels, dataset, location_kind=None, batch=1, nbatches=None,seed=None, verbose=False):
#     tmp = IMG_SIZE - 20 # -20 remove logo at the bottom
#     assert tmp == pixels
#     path_cnn = os.path.join(root, 'cnn', '{}c_{}x{}{}'.format(nclasses, pixels, pixels, '' if location_kind is None else '_{}'.format(location_kind)))
#     print("CNN folder: {}".format(path_cnn))
    
#     # 1. if already exist, return X,y for train, val and test
#     if dataset != 'pplaces':
#         fn_img_compressed = os.path.join(path_cnn, '{}_images.npz'.format(dataset))
#         if ios.exists(fn_img_compressed):
#             print("Loading X (images)...")
#             X = np.load(fn_img_compressed)['arr_0']
#             if verbose:
#                 print("{} loaded.".format(fn_img_compressed))
#                 print("X: {}".format(X.shape))
#             return X, path_cnn

#     # 2. otherwise create X for pplaces
#     if verbose:
#         print("Creating images tensor for pplaces...")
    
#     fn_pplaces = glob.glob(os.path.join(root,'features','pplaces','PPLACES.csv'))[0]
#     df = ios.load_csv(fn_pplaces)
    
#     if location_kind is not None:
#         # load mapping to pplace (OSMID)
#         fn_map = glob.glob(os.path.join(root,'features','{}_cluster_pplace_ids.csv'.format(location_kind)))[0]
#         if verbose:
#             print("Mapping IDs: ", fn_map)
#         df_mappings = ios.load_csv(fn_map)
        
#         # removing pplaces that are now cluster locations
#         df.drop(df_mappings.closest_pplace.dropna().values, inplace=True)
        
#     df = np.array_split(df,nbatches)[batch-1]
    
#     path_img = os.path.join(root, 'staticmaps', 'pplaces')
#     photos, targets = list(), list()
#     # enumerate files in the directory
#     for id,row in df.iterrows():
#         prefix = "OSMID{}".format(row.id)
#         fn = glob.glob(os.path.join(path_img,"{}-LA*-ZO{}-SC{}-{}x{}-{}.png".format(prefix, ZOOM, SCALE, IMG_SIZE, IMG_SIZE, MAPTYPE)))
#         photo = load_img(fn[0], target_size=(IMG_SIZE,IMG_SIZE))
#         # convert to numpy array
#         photo = img_to_array(photo, dtype='uint8')
#         photo = photo[0:-20,0:-20,:] # removing logo and keeping it squared
#         photos.append(photo)
#     X = np.asarray(photos, dtype='uint8')
#     #savez_compressed(fn_img_compressed, X) 
#     if verbose:
#         print("X: {}".format(X.shape))
#         #print("{} saved!".format(fn_img_compressed))
            
#     return X, path_cnn

# def predict(model, X):
#     pred = model.predict(X)
#     return pred

# def extract_features_last_layer(model, X, verbose=False):
#     layer_id = 23
#     remodel = Model(inputs=model.inputs, outputs=model.layers[layer_id].output)
#     if verbose:
#         remodel.summary()
#     feature_maps = remodel.predict(X)
#     return feature_maps


