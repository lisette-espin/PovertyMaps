#export PYTHONPATH=/env/python:/home/leespinn/code/SES-Inference/libs/

# README:
# if folds have been trained separately, after all folds are done, run notebooks/_MergeTuningFiles.ipynb
# Then, continue runing each epoch (this should be a TODO automatically)

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

def run(root, years,  model_name, y_attributes, dhsloc, traintype, kfold=4, epochs=100, patience=50, 
        class_weight=False, n_jobs=1, retrain=False, offaug=False, gpus='0', specific_run=None, specific_fold=None, 
        img_width=None, img_height=None):
  # validation
  validations.validate_not_empty(root,'root')
  img_width = IMG_WIDTH if img_width is None else img_width
  img_height = IMG_HEIGHT if img_height is None else img_height
  
  ### 0. Pre-validation
  isregression = model_name.endswith("_regression")
  y_attributes = validations.get_valid_output_names(y_attributes)
  print("INFO: {} | {} augmentation | predict: {}".format('regression' if isregression else 'classification', 'offline' if offaug else 'online' if model_name.startswith('aug_') else 'no', y_attributes))
  
  ### 1. Hyper-param tunning
  data = Data(root, years, dhsloc, traintype, img_width=img_width, img_height=img_height, 
              model_name=model_name, offaug=offaug, isregression=isregression)
  for train, val, path, runid, rs, fold in data.iterate_train_val(specific_run=specific_run, specific_fold=specific_fold):

    ### 0a. skip runids if specific_run is set, otherwise do all
    if specific_run is None:
      pass
    elif specific_run != runid:
      continue
      
    ### 0b. skip fold if specific_fold is set, otherwise do all
    if specific_fold is None:
      pass
    elif specific_fold != fold:
      continue
      
    print("==========================================")
    print(f"1. LOADING: runid:{runid}, foldid:{fold}")
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
  data = Data(root, years, dhsloc, traintype, img_width=img_width, img_height=img_height, 
              model_name=model_name, isregression=isregression)
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
  parser.add_argument("-imgwidth", help="Image width", type=int, default=None)
  parser.add_argument("-imgheight", help="Image width", type=int, default=None)
  parser.add_argument("-shutdown", help="Python script that shutsdown the server after training.", type=str, default=None, required=False)
    
  args = parser.parse_args()
  for arg in vars(args):
    print("{}: {}".format(arg, getattr(args, arg)))

  start_time = time.time()
  try:
    run(args.r, args.years, args.model, args.yatt, args.dhsloc, args.traintype, args.kfold, args.epochs, args.patience, args.cw, args.njobs, args.retrain, args.offaug, args.gpus, args.runid, args.foldid, args.imgwidth, args.imgheight)
  except Exception as ex:
    print(ex)
  print("--- %s seconds ---" % (time.time() - start_time))

  if args.shutdown:
    system.google_cloud_shutdown(args.shutdown)


    
    

#       ses.define_model()   
#       ses.define_callbacks()
#       ses.define_class_weights(y_train)

#       ### 3. training
#       start_time_training = time.time()
#       njobs = multiprocessing.cpu_count()
#       ses.fit(X_train, y_train, X_val, y_val, njobs)
#       duration = time.time() - start_time_training

#       ### 4. results
#       ses.plot_learning_curves()
#       ses.evaluate(X_val, y_val, 'val')
#       ses.evaluate(X_test, y_test, 'test')
#       ses.log(duration)
#       ses.model_summary()


    # del(ses)
    # del(train)
    # del(val)
    # del(test)
    # del(X_train)
    # del(X_val)
    # del(X_test)
    # del(y_train)
    # del(y_val)
    # del(y_test)
    

# def skip(path, fold):
#   ### @TODO: make this automatic by checking if summary.json exists
  
#   # temporal code when google cloud serivce terminates and need to be restarted.
#   epoch = int(path.split("/")[-1].split('-rs')[0].replace("epoch",""))
#   fold = int(fold) #.replace("fold",""))
#   cont = False
  
#   # if epoch in [2,4]:
#   #   print(f'epoch {epoch} already done!')
#   #   cont = True
  
#   # if epoch == 3 and fold in [1]:
#   #   print(f"epoch {epoch} fold {fold} already done!")
#   #   cont = True
  
#   if not cont:
#     print('to do: ', epoch, fold)
    
#   return cont

# def plot_learning_curves(history, fn):
#     metrics = ['loss','accuracy','auc']
#     fig,axes = plt.subplots(len(metrics),1,figsize=(6,len(metrics)*2),sharex=True,sharey=False)
#     for i, k in enumerate(metrics):
#         axes[i].plot(history.history[k], color='blue', label='train')
#         axes[i].plot(history.history['val_{}'.format(k)], color='orange', label='val')
#         axes[i].set_title(k)
#     plt.legend()
#     plt.savefig(fn)
#     plt.close()
    

    

# def load_data(root, location_kind=None, seed=None, verbose=False):
#     pixels = IMG_SIZE - 20 # -20 to remove logo at the bottom
#     path_cnn = os.path.join(root, 'cnn', '{}c_{}x{}{}'.format(NCLASSES, pixels, pixels,'' if location_kind is None else '_{}'.format(location_kind)))
#     ios.create_path(path_cnn)
#     print("Results folder: {}".format(path_cnn))
    
#     # 1. if already exist, return X,y for train, val and test
#     fn_img_compressed_train = os.path.join(path_cnn, 'train_images.npz')
#     fn_img_compressed_val = os.path.join(path_cnn, 'val_images.npz')
#     fn_img_compressed_test = os.path.join(path_cnn, 'test_images.npz')
#     if ios.exists(fn_img_compressed_train) and  ios.exists(fn_img_compressed_val) and  ios.exists(fn_img_compressed_test):
#         print("Loading datasets...")
        
#         X = {'train':None, 'val':None, 'test':None}
#         y = {'train':None, 'val':None, 'test':None}
#         for kind in ['train','val','test']:
#             fn_img_compressed = os.path.join(path_cnn, '{}_images.npz'.format(kind))
#             data = np.load(fn_img_compressed)
#             X[kind] = data['arr_0']
#             y[kind] = data['arr_1']
            
#             if verbose:
#                 print("{} loaded.".format(fn_img_compressed))
            
#         return X['train'], X['val'], X['test'], y['train'], y['val'], y['test'], path_cnn, pixels
    
    
#     # 2. otherwise create X, y    
#     fn_cluster = glob.glob(os.path.join(root,'features','clusters','*_iwi_cluster.csv'))[0] # cluster location unchanged
   
#     if verbose:
#         print("Survey data: {}".format(fn_cluster))
        
#     fn_cluster_cat = os.path.join(path_cnn,os.path.basename(fn_cluster).replace(".csv","_cat.csv"))
#     fn_cluster_cat_train = fn_cluster_cat.replace(".csv","_train.csv")
#     fn_cluster_cat_val = fn_cluster_cat.replace(".csv","_val.csv")
#     fn_cluster_cat_test = fn_cluster_cat.replace(".csv","_test.csv")
    
#     if ios.exists(fn_cluster_cat_train) and ios.exists(fn_cluster_cat_val) and ios.exists(fn_cluster_cat_test):
#         # 2.1 if train, val, test exist
#         if verbose:
#             print("Loading train, val and test sets...")
#             train = ios.load_csv(fn_cluster_cat_train)
#             val = ios.load_csv(fn_cluster_cat_val)
#             test = ios.load_csv(fn_cluster_cat_test)
#     else:
    
#         if ios.exists(fn_cluster_cat):
#             # 2.2 if categories already exist
#             if verbose:
#                 print("Loading categories...")
#             df = ios.load_csv(fn_cluster_cat)
#         else:
#             # 2.3 generate categories
#             if verbose:
#                 print("Generating categories...")

#             # assigning categories to each record
#             df = ios.load_csv(fn_cluster)

#             if location_kind is not None:
#                 # load mapping to pplace (OSMID)
#                 fn_map = glob.glob(os.path.join(root,'features','{}_cluster_pplace_ids.csv'.format(location_kind)))[0]
#                 if verbose:
#                     print("Mapping IDs: ", fn_map)
#                 df_mappings = ios.load_csv(fn_map)
                
#                 print(df_mappings.shape[0], df_mappings.dhs.max(), df_mappings.dhs.min())
                
#                 ### @TODO: for gc and gcur there is no dhs (then reconstruct dataframe)
                
#                 df.loc[df_mappings.dhs,'OSMID'] = df_mappings.OSMID.astype(pd.Int64Dtype())
#             else:
#                 df.loc[:,'OSMID'] = pd.NA

#             df.loc[:,'category'] = pd.cut(df[COLUMN_TARGET], bins=NCLASSES, labels=LABELS, include_lowest=True, precision=0, right=False)
#             df.loc[:,'category_numeric'] = df.category.apply(lambda c: LABELS.index(c))
#             df.rename(columns={'DHSYEAR':'year',"DHSCLUST":"cluster","URBAN_RURA":"urban","LATNUM":"lat","LONGNUM":"lon"}, inplace=True)
#             ios.save_csv(df, fn_cluster_cat)
            
#             # keeping track of bins (for later apply the same ranges to pplaces)
#             bins = pd.cut(df[COLUMN_TARGET], bins=NCLASSES, include_lowest=True, precision=0, right=False)
#             bins = bins.values.categories
#             bins = {'left':'closed', 'right':'open', 'bins':{i:[b.left, b.right] for i,b in enumerate(bins)}}
#             ios.save_json(bins, fn_cluster_cat.replace(".csv","_bins.json"))
            
#         # 2.4. Splitting train, val, test (stratified and balanced according to each class representation)
#         train, test = utils.stratify_sampling(df, 'category', 1-VAL_FRAC, seed)
#         train, val = utils.stratify_sampling(train, 'category', 1-VAL_FRAC, seed)
#         ios.save_csv(train, fn_cluster_cat_train)
#         ios.save_csv(val, fn_cluster_cat_val)
#         ios.save_csv(test, fn_cluster_cat_test)
#         del(df)
    
#     if verbose:
#         print(train.category.unique())
#         print(train.category_numeric.unique())

#     # 3. Create X and y
#     X = {'train':None, 'val':None, 'test':None}
#     y = {'train':None, 'val':None, 'test':None}
#     path_staticmaps = os.path.join(root, 'staticmaps')
#     counter = 0
#     for kind,df in [('train',train), ('val',val), ('test',test)]:
#         photos, targets = list(), list()
#         # enumerate files in the directory
#         for id,row in df.iterrows():

#             # load image
#             if row.OSMID is pd.NA:
#                 # cluster info
#                 path_img = os.path.join(path_staticmaps,'clusters')
#                 prefix = "Y{}-C{}-U{}".format(row.year, row.cluster, row.urban)
#             else:
#                 # PPlace (OSMID) info
#                 path_img = os.path.join(path_staticmaps,'pplaces')
#                 prefix = "OSMID{}".format(row.OSMID)

#             fn = glob.glob(os.path.join(path_img,"{}-LA*-ZO{}-SC{}-{}x{}-{}.png".format(prefix, ZOOM, SCALE, IMG_SIZE, IMG_SIZE, MAPTYPE)))
            
#             # if len(fn) == 0:
#             #     ### this is temporary. The downloading of maps should be done separately before.
#             #     if verbose:
#             #         print("downloading image: {}...".format(prefix))
#             #     size = "{}x{}".format(IMG_SIZE,IMG_SIZE)
#             #     sm = StaticMaps(key=API_KEY, secret=SECRET_KEY, lat=row.lat, lon=row.lon, size=size, 
#             #                     zoom=ZOOM, scale=SCALE, maptype=MAPTYPE)
#             #     sm.retrieve_and_save(path_img, prefix=prefix, verbose=False)
#             #     fn = glob.glob(os.path.join(path_img,"{}-LA*-ZO{}-SC{}-{}x{}-{}.png".format(prefix, ZOOM, SCALE, 
#             #                                                                                        IMG_SIZE, IMG_SIZE, MAPTYPE)))
            
#             photo = load_img(fn[0], target_size=(IMG_SIZE,IMG_SIZE))
#             # convert to numpy array
#             photo = img_to_array(photo, dtype='uint8')
#             photo = photo[0:-20,0:-20,:] # removing logo and keeping it squared
#             # get tags
#             target = to_categorical(row.category_numeric, NCLASSES)
#             # store
#             photos.append(photo)
#             targets.append(target)

#         X[kind] = np.asarray(photos, dtype='uint8')
#         y[kind] = np.asarray(targets, dtype='uint8')
#         fn_img_compressed = os.path.join(path_cnn, '{}_images.npz'.format(kind))
#         savez_compressed(fn_img_compressed, X[kind], y[kind]) 
#         if verbose:
#             print("{} saved!".format(fn_img_compressed))
            
#     return X['train'], X['val'], X['test'], y['train'], y['val'], y['test'], path_cnn, pixels


# def get_class_weight(y):
#     vals = np.argmax(y,axis=1)
#     labels = np.unique(vals)
#     weights = compute_class_weight('balanced',classes=labels, y=vals)
#     weights = {l:w for l,w in zip(*[labels,weights])}
#     return weights
