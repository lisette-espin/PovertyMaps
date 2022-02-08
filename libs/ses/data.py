################################################################################
# Dependencies
################################################################################

import os
import glob
import numpy as np
import pandas as pd

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from utils import ios
from ses import locations as loc 
from ses.images import SESImages
from ses.metadata import SESMetadata

################################################################################
# Constants
################################################################################

from utils.constants import *

################################################################################
# Main
################################################################################

class Data(object):

  def __init__(self, root, years, dhsloc, traintype, epoch=None, rs=None, fold=None, model_name=None, cnn_path=None, isregression=None):
    self.root = root
    self.years = years if type(years)==list else years.strip('').replace(' ','').split(',')
    self.dhsloc = dhsloc
    self.traintype = traintype
    self.epoch = epoch
    self.rs = rs
    self.fold = fold
    self.isregression = isregression
    self.prefix = "_".join([pre for pre in ios.get_prefix_surveys(root=root, years=years).split('_') for y in self.years if y in pre])
    # cnn-images
    self.model_name = model_name
    self.cnn_path = cnn_path
    self.n_classes = None
    self.path_augmented_images = os.path.join(self.root,'results','staticmaps',"augmented")  
    self.df_evaluation = None  
    # cnn-metadata
    self.df_clusters = None
    self.df_pplaces = None
    validate_traintype(traintype)
    validate_years_traintype(years, traintype)

  ####################################################################################################################################
  # GENERAL
  ####################################################################################################################################
  
  def iterate_runid(self):
    path = os.path.join(self.root,'results','samples',f"{self.prefix}_{self.traintype}_{self.dhsloc}","epoch*-rs*" if self.epoch is None else f"epoch{self.epoch}-rs{self.rs}" ,"data.csv")
    files = glob.glob(path)
    
    # epochs
    for fn in files:
      print("FN: ",fn)
      path = os.path.dirname(fn) # until epoch*-rs* folder
      epoch = int(path.split("/")[-1].split("-rs")[0].replace("epoch","")) if self.epoch is None else self.epoch
      
      if epoch != 2: ################################################################################################################## REMOVE THIS (THIS IS JUST FOR DEBUGGING)
          continue
          
      rs = int(path.split("/")[-1].split("-rs")[-1].replace(".csv","")) if self.rs is None else self.rs
      yield path, epoch, rs

  def iterate_train_test(self):
    path = os.path.join(self.root,'results','samples',f"{self.prefix}_{self.traintype}_{self.dhsloc}","epoch*-rs*" if self.epoch is None else f"epoch{self.epoch}-rs{self.rs}" ,"data.csv")
    files = glob.glob(path)
    
    # epochs
    for fn in files:
      print("FN: ",fn)
      df = ios.load_csv(fn)
      df.loc[:,'pplace_rural'] = df.pplace_rural.astype(pd.Int64Dtype())
      df.loc[:,'OSMID'] = df.OSMID.astype(pd.Int64Dtype())
      path = os.path.dirname(fn) # until epoch*-rs* folder
      epoch = int(path.split("/")[-1].split("-rs")[0].replace("epoch","")) if self.epoch is None else self.epoch
      rs = int(path.split("/")[-1].split("-rs")[-1].replace(".csv","")) if self.rs is None else self.rs

      if epoch != 2: ################################################################################################################## REMOVE THIS (THIS IS JUST FOR DEBUGGING)
        continue
          
      # train / test
      test = df.loc[df.test.dropna().index.values]
      train = df.drop(test.index)
      yield train, test, path, epoch, rs

  def iterate_train_val_test(self, tune_img=True):
    print('prefix',self.prefix)
    print('traintype',self.traintype)
    print('root',self.root)
    print('dhsloc',self.dhsloc)
    print('epoch',self.epoch)
    print('rs',self.rs)
    path = os.path.join(self.root,'results','samples',f"{self.prefix}_{self.traintype}_{self.dhsloc}","epoch*-rs*" if self.epoch is None else f"epoch{self.epoch}-rs{self.rs}" ,"data.csv")
    files = glob.glob(path)
    
    # epochs
    for fn in files:
      print("FN: ",fn)
      df = ios.load_csv(fn)
      df.loc[:,'pplace_rural'] = df.pplace_rural.astype(pd.Int64Dtype())
      df.loc[:,'OSMID'] = df.OSMID.astype(pd.Int64Dtype())
      path = os.path.dirname(fn) # until epoch*-rs* folder
      epoch = int(path.split("/")[-1].split("-rs")[0].replace("epoch","")) if self.epoch is None else self.epoch
      rs = int(path.split("/")[-1].split("-rs")[-1].replace(".csv","")) if self.rs is None else self.rs
      
      # folds
      test = df.loc[df.test.dropna().index.values]
      foldcol = [c.replace('fold','') for c in df.columns if c.startswith("fold")] if self.fold is None else [self.fold]
      for fold in foldcol:
        
        if epoch != 2: ################################################################################################################## REMOVE THIS
          continue
        
        if (tune_img and SESImages.needs_tuning_runid_fold(path, self.model_name, epoch, rs, fold)) or not tune_img:
          tmp = df.loc[df[f'fold{fold}'].dropna().index.values].copy()
          train = tmp.query(f"fold{fold}=='train'")
          val = tmp.query(f"fold{fold}=='val'")
          yield train, val, test, path, epoch, rs, fold

  def set_nclasses(self, y_attribute, df=None):
    y_attribute = Data.get_valid_output_names(y_attribute)
    if not self.isregression and len(y_attribute) > 1:
      raise Exception("Multiple output classification is not supported yet.")
    if not self.isregression and df is None:
      raise Exception("Classification requires data to infer number of unique classes.")
    self.n_classes = len(y_attribute) if self.isregression else df.loc[:,y_attribute].nunique()
    print("nclasses:", self.n_classes)

  @staticmethod
  def get_valid_output_names(y_attributes):
    y_attributes = y_attributes if type(y_attributes)==list else y_attributes.strip('').replace(' ','').split(',')
    if len(y_attributes) not in [1,2]:
      raise Exception("Only 1 or 2 outputs are supported at the moment")
    return y_attributes

  ####################################################################################################################################
  # CNN-IMAGES
  ####################################################################################################################################

  ############################################
  # CLusters
  ############################################

  def cnn_get_X(self, df):
    photos = []
    root = os.path.join(self.root, 'results', 'staticmaps')
    
    for i,row in df.iterrows():
      prefix = f"OSMID{row.id}" 
      path_img = os.path.join(root, 'pplaces')
      fn = glob.glob(os.path.join(path_img,"{}-LA*-ZO{}-SC{}-{}x{}-{}.png".format(prefix, ZOOM, SCALE, IMG_WIDTH, IMG_HEIGHT, MAPTYPE)))
      # load as matrix
      photo = load_img(fn[0], target_size=(IMG_WIDTH, IMG_HEIGHT))
      # convert to numpy array
      photo = img_to_array(photo, dtype='uint8')
      photo = photo[0:-PIXELS_LOGO,0:-PIXELS_LOGO,:] # removing logo and keeping it squared
      # append
      photos.append(photo)
      
    X = np.asarray(photos, dtype='uint8')
    return X

  def cnn_get_X_y(self, df, y_attribute, offlineaug=False):
    photos = []
    targets = []
    root = os.path.join(self.root, 'results', 'staticmaps')
    y_attribute = Data.get_valid_output_names(y_attribute)
    self.set_nclasses(y_attribute, df)

    for i,row in df.iterrows():
      prefix = f"OSMID{row.OSMID}" if not pd.isna(row.OSMID) else f"Y{row.dhs_year}-C{row.dhs_cluster}-U{row.dhs_rural+1}" 
      path_img = os.path.join(root, 'pplaces' if not pd.isna(row.OSMID) else 'clusters')
      fn = glob.glob(os.path.join(path_img,"{}-LA*-ZO{}-SC{}-{}x{}-{}.png".format(prefix, ZOOM, SCALE, IMG_WIDTH, IMG_HEIGHT, MAPTYPE)))[0]
      # load as matrix
      photo = load_img(fn, target_size=(IMG_WIDTH, IMG_HEIGHT))
      # convert to numpy array
      photo = img_to_array(photo, dtype='uint8')
      photo = photo[0:-PIXELS_LOGO,0:-PIXELS_LOGO,:] # removing logo and keeping it squared
      photo = tf.image.resize(photo, size = [IMG_WIDTH, IMG_HEIGHT], method='nearest', antialias=True, preserve_aspect_ratio=True) #going back to normal size
      # get tags
      target = row[y_attribute] if self.isregression else to_categorical(row[y_attribute], self.n_classes)
      # append
      photos.append(photo)
      targets.append(target)
      
      if offlineaug:
        ### getting augmented photos
        augmented_photos = self.get_augmented_photos(fn, row)
        if len(augmented_photos) > 0 :
          photos.extend(augmented_photos)
          targets.extend([target]*len(augmented_photos))
        
    X = np.asarray(photos, dtype='uint8')
    y = np.asarray(targets, dtype='float32')
    return X,y
  
  def get_number_of_augmentations(self, fn, row):
    afn = os.path.basename(fn)
    afn = os.path.join(self.path_augmented_images,afn.replace(f".{IMG_TYPE}",f"-*.{IMG_TYPE}"))
    files = glob.glob(afn)
    return len(files)

  def get_augmented_photos(self, fn, row):
    afn = os.path.basename(fn)
    afn = os.path.join(self.path_augmented_images,afn.replace(f".{IMG_TYPE}",f"-*.{IMG_TYPE}"))
    files = glob.glob(afn)
    images = []
    
    for fn in files:
      photo = load_img(fn, target_size=(IMG_WIDTH, IMG_HEIGHT))
      photo = img_to_array(photo, dtype='uint8')
      images.append(tf.convert_to_tensor(photo))
    
    return images
  
  ############################################
  # PPlaces
  ############################################
  
  def iterate_pplaces(self, y_attribute=None):
    # @TODO: check if this code still makes and sense and where is used.    

    # pplaces
    fn = glob.glob(os.path.join(self.root,'results','features',"pplaces","PPLACES.csv"))[0]
    df_pplaces = ios.load_csv(fn)
    df_pplaces.loc[:,'rural'] = df_pplaces.place.apply(lambda c: int(c in PPLACE_RURAL))
    print('pplaces: ', df_pplaces.shape)
    
    # clusters
    fn = glob.glob(os.path.join(self.root,'results','samples',self.dhsloc,f"epoch{self.epoch}-rs*" ,"data.csv"))[0]
    df_clusters = ios.load_csv(fn)
    df_clusters.loc[:,'pplace_id'] = df_clusters.pplace_id.astype(pd.Int64Dtype())
    df_clusters.loc[:,'pplace_rural'] = df_clusters.pplace_rural.astype(pd.Int64Dtype())
    df_clusters.loc[:,'OSMID'] = df_clusters.OSMID.astype(pd.Int64Dtype())
    print('clusters: ', df_clusters.shape, 'pplaces as clusters', df_clusters.pplace_id.dropna().shape)
    y_attribute = Data.get_valid_output_names(y_attribute)
    self.set_nclasses(y_attribute, df_clusters)
    
    # remove
    df_pplaces = df_pplaces.loc[~df_pplaces.index.isin(df_clusters.pplace_id.dropna().index.values)]
    print('new pplaces: ', df_pplaces.shape)
    
    return df_pplaces

  ####################################################################################################################################
  # CNN-METADATA
  ####################################################################################################################################

  def load_metadata(self, viirsnorm=False):
    path = os.path.join(self.root, 'results', 'features')
    print(path)
    print(self.prefix)

    # clusters
    self.df_clusters = None
    for fn in sorted(glob.glob(os.path.join(path,'clusters',f'*{self.prefix}*.csv'))):
      print(fn)
      tmp = ios.load_csv(fn)
      tmp.set_index('DHSID',inplace=True)
      self.df_clusters = tmp.copy() if self.df_clusters is None else self.df_clusters.join(tmp, on='DHSID')
    cols = ['DHSCC','DHSCLUST','CCFIPS','ADM1FIPS','ADM1FIPSNA','ADM1SALBNA','ADM1SALBCO','ADM1DHS','ADM1NAME','DHSREGCO','DHSREGNA','SOURCE','URBAN_RURA','LATNUM','LONGNUM','ALT_GPS','ALT_DEM','DATUM','geometry','SURVEY','iwi_bin','iwi_cat','iwi_cat_id','nodeid']
    self.df_clusters.drop(columns=cols, inplace=True)
    
    # pplaces
    self.df_pplaces = None
    for fn in sorted(glob.glob(os.path.join(path,'pplaces','PPLACES*.csv'))):
      print(fn)
      tmp = ios.load_csv(fn)
      tmp.set_index('OSMID',inplace=True)
      self.df_pplaces = tmp.copy() if self.df_pplaces is None else self.df_pplaces.join(tmp, on='OSMID')
    cols = ['lat','lon','loc_name','name','type','place','population','capital','is_capital','admin_level','GNS:dsg_code','nodeid']
    self.df_pplaces.drop(columns=cols, inplace=True)
    
    if viirsnorm:
      self.viirs_validation()
    
    print("DHSYEARs:",self.df_clusters.DHSYEAR.unique())
    self.df_clusters.drop(columns='DHSYEAR', inplace=True)
    print(f"CLUSTERS: {self.df_clusters.shape}")
    print(f"PPLACES: {self.df_pplaces.shape}")

  def viirs_validation(self):
    # VIIRS: mean0 to avoid big differences across years
    viirscols = [c for c in self.df_clusters.columns if c.startswith("NTLL")]
    # clusters
    tmp = self.df_clusters.groupby("DHSYEAR")[viirscols].transform(lambda x: (x - x.mean()) / x.std())
    self.df_clusters.loc[tmp.index,tmp.columns] = tmp
    # pplaces
    tmp = self.df_pplaces[viirscols].transform(lambda x: (x - x.mean()) / x.std())
    self.df_pplaces.loc[tmp.index,tmp.columns] = tmp

  @staticmethod
  def is_from_source(column, source):
    if source == 'cells':
      return column.startswith('cells_') or column.startswith('towers_') or column == 'distance_closest_cell'
    if source == 'FBM':
      return column.startswith('FBM_')
    if source == 'NTLL':
      return column.startswith('NTLL_')
    if source == 'FBMV':
      return column.startswith('FBMV_')
    if source == 'FBP':
      return column.startswith('population_') or column == 'distance_closest_tile'
    if source == 'OSM':
      return not column.startswith('cells_') and not column.startswith('towers_') and column != 'distance_closest_cell' and not column.startswith('FBM_') and not column.startswith('NTLL_') and not column.startswith('FBMV_') and not column.startswith('population_') and column != 'distance_closest_tile'

    raise Exception("source does not exist")

  def metadata_get_X_y(self, df, y_attribute, fmaps, offlineaug=False, features_source='all'):
    df_data = df.copy()
    osmids = df_data.OSMID.dropna()
    y_attribute = Data.get_valid_output_names(y_attribute)
    
    # cluster features
    df_data = df_data.join(self.df_clusters, on="dhs_id", how='inner')
    print("df_data.dhs_years: ", df_data.dhs_year.unique())
    
    # update features from OSMID
    self.feature_names = [c for c in self.df_clusters.columns if c not in ['mean_iwi','std_iwi','rural']]
    print("----------------->",features_source)
    if features_source == 'all':
      pass
    else:
      self.feature_names = [c for c in self.feature_names if Data.is_from_source(c,features_source)]
      print(self.feature_names)

    df_data.loc[osmids.index,self.feature_names] = self.df_pplaces.loc[osmids,self.feature_names].values
    
    if offlineaug:
      features = []
      targets = []
      root = os.path.join(self.root, 'results', 'staticmaps')
      print('augmented')
      # N_AUGMENTATIONS
      tmp = ios.load_csv(os.path.join(root, 'augmented', '_summary.csv'))
      for i,row in df_data.iterrows():
        features.extend([row.loc[self.feature_names].values])
        targets.extend([row.loc[y_attribute].astype(np.float32).values]) 
        if tmp.query(f"dhs_id == '{row.dhs_id}'").iloc[0].augmented:
          features.extend([row.loc[self.feature_names].values]*N_AUGMENTATIONS)
          targets.extend([row.loc[y_attribute].astype(np.float32).values]*N_AUGMENTATIONS) 

      X = np.nan_to_num(np.asarray(features, dtype='float32')) #.round(PRECISION)
      y = np.nan_to_num(np.asarray(targets, dtype='float32')) #.round(PRECISION)
      
    else:
      # get X and y
      df_data = df_data.fillna(0)
      y = df_data.loc[:,y_attribute].values #.round(PRECISION)
      X = df_data.loc[:,self.feature_names] #.values.round(PRECISION)
    
    print('original shapes:', X.shape, y.shape)
    if fmaps is not None:
      X = np.append(X, fmaps, axis=1)
      print('new shapes:', X.shape, y.shape)
      self.feature_names.extend([f'cnn{i}' for i in np.arange(fmaps.shape[1])])
    print(f'{len(self.feature_names)} total features.')
    
    return X, y, self.feature_names

###############################################################################
# GENERAL
###############################################################################
def delete_nonprojected_variables(df, fn, del_geometry=False):
  # This ie being used in: libs.maps.opencellid and libs.facebook.population

  import gc
  # removing these columns
  dhs_columns = ['DHSCC', 'DHSYEAR', 'DHSCLUST', 'CCFIPS', 'ADM1FIPS',
       'ADM1FIPSNA', 'ADM1SALBNA', 'ADM1SALBCO', 'ADM1DHS', 'ADM1NAME',
       'DHSREGCO', 'DHSREGNA', 'SOURCE', 'URBAN_RURA', 'LATNUM', 'LONGNUM',
       'ALT_GPS', 'ALT_DEM', 'DATUM', 'SURVEY', 'mean_iwi',
       'std_iwi', 'iwi_bin', 'iwi_cat', 'iwi_cat_id'] # keeps DHSID
  pplaces_columns = ['lat', 'lon', 'loc_name', 'name', 'type', 'place', 'population',
       'capital', 'is_capital', 'admin_level', 'GNS:dsg_code'] # keeps OSMID
  fb_mv = ['original_index']
  try:
    cols_to_remove = dhs_columns + pplaces_columns + fb_mv
    if del_geometry:
      cols_to_remove += ['geometry']
    df = df.drop(columns=cols_to_remove, errors='ignore')
    gc.collect()
  except Exception as ex:
    print(f"{fn} | delete_nonprojected_variables | ",ex)
  
  cols = df.columns.values.tolist()
  first_col = [c for c in cols if c in ['DHSID','OSMID']]
  cols = [c for c in cols if c not in first_col]
  cols = first_col +  cols
  return df.loc[:,cols]



  # def xgboost_get_X_y(self, df, y_attribute):

  #   X = None
  #   y = None
  #   kind = None
  #   feature_names = None

  #   for i,row in df.iterrows():
      
  #     if pd.isna(row.OSMID):
  #       # cluster
  #       tmp = self.df_dhs.loc[row.dhs_id,:].copy()
  #       tmp.drop(labels=COLS_DHS_REMOVE, inplace=True)
  #       tmp.loc['rural'] = row.dhs_rural
  #     else:
  #       # pplace
  #       tmp = self.df_pplaces.loc[row.pplace_id,:].copy()
  #       tmp.drop(labels=COLS_PPLACE_REMOVE, inplace=True)
  #       tmp.loc['rural'] = row.pplace_rural
        
  #     # append
  #     tmp = tmp.fillna(0)
  #     feature = tmp.copy().values.reshape(1,-1)
  #     target = np.array([row[y_attribute]])
  #     X = feature if X is None else np.append(X, feature, axis=0)
  #     y =  target if y is None else np.append(y, target, axis=0)

  #     if kind is None:
  #       kind = 'test' if pd.isna(row[f'fold{self.fold}']) else row[f'fold{self.fold}']
  #       feature_names = tmp.index.values

  #   # cnn features
  #   fmap = np.load(os.path.join(self.cnn_path, f"fmap_{kind}.npz"))['arr_0']
  #   print("{} features from NPZ file".format(fmap.shape))
  #   X = np.append(X, fmap, axis=1)
    
  #   if kind == 'train':
  #     feature_names = tmp.index.values.tolist()
  #     feature_names.extend(["cnn{}".format(c) for c in np.arange(fmap.shape[1])])

  #   return X, y.flatten().reshape(-1,1).round(PRECISION), feature_names

  # def xgboost_get_X(self, df):
  #   X = None
  #   y = None
  #   kind = None
  #   feature_names = None

  #   for i,row in df.iterrows():
  #     tmp = self.df_pplaces.loc[i,:].copy()
  #     tmp.drop(labels=COLS_PPLACE_REMOVE, inplace=True)

  #     # append
  #     tmp = tmp.fillna(0)
  #     feature = tmp.copy().values.reshape(1,-1)
  #     X = feature if X is None else np.append(X, feature, axis=0) # here there should be a join by OSMID
      
  #   # cnn features
  #   fmap = np.load(os.path.join(self.cnn_path, "fmap_pplaces.npz"))['arr_0']
  #   print("{} features from NPZ file".format(fmap.shape))
  #   X = np.append(X, fmap, axis=1)
    
  #   return X




  
  

  
