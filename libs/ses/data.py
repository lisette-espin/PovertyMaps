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
from utils import validations
from google.staticmaps import StaticMaps

################################################################################
# Constants
################################################################################

from utils.constants import *

################################################################################
# Main
################################################################################

class Data(object):

  def __init__(self, root, years, dhsloc, traintype, epoch=None, rs=None, fold=None, model_name=None, offaug=None, cnn_path=None, isregression=None):
    self.root = root
    self.years = validations.validate_years(years) #years if type(years)==list else years.strip('').replace(' ','').split(',')
    self.dhsloc = dhsloc
    self.traintype = traintype
    self.epoch = epoch
    self.rs = rs
    self.fold = fold
    self.isregression = isregression
    self.prefix = "_".join([pre for pre in ios.get_prefix_surveys(root=root, years=years).split('_') for y in self.years if str(y) in pre])
    # cnn-images
    self.model_name = model_name
    self.offaug = offaug
    self.cnn_path = cnn_path
    self.n_classes = None
    self.path_augmented_images = os.path.join(self.root,'results','staticmaps',"augmented")  
    self.df_evaluation = None  
    # cnn-metadata
    self.df_clusters = None
    self.df_pplaces = None
    validations.validate_traintype(traintype)
    validations.validate_years_traintype(years, traintype)

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
       
      rs = int(path.split("/")[-1].split("-rs")[-1].replace(".csv","")) if self.rs is None else self.rs
      yield path, epoch, rs
    
  def iterate_train_test(self, specific_run=None):
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

      if specific_run is None:
        pass
      elif specific_run != int(epoch):
        continue 
          
      # train / test
      test = df.loc[df.test.dropna().index.values]
      train = df.drop(test.index)
      yield train, test, path, epoch, rs

  def iterate_train_val(self, tune_img=True, specific_run=None, specific_fold=None):
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
      
      if specific_run is None:
        pass
      elif specific_run != int(epoch):
        continue 
        
      # folds
      foldcol = [c.replace('fold','') for c in df.columns if c.startswith("fold")] if self.fold is None else [self.fold]
      for fold in foldcol:
        
        if specific_fold is None:
          pass
        elif specific_fold != int(fold):
          continue
          
        if (tune_img and SESImages.needs_tuning_runid_fold(path, self.model_name, self.offaug, epoch, rs, fold)) or not tune_img:
          tmp = df.loc[df[f'fold{fold}'].dropna().index.values].copy() # to remove test instances
          train = tmp.query(f"fold{fold}=='train'")
          val = tmp.query(f"fold{fold}=='val'")
          yield train, val, path, epoch, rs, int(fold)

  def iterate_train_val_test(self, tune_img=True, specific_run=None):
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
      
      if specific_run is None:
        pass
      elif specific_run != int(epoch):
        continue 
        
      # folds
      test = df.loc[df.test.dropna().index.values]
      foldcol = [c.replace('fold','') for c in df.columns if c.startswith("fold")] if self.fold is None else [self.fold]
      for fold in foldcol:
        
        if (tune_img and SESImages.needs_tuning_runid_fold(path, self.model_name, self.offaug, epoch, rs, fold)) or not tune_img:
          tmp = df.loc[df[f'fold{fold}'].dropna().index.values].copy()
          train = tmp.query(f"fold{fold}=='train'")
          val = tmp.query(f"fold{fold}=='val'")
          yield train, val, test, path, epoch, rs, int(fold)

  def set_nclasses(self, y_attribute, df=None):
    y_attribute = validations.get_valid_output_names(y_attribute)
    if not self.isregression and len(y_attribute) > 1:
      raise Exception("Multiple output classification is not supported yet.")
    if not self.isregression and df is None:
      raise Exception("Classification requires data to infer number of unique classes.")
    self.n_classes = len(y_attribute) if self.isregression else df.loc[:,y_attribute].nunique()
    print("nclasses:", self.n_classes)

  ####################################################################################################################################
  # CNN-IMAGES
  ####################################################################################################################################

  ############################################
  # CLusters
  ############################################

  @staticmethod
  def cnn_get_X(root, df):
    photos = []
    root = os.path.join(root, 'results', 'staticmaps')
    
    for i,row in df.iterrows():
      prefix = f"OSMID{row.OSMID}" 
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
    y_attribute = validations.get_valid_output_names(y_attribute)
    self.set_nclasses(y_attribute, df)

    for i,row in df.iterrows():
      
      prefix = StaticMaps.get_prefix(row)
      path_img = os.path.join(root, 'pplaces' if not pd.isna(row.OSMID) else 'clusters')
      fn = StaticMaps.get_satellite_img_filename(prefix, ZOOM, SCALE, MAPTYPE, IMG_TYPE, path=path_img, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)
      fn = glob.glob(fn)[0]
      
      # load as matrix
      photo = load_img(fn, target_size=(IMG_WIDTH, IMG_HEIGHT))
      # convert to numpy array
      photo = img_to_array(photo, dtype='uint8')
      photo = photo[0:-PIXELS_LOGO,0:-PIXELS_LOGO,:] # removing logo and keeping it squared
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
  
      
  def get_augmented_photos(self, fn, row):
    afn = os.path.basename(fn)
    afn = os.path.join(self.path_augmented_images,afn.replace(f".{IMG_TYPE}",f"-*.{IMG_TYPE}"))
    files = glob.glob(afn)
    images = []
    
    for fn in files:
      photo = load_img(fn, target_size=(IMG_WIDTH-PIXELS_LOGO, IMG_HEIGHT-PIXELS_LOGO))
      photo = img_to_array(photo, dtype='uint8')
      images.append(tf.convert_to_tensor(photo))
    
    return images

  
  ############################################
  # PPlaces
  ############################################
  
  @staticmethod
  def get_pplaces(root, metadata=True):
    # @TODO: check if this code still makes and sense and where is used.    
    # 2022-05-10 This does not make sene, in load_metadata pplaces are loaded with all metadata
    # 2022-10-08 Used to infer wealth on pplaces only

    # pplaces & features
    if metadata:
      df_pplaces = None
      for fn in sorted(glob.glob(os.path.join(root,'results','features','pplaces','PPLACES*.csv'))):
        print(fn)
        tmp = ios.load_csv(fn)
        if fn.endswith("PPLACES.csv"):
          cols = [c for c in tmp.columns if c not in [OSMID,'lon','lat','rural','place','name']]
        tmp.set_index(OSMID,inplace=True)
        df_pplaces = tmp.copy() if df_pplaces is None else df_pplaces.join(tmp, on=OSMID)
      df_pplaces.drop(columns=cols, inplace=True)
    else:
      fn = glob.glob(os.path.join(root,'results','features',"pplaces","PPLACES.csv"))[0]
      df_pplaces = ios.load_csv(fn)
    print('pplaces: ', df_pplaces.shape)
    return df_pplaces
  


  ####################################################################################################################################
  # XGB-METADATA
  ####################################################################################################################################

  def load_metadata(self, viirsnorm=False, dropyear=True):
    path = os.path.join(self.root, 'results', 'features')
    print(path)
    print(self.prefix)

    # clusters
    self.df_clusters = None
    for fn in sorted(glob.glob(os.path.join(path,'clusters',f'*{self.prefix}*.csv'))):
      
      if validations.valid_file_year(fn, self.years):
        print(fn)
        tmp = ios.load_csv(fn)
        if fn.endswith("_cluster.csv"):
          cols = [c for c in tmp.columns if c not in [GTID,'year','mean_wi','std_wi','lon','lat','rural']] #mean_wi and std_wi?
        tmp.set_index(GTID,inplace=True)
        self.df_clusters = tmp.copy() if self.df_clusters is None else self.df_clusters.join(tmp, on=GTID)
    self.df_clusters.drop(columns=cols, inplace=True)
    
    # pplaces
    self.df_pplaces = None
    for fn in sorted(glob.glob(os.path.join(path,'pplaces','PPLACES*.csv'))):
      print(fn)
      tmp = ios.load_csv(fn)
      if fn.endswith("PPLACES.csv"):
        cols = [c for c in tmp.columns if c not in [OSMID,'lon','lat','rural']] # lon, lat, rural?
      tmp.set_index(OSMID,inplace=True)
      self.df_pplaces = tmp.copy() if self.df_pplaces is None else self.df_pplaces.join(tmp, on=OSMID)
    self.df_pplaces.drop(columns=cols, inplace=True)
    
    if viirsnorm:
      self.viirs_validation()
    
    print("GT-YEARS:",self.df_clusters.year.unique())
    if dropyear:
      self.df_clusters.drop(columns='year', inplace=True)
    print(f"CLUSTERS: {self.df_clusters.shape}")
    print(f"PPLACES: {self.df_pplaces.shape}")

  def viirs_validation(self):
    # VIIRS: mean0 to avoid big differences across years
    viirscols = [c for c in self.df_clusters.columns if c.startswith("NTLL")]
    # clusters
    tmp = self.df_clusters.groupby("year")[viirscols].transform(lambda x: (x - x.mean()) / x.std())
    self.df_clusters.loc[tmp.index,tmp.columns] = tmp
    # pplaces
    #viirscols = [c.replace('1.61km','1.6km') for c in viirscols if '1.61km' in c] # deleteme
    tmp = self.df_pplaces[viirscols].transform(lambda x: (x - x.mean()) / x.std())
    self.df_pplaces.loc[tmp.index,tmp.columns] = tmp

    
  @staticmethod
  def is_feature_from_source(column, source):
    if source == SOURCE_ANTENNA:
      # opencellid
      return column.startswith('cells_') or column.startswith('towers_') or column == 'distance_closest_cell'
    if source == SOURCE_FBMARKETING:
      # facebook marketing
      return column.startswith('FBM_')
    if source == SOURCE_NIGHTLIGHT:
      # nightlight intensity
      return column.startswith('NTLL_')
    if source == SOURCE_FBMOVEMENT:
      # facebook movement
      return column.startswith('FBMV_')
    if source == SOURCE_FBPOPULATION:
      # facebook population
      return column.startswith('population_') or column == 'distance_closest_tile'
    if source == SOURCE_OPENSTREETMAP:
      # openstretmap
      return not column.startswith('cells_') and not column.startswith('towers_') and \
                 column != 'distance_closest_cell' and not column.startswith('FBM_') and not \
                 column.startswith('NTLL_') and not column.startswith('FBMV_') and not \
                 column.startswith('population_') and column != 'distance_closest_tile'

    print(f'[ERROR] source: {source}')
    raise Exception("source does not exist")

  @staticmethod
  def get_timevar(df, timevar):
    # @TODO: make this dynzmic

    # if timevar == 'gdp': # gdp current year  
    #   v = df.dhs_year.apply(lambda y: 525.95 if y==2016 else 527.17) 
    # if timevar == 'gdpp': # gdp previous year
    #   v = df.dhs_year.apply(lambda y: 592.9 if y==2016 else 533.97) 

    if timevar == 'deltatime': # delta year
      v = df.cluster_year.apply(lambda y: 2021-y)

    # GDP (current US $)
    if timevar == 'gdp': # gdp current year  
      v = df.cluster_year.apply(lambda y: 3.675 if y==2016 else 4.077) 
    if timevar == 'gdpp': # gdp previous year
      v = df.cluster_year.apply(lambda y: 4.219 if y==2016 else 4.085) 
    
    # GDP per capita growth (anual %)
    if timevar == 'gdppg': # growth gdp current year  
      v = df.cluster_year.apply(lambda y: 3.784 if y==2016 else 3.058) 
    if timevar == 'gdppgp': # growth gdp previous year  
      v = df.cluster_year.apply(lambda y: -22.312 if y==2016 else 1.277) 

    # GDP growth (anual %)
    if timevar == 'gdpg': # growth gdp current year  
      v = df.cluster_year.apply(lambda y: 6.055 if y==2016 else 5.254) 
    if timevar == 'gdpgp': # growth gdp previous year  
      v = df.cluster_year.apply(lambda y: -20.599 if y==2016 else 3.465) 

    # GNI per capita, Atlas method (current US$)
    if timevar == 'gni': # gdp current year  
      v = df.cluster_year.apply(lambda y: 490 if y==2016 else 530) 
    if timevar == 'gnip': # gdp previous year
      v = df.cluster_year.apply(lambda y: 550 if y==2016 else 490)

    return v

  @staticmethod
  def metadata_get_X(root, df, feature_names, features_source='all', fmaps=None):
    ## 2022-10-07 only used by pplace inference
    
    df_data = df.copy()
    feature_names = feature_names if features_source=='all' else sorted([c for c in feature_names if Data.is_feature_from_source(c,features_source)])
    
    df_data = df_data.loc[:,feature_names]
    df_data = df_data.fillna(0)
    X = df_data.loc[:,feature_names]
    
    if fmaps is not None:
      print('original shape:', X.shape)
      print('fmap shape:',fmaps.shape)
      X = np.append(X, fmaps, axis=1)
      print('new shape:', X.shape)
      feature_names.extend([f'cnn{i}' for i in np.arange(fmaps.shape[1])])
      print(f'{len(feature_names)} total features.')
      
    return X
  

  def validate_feature_names(self, features_source='all'):
    # all features
    self.feature_names = sorted([c for c in self.df_clusters.columns if c not in ['mean_wi','std_wi','lon','lat']])
    print("-----------------------------")
    print("- All features")
    print(self.feature_names)
    print("-----------------------------")
    
    # subset of features
    if features_source != 'all':
      self.feature_names = sorted([c for c in self.feature_names if Data.is_feature_from_source(c,features_source)])
      print(f"Only {features_source}: {self.feature_names}")
      
    print(f"{len(self.feature_names)} metadata features.")
    
  def metadata_get_X_y(self, df, y_attribute, fmaps, offlineaug=False, features_source='all', timevar=None):
    df_data = df.copy()
    df_data.set_index("cluster_id", inplace=True)
    osmids = df_data.OSMID.dropna()
    y_attribute = validations.get_valid_output_names(y_attribute)
    
    # validate features (Xs)
    self.validate_feature_names(ALL_FEATURES)
    
    # join GT data (sample instances) with features
    df_data = df_data.join(self.df_clusters.loc[:,self.feature_names], how='inner')
    print("df_data.cluster_year: ", df_data.cluster_year.unique())
      
    # update OSMID places
    if osmids.shape[0] > 0:
      df_data.loc[osmids.index,self.feature_names] = self.df_pplaces.loc[osmids,self.feature_names].values
    
    # validate settlement
    if df_data.query("pplace_rural != rural and OSMID not in @NONE").shape[0] > 0:
      raise Exception("[ERROR] metadata_get_X_y | data.py | pplace_rural must be the same as rural.")
    
    # drop unnecesary columns
    df_data.drop(columns=['pplace_rural','cluster_year','cluster_number','pplace_cluster_distance','ses','cluster_rural'], inplace=True)
    #df_data.rename(columns={'cluster_rural':'rural'}, inplace=True)
    #self.feature_names.append("rural")
    #print(f"{len(self.feature_names)} metadata features.")
    
    # validate features (Xs)
    self.validate_feature_names(features_source)
    
    print("-----------------------------")
    print(df_data.head(5))
    print("-----------------------------")
    print(', '.join(df_data.columns.values))
    print("-----------------------------")

    # delta time
    if timevar is not None:
      print("================= TIMEVAR =================")
      df_data.loc[:,timevar] = Data.get_timevar(df_data, timevar)
      self.feature_names.append(timevar)
      print(df_data.head(2))
      print(df_data.loc[:5,[timevar]])

    # offline augmentation (if combined with CNN)
    if offlineaug:
      features = []
      targets = []
      root = os.path.join(self.root, 'results', 'staticmaps')
      print('[INFO] Data will be augmented')
      # N_AUGMENTATIONS
      tmp = ios.load_csv(os.path.join(root, 'augmented', '_summary.csv'))
      print('tmp',tmp.columns)
      print('df_data',df_data.columns)
      
      for ci,row in df_data.iterrows():
        features.extend([row.loc[self.feature_names].values])
        targets.extend([row.loc[y_attribute].astype(np.float32).values]) 
        if tmp.query(f"cluster_id == @ci").iloc[0].augmented:
          features.extend([row.loc[self.feature_names].values]*N_AUGMENTATIONS)
          targets.extend([row.loc[y_attribute].astype(np.float32).values]*N_AUGMENTATIONS) 

      X = np.nan_to_num(np.asarray(features, dtype='float32')) #.round(PRECISION)
      y = np.nan_to_num(np.asarray(targets, dtype='float32')) #.round(PRECISION)
      
    else:
      # get X and y without augmented (image) features
      df_data = df_data.fillna(0)
      y = df_data.loc[:,y_attribute].values #.round(PRECISION)
      X = df_data.loc[:,self.feature_names].values #.values.round(PRECISION)
    
    # feature maps (if combined with CNN)
    if fmaps is not None:
      print('original shapes:', X.shape, y.shape)
      print('fmap shape:',fmaps.shape)
      X = np.append(X, fmaps, axis=1)
      print('new shapes:', X.shape, y.shape)
      self.feature_names.extend([f'cnn{i}' for i in np.arange(fmaps.shape[1])])
      print(f'{len(self.feature_names)} total features.')
    
    return X, y, self.feature_names
