###############################################################################
# Dependencies
###############################################################################
import gc
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import StratifiedKFold

from maps import geo
from utils import ios

###############################################################################
# Constants
###############################################################################

from utils.constants import *

###############################################################################
# Functions: Sampling
###############################################################################

def split_kfold_stratified(df_mapping, catcol, dhsloc, traintype, kfolds, random_state):
  
  df = df_mapping.copy()

  if random_state is None:
    random_state = np.random.randint(0,2**32 - 1,1)[0]

  # separating test sample
  if traintype == TTYPE_ALL:
    frac_train = round(1 - (1/float(kfolds)),2)
    train, test = stratify_sampling(df, 'ses', frac_train, random_state, shuffle=True, verbose=False)
    test = test.reset_index(drop=True)
  else:
    train, test = split_by_year(df, traintype)

  # splitting train/val
  train = train.reset_index(drop=True)
  skf = StratifiedKFold(n_splits=kfolds-1, shuffle=True, random_state=random_state)
  X = train.drop(columns='ses')
  y = train.loc[:,'ses']
  indexes = skf.split(X, y)
  
  # adding info to csv records
  ids = test.dhs_id.values
  df.loc[df.query("dhs_id in @ids").index.values, 'test'] = 1
  for fold, (train_index, val_index) in enumerate(indexes):
    ids = X.loc[train_index,'dhs_id'].values
    df.loc[df.query("dhs_id in @ids").index.values,f'fold{fold+1}'] = 'train'

    ids = X.loc[val_index,'dhs_id'].values
    df.loc[df.query("dhs_id in @ids").index.values,f'fold{fold+1}'] = 'val'

  return df, random_state

def stratify_sampling(df_mapping, categorical_column, frac_train, random_state, shuffle=False, verbose=True):
  df = df_mapping.copy()
  if shuffle:
    df = df.sample(frac=1,random_state=random_state) #.reset_index(drop=True)
  size_train = int(round(df.shape[0] * frac_train))
  train = df.groupby(categorical_column, group_keys=False).apply(lambda x: x.sample(int(np.trunc(size_train*len(x)/len(df))), random_state=random_state)).sample(frac=1) #.reset_index(drop=True)
  test = df.drop(train.index)
  if verbose:
    print(train[categorical_column].value_counts())
    print(test[categorical_column].value_counts())
  return train, test

def split_by_year(df_mapping, traintype):
  # newest, oldest
  years = df_mapping.dhs_year.drop_duplicates().sort_values(ascending=True).values # oldest to newest
  index = 0 if traintype==TTYPE_OLDEST else -1
  year = years[index]
  train = df_mapping.query("dhs_year == @year").copy()
  test = df_mapping.query("dhs_year != @year").copy()
  return train, test

###############################################################################
# Functions: Change location
###############################################################################

def validate(option):
  option = option.lower()
  if option not in DHSLOC_OPTIONS.keys():
    raise Exception("Option or mode to change locations does not exist.") 
  return option
  
def get_data(fn_cluster, fn_pplaces=None):
  # 1. load data
  gdf_cluster = geo.load_as_GeoDataFrame(fn_cluster, index_col=0, lat='LATNUM', lon='LONGNUM', crs=geo.PROJ_DEG)
  # 2. rural/urban flag
  gdf_cluster.loc[:,'rural'] = gdf_cluster.URBAN_RURA.apply(lambda c: int(c==DHS_RURAL))
   # 3. projection to meters
  gdf_cluster_m = geo.get_projection(gdf_cluster, geo.PROJ_MET)
  del(gdf_cluster)

  gdf_pplaces_m = None
  if fn_pplaces:
    # 1. load data
    gdf_pplaces = geo.load_as_GeoDataFrame(fn_pplaces, index_col=0, lat='lat', lon='lon', crs=geo.PROJ_DEG)
    # 2. rural/urban flag
    gdf_pplaces.loc[:,'rural'] = gdf_pplaces.place.apply(lambda c: int(c in PPLACE_RURAL))
    # 3. projection to meters
    gdf_pplaces_m = geo.get_projection(gdf_pplaces, geo.PROJ_MET)
    del(gdf_pplaces)

  gc.collect()
  return gdf_cluster_m, gdf_pplaces_m

def distances(gdf_cluster_m, gdf_pplaces_m, option):
  print("=============================")
  print("{}: {}".format(option, DHSLOC_OPTIONS[option]))
  print("=============================")
  
  ### First: distances
  if option == 'none':
    ### Original
    return distances_none(gdf_cluster_m, gdf_pplaces_m)
  
  if option == 'cc':
    ### Closest PPlace (iterative - no duplicates)
    return distances_cc(gdf_cluster_m, gdf_pplaces_m)

  if option == 'ccur':
    ### Closest PPlace Urban/Rural (iterative - no duplicates)
    return distances_ccur(gdf_cluster_m, gdf_pplaces_m)
  
  if option == 'gc':
    ### Group Clusters same PPlace
    return distances_gc(gdf_cluster_m, gdf_pplaces_m)

  if option == 'gcur':
    ### Group Clusters same PPlace Urban/Rural
    return distances_gcur(gdf_cluster_m, gdf_pplaces_m)

  if option == 'rc':
    ### Urban no change. Rural, closest PPlace (incremental based on radius)
    return distances_rc(gdf_cluster_m, gdf_pplaces_m)

  #else:
  raise Exception("Option does not exist.")

def distances_none(gdf_cluster_m, gdf_pplaces_m):
  df_results = pd.DataFrame(columns=DISTCOLS)
  df_results.loc[:,'dhs_id'] = gdf_cluster_m.DHSID.values
  df_results.loc[:,'dhs_year'] = gdf_cluster_m.DHSYEAR.values
  df_results.loc[:,'dhs_cluster'] = gdf_cluster_m.DHSCLUST.values
  df_results.loc[:,'dhs_rural'] = gdf_cluster_m.rural.values
  df_results.loc[:,'dhs_mean_iwi'] = gdf_cluster_m.mean_iwi.values
  df_results.loc[:,'dhs_std_iwi'] = gdf_cluster_m.std_iwi.values
  df_results.loc[:,'pplace_dhs_distance'] = 0
  df_results.loc[:,'pplace_rural'] = None
  df_results.loc[:,'OSMID'] = None

  return df_results,None

def distances_cc(gdf_cluster_m, gdf_pplaces_m):
  df_results = pd.DataFrame(columns=DISTCOLS)

  for k,v in DISPLACEMENT_M.items():
    print("\n\n{}:{}".format(k,'RURAL' if k else 'URBAN'))

    for y in sorted(gdf_cluster_m.DHSYEAR.unique(), reverse=True):
      print("year:{}".format(y))

      gdf_cluster_rural = gdf_cluster_m.query('rural==@k & DHSYEAR==@y')
      gdf_pplaces_rural = gdf_pplaces_m.copy()
      
      print('cluster: ', gdf_cluster_m.shape, gdf_cluster_rural.shape)
      print('pplaces: ', gdf_pplaces_m.shape, gdf_pplaces_rural.shape)
        
      while True: 
        
        # remove those who already found a match
        if df_results.shape[0] > 0:
          gdf_cluster_rural = gdf_cluster_rural[~gdf_cluster_rural.DHSID.isin(df_results.dhs_id.values)]
          gdf_pplaces_rural = gdf_pplaces_rural[~gdf_pplaces_rural.OSMID.isin(df_results.OSMID.values)]
        
        if gdf_cluster_rural.shape[0]==0 or gdf_pplaces_rural.shape[0]==0:
          break

        gdf_nearest_rural, dist_rural = geo.fast_find_nearest_per_record(gdf_cluster_rural,gdf_pplaces_rural) #original_index (index from gdf_pplaces_rural)
        gdf_nearest_rural.loc[:,'dhs_id'] = gdf_cluster_rural.DHSID.values
        gdf_nearest_rural.loc[:,'dhs_year'] = gdf_cluster_rural.DHSYEAR.values
        gdf_nearest_rural.loc[:,'dhs_cluster'] = gdf_cluster_rural.DHSCLUST.values
        gdf_nearest_rural.loc[:,'dhs_rural'] = gdf_cluster_rural.rural.values
        gdf_nearest_rural.loc[:,'dhs_mean_iwi'] = gdf_cluster_rural.mean_iwi.values
        gdf_nearest_rural.loc[:,'dhs_std_iwi'] = gdf_cluster_rural.std_iwi.values
        gdf_nearest_rural.loc[:,'pplace_dhs_distance'] = dist_rural
        gdf_nearest_rural.drop(columns=['lat','lon','loc_name','name','type','place','population','capital','is_capital','admin_level','GNS:dsg_code','original_index'], inplace=True)
        gdf_nearest_rural.rename(columns={'rural':'pplace_rural'}, inplace=True)

        gdf_nearest_rural = gdf_nearest_rural[DISTCOLS]
        df_results = df_results.append(gdf_nearest_rural.sort_values('pplace_dhs_distance').drop_duplicates('OSMID',keep='first'), ignore_index=True)
        print('results (pplaces)', df_results.shape)
  
  return df_results,None

def distances_ccur(gdf_cluster_m, gdf_pplaces_m):
  df_results = pd.DataFrame(columns=DISTCOLS)

  for k,v in DISPLACEMENT_M.items():
    print("\n\n{}:{}".format(k,'RURAL' if k else 'URBAN'))

    for y in sorted(gdf_cluster_m.DHSYEAR.unique(), reverse=True):
      print("year:{}".format(y))

      gdf_cluster_rural = gdf_cluster_m.query('rural==@k & DHSYEAR==@y')
      gdf_pplaces_rural = gdf_pplaces_m.query("rural==@k")
      
      print('cluster: ', gdf_cluster_m.shape, gdf_cluster_rural.shape)
      print('pplaces: ', gdf_pplaces_m.shape, gdf_pplaces_rural.shape)
        
      while True: 
        
        # remove those who already found a match
        if df_results.shape[0] > 0:
          gdf_cluster_rural = gdf_cluster_rural[~gdf_cluster_rural.DHSID.isin(df_results.dhs_id.values)]
          gdf_pplaces_rural = gdf_pplaces_rural[~gdf_pplaces_rural.OSMID.isin(df_results.OSMID.values)]
        
        if gdf_cluster_rural.shape[0]==0 or gdf_pplaces_rural.shape[0]==0:
          break

        gdf_nearest_rural, dist_rural = geo.fast_find_nearest_per_record(gdf_cluster_rural,gdf_pplaces_rural) #original_index (index from gdf_pplaces_rural)
        gdf_nearest_rural.loc[:,'dhs_id'] = gdf_cluster_rural.DHSID.values
        gdf_nearest_rural.loc[:,'dhs_year'] = gdf_cluster_rural.DHSYEAR.values
        gdf_nearest_rural.loc[:,'dhs_cluster'] = gdf_cluster_rural.DHSCLUST.values
        gdf_nearest_rural.loc[:,'dhs_rural'] = gdf_cluster_rural.rural.values
        gdf_nearest_rural.loc[:,'dhs_mean_iwi'] = gdf_cluster_rural.mean_iwi.values
        gdf_nearest_rural.loc[:,'dhs_std_iwi'] = gdf_cluster_rural.std_iwi.values
        gdf_nearest_rural.loc[:,'pplace_dhs_distance'] = dist_rural
        gdf_nearest_rural.drop(columns=['lat','lon','loc_name','name','type','place','population','capital','is_capital','admin_level','GNS:dsg_code','original_index'], inplace=True)
        gdf_nearest_rural.rename(columns={'rural':'pplace_rural'}, inplace=True)

        gdf_nearest_rural = gdf_nearest_rural[DISTCOLS]
        df_results = df_results.append(gdf_nearest_rural.sort_values('pplace_dhs_distance').drop_duplicates('OSMID',keep='first'), ignore_index=True)
        print('results (pplaces)', df_results.shape)
  
  return df_results,None

def distances_gc(gdf_cluster_m, gdf_pplaces_m):
  df_results = pd.DataFrame(columns=DISTCOLS)

  for k,v in DISPLACEMENT_M.items():
    print("\n\n{}:{}".format(k,LABEL[k]))

    for y in sorted(gdf_cluster_m.DHSYEAR.unique(), reverse=True):
      print("year:{}".format(y))
      
      gdf_cluster_rural = gdf_cluster_m.query('rural==@k & DHSYEAR==@y').copy()
      gdf_pplaces_rural = gdf_pplaces_m.copy()
    
      # remove those who already found a match
      if df_results.shape[0] > 0:
        gdf_pplaces_rural = gdf_pplaces_rural[~gdf_pplaces_rural.OSMID.isin(df_results.OSMID.values)]
    
      print('cluster: ', gdf_cluster_m.shape, gdf_cluster_rural.shape)
      print('pplaces: ', gdf_pplaces_m.shape, gdf_pplaces_rural.shape)
      
      gdf_nearest_rural, dist_rural = geo.fast_find_nearest_per_record(gdf_cluster_rural,gdf_pplaces_rural) #original_index (index from gdf_pplaces_rural)
      gdf_nearest_rural.loc[:,'dhs_id'] = gdf_cluster_rural.DHSID.values
      gdf_nearest_rural.loc[:,'dhs_year'] = gdf_cluster_rural.DHSYEAR.values
      gdf_nearest_rural.loc[:,'dhs_cluster'] = gdf_cluster_rural.DHSCLUST.values
      gdf_nearest_rural.loc[:,'dhs_rural'] = gdf_cluster_rural.rural.values
      gdf_nearest_rural.loc[:,'dhs_mean_iwi'] = gdf_cluster_rural.mean_iwi.values
      gdf_nearest_rural.loc[:,'dhs_std_iwi'] = gdf_cluster_rural.std_iwi.values
      gdf_nearest_rural.loc[:,'pplace_dhs_distance'] = dist_rural
      gdf_nearest_rural.drop(columns=['lat','lon','loc_name','name','type','place','population','capital','is_capital','admin_level','GNS:dsg_code','original_index'], inplace=True)
      gdf_nearest_rural.rename(columns={'rural':'pplace_rural'}, inplace=True)
      
      gdf_nearest_rural = gdf_nearest_rural[DISTCOLS]
      df_results = df_results.append(gdf_nearest_rural, ignore_index=True)
      print('results (pplaces)', df_results.shape)

  return df_results,None

def distances_gcur(gdf_cluster_m, gdf_pplaces_m):
  df_results = pd.DataFrame(columns=DISTCOLS)

  for k,v in DISPLACEMENT_M.items():
    print("\n\n{}:{}".format(k,LABEL[k]))

    for y in sorted(gdf_cluster_m.DHSYEAR.unique(), reverse=True):
      print("year:{}".format(y))
      
      gdf_cluster_rural = gdf_cluster_m.query('rural==@k & DHSYEAR==@y')
      gdf_pplaces_rural = gdf_pplaces_m.query("rural==@k")
    
      # remove those who already found a match
      if df_results.shape[0] > 0:
        gdf_pplaces_rural = gdf_pplaces_rural[~gdf_pplaces_rural.OSMID.isin(df_results.OSMID.values)]
    
      print('cluster: ', gdf_cluster_m.shape, gdf_cluster_rural.shape)
      print('pplaces: ', gdf_pplaces_m.shape, gdf_pplaces_rural.shape)
      
      gdf_nearest_rural, dist_rural = geo.fast_find_nearest_per_record(gdf_cluster_rural,gdf_pplaces_rural) #original_index (index from gdf_pplaces_rural)
      gdf_nearest_rural.loc[:,'dhs_id'] = gdf_cluster_rural.DHSID.values
      gdf_nearest_rural.loc[:,'dhs_year'] = gdf_cluster_rural.DHSYEAR.values
      gdf_nearest_rural.loc[:,'dhs_cluster'] = gdf_cluster_rural.DHSCLUST.values
      gdf_nearest_rural.loc[:,'dhs_rural'] = gdf_cluster_rural.rural.values
      gdf_nearest_rural.loc[:,'dhs_mean_iwi'] = gdf_cluster_rural.mean_iwi.values
      gdf_nearest_rural.loc[:,'dhs_std_iwi'] = gdf_cluster_rural.std_iwi.values
      gdf_nearest_rural.loc[:,'pplace_dhs_distance'] = dist_rural
      gdf_nearest_rural.drop(columns=['lat','lon','loc_name','name','type','place','population','capital','is_capital','admin_level','GNS:dsg_code','original_index'], inplace=True)
      gdf_nearest_rural.rename(columns={'rural':'pplace_rural'}, inplace=True)
      
      gdf_nearest_rural = gdf_nearest_rural[DISTCOLS]
      df_results = df_results.append(gdf_nearest_rural, ignore_index=True)
      print('results (pplaces)', df_results.shape)

  return df_results,None

def distances_rc(gdf_cluster_m, gdf_pplaces_m):
  df_results = pd.DataFrame(columns=DISTCOLS)
  dhs_notchanged = []

  for k,v in DISPLACEMENT_M.items():
    print("\n\n{}:{},{}".format(k,LABEL[k],v))

    for y in sorted(gdf_cluster_m.DHSYEAR.unique(), reverse=True):
      print("year:{}".format(y))
      
      gdf_cluster_rural = gdf_cluster_m.query('rural==@k & DHSYEAR==@y')
      print('cluster: ', gdf_cluster_m.shape, gdf_cluster_rural.shape)

      if LABEL[k] == 'URBAN':
        # keep location
        gdf_nearest_rural = gdf_cluster_rural.copy()
        gdf_nearest_rural.loc[:,'OSMID'] = None
        gdf_nearest_rural.loc[:,'pplace_dhs_distance'] = 0
        gdf_nearest_rural.loc[:,'pplace_rural'] = None
        gdf_nearest_rural.rename(columns={'DHSID':'dhs_id',
                                  'DHSYEAR':'dhs_year',
                                  'DHSCLUST':'dhs_cluster',
                                  'rural':'dhs_rural',
                                  'mean_iwi':'dhs_mean_iwi',
                                  'std_iwi':'dhs_std_iwi'}, inplace=True)
        
        # concatenate results
        gdf_nearest_rural = gdf_nearest_rural.loc[:,DISTCOLS]
        df_results = df_results.append(gdf_nearest_rural, ignore_index=True)
        print('results (clusters)', df_results.shape)
        
      else:
        # change location
        gdf_pplaces_rural = gdf_pplaces_m.query("rural==@k")
        print('pplaces: ', gdf_pplaces_m.shape, gdf_pplaces_rural.shape)

        # find closest progressively
        tmpcols = ['OSMID','dhs_id','dhs_year','dhs_cluster','dhs_rural','dhs_mean_iwi','dhs_std_iwi','distance']
        tmp_distances = pd.DataFrame(columns=tmpcols)
        
        for i,row in gdf_cluster_rural.iterrows():
            pplaces_in_radius = geo.find_places_sqm(gdf_pplaces_rural, row.geometry, width=v + EXTRA[k]).copy()

            if pplaces_in_radius.shape[0]==0:
              print('no-change',row.DHSID)
              dhs_notchanged.append(row.DHSID)

              # if no pplace in radius, keep cluster location
              tmp = pd.DataFrame(columns=DISTCOLS)
              tmp.loc[0,'OSMID'] = None
              tmp.loc[0,'pplace_dhs_distance'] = 0
              tmp.loc[0,'pplace_rural'] = None
              tmp.loc[0,'dhs_id'] = row.DHSID
              tmp.loc[0,'dhs_year'] = row.DHSYEAR
              tmp.loc[0,'dhs_cluster'] = row.DHSCLUST
              tmp.loc[0,'dhs_rural'] = row.rural
              tmp.loc[0,'dhs_mean_iwi'] = row.mean_iwi
              tmp.loc[0,'dhs_std_iwi'] = row.std_iwi
              df_results = df_results.append(tmp[DISTCOLS], ignore_index=True)

            else:
              pplaces_in_radius.loc[:,'dhs_id'] = row.DHSID
              pplaces_in_radius.loc[:,'dhs_year'] = row.DHSYEAR
              pplaces_in_radius.loc[:,'dhs_cluster'] = row.DHSCLUST
              pplaces_in_radius.loc[:,'dhs_rural'] = row.rural
              pplaces_in_radius.loc[:,'dhs_mean_iwi'] = row.mean_iwi
              pplaces_in_radius.loc[:,'dhs_std_iwi'] = row.std_iwi
              pplaces_in_radius.loc[:,'distance'] = pplaces_in_radius.geometry.distance(row.geometry)
              tmp_distances = tmp_distances.append(pplaces_in_radius.loc[:,tmpcols], ignore_index=True)
        
        tmp = tmp_distances.groupby('dhs_id').size().reset_index().rename(columns={0:'pplaces'}).sort_values('pplaces', ascending=True) # from 1 to farest
        for i,row in tmp.iterrows():
          targets = tmp_distances.query("dhs_id=='{}'".format(row.dhs_id)).sort_values('distance', ascending=True) # from closest to farest

          counter = 0 
          for j,target in targets.iterrows(): 
            
            if gdf_pplaces_rural.query("OSMID=={}".format(target.OSMID)).shape[0] == 0:
              # there is at least 1 pplace, but already has been taken by another cluster with shorter distance
              continue 

            counter += 1
            gdf_nearest_rural = gdf_pplaces_rural.query("OSMID=={}".format(target.OSMID)).copy()
            gdf_nearest_rural.loc[:,'dhs_id'] = target.dhs_id
            gdf_nearest_rural.loc[:,'dhs_year'] = target.dhs_year
            gdf_nearest_rural.loc[:,'dhs_cluster'] = target.dhs_cluster
            gdf_nearest_rural.loc[:,'dhs_rural'] = target.dhs_rural
            gdf_nearest_rural.loc[:,'dhs_mean_iwi'] = target.dhs_mean_iwi
            gdf_nearest_rural.loc[:,'dhs_std_iwi'] = target.dhs_std_iwi
            gdf_nearest_rural.loc[:,'pplace_dhs_distance'] = target.distance
            gdf_nearest_rural.drop(columns=['lat','lon','loc_name','name','type','place','population','capital','is_capital','admin_level','GNS:dsg_code'], inplace=True)
            gdf_nearest_rural.rename(columns={'rural':'pplace_rural'}, inplace=True)
            
            # concatenate results
            gdf_nearest_rural = gdf_nearest_rural[DISTCOLS]
            df_results = df_results.append(gdf_nearest_rural, ignore_index=True)
            
            # remove those who already found a match
            gdf_pplaces_rural = gdf_pplaces_rural[~gdf_pplaces_rural.OSMID.isin(df_results.OSMID.values)]
            break
          
          if counter==0 and targets.shape[0]>0:
            # all possible pplaces are already taken by other clusters
            # then add cluster
            tmp = pd.DataFrame(columns=DISTCOLS)
            tmp.loc[0,'OSMID'] = None
            tmp.loc[0,'dhs_id'] = target.dhs_id
            tmp.loc[0,'dhs_year'] = target.dhs_year
            tmp.loc[0,'dhs_cluster'] = target.dhs_cluster
            tmp.loc[0,'dhs_rural'] = target.dhs_rural
            tmp.loc[0,'dhs_mean_iwi'] = target.dhs_mean_iwi
            tmp.loc[0,'dhs_std_iwi'] = target.dhs_std_iwi
            tmp.loc[0,'pplace_id'] = None
            tmp.loc[0,'pplace_dhs_distance'] = 0
            tmp.loc[0,'pplace_rural'] = None
            df_results = df_results.append(tmp[DISTCOLS], ignore_index=True)
            dhs_notchanged.append(target.dhs_id)

        print('results (clusters+pplaces)', df_results.shape)
        
  return df_results, dhs_notchanged





def groupby(gdf_cluster_m, gdf_pplaces_m, df_valid, option):
  if option in ['gc','gcur']:
    #gdf_cluster_m_new = df_valid.groupby(['OSMID','dhs_rural','pplace_id','pplace_rural'],dropna=False)[['dhs_mean_iwi','dhs_std_iwi','pplace_dhs_distance']].mean().reset_index()
    gdf_cluster_m_new = df_valid.groupby(['OSMID','dhs_rural','pplace_rural'],dropna=False)[['dhs_mean_iwi','dhs_std_iwi','pplace_dhs_distance']].mean().reset_index()
    gdf_cluster_m_new.loc[:,'dhs_id'] = None
    gdf_cluster_m_new.loc[:,'dhs_rural'] = None
    gdf_cluster_m_new.loc[:,'dhs_year'] = None
    gdf_cluster_m_new.loc[:,'dhs_cluster'] = None
  else:
    cols = [c for c in DISTCOLS if c not in ['dhs_mean_iwi','dhs_std_iwi'] ]
    df_valid.loc[:,'dhs_mean_iwi'] = df_valid.dhs_mean_iwi.astype(np.float16)
    df_valid.loc[:,'dhs_std_iwi'] = df_valid.dhs_std_iwi.astype(np.float16)
    gdf_cluster_m_new = df_valid.groupby(cols,dropna=False)[['dhs_mean_iwi','dhs_std_iwi']].mean().reset_index()
   
    if option in ['rc']:
      gdf_cluster_m_new.loc[:,'OSMID'] = gdf_cluster_m_new.loc[:,'OSMID'].astype(pd.Int64Dtype())
      #gdf_cluster_m_new.loc[:,'pplace_id'] = gdf_cluster_m_new.loc[:,'pplace_id'].astype(pd.Int64Dtype())
      gdf_cluster_m_new.loc[:,'pplace_rural'] = gdf_cluster_m_new.loc[:,'pplace_rural'].astype(pd.Int64Dtype())
  
  # gdf_cluster_m_new = gdf_cluster_m.loc[df_valid.dhs.values,:]
  # gdf_cluster_m_new.loc[:,'PPLACE'] = df_valid.closest_pplace.values
  # gdf_cluster_m_new.loc[:,'OSMID'] = gdf_pplaces_m.loc[df_valid.closest_pplace.values].id.values
  # gdf_cluster_m_new.loc[:,'PPLACE_LAT'] = gdf_pplaces_m.loc[df_valid.closest_pplace.values].lat.values
  # gdf_cluster_m_new.loc[:,'PPLACE_LON'] = gdf_pplaces_m.loc[df_valid.closest_pplace.values].lon.values
  # gdf_cluster_m_new = gdf_cluster_m_new.groupby(['DHSCC','DHSYEAR','URBAN_RURA','rural','PPLACE','PPLACE_OSMID','PPLACE_LAT','PPLACE_LON']).mean_iwi.mean().reset_index()
  return gdf_cluster_m_new[DISTCOLS]

def restrict_displacement(df_results):
  tmp = df_results.query("(dhs_rural==0 & pplace_dhs_distance>{}) or (dhs_rural==1 & pplace_dhs_distance>{})".format(DISPLACEMENT_M[0]+EXTRA[0], 
                                                                                                                     DISPLACEMENT_M[1]+EXTRA[1]))
  # # print(tmp.head())
  # # print(tmp.index)
  # print('tmp',tmp.shape)

  # if tmp.shape[0]>1:
  #   df_results.loc[tmp.index,'pplace_id'] = None
  #   df_results.loc[tmp.index,'pplace_dhs_distance'] = 0
  #   df_results.loc[tmp.index,'pplace_rural'] = None
  #   df_results.loc[tmp.index,'OSMID'] = None

  within = df_results.query("(dhs_rural==0 & pplace_dhs_distance<={}) or (dhs_rural==1 & pplace_dhs_distance<={})".format(DISPLACEMENT_M[0]+EXTRA[0], 
                                                                                                                          DISPLACEMENT_M[1]+EXTRA[1])).copy()
  print('df_results',df_results.shape)
  print('within',within.shape)
  return within



