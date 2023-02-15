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

  np.random.seed(random_state)
  
  # is downsampling?
  if traintype == TTYPE_DOWNSAMPLE:
    df = df.sample(n=DOWNSAMPLE_SIZE,random_state=random_state).copy()
  
  # separating test sample
  if traintype in [TTYPE_ALL,TTYPE_DOWNSAMPLE]:
    frac_train = round(1 - (1/float(kfolds)),2)
    train, test = stratify_sampling(df, catcol, frac_train, random_state, shuffle=True, verbose=False)
    test = test.reset_index(drop=True)
  else:
    train, test = split_by_year(df, traintype)

  # splitting train/val
  train = train.reset_index(drop=True)
  skf = StratifiedKFold(n_splits=kfolds-1, shuffle=True, random_state=random_state)
  X = train.drop(columns=catcol)
  y = train.loc[:,catcol]
  indexes = skf.split(X, y)
  
  # adding info to csv records
  ids = test.cluster_id.values
  df.loc[df.query("cluster_id in @ids").index.values, 'test'] = 1
  for fold, (train_index, val_index) in enumerate(indexes):
    ids = X.loc[train_index,'cluster_id'].values
    df.loc[df.query("cluster_id in @ids").index.values,f'fold{fold+1}'] = 'train'

    ids = X.loc[val_index,'cluster_id'].values
    df.loc[df.query("cluster_id in @ids").index.values,f'fold{fold+1}'] = 'val'

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
  years = df_mapping['cluster_year'].drop_duplicates().sort_values(ascending=True).values # oldest to newest
  index = 0 if traintype==TTYPE_OLDEST else -1
  year = years[index]
  train = df_mapping.query("cluster_year == @year").copy()
  test = df_mapping.query("cluster_year != @year").copy()
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
  gdf_cluster = geo.load_as_GeoDataFrame(fn_cluster, index_col=0, lat=LAT, lon=LON, crs=geo.PROJ_DEG)
  # 2. rural/urban flag
  #gdf_cluster.loc[:,'rural'] = gdf_cluster[RURAL].apply(lambda c: int(c==DHS_RURAL)) # 2022-03-15: moved this to the GT generation
   # 3. projection to meters
  gdf_cluster_m = geo.get_projection(gdf_cluster, geo.PROJ_MET)
  del(gdf_cluster)

  gdf_pplaces_m = None
  if fn_pplaces:
    # 1. load data
    gdf_pplaces = geo.load_as_GeoDataFrame(fn_pplaces, index_col=0, lat=LAT, lon=LON, crs=geo.PROJ_DEG)
    # 2. rural/urban flag
    #gdf_pplaces.loc[:,'rural'] = gdf_pplaces.place.apply(lambda c: int(c in PPLACE_RURAL))  # 2022-03-15: moved this to the pplace extraction
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
    return distances_rc(gdf_cluster_m, gdf_pplaces_m, only_rural=True)
  
  if option == 'ruc':
    ### Rural and Urban changed, closest PPlace (incremental based on radius)
    return distances_rc(gdf_cluster_m, gdf_pplaces_m, only_rural=False)

  #else:
  raise Exception("Option does not exist.")

def distances_none(gdf_cluster_m, gdf_pplaces_m):
  df_results = pd.DataFrame(columns=DISTCOLS)
  df_results.loc[:,'cluster_id'] = gdf_cluster_m[GTID].values
  df_results.loc[:,'cluster_year'] = gdf_cluster_m[YEAR].values
  df_results.loc[:,'cluster_number'] = gdf_cluster_m[CLUSTER].values
  df_results.loc[:,'cluster_rural'] = gdf_cluster_m[RURAL].values
  df_results.loc[:,f'mean_{WEALTH}'] = gdf_cluster_m[f'mean_{WEALTH}'].values
  df_results.loc[:,f'std_{WEALTH}'] = gdf_cluster_m[f'std_{WEALTH}'].values
  df_results.loc[:,'pplace_cluster_distance'] = 0
  df_results.loc[:,'pplace_rural'] = None
  df_results.loc[:,'OSMID'] = None

  return df_results,None



def distances_rc(gdf_cluster_m, gdf_pplaces_m, only_rural=True):
  
  # @TODO: for rural validate that only 1% are between 0 and 10Km
  
  df_results = pd.DataFrame(columns=DISTCOLS)
  dhs_notchanged = []

  for k,v in DISPLACEMENT_M.items():
    print("\n\n{}:{},{}".format(k,LABEL[k],v))

    for y in sorted(gdf_cluster_m[YEAR].unique(), reverse=True):
      print("year:{}".format(y))
      
      gdf_cluster_rural = gdf_cluster_m.query(RURAL+'==@k & '+YEAR+'==@y')
      print('cluster: ', gdf_cluster_m.shape, gdf_cluster_rural.shape)

      if LABEL[k] == 'URBAN' and only_rural:
        print('keep location:',y,k,v,LABEL[k])
        # keep location
        gdf_nearest_rural = gdf_cluster_rural.copy()
        gdf_nearest_rural.loc[:,'OSMID'] = None
        gdf_nearest_rural.loc[:,'pplace_cluster_distance'] = 0
        gdf_nearest_rural.loc[:,'pplace_rural'] = None
        gdf_nearest_rural.rename(columns={GTID:'cluster_id',
                                          YEAR:'cluster_year',
                                          CLUSTER:'cluster_number',
                                          RURAL:'cluster_rural',
                                          #f'mean_{WEALTH}':'cluster_mean',
                                          #f'std_{WEALTH}':'cluster_std'
                                          }, inplace=True)
        
        # concatenate results
        gdf_nearest_rural = gdf_nearest_rural.loc[:,DISTCOLS]
        df_results = df_results.append(gdf_nearest_rural, ignore_index=True)
        print('results (clusters)', df_results.shape)
        
      else:
        # change location
        print('change location:',y,k,v,LABEL[k])
        
        gdf_pplaces_rural = gdf_pplaces_m.query(RURAL+"==@k")
        print('pplaces: ', gdf_pplaces_m.shape, gdf_pplaces_rural.shape)

        # find closest progressively
        tmpcols = ['OSMID','cluster_id','cluster_year','cluster_number','cluster_rural',f"mean_{WEALTH}",f"std_{WEALTH}",'distance']
        tmp_distances = pd.DataFrame(columns=tmpcols)
        
        for i,row in gdf_cluster_rural.iterrows():
            pplaces_in_radius = geo.find_places_sqm(gdf_pplaces_rural, row.geometry, width=v + EXTRA[k]).copy()

            if pplaces_in_radius.shape[0]==0:
              print('no-change',row[GTID])
              dhs_notchanged.append(row[GTID])

              # if no pplace in radius, keep cluster location
              tmp = pd.DataFrame(columns=DISTCOLS)
              tmp.loc[0,'OSMID'] = None
              tmp.loc[0,'pplace_cluster_distance'] = 0
              tmp.loc[0,'pplace_rural'] = None
              tmp.loc[0,'cluster_id'] = row[GTID]
              tmp.loc[0,'cluster_year'] = row[YEAR]
              tmp.loc[0,'cluster_number'] = row[CLUSTER]
              tmp.loc[0,'cluster_rural'] = row[RURAL]
              tmp.loc[0,f'mean_{WEALTH}'] = row[f'mean_{WEALTH}']
              tmp.loc[0,f'std_{WEALTH}'] = row[f'std_{WEALTH}']
              df_results = df_results.append(tmp[DISTCOLS], ignore_index=True)

            else:
              pplaces_in_radius.loc[:,'cluster_id'] = row[GTID]
              pplaces_in_radius.loc[:,'cluster_year'] = row[YEAR]
              pplaces_in_radius.loc[:,'cluster_number'] = row[CLUSTER]
              pplaces_in_radius.loc[:,'cluster_rural'] = row[RURAL]
              pplaces_in_radius.loc[:,f"mean_{WEALTH}"] = row[f"mean_{WEALTH}"]
              pplaces_in_radius.loc[:,f"std_{WEALTH}"] = row[f"std_{WEALTH}"]
              pplaces_in_radius.loc[:,'distance'] = pplaces_in_radius.geometry.distance(row.geometry)
              tmp_distances = tmp_distances.append(pplaces_in_radius[tmpcols], ignore_index=True)

        # from 1 to farest
        tmp = tmp_distances.groupby('cluster_id').size().reset_index().rename(columns={0:'pplaces'}).sort_values('pplaces', ascending=True) 
        
        for i,row in tmp.iterrows():
          # from closest to farest
          targets = tmp_distances.query("cluster_id=='{}'".format(row.cluster_id)).sort_values('distance', ascending=True) 

          counter = 0 
          for j,target in targets.iterrows(): 
            
            if gdf_pplaces_rural.query("{}=={}".format(OSMID,target[OSMID])).shape[0] == 0:
              # there is at least 1 pplace, but already has been taken by another cluster with shorter distance
              continue 

            counter += 1
            gdf_nearest_rural = gdf_pplaces_rural.query("{}=={}".format(OSMID,target[OSMID])).copy()
            gdf_nearest_rural.loc[:,'cluster_id'] = target.cluster_id
            gdf_nearest_rural.loc[:,'cluster_year'] = target.cluster_year
            gdf_nearest_rural.loc[:,'cluster_number'] = target.cluster_number
            gdf_nearest_rural.loc[:,'cluster_rural'] = target.cluster_rural
            gdf_nearest_rural.loc[:,f"mean_{WEALTH}"] = target[f"mean_{WEALTH}"]
            gdf_nearest_rural.loc[:,f"std_{WEALTH}"] = target[f"std_{WEALTH}"]
            gdf_nearest_rural.loc[:,'pplace_cluster_distance'] = target.distance
            gdf_nearest_rural.rename(columns={RURAL:'pplace_rural'}, inplace=True)
            gdf_nearest_rural.drop(columns=[c for c in gdf_nearest_rural.columns if not c.startswith('cluster') 
                                            and not c.startswith('pplace') and not c.startswith(OSMID) and not c.startswith("mean_") 
                                            and not c.startswith("std_")], inplace=True)
            
            # concatenate results
            gdf_nearest_rural = gdf_nearest_rural[DISTCOLS]
            df_results = df_results.append(gdf_nearest_rural, ignore_index=True)
            
            # remove those who already found a match
            gdf_pplaces_rural = gdf_pplaces_rural[~gdf_pplaces_rural[OSMID].isin(df_results[OSMID].values)]
            break
          
          if counter==0 and targets.shape[0]>0:
            # all possible pplaces are already taken by other clusters
            # then add cluster
            tmp = pd.DataFrame(columns=DISTCOLS)
            tmp.loc[0,'OSMID'] = None
            tmp.loc[0,'cluster_id'] = target.cluster_id
            tmp.loc[0,'cluster_year'] = target.cluster_year
            tmp.loc[0,'cluster_number'] = target.cluster_number
            tmp.loc[0,'cluster_rural'] = target.cluster_rural
            tmp.loc[0,f"mean_{WEALTH}"] = target[f"mean_{WEALTH}"]
            tmp.loc[0,f"std_{WEALTH}"] = target[f"std_{WEALTH}"]
            tmp.loc[0,'pplace_cluster_distance'] = 0
            tmp.loc[0,'pplace_rural'] = None
            df_results = df_results.append(tmp[DISTCOLS], ignore_index=True)
            dhs_notchanged.append(target.cluster_id)

        print('results (clusters+pplaces)', df_results.shape)
    
    
  print("[INFO] not changed before validating extras: ",len(list(set(dhs_notchanged))))
  df_results, reverted = validate_extras(df_results) # rural 1, urban 0
  dhs_notchanged = list(set(dhs_notchanged) | set(reverted))
  print("[INFO] not changed after validating extras: ",len(dhs_notchanged))  
  return df_results, dhs_notchanged


def validate_extras(df):
  # Validate mximuns: urban [0,2] and rural [0,5], and 1% rural [0,10]
  print("*************************")
  print("VALIDATING EXTRA DISTANCE")
  print("*************************")
  reverted = []
  for settlement, tmp in df.groupby("cluster_rural"):
    ns = tmp.shape[0]
    dmax = DISPLACEMENT_M[settlement]
    dextra = dmax + EXTRA[settlement]
    label = 'rural' if settlement in [1,'1'] else 'urban'

    nmax = tmp.query("pplace_cluster_distance <= @dmax").shape[0]
    pmax = nmax*100/ns
    print(f"[INFO] {label} ({ns}): d<={dmax}: {nmax} ({pmax:.2f}%)")

    nextra = tmp.query("pplace_cluster_distance > @dmax and pplace_cluster_distance<= @dextra").shape[0]
    pextra = nextra*100/ns
    nextraallowed = int(round(EXTRAS_ALLOWED[settlement]*ns))
    remove = nextra > nextraallowed
    print(f"[INFO] {label} ({ns}): d>{dmax} and d<={dextra}: {nextra} ({nextra/ns:.2f}%) ({nextraallowed} max allowed) -> {'exceeds!' if remove else 'ok'}")

    if remove:
      nremove = nextra-nextraallowed
      print(f"[INFO] {label}: {nextra}-{nextraallowed} = {nremove} to go back to noisy location")
      
      indexes_to_remove = tmp.query("pplace_cluster_distance > @dmax").index.values.copy()
      np.random.shuffle(indexes_to_remove)
      indexes_to_remove = indexes_to_remove[:nremove]
      
      print(f"[INFO] removing ({len(indexes_to_remove)}):")
      print(df.loc[indexes_to_remove,['cluster_id','cluster_rural','OSMID','pplace_cluster_distance']])
      df.loc[indexes_to_remove,'OSMID'] = None
      df.loc[indexes_to_remove,'pplace_cluster_distance'] = 0
      df.loc[indexes_to_remove,'pplace_rural'] = None
      reverted.extend(tmp.loc[indexes_to_remove,'cluster_id'].values.tolist())
  return df, reverted


def groupby(gdf_cluster_m, gdf_pplaces_m, df_valid, option):
  cols = [c for c in DISTCOLS if c not in [f"mean_{WEALTH}",f"std_{WEALTH}"] ]
  df_valid.loc[:,f"mean_{WEALTH}"] = df_valid.loc[:,f"mean_{WEALTH}"].astype(np.float16)
  df_valid.loc[:,f"std_{WEALTH}"] = df_valid.loc[:,f"std_{WEALTH}"].astype(np.float16)
  gdf_cluster_m_new = df_valid.groupby(cols,dropna=False)[[f"mean_{WEALTH}",f"std_{WEALTH}"]].mean().reset_index()

  if option in ['rc','ruc']:
    gdf_cluster_m_new.loc[:,OSMID] = gdf_cluster_m_new.loc[:,OSMID].astype(pd.Int64Dtype())
    gdf_cluster_m_new.loc[:,'pplace_rural'] = gdf_cluster_m_new.loc[:,'pplace_rural'].astype(pd.Int64Dtype())
  
  return gdf_cluster_m_new[DISTCOLS]

def restrict_displacement(df_results):
  queries = []
  for k,v in DISPLACEMENT_M.items():
    d = v+EXTRA[k]
    queries.append(f" (cluster_rural=={k} and pplace_cluster_distance<={d}) ")
    
  query = " or ".join(queries)
  print(query)
  within = df_results.query(query).copy()
  
  print('df_results',df_results.shape)
  print('within',within.shape)
  
  return within

