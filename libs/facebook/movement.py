import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

###############################################################################
# Dependencies
###############################################################################

import os
import gc
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import geopandas as gpd
from joblib import delayed
from joblib import Parallel
from functools import partial
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix
from fast_pagerank import pagerank

from maps import geo
from utils import validations

###############################################################################
# Constants
###############################################################################

from utils.constants import *

###############################################################
# Class
###############################################################
class FacebookMovement(object):

  def __init__(self, country_name, country_code, path, testing=False):
    self.country_name = country_name  # country (string) name
    self.country_code = country_code  # code (string) code, eg. SL for Sierra Leone and UG for Uganda
    self.path = path        # folder path that contains all zip files (each file is 1 month)
    self.gdf_data = None    # geodataframe with all data
    self.M = None           # network nxn, cell ij contains the weight wij of edge i-->j
    self.A = None           # network nxn, cell ij contains the edge i-->j (no edge, just 1 = adjacency matrix)
    self.D = None           # distance (length_km) between (traversed) ij
    self.testing = testing  # id True, it will load only a few files (not all)

  def load_movements(self, njobs=1, querystr=None, parallel=True):
    self.gdf_data = load_files_from_folder(path=self.path, njobs=njobs, togeopandas=True, testing=self.testing, parallel=parallel)
    
    if querystr is not None:
      self.gdf_data = self.gdf_data.query(querystr, engine='python').copy()

    self.gdf_data.drop(columns=['start_polygon_name','end_polygon_name','start_polygon_id','end_polygon_id',
                                'tile_size','level','n_difference','percent_change','is_statistically_significant',
                                'z_score','start_lon','start_lat','end_lat','end_lon','start_quadkey','end_quadkey'], inplace=True)
    gc.collect()

  def generate_network(self, includenan=True):
    
    # 1. filter out (or not) records with no n_baseline (mean # of people prior crisis)
    query = []
    if not includenan:
      query.append("not n_baseline.isnull()")
    else:
      self.gdf_data.loc[:,'n_baseline'] = self.gdf_data.n_baseline.apply(lambda c: 1 if c is None or np.isnan(c) else c)

    # 2. country
    if self.country_code is not None:
      query.append("country == '{}'".format(self.country_code))

    if len(query) > 0:
      self.gdf_data = self.gdf_data.query(" and ".join(query), engine='python').copy()

    # 2. aggregate 
    # 2.1 for those cases with multiple unique baselines, use the mode
    #     sum all nbaselines for each edge (Regardles or dow or time)
    # 2.2 unique length_km for each edge
    edges = self.gdf_data[['start','end','dow','time','n_baseline','length_km']]
    edges = edges.groupby(['start','end','dow','time']).agg({'length_km':lambda x:x.unique(),
                                                             'n_baseline':lambda x: 1.0 if x.nunique() == 1 and x.unique().min() == 1.0 else 
                                                             x[x>1].mode()[0] if x.mode()[0]==1.0 else x.mode()[0]}).reset_index()
    edges = edges.groupby(['start','end']).agg({'length_km':lambda x:x.unique(),
                                                'n_baseline':'sum'}).reset_index()
    edges.rename(columns={'start':'source', 'end':'target', 'n_baseline':'weight', 'length_km':'distance'}, inplace=True)

    # 3. nodes
    self.nodes = edges.source.append(edges.target).unique()
    n = self.nodes.shape[0]

    # 4. edge list (source, target, weight)
    edges.loc[:,'source'] = edges.source.apply(lambda c: np.where(self.nodes == c)[0][0])
    edges.loc[:,'target'] = edges.target.apply(lambda c: np.where(self.nodes == c)[0][0])

    # 5. create sparse matrix
    source_index, target_index, distances, weights = np.hsplit(edges.values, 4)
    self.M = csr_matrix((weights.ravel(), (source_index.ravel(), target_index.ravel())), shape=(n, n))                  # number of people
    self.A = csr_matrix((np.ones(self.M.count_nonzero()), (source_index.ravel(), target_index.ravel())), shape=(n, n))  # number of places
    self.D = csr_matrix((distances.ravel(), (source_index.ravel(), target_index.ravel())), shape=(n, n))                # average traversed distance

  def get_features(self, df):
    ### 0. lon,lat columns
    id = validations.get_column_id(df)
    lon,lat = LON,LAT

    ### 1. geometry lon,lat from survey
    gdf = gpd.GeoDataFrame(df[[id,lon,lat]], geometry=gpd.points_from_xy(df[lon], df[lat]), crs=geo.PROJ_DEG)

    ### 2. geometry lon,lat places from movement data
    gdf_tiles = gpd.GeoDataFrame()
    gdf_tiles.loc[:,'geometry'] = gpd.GeoSeries.from_wkt(self.nodes, crs=geo.PROJ_DEG)
    gdf_tiles.loc[:,'nodeid'] = np.arange(0,gdf_tiles.shape[0])

    ### 3. Projection to meters
    gdf_tiles_proj = geo.get_projection(gdf_tiles, geo.PROJ_MET)
    gdf_proj = geo.get_projection(gdf, geo.PROJ_MET)
    del(gdf)
    del(gdf_tiles)
    gc.collect()

    ### 4. closest tile per cluster
    gdf_nearest, dist = geo.fast_find_nearest_per_record(gdf_proj, gdf_tiles_proj)
    gdf_nearest.loc[:,id] = gdf_proj.loc[:,id]
    del(gdf_proj)
    del(gdf_tiles_proj)
    gc.collect()

    ### 5. Features:
    # F1. out-degree: distinct places -->
    O = self.A.sum(axis=1).A.ravel()

    # F2. in-degree: distinct places <--
    I = self.A.sum(axis=0).A.ravel()

    # F3. weighted out-degree: (flow) number of people -->
    FO = self.M.sum(axis=1).A.ravel() # row

    # F4. weighted in-degree: (flow) number of people <--
    FI = self.M.sum(axis=0).A.ravel() # col

    # F5. average distance OUT
    DO = self.D.mean(axis=1).A.ravel()

    # F6. averange distance IN
    DI = self.D.mean(axis=0).A.ravel()

    # F7. weighted pagerank
    WPR = pagerank(self.M, p=0.85)

    # F8. pagerank
    PR = pagerank(self.A, p=0.85)

    ### 6. adding features to final df
    # raw numbers:
    gdf_nearest.loc[:, 'FBMV_OUTdeg'] = gdf_nearest.nodeid.apply(lambda c: O[c])
    gdf_nearest.loc[:, 'FBMV_INdeg'] = gdf_nearest.nodeid.apply(lambda c: I[c])
    gdf_nearest.loc[:, 'FBMV_fOUTdeg'] = gdf_nearest.nodeid.apply(lambda c: FO[c])
    gdf_nearest.loc[:, 'FBMV_fINdeg'] = gdf_nearest.nodeid.apply(lambda c: FI[c])
    gdf_nearest.loc[:, 'FBMV_dOUTdeg'] = gdf_nearest.nodeid.apply(lambda c: DO[c])
    gdf_nearest.loc[:, 'FBMV_dINdeg'] = gdf_nearest.nodeid.apply(lambda c: DI[c])
    gdf_nearest.loc[:, 'FBMV_WPR'] = gdf_nearest.nodeid.apply(lambda c: WPR[c])
    gdf_nearest.loc[:, 'FBMV_PR'] = gdf_nearest.nodeid.apply(lambda c: PR[c])
    gdf_nearest.loc[:, 'FBMV_dist'] = dist

    # gravitational: divided by distance to closest tile
    for exponent in [1,1.5,2]:
      gdf_nearest.loc[:, 'FBMV_OUTdeg_grav_{}'.format(exponent)] = gdf_nearest.apply(lambda row: row.FBMV_OUTdeg / (row.FBMV_dist ** exponent), axis=1)
      gdf_nearest.loc[:, 'FBMV_INdeg_grav_{}'.format(exponent)] = gdf_nearest.apply(lambda row: row.FBMV_INdeg / (row.FBMV_dist ** exponent), axis=1)
      gdf_nearest.loc[:, 'FBMV_fOUTdeg_grav_{}'.format(exponent)] = gdf_nearest.apply(lambda row: row.FBMV_fOUTdeg / (row.FBMV_dist ** exponent), axis=1)
      gdf_nearest.loc[:, 'FBMV_fINdeg_grav_{}'.format(exponent)] = gdf_nearest.apply(lambda row: row.FBMV_fINdeg /(row.FBMV_dist ** exponent), axis=1)
      gdf_nearest.loc[:, 'FBMV_WPR_grav_{}'.format(exponent)] = gdf_nearest.apply(lambda row: row.FBMV_WPR / (row.FBMV_dist ** exponent), axis=1)
      gdf_nearest.loc[:, 'FBMV_PR_grav_{}'.format(exponent)] = gdf_nearest.apply(lambda row: row.FBMV_PR / (row.FBMV_dist ** exponent), axis=1)

    gdf_nearest = validations.delete_nonprojected_variables(gdf_nearest, os.path.basename(__file__), True)
    return gdf_nearest
  
###############################################################
# Functions
###############################################################
def load_files_from_folder(path, njobs=1, togeopandas=True, testing=False, parallel=True):
  files = os.listdir(path)
  files = files[:2] if testing else files
  print(f"{len(files)} files.")
  
  ## PARALLEL
  if parallel:
    fnc = partial(load_files_from_zip, togeopandas=togeopandas, parallel=True, testing=testing)
    results = Parallel(n_jobs=njobs)(delayed(fnc)(os.path.join(path,fn)) for fn in files if fn.endswith(".zip"))
    return pd.concat(results)
  ## SERIAL
  else:
    df = None
    for fn in files:
      tmp = load_files_from_zip(os.path.join(path,fn), togeopandas=togeopandas, parallel=False, testing=testing)
      if tmp is None:
        print(fn, 'is None')
      else:
        df = tmp.copy() if df is None else pd.concat([df,tmp], ignore_index=True) # df.append(tmp, ignore_index=True)
    return df

def load_files_from_zip(fn, togeopandas=False, parallel=True, testing=False):
  df = pd.DataFrame()
  try:
    with zipfile.ZipFile(fn) as z:
      its = z.namelist()
      its = its[:2] if testing else its
      its =  its if parallel else tqdm(its)
      for filename in its:
        if not os.path.isdir(filename):
          with z.open(filename, mode='r') as f:
            tmp = pd.read_csv(f)
            if 'GEOMETRY' in tmp:
              tmp.rename(columns={'GEOMETRY':'geometry'}, inplace=True)
              
            # if tmp.query("not geometry.str.startswith('LINE')", engine='python').shape[0] > 0:
            #   ### when no escaping (then correct geometry)
            #   for index,row in tmp.iterrows():
            #     tmp.loc[index,'geometry'] = '{},{}'.format(index,row.geometry)
            #   tmp.reset_index(drop=True, inplace=True)
              
            #df = df.append(tmp, ignore_index=True)
            df = pd.concat([df,tmp], ignore_index=True)
  except Exception as ex:
    print(f"[ERROR] movement.py | load_files_from_zip #1 | {ex} | fn:{fn}")
    df = None
  
  try:
    # disentangle date/time (faster to handle)
    df.loc[:,'year'] = df.date_time.apply(lambda c: int(c.split("-")[0]))
    df.loc[:,'month'] = df.date_time.apply(lambda c: int(c.split("-")[1]))
    df.loc[:,'date'] = df.date_time.apply(lambda c: c.split(" ")[0])
    df.loc[:,'time'] = df.date_time.apply(lambda c: c.split(" ")[1])
    df.loc[:,'dow'] = pd.to_datetime(df.loc[:,'date']).dt.day_name()

    # disentangle start and end from geometry  
    df.loc[:,'start'] = df.geometry.apply(lambda c: None if 'LINE' not in c else "POINT ({})".format(  " ".join([str(round(float(f),5)) for f in c.split(", ")[0].replace("LINESTRING (","").split(' ')]) )) #lon,lat
    df.loc[:,'end'] = df.geometry.apply(lambda c: None if 'LINE' not in c else "POINT ({})".format( " ".join([str(round(float(f),5)) for f in c.split(", ")[1].replace(")","").split(" ")]) )) #lon,lat

    if togeopandas:
      df = df.query("geometry.str.startswith('LINE')", engine='python').copy()
      df = gpd.GeoDataFrame(df)
      df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
      df['geo_start'] = gpd.GeoSeries.from_wkt(df['start'])
      df['geo_end'] = gpd.GeoSeries.from_wkt(df['end'])

  except Exception as ex:
    print(f"[ERROR] movement.py | load_files_from_zip #2| {ex} | fn:{fn}")
    df = None
    
  return df

def get_stats(df, groupby=['year']):

  def stats(x):
    d = {}
    d['all_locations'] = x.geo_start.append(x.geo_end, ignore_index=True).nunique()
    d['all_movements'] = x.shape[0]
    d['all_mv_within'] = x.query("start_polygon_id == end_polygon_id").shape[0]
    d['all_mv_across'] = x.query("start_polygon_id != end_polygon_id").shape[0]
    d['all_mv_with_nbaseline'] = x.query("not n_baseline.isnull()", engine='python').shape[0]
    d['all_mv_min_nbaseline'] = x.n_baseline.min()
    d['all_mv_max_nbaseline'] = x.n_baseline.max()
    d['all_mv_sum_nbaseline'] = x.n_baseline.sum()
    d['all_mv_with_no_nbaseline'] = x.n_baseline.isna().sum()
    return pd.Series(d, index=['all_locations','all_movements','all_mv_within','all_mv_across','all_mv_with_nbaseline','all_mv_with_no_nbaseline', 'all_mv_min_nbaseline','all_mv_max_nbaseline','all_mv_sum_nbaseline'])

  return df.groupby(groupby).apply(stats)
