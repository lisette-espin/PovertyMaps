import os
import gc
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import multiprocessing
from joblib import delayed
from joblib import Parallel
from functools import partial
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix

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

  def load_movements(self, njobs=1, querystr=None):
    self.gdf_data = load_files_from_folder(path=self.path, njobs=njobs, togeopandas=True, testing=self.testing)
    if querystr is not None:
      self.gdf_data = self.gdf_data.query(querystr).copy()

    self.gdf_data.drop(columns=['start_polygon_name','end_polygon_name','start_polygon_id','end_polygon_id','tile_size','level','n_difference','percent_change','is_statistically_significant','z_score','start_lon','start_lat','end_lat','end_lon','start_quadkey','end_quadkey'], inplace=True)
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
                                                             'n_baseline':lambda x: 1.0 if x.nunique() == 1 and x.unique().min() == 1.0 else x[x>1].mode()[0] if x.mode()[0]==1.0 else x.mode()[0]}).reset_index()
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
    self.D = csr_matrix((distances.ravel(), (source_index.ravel(), target_index.ravel())), shape=(n, n))                 # average traversed distance

###############################################################
# Functions
###############################################################
def load_files_from_folder(path, njobs=1, togeopandas=True, testing=False):
  ## PARALLEL
  files = os.listdir(path)
  files = files[:1] if testing else files
  fnc = partial(load_files_from_zip, togeopandas=togeopandas)
  results = Parallel(n_jobs=njobs)(delayed(fnc)(os.path.join(path,fn)) for fn in files if fn.endswith(".zip"))
  return pd.concat(results)
  ## SERIAL
  # df = None
  # for fn in os.listdir(path):
  #   tmp = load_files_from_zip(os.path.join(path,fn), togeopandas=togeopandas)
  #   if tmp is None:
  #     print(fn)
  #   else:
  #     df = tmp.copy() if df is None else df.append(tmp, ignore_index=True)
  # return df

def load_files_from_zip(fn, togeopandas=False):
  df = pd.DataFrame()
  try:
    with zipfile.ZipFile(fn) as z:
      for filename in z.namelist():
          if not os.path.isdir(filename):
            with z.open(filename, mode='r') as f:
              tmp = pd.read_csv(f)
              # if tmp.query("not geometry.str.startswith('LINE')", engine='python').shape[0] > 0:
              #   ### when no escaping (then correct geometry)
              #   for index,row in tmp.iterrows():
              #     tmp.loc[index,'geometry'] = '{},{}'.format(index,row.geometry)
              #   tmp.reset_index(drop=True, inplace=True)
              df = df.append(tmp, ignore_index=True)
    
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
    print(ex)
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
