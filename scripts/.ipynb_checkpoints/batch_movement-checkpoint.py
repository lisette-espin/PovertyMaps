###############################################################################
# Dependencies
###############################################################################
import os
import gc
import sys
import glob
import time
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from collections import Counter
from fast_pagerank import pagerank
from joblib import Parallel, delayed

from maps import geo
from utils import ios
from facebook import movement as mv
from maps.reversegeocode import ReverseGeocode
from facebook.movement import FacebookMovement
from ses.data import delete_nonprojected_variables

###############################################################################
# Constants
###############################################################################

from utils.constants import *

###############################################################################
# Functions
###############################################################################

def run(root, years, njobs=1):
  # survey data
  fn_places = ios.get_places_file(root, years)
  df_places = ios.load_csv(fn_places, index_col=0)

  # params setup
  country = root.split("/")[-1]
  code = COUNTRIES[country]
  path = os.path.join(root, "movement/Facebook")

  # generate movement graph
  mvnet = FacebookMovement(country_name=country, country_code=code, path=path)
  mvnet.load_movements(njobs=njobs)
  mvnet.generate_network(includenan=True)
  print(mvnet.M.shape, mvnet.A.shape, mvnet.D.shape)
  print(mvnet.M.count_nonzero(), mvnet.A.sum())
  print(mvnet.M.sum())
  print(mvnet.D.min(), mvnet.D.max(), mvnet.D.mean())

  # generate features for each cluster
  fn_places_new = fn_places.replace(".csv","_FBMV.csv")
  df_places_new = query(df_places, mvnet)

  # final save
  ios.save_csv(df_places_new, fn_places_new)

def query(df_places, mvnet):
  ### 0. lon,lat columns
  id,lon,lat = ("DHSID","LONGNUM","LATNUM") if 'LATNUM' in df_places.columns else ('OSMID','lon','lat')

  ### 1. geometry lon,lat from survey
  gdf_places = gpd.GeoDataFrame(df_places[[id,lon,lat]], geometry=gpd.points_from_xy(df_places[lon], df_places[lat]), crs=geo.PROJ_DEG)
  del(df_places)

  ### 2. geometry lon,lat places from movement data
  gdf_tiles = gpd.GeoDataFrame()
  gdf_tiles.loc[:,'geometry'] = gpd.GeoSeries.from_wkt(mvnet.nodes, crs=geo.PROJ_DEG)
  gdf_tiles.loc[:,'nodeid'] = np.arange(0,gdf_tiles.shape[0])

  ### 3. Projection to meters
  gdf_tiles_proj = geo.get_projection(gdf_tiles, geo.PROJ_MET)
  gdf_places_proj = geo.get_projection(gdf_places, geo.PROJ_MET)
  del(gdf_places)
  del(gdf_tiles)
  gc.collect()

  ### 4. closest tile per cluster
  gdf_nearest, dist = geo.fast_find_nearest_per_record(gdf_places_proj, gdf_tiles_proj)
  gdf_nearest.loc[:,id] = gdf_places_proj.loc[:,id]
  del(gdf_places_proj)
  del(gdf_tiles_proj)
  gc.collect()

  ### 5. Features:
  # F1. out-degree: distinct places -->
  O = mvnet.A.sum(axis=1).A.ravel()

  # F2. in-degree: distinct places <--
  I = mvnet.A.sum(axis=0).A.ravel()

  # F3. weighted out-degree: (flow) number of people -->
  FO = mvnet.M.sum(axis=1).A.ravel() # row

  # F4. weighted in-degree: (flow) number of people <--
  FI = mvnet.M.sum(axis=0).A.ravel() # col

  # F5. average distance OUT
  DO = mvnet.D.mean(axis=1).A.ravel()

  # F6. averange distance IN
  DI = mvnet.D.mean(axis=0).A.ravel()

  # F7. weighted pagerank
  WPR = pagerank(mvnet.M, p=0.85)
  
  # F8. pagerank
  PR = pagerank(mvnet.A, p=0.85)

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

  gdf_nearest = delete_nonprojected_variables(gdf_nearest, os.path.basename(__file__), True)
  return gdf_nearest



###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Country's main folder.", type=str, required=True)
    parser.add_argument("-y", help="Year or years separated by comma (E.g. 2016,2019).", type=str, required=True)
    parser.add_argument("-j", help="n_jobs for parallel computing (E.g. 10)", type=int, default=1)
    
    args = parser.parse_args()
    for arg in vars(args):
      print("{}: {}".format(arg, getattr(args, arg)))
    
    start_time = time.time()
    run(args.r, args.y, args.j)
    print("--- %s seconds ---" % (time.time() - start_time))