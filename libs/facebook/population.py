### Resources:
# https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1

### Data: 
# Facebook Connectivity Lab and Center for International Earth Science 
# Information Network - CIESIN - Columbia University. 2016. High Resolution 
# Settlement Layer (HRSL). Source imagery for HRSL © 2016 DigitalGlobe. 
# Accessed 10 MAY 2021.
# https://data.humdata.org/dataset/highresolutionpopulationdensitymaps-uga

###########################################################################
# Dependencies
###########################################################################
import random
import numpy as np
import pandas as pd
from maps import geo
from utils import ios
from tqdm import tqdm 
from utils.validations import delete_nonprojected_variables

###########################################################################
# Constants
###########################################################################

from utils.constants import METERS

###########################################################################
# Functions
###########################################################################

class FacebookPopulation(object):

  def __init__(self, fn_pop=None, fn_places=None):
    self.fn_places = fn_places  # clusters or populated places
    self.fn_pop = fn_pop
    self.gdf_places = None
    self.gdf_pop = None
    self.gdf_places_proj = None
    self.gdf_pop_proj = None
    self.pop2019 = '2019' in fn_pop
    self.lat = 'Lat' if self.pop2019 else 'latitude'
    self.lon = 'Lon' if self.pop2019 else 'longitude'
    self.population = 'Population' if self.pop2019 else fn_pop.split('/')[-1].split('_csv.zip')[0]
    print(f"[INFO] {self.lon}, {self.lat}, {self.population}")

  def load_data(self, crs_from=geo.PROJ_DEG):
    '''
    Loading ground-truth (population data and dhs survey - or populated places).
    By default it assumes that both geometry data are given in degrees (lon,lat)
    '''
    ### population data
    if self.fn_pop is not None:
      self.gdf_pop = geo.load_as_GeoDataFrame(fn=self.fn_pop, index_col=None, lat=self.lat, lon=self.lon, crs=crs_from)

    ### clusters or poppulated places
    if self.fn_places is not None:
      self.gdf_places = geo.load_as_GeoDataFrame(fn=self.fn_places, index_col=0, lat='lat', lon='lon', crs=crs_from)
      # try:
      #   # dhs
      #   self.gdf_places = geo.load_as_GeoDataFrame(fn=self.fn_places, index_col=0, lat='LATNUM', lon='LONGNUM', crs=crs_from)
      # except:
      #   # populated places
      #   self.gdf_places = geo.load_as_GeoDataFrame(fn=self.fn_places, index_col=0, lat='lat', lon='lon', crs=crs_from)

  def project_data(self, crs_to=geo.PROJ_MET):
    '''
    Projects the data: eg. from degrees to metric
    '''
    import os
    import gc
    self.gdf_places_proj = geo.get_projection(self.gdf_places, crs_to)
    self.gdf_pop_proj = geo.get_projection(self.gdf_pop, crs_to)
    self.index_pop = self.gdf_pop_proj.sindex
    # deleting unnecesary variables
    del(self.gdf_pop)
    del(self.gdf_places)
    gc.collect()
    # deleting unnecesary columns
    self.gdf_places_proj = delete_nonprojected_variables(self.gdf_places_proj, os.path.basename(__file__))
    
  def update_population_features(self, meters=METERS):
    '''
    For each survey record (gt-cluster or populated place) it adds all features about population
    from facebook data (high resolution maps).
    '''
    # nearest
    print('Matching nearest data point... (it might take a while)')
    with tqdm(total=1) as pbar:
      # for each place find nearest population_cell
      gdf_nearest, dist = geo.fast_find_nearest_per_record(self.gdf_places_proj, self.gdf_pop_proj)
      pbar.update(1)  
    print(gdf_nearest.head(10))
    print("gdf_nearest:", gdf_nearest.shape, "| dist:", dist.shape)
    
    # within area
    within = []
    for m in tqdm(meters, total=len(meters)):
      km = round(m/1000.,2)
      within.append([self.gdf_pop_proj.iloc[list(self.index_pop.intersection(poly.bounds))].loc[:,self.population].sum() for poly in self.gdf_places_proj.geometry.buffer(m/2., cap_style=3, join_style=1)])

    # results
    results = []
    results.append(self.gdf_places_proj)
    results.append(pd.Series(dist, name='distance_closest_tile'))
    results.append(pd.Series(gdf_nearest.loc[:,self.population], name='population_closest_tile'))
    results.append(pd.Series(gdf_nearest.loc[:,self.population] / dist, name='population_grav_1'))
    results.append(pd.Series(gdf_nearest.loc[:,self.population] / dist**1.5, name='population_grav_1.5'))
    results.append(pd.Series(gdf_nearest.loc[:,self.population] / dist**2, name='population_grav_2'))
    for w,k in zip(*[within,meters]):
      k = round(k/1000.,2)
      results.append(pd.Series(w, name='population_in_{}km'.format(k)))
    
    for i,r in enumerate(results):
      print(i, r.shape)
      
    print(gdf_nearest.head(10))
    print("gdf_nearest:", gdf_nearest.shape, "| dist:", dist.shape)
    
    ignore_index = False
    return pd.concat(results, axis=1, ignore_index=ignore_index).drop(columns='geometry')



###########################################################################
# Some old code
###########################################################################

# from scipy.spatial import cKDTree
# from tqdm import tqdm
# from p_tqdm import p_map
# import functools
# import multiprocessing
# from functools import partial
# from multiprocessing import  Pool


  # # @staticmethod
# def _population_in_tile(row, gdf_pop_proj, places_proj):
#   nearest, distance_nearest = geo.find_nearest_place(places_proj, row.geometry)
#   population =  gdf_pop_proj[~gdf_pop_proj.intersection(nearest).is_empty].Population.values[0]
#   return population, distance_nearest

# # @staticmethod
# def add_population_features(df, gdf_pop_proj, places_proj):

#   # num. of people within nearest tile  
#   df.loc[:,['population_closest_tile','distance_closest_tile']] = df.apply(lambda row:_population_in_tile(row, gdf_pop_proj, places_proj), axis=1)
  
#   # sum of num of people within meter-square area
#   for meters in geo.METERS:
#     k = round(meters/1000.,1)
#     df.loc[:,'population_in_{}km'.format(k)] = df.geometr.apply(lambda g:geo.find_places_sqm(gdf_proj=gdf_pop_proj, point_proj=g, width=meters).Population.sum())
    
#   return df
   

  # def _population_in_tile(self, row):
  #   nearest, distance_nearest = geo.find_nearest_place(self.places_proj, row.geometry)
  #   population =  self.gdf_pop_proj[~self.gdf_pop_proj.intersection(nearest).is_empty].Population.values[0]
  #   return population, distance_nearest

  # def add_population_features(self, df):

  #   # num. of people within nearest tile  
  #   df.loc[:,['population_closest_tile','distance_closest_tile']] = df.apply(lambda row:self._population_in_tile(row), axis=1)
    
  #   # sum of num of people within meter-square area
  #   for meters in geo.METERS:
  #     k = round(meters/1000.,1)
  #     df.loc[:,'population_in_{}km'.format(k)] = df.geometry.apply(lambda g:geo.find_places_sqm(gdf_proj=self.gdf_pop_proj, point_proj=g, width=meters).Population.sum())
      
  #   return df

  # def update_features(self, func, n_cores=4):
  #   df_split = np.array_split(self.df_dhs, 1014)
  #   df = pd.concat(p_map(func, df_split[:1], num_cpus=n_cores))

  #   #df = pd.concat(p_map(partial(func, self.gdf_pop_proj, self.places_proj), df_split[:1], num_cpus=n_cores))

  #   # pool = Pool(n_cores)
  #   # df = pd.concat(pool.map(func, df_split))
  #   # pool.close()
  #   # pool.join()
  #   # return df
  #   return df



# def add_population_features(gdf_dhs_proj, gdf_pop_proj, places_proj):
  
#   df_dhs = pd.DataFrame(gdf_dhs_proj)

#   for index, row in gdf_dhs_proj.iterrows():
#     # num. of people within nearest tile 
#     nearest, distance_nearest = geo.find_nearest_place(places_proj, row.geometry)
#     population =  gdf_pop_proj[~gdf_pop_proj.intersection(nearest).is_empty].Population.values[0]

#     # sum of num of people within meter-square area
#     population_area = {}
#     for meters in METERS:
#       k = round(meters/1000.,1)
#       places_in_area = geo.find_places_sqm(gdf_proj=gdf_pop_proj, point_proj=row.geometry, width=meters)
#       population_area[k] = places_in_area.Population.sum()

#     # update
#     df_dhs.loc[index,'population_closest_tile'] = population
#     df_dhs.loc[index,'distance_closest_tile'] = distance_nearest
#     for k,v in population_area.items():
#       df_dhs.loc[index,'population_in_{}km'.format(k)] = v
  
#   return df_dhs

# def update_population_features_parallel(gdf_dhs_proj, gdf_pop_proj, places_proj, n_cores=5):
  
#   ### 3. Computation
#   ### - number of people within sqr-meters
#   ### - number of people within tile (30x30 m2)
#   pool = multiprocessing.Pool(processes=n_cores)
#   partial_population_features = partial(add_population_features, gdf_pop_proj, places_proj) # prod_x has only one argument x (y is fixed to 10)
#   df = pool.map(partial_population_features, gdf_dhs_proj)

#   # partial_add_population_features = functools.partial(add_population_features, gdf_pop_proj, places_proj)
#   # with Pool(processes=n_cores) as pool:
#   #   res = pool.map(partial_add_population_features, gdf_dhs_proj)

#   return df

# def update_population_features(fn_pop, fn_dhs):

#   ### 1. load data
#   gdf_pop = geo.load_as_GeoDataFrame(fn=fn_pop, index_col=None, lat='Lat', lon='Lon', crs=geo.PROJ_DEG)
#   df_dhs = ios.load_csv(fn_dhs, index_col=0)
#   gdf_dhs = geo.get_GeoDataFrame(df=df_dhs, lat='LATNUM', lon='LONGNUM', crs=geo.PROJ_DEG)

#   ### 2. projection
#   gdf_dhs_proj = geo.get_projection(gdf_dhs, geo.PROJ_MET)
#   gdf_pop_proj = geo.get_projection(gdf_pop, geo.PROJ_MET)
#   places_proj = geo.get_STRtree(gdf_pop_proj)

#   ### 3. Computation
#   ### - number of people within sqr-meters
#   ### - number of people within tile (30x30 m2)
#   for index, row in tqdm(gdf_dhs_proj.iterrows(), total=gdf_dhs_proj.shape[0]):
#     # num. of people within nearest tile 
#     nearest, distance_nearest = geo.find_nearest_place(places_proj, row.geometry)
#     population =  gdf_pop_proj[~gdf_pop_proj.intersection(nearest).is_empty].Population.values[0]

#     # sum of num of people within meter-square area
#     population_area = {}
#     for meters in METERS:
#       k = round(meters/1000.,1)
#       places_in_area = geo.find_places_sqm(gdf_proj=gdf_pop_proj, point_proj=row.geometry, width=meters)
#       population_area[k] = places_in_area.Population.sum()

#     # update
#     df_dhs.loc[index,'population_closest_tile'] = population
#     df_dhs.loc[index,'distance_closest_tile'] = distance_nearest
#     for k,v in population_area.items():
#       df_dhs.loc[index,'population_in_{}km'.format(k)] = v
  
#   return df_dhs.drop(columns='geometry')