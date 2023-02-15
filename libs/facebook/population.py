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
      gdf_nearest, dist = geo.fast_find_nearest_per_record(self.gdf_places_proj, self.gdf_pop_proj)
      pbar.update(1)
      
    # within area
    within = []
    for m in tqdm(meters, total=len(meters)):
      km = round(m/1000.,2)
      within.append([self.gdf_pop_proj.iloc[list(self.index_pop.intersection(poly.bounds))].loc[:,self.population].sum() 
                     for poly in self.gdf_places_proj.geometry.buffer(m/2., cap_style=3, join_style=1)])

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

    return pd.concat(results, axis=1).drop(columns='geometry')

