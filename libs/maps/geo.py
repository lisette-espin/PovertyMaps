###########################################################################
# Dependencies
###########################################################################

import numpy as np
import pandas as pd
from tqdm import tqdm
# from osgeo import ogr
# from osgeo import osr
# from osgeo import gdal
import geopandas as gpd
from pyproj import Transformer
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from shapely.ops import transform
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely.geometry import Polygon
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

###########################################################################
# Constants
###########################################################################

from utils.constants import *

###########################################################################
# Functions
###########################################################################

def convert(geometry, proj_from=PROJ_MET, proj_to=PROJ_DEG):
  '''
  Porjection from-to of geometry
  '''
  transformer = Transformer.from_crs(proj_from, proj_to, always_xy=True)
  return transform(transformer.transform, geometry)


def find_nearest_place(places, point, return_distance=True):
  '''
  Make sure that places and point are in the same crs
  '''
  nearest = places.nearest(point)
  if return_distance:
    distance_nearest = nearest.distance(point)
    return nearest, distance_nearest
  return nearest

def find_places_sqm(gdf_proj, point_proj, width=MILE_TO_M):
  '''
  Make sure gdf and point are in meters (crs=PROJ_MET)
  buffer: https://shapely.readthedocs.io/en/stable/manual.html#object.buffer
  '''
  area = point_proj.buffer(width / 2., cap_style=3, join_style=1) #square cap_style, round join_style
  tmp = gdf_proj["geometry"].intersection(area)
  return gdf_proj[~tmp.is_empty]

def fast_find_nearest_per_record(gdfa, gdfb, dropgeometry=True):
  ### for each a, find closest in b
  pointsa = np.array(list(gdfa.geometry.apply(lambda x: (x.x, x.y))))
  pointsb = np.array(list(gdfb.geometry.apply(lambda x: (x.x, x.y))))
  treea = cKDTree(pointsa)
  treeb = cKDTree(pointsb)

  # nearest
  dist, idx = treeb.query(pointsa, k=1)
  if dropgeometry:
    gdf_nearest = gdfb.iloc[idx].drop(columns="geometry").reset_index(drop=True)
  else:
    gdf_nearest = gdfb.iloc[idx].reset_index(drop=True)
  
  gdf_nearest.loc[:,'original_index'] = idx #ids from b
  return gdf_nearest, dist

def load_as_GeoDataFrame(fn, index_col, lat, lon, crs):
  return get_GeoDataFrame(pd.read_csv(fn, index_col=index_col), lat, lon, crs)

def get_GeoDataFrame(df, lat, lon, crs=PROJ_DEG):
  return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df[lon], y=df[lat]), crs=crs)
  
def get_STRtree(gdf):
  return STRtree(gdf.geometry.values)

def get_cKDTree(gdf):
  return cKDTree(gdf.geometry.values)

def get_projection(gdf, crs=PROJ_MET):
  return gdf.to_crs(crs)

def fast_identify_clusters_within_distance(gdf_proj, max_distance=5.0, n_jobs=1):
  X = np.array([[g.x, g.y] for g in gdf_proj.geometry]) 
  clustering = DBSCAN(eps=max_distance, min_samples=1, algorithm='kd_tree', n_jobs=n_jobs).fit(X)
  gdf_proj.loc[:,'CLUSTER_ID'] = clustering.labels_
  return gdf_proj


# ############################################################################
# ### Adapted from: https://gis.stackexchange.com/questions/57834/how-to-get-raster-corner-coordinates-using-python-gdal-bindings
# ############################################################################

# def GetExtentAndCentroid(gt,cols,rows):
#     ''' Return list of corner coordinates from a geotransform

#         @type gt:   C{tuple/list}
#         @param gt: geotransform
#         @type cols:   C{int}
#         @param cols: number of columns in the dataset
#         @type rows:   C{int}
#         @param rows: number of rows in the dataset
#         @rtype:    C{[float,...,float]}
#         @return:   coordinates of each corner
#     '''
#     ext=[]
#     xarr=[0,cols]
#     yarr=[0,rows]

#     ### 1. get corners in (lat,lon)
#     for px in xarr:
#         for py in yarr:
#             x=gt[0]+(px*gt[1])+(py*gt[2])
#             y=gt[3]+(px*gt[4])+(py*gt[5])
#             ext.append([y,x]) # lat, lon
#         yarr.reverse()
#     ext = np.array(ext)

#     ### 2. get centroid
#     centroid = (ext[:,0].min() + ext[:,0].max()) / 2, (ext[:,1].min() + ext[:,1].max()) / 2
#     return ext, centroid

# def ReprojectCoords(coords,src_srs,tgt_srs):
#     ''' Reproject a list of x,y coordinates.

#         @type geom:     C{tuple/list}
#         @param geom:    List of [[x,y],...[x,y]] coordinates
#         @type src_srs:  C{osr.SpatialReference}
#         @param src_srs: OSR SpatialReference object
#         @type tgt_srs:  C{osr.SpatialReference}
#         @param tgt_srs: OSR SpatialReference object
#         @rtype:         C{tuple/list}
#         @return:        List of transformed [[x,y],...[x,y]] coordinates
#     '''
#     trans_coords=[]
#     transform = osr.CoordinateTransformation( src_srs, tgt_srs)
#     for x,y in coords:
#         x,y,z = transform.TransformPoint(x,y)
#         trans_coords.append([x,y])
#     return trans_coords

# def get_corners_and_centroid(fn_image):
#   ds = gdal.Open(fn_image)
#   gt = ds.GetGeoTransform()
#   cols = ds.RasterXSize
#   rows = ds.RasterYSize
#   ext,centroid = GetExtentAndCentroid(gt,cols,rows)
#   return ext, centroid


# def get_locations(df_geo, geopy_user_agent='app_name_here', copy=False):
    
#     geolocator = Nominatim(user_agent=geopy_user_agent)
#     geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2)

#     df = df_geo if not copy else df_geo.copy()
    
#     for index, row in tqdm(df.iterrows()):
#         location = geolocator.reverse("{}, {}".format(row.lat, row.lon), language='en', exactly_one=True)  
#         # , addressdetails=False, zoom=10)

#         try:
#             address = location.raw['display_name']
#             try:
#                 country = location.raw['address']['country']
#             except:
#                 country = '-'

#             try:
#                 city = location.raw['address']['city']
#             except:
#                 try:
#                     city = location.raw['address']['suburb']
#                 except:
#                     try:
#                         city = location.raw['address']['state']
#                     except:
#                         try:
#                             city = location.raw['address']['county']
#                         except:
#                             try:
#                                 city = location.raw['address']['town']
#                             except:
#                                 try:
#                                     city = location.raw['address']['province']
#                                 except:
#                                     city = '-'

#             try:
#                 postcode = location.raw['address']['postcode']
#             except:
#                 postcode = '-'

#             df.loc[index,'address'] = address
#             df.loc[index,'country'] = country
#             df.loc[index,'city'] = city
#             df.loc[index,'postcode'] = postcode

#         except Exception as ex:
#             print(ex, location.raw)

#     return df