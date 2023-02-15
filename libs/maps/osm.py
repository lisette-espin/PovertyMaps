### Documentation
### OSM data model: https://labs.mapbox.com/mapping/osm-data-model
### Nominatim: https://nominatim.org/release-docs/develop/api/Reverse/
### Overpass: https://github.com/mocnik-science/osm-python-tools/blob/master/docs/overpass.md
### Projection by deafult: Mercator EPSG:3857 = WGS84 (units: meters) https://epsg.io/3857
### OSM: lat,lon
### shapely: lon, lat
### roads vs multiple ways: https://help.openstreetmap.org/questions/63337/getting-complete-roadsstreets-instead-of-multiple-ways-per-road

import os
import time
import urllib
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd 
from functools import partial
from collections import defaultdict

import pyproj
from pyproj import Geod

from shapely import wkt
from shapely import ops
from shapely.ops import transform
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from shapely.geometry import LineString
from shapely.geometry import MultiLineString

from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.overpass import overpassQueryBuilder
from OSMPythonTools.overpass import Overpass
from OSMPythonTools.cachingStrategy import CachingStrategy, JSON
  
from utils import ios
from utils.validations import delete_nonprojected_variables

#############################################################################
# Constants
#############################################################################

from utils.constants import PROJ_DEG
from utils.constants import PROJ_MET 
from utils.constants import MILE_TO_M
from utils.constants import NO
from utils.constants import NONE
from utils.constants import PPLACE_RURAL_BY_TYPE
from utils.constants import LAT
from utils.constants import LON
from utils.constants import RURAL
from utils.constants import OSMID

OSMPT_MIN_WAIT = 1
OSMPT_WAIT_BETWEEN_QUERIES_MAX = 5
OSMPT_TIMEOUT = 30 #360
ZOOM_MAJOR_AND_MINOR_STREETS = 17
ZOOM_BUILDINGS = 18
ZOOM_POIS = 18
SLEEP = 10
MAX_TRIES = 10

TAGS = {'place':['city','town','neighbourhood','village','hamlet','isolated_dwelling'],
        'highway':['primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link', 'motorway'],
        'amenity':['bar','cafe','fast_food','pub','college','kindergarten','library','school','university','bus_station','atm','bank','clinic','dentist','hospital','pharmacy','veterinari','cinema','community_centre','courthouse','embassy','marketplace','police','townhall']
       }
PPL_COLS = [OSMID,LAT,LON,RURAL,'place','loc_name', 'name', 'type', 'population','capital','is_capital','admin_level','GNS:dsg_code']
FEAT_COLS = ['id','total_length_roads','distance_closest_road','num_junctions','distance_closest_junction', 'total_building_area', 'num_buildings'] #lat,lon

for t in TAGS['amenity']:
  FEAT_COLS.append(t)
  FEAT_COLS.append("{}_dist".format(t))

#############################################################################
# Class
#############################################################################
class OSM(object):

  def __init__(self):
    self.ppl = None
    self.features = None
    self.roads = []
    self.junctions = []
    self.buildings = []
    self.pois = []
    self.bbox = []

  def get_populated_places_by_country(self, country, fn, load=False, cache_dir='/mydrive/cache'):
    '''
    Loads into self.ppl all populated places found in a country using OSM.
    Such places must be of any tag within TAGS['place'].

    @param country: country label/name/initials
    @param fn: file path where to store dataframe
    @param cache_dir: directory where to cache OSM data
    @return 
    '''
    from urllib.error import HTTPError
      
    ### if exists and load, then load
    if os.path.exists(fn) and load:
      self.ppl = ios.load_csv(fn, index_col=0)
      return

    ### osm area id
    CachingStrategy.use(JSON, cacheDir=cache_dir)
    nominatim = Nominatim()
    area_id = nominatim.query(country).areaId()

    ### query
    kind = 'place'
    tags = '|'.join(TAGS[kind])
    selector = "'{}'~'{}'".format(kind,tags)
    CachingStrategy.use(JSON, cacheDir=cache_dir)
    timeout = OSMPT_TIMEOUT
    
    while True:
      try:
        ### builder
        wait = np.random.randint(OSMPT_MIN_WAIT, OSMPT_WAIT_BETWEEN_QUERIES_MAX+1)
        overpass = Overpass(waitBetweenQueries=wait)
        query = overpassQueryBuilder(area=area_id, elementType='node', selector=selector, out='body')

        ### retrieve data
        self.ppl = pd.DataFrame(columns=PPL_COLS)
        results = overpass.query(query, timeout=timeout).toJSON()['elements']
      except HTTPError as httpex:
        print(f"[HTTPError] osm.py | get_populated_places_by_country | {httpex}")
        timeout += 30
      except Exception as ex:
        print(f"[ERROR] osm.py | get_populated_places_by_country | {ex}")
        timeout += 30
      else:
        break
    
    for r in tqdm(results):
      obj = r['tags']
      obj[OSMID] = r['id']
      obj[LAT] = r['lat']
      obj[LON] = r['lon']
      obj['type'] = r['type']
      obj[RURAL] = int(r['tags']['place'] in PPLACE_RURAL_BY_TYPE)
      self.ppl = self.ppl.append(pd.DataFrame(obj, index=[0], columns=PPL_COLS), ignore_index=True)

    ### save
    if fn is not None:
      ios.save_csv(self.ppl[PPL_COLS], fn)

  def get_populated_places_id_lat_lon(self):
    return self.ppl[[OSMID,LAT,LON]].to_numpy()

  def get_features(self, places, width=MILE_TO_M, fn=None, overwrite=False, cache_dir='/mydrive/cache'):
    '''
    Loads into self.features all features found within each place in places.
    Such places must be of the form (id, lat, lon).
    
    @param places: tuple (id, lat, lon)
    @param width: width (and thus height) of the bounding box
    @param fn: file path where to store dataframe
    @param cache_dir: directory where to cache OSM data
    @return 
    '''

    ### if exists, load
    if fn is not None and os.path.exists(fn) and not overwrite:
      self.features = ios.load_csv(fn, index_col=0)
      print('loaded')
      return

    if width in NO or width in NONE:
      width = MILE_TO_M
    print(f"Using bounding-box width: {width}")
    
    self.features = pd.DataFrame(columns=FEAT_COLS)
    for id,lat,lon in tqdm(places, total=len(places)):
      
      roads, junctions, d1, d2, d3, d4 = OSM.get_road_features(lat, lon, width, cache_dir)
      buildings, d5, d6 = OSM.get_building_features(lat, lon, width, cache_dir)
      pois, pois_summary = OSM.get_poi_features(lat, lon, width, cache_dir)

      obj = {}
      obj['id'] = id
      obj['total_length_roads'] = d1
      obj['distance_closest_road'] = d2
      obj['num_junctions'] = d3
      obj['distance_closest_junction'] = d4
      obj['total_building_area'] = d5
      obj['num_buildings'] = d6

      for t in TAGS['amenity']:
        obj[t] = pois_summary[t]['count']
        obj["{}_dist".format(t)] = pois_summary[t]['distance']
      
      self.roads.append(roads)
      self.junctions.append(junctions)
      self.buildings.append(buildings)
      self.pois.append(pois)
      self.features = self.features.append(pd.DataFrame(obj, index=[0], columns=FEAT_COLS), ignore_index=True)
      
    ### save
    if fn is not None:
      ios.save_csv(self.features[FEAT_COLS], fn=fn)

  #############################################################################
  # Static methods
  #############################################################################
  
  @staticmethod
  def query_nominatim(lat, lon, reverse, zoom, cache_dir='/mydrive/cache'):
    tries = 0
    sleep = SLEEP

    while tries <= MAX_TRIES:
      try:
        CachingStrategy.use(JSON, cacheDir=cache_dir)
        nominatim = Nominatim()
        q = nominatim.query(lat, lon, reverse=reverse, zoom=zoom).toJSON()
        return q
      except Exception as exception:
        print(exception)
        time.sleep(sleep)
        print("\nRETRY nominatim: {}".format(tries))
      tries+=1
      sleep+=1
      
    return None
      
  @staticmethod
  def get_query_builder_results(bbox, selector, elementType, out, includeGeometry, cache_dir='/mydrive/cache'):
    tries = 0
    sleep = SLEEP

    while tries <= MAX_TRIES:
      try:
        wait = np.random.randint(1,OSMPT_WAIT_BETWEEN_QUERIES_MAX+1)
        CachingStrategy.use(JSON, cacheDir=cache_dir)
        overpass = Overpass(waitBetweenQueries=wait)
        query = overpassQueryBuilder(bbox=bbox, elementType=elementType, selector=selector, out=out, includeGeometry=includeGeometry)
        results = overpass.query(query, timeout=OSMPT_TIMEOUT).toJSON()['elements']
        return results
      except Exception as exception:
        print(exception)
        time.sleep(sleep)
        print("\nRETRY query_builder: {}".format(tries))
      tries+=1
      sleep+=1

    return None

  @staticmethod
  def get_road_features(lat, lon, width=MILE_TO_M, cache_dir='/mydrive/cache'):
    ### osm area id
    bbox = OSM.get_bbox_by_width(lat, lon, width)
    
    ### query results
    kind = 'highway'
    tags = '|'.join(TAGS[kind])
    selector = "'{}'~'{}'".format(kind,tags)
    results = OSM.get_query_builder_results(bbox=bbox, selector=selector, elementType='way', out='body', 
                                            includeGeometry='True', cache_dir=cache_dir)

    ### length of road
    roads = defaultdict(list)
    d1 = 0
    for i, r in enumerate(results):

      k = r['tags']['name']  if 'name' in r['tags'] else 'ID{}'.format(i)
      road = OSM.get_road(r['geometry'])
      roads[k].append(road)
      d1 += OSM.get_length(road)

    geometries = list(itertools.chain(*roads.values()))
    d2 = OSM.get_distance_to_closest_objects(lat,lon,geometries)
    junctions = OSM.get_junctions(geometries)
    if type(junctions) == Point:
      junctions = [junctions]
      d3 = len(junctions)
    else:
      d3 = len(list(junctions.geoms))
    d4 = OSM.get_distance_to_closest_objects(lat,lon,junctions)

    return roads, junctions, d1, d2, d3, d4

  @staticmethod
  def get_building_features(lat, lon, width=MILE_TO_M, cache_dir='/mydrive/cache'):
    ### osm area id
    bbox = OSM.get_bbox_by_width(lat, lon, width)

    ### query
    kind = 'building'
    selector = "'{}'='yes'".format(kind)
    results = OSM.get_query_builder_results(bbox=bbox, selector=selector, elementType='way', out='body', 
                                            includeGeometry='True', cache_dir=cache_dir)

    buildings = []
    d5 = 0
    for r in results:
      poly = OSM.get_polygon(r['geometry'])
      if poly is not None:
        buildings.append(poly)
        d5 += OSM.get_area(poly)
      else:
        print(f"[ERROR] osm.py | get_building_features | lon:{lon}, lat:{lat} | {r}")
    d6 = len(buildings)

    return buildings, d5, d6

  @staticmethod
  def get_poi_features(lat, lon, width=MILE_TO_M, cache_dir='/mydrive/cache'):
    ### osm area id
    bbox = OSM.get_bbox_by_width(lat, lon, width)

    ### query
    kind = 'amenity'
    tags = '|'.join(TAGS[kind])
    selector = "'{}'~'{}'".format(kind,tags)
    results = OSM.get_query_builder_results(bbox=bbox, selector=selector, elementType='node', out='body', 
                                            includeGeometry='True', cache_dir=cache_dir)

    pois = defaultdict(lambda:[])
    for r in results:
      p = Point(r['lon'],r['lat'])
      pois[r['tags']['amenity']].append(p)
      
    ### distances
    pois_summary = defaultdict(lambda:{})
    for a in TAGS['amenity']:
      pois_summary[a] = {'count':len(pois[a]), 'distance':OSM.get_distance_to_closest_objects(lat,lon,pois[a])}

    return pois, pois_summary

  @staticmethod
  def get_road(geometry):
    coordinates = [Point(obj['lon'],obj['lat']) for obj in geometry]
    road = LineString(coordinates)
    return road

  @staticmethod
  def get_polygon(geometry):
    try:
      coordinates = [Point(obj['lon'],obj['lat']) for obj in geometry]
      poly = Polygon(coordinates)
    except Exception as ex:
      print("coordinates:", coordinates)
      print(f"[ERROR] osm.py | get_polygon | {ex}")
      poly = None
    return poly

  @staticmethod
  def get_length(road):
      distance = road.length # in degrees 
      geod = Geod(ellps="WGS84")
      distance = geod.geometry_length(road) # in meters
      return distance

  @staticmethod 
  def get_bbox_by_width(clat, clon, width=MILE_TO_M):
    '''
    Return a bounding box around a centroid (clon,clat)
    '''
    #width = MILE_TO_M
    p = Point(clon, clat)
    project = pyproj.Transformer.from_crs(crs_from=PROJ_DEG, crs_to=PROJ_MET, always_xy=True).transform
    p_proj = transform(project, p)
    bbox_proj = p_proj.buffer(width/2., cap_style=3, join_style=1)

    project = pyproj.Transformer.from_crs(crs_from=PROJ_MET, crs_to=PROJ_DEG, always_xy=True).transform
    bbox = transform(project, bbox_proj)

    minx,miny,maxx,maxy = bbox.bounds  #minx, miny, maxx, maxy
    return miny, minx, maxy, maxx

  @staticmethod
  def get_area(polygon):
    area = polygon.area # in degrees 
    geod = Geod(ellps="WGS84")
    area = geod.geometry_area_perimeter(polygon)[0] # in meters
    return abs(area)

  @staticmethod
  def get_distance_to_closest_objects(lat, lon, objects):
    mindistance = 0
    for i,obj in enumerate(objects.geoms if type(objects)!=list else objects):
      points = nearest_points(Point(lon,lat), obj)
      distance = OSM.get_length(LineString(points))
      mindistance = distance if i==0 or distance < mindistance else mindistance
    return mindistance

  @staticmethod
  def get_junctions(roads):
    junctions = []
    for r1 in np.arange(len(roads)):
      for r2 in np.arange(r1+1, len(roads)):
        j = roads[r1].intersection(roads[r2])
        if not j.is_empty:
          junctions.append(j)
    return unary_union(junctions)

