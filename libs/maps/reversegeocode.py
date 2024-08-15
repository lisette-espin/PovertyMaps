### https://stackoverflow.com/questions/20169467/how-to-convert-from-longitude-and-latitude-to-country-or-city
### A QUICK SOLUTION

import requests
from tqdm import tqdm
import pandas as pd

from shapely.geometry import mapping
from shapely.geometry import shape
from shapely.prepared import prep
from shapely.geometry import Point

class ReverseGeocode(object):

  def __init__(self):
    self.countries = {}

  def load(self, verbose=True):
    data = requests.get("https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson").json()

    for feature in tqdm(data["features"], total=len(data["features"]), disable=not verbose):
      geom = feature["geometry"]
      country = feature["properties"]["ADMIN"]
      self.countries[country] = prep(shape(geom))

  def get_country(self, lon, lat):
    point = Point(lon, lat)
    for country, geom in self.countries.items():
      if geom.contains(point):
        return country
    return "unknown"

  @staticmethod
  def assign_country(fn):

    cols = ['radio', 'mcc', 'net', 'area', 'cell', 'unit', 'lon', 'lat', 'range', 'samples', 'changeable', 'created',
            'updated', 'averageSignal']

    if fn.endswith('.aa'):
      # if it is the first one (with header)
      df = pd.read_csv(fn, low_memory=False, index_col=None)
    else:
      # not the first one (without header)
      df = pd.read_csv(fn, low_memory=False, index_col=None, names=cols)

    # if already has country column, skip
    if df.columns.shape[0] == 15:
      return

    # load country geometries
    rgc = ReverseGeocode()
    rgc.load(verbose=False)

    # assign countries to each cell
    df.loc[:, 'country'] = df.apply(lambda row: rgc.get_country(row.lon, row.lat), axis=1)
    df.to_csv(fn, index=False) # updates the file directly