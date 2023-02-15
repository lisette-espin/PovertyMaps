### https://stackoverflow.com/questions/20169467/how-to-convert-from-longitude-and-latitude-to-country-or-city
### A QUICK SOLUTION

import requests
from tqdm import tqdm

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