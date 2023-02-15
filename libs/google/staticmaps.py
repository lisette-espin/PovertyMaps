################################################################
# Dependencies: System
################################################################
import os
import glob
import hmac
import base64
import hashlib
import pandas as pd
import urllib.parse as urlparse
from urllib.parse import urlencode
from urllib.request import urlretrieve

################################################################
# Dependencies: Local
################################################################
from utils import ios

################################################################
# Constants
################################################################
from utils.constants import NONE
from utils.constants import IMG_TYPE
from utils.constants import GTID
from utils.constants import OSMID

################################################################
# Documentation (Google Developers)
################################################################    
# Zoom levels: https://developers.google.com/maps/documentation/maps-static/start#Zoomlevels
# API Keys & Secret: https://developers.google.com/maps/documentation/maps-static/get-api-key

################################################################
# Class
################################################################    

class StaticMaps(object):

  URI = "https://maps.googleapis.com/maps/api/staticmap?"

  def __init__(self, key, secret, lat, lon, size, zoom, scale, maptype, img_type=IMG_TYPE):
    self.key = key
    self.secret = secret
    self.lat = lat
    self.lon = lon
    self.img_size = size
    self.zoom = zoom
    self.scale = scale
    self.maptype = maptype
    self.img_type = img_type
    self.url = None

  def get_signed_url(self, url):
      # adapted from: https://github.com/googlemaps/url-signing/blob/gh-pages/urlsigner.py
      if not url or not self.secret:
          raise Exception("Both url and secret are required")

      url = urlparse.urlparse(url)

      # We only need to sign the path+query part of the string
      url_to_sign = url.path + "?" + url.query

      # Decode the private key into its binary format
      # We need to decode the URL-encoded private key
      decoded_key = base64.urlsafe_b64decode(self.secret)

      # Create a signature using the private key and the URL-encoded
      # string using HMAC SHA1. This signature will be binary.
      signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

      # Encode the binary signature into base64 for use within a URL
      encoded_signature = base64.urlsafe_b64encode(signature.digest())

      original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

      # Return signed URL
      return original_url + "&signature=" + encoded_signature.decode()

  def set_url(self):
    params = {"center":"{}, {}".format(self.lat,self.lon),
              "zoom": 18 if self.zoom is None else self.zoom, 
              "size": "416x436" if self.img_size is None else self.img_size,
              "scale":2 if self.scale is None else self.scale,
              "maptype":"satellite" if self.maptype is None else self.maptype,
              "key":self.key}
    input_url = self.URI + urlencode(params)
    self.url = self.get_signed_url(input_url)

  def retrieve_and_save(self,path,prefix=None,verbose=True,load=True):
    self.set_url()
    fn = self.get_image_filename(path, prefix)
    
    if os.path.exists(fn) and load:
      return 0

    if not os.path.exists(fn) and fn not in NONE:
      try:
        urlretrieve(self.url, fn)
        if verbose:
          print("{} saved!".format(fn))
        return 1
      except Exception as ex:
        print(ex)
        
    return -1

  def get_image_filename(self, path, prefix=None):
    # fn = "{}*-ZO{}-SC{}-{}-{}.png".format('' if prefix is None else prefix,
    #                                       self.zoom, self.scale, 
    #                                       self.img_size, self.maptype)
    
    fn = StaticMaps.get_satellite_img_filename(prefix, self.zoom, self.scale, self.maptype, self.img_type, img_size=self.img_size)
    files = glob.glob(os.path.join(path, fn))

    if len(files) > 1:
      raise Exception("More than 1 image for the same cluster and settings.")

    if len(files) == 0:
      fn = StaticMaps.get_satellite_img_filename(prefix, self.zoom, self.scale, self.maptype, 
                                                 self.img_type, path=path, img_size=self.img_size, lat=self.lat, lon=self.lon)
      files = [fn]
      
    return files[0]

  def exists(self, path, prefix=None):
    return ios.exists(self.get_image_filename(path, prefix))
    
  @staticmethod
  def get_prefix(row):
    
    if OSMID in row:
      if not pd.isna(row.OSMID) and row.OSMID not in NONE:
        return 'OSMID{}'.format(row.OSMID)
    
    if GTID in row:
      return 'Y{}-C{}-U{}'.format(int(row.year), int(row.cluster), int(row.rural)) 
    
    if "cluster_id" in row: 
      return 'Y{}-C{}-U{}'.format(int(row.cluster_year), int(row.cluster_number), int(row.cluster_rural)) 
    
    
    
    print(row)
    raise Exception(f"[ERROR] staticmaps.py | get_prefix | prefix could not be built")
    
  @staticmethod
  def get_satellite_img_filename(prefix, zoom, scale, maptype, img_type, path=None, img_size=None, 
                                 img_width=None, img_height=None, lat='*', lon='*'):
    if img_size is None and img_width is None and img_height is None:
      raise Exception("[ERROR] staticmaps.py | get_satellite_img_filename | cannot be None when img_width and img_height are also None.")
      
    lat = round(lat,10) if type(lat) != str else lat
    lon = round(lon,10) if type(lon) != str else lon
    
    fn = "{}LA{}-LO{}-ZO{}-SC{}-{}-{}.{}".format('' if prefix is None else '{}-'.format(prefix),
                                                    lat,lon,
                                                    zoom, 
                                                    scale, 
                                                    img_size if img_size is not None else "{}x{}".format(img_width,img_height), 
                                                    maptype,
                                                    img_type)
    if path is not None:
      fn = os.path.join(path, fn)
    return fn
    