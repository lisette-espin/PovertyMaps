### distance between antenas: https://www.telco.nsw.gov.au/sites/default/files/4-3-2-001%20Antenna%20Placement%20Guideline%20v1.0.pdf 
### minimum: 1.1m vertical, 1.3m horizontal
### FAQ OpenCelliD: http://wiki.opencellid.org/wiki/FAQ

###########################################################################
# Dependencies
###########################################################################
import os
from tqdm import tqdm
from pqdm.threads import pqdm
from utils import ios
from maps import geo
from utils.validations import delete_nonprojected_variables

###########################################################################
# Constants
###########################################################################

from utils.constants import METERS

###########################################################################
# Functions
###########################################################################

def update_opencellid_features(fn_cells, fn_places, antenna_same_tower_max_distance=5.0, meters=METERS, n_jobs=1):
  ### 1. load cell ids
  print(fn_cells)
  gdf_cells =  geo.load_as_GeoDataFrame(fn=fn_cells, index_col=0, lat='lat', lon='lon', crs=geo.PROJ_DEG)
  
  ### 2. load survey data (DHS) or populated place
  df_places = ios.load_csv(fn_places, index_col=0)
  lat,lon = ('lat','lon') #('LATNUM', 'LONGNUM') if 'LATNUM' in df_places.columns else ('lat','lon')
  gdf_places = geo.get_GeoDataFrame(df=df_places, lat=lat, lon=lon, crs=geo.PROJ_DEG)
  df_places = delete_nonprojected_variables(df_places, os.path.basename(__file__))
  
  ### 3. Projection to meters
  gdf_cells_proj = geo.get_projection(gdf_cells, geo.PROJ_MET)
  gdf_places_proj = geo.get_projection(gdf_places, geo.PROJ_MET)
  all_cells = geo.get_STRtree(gdf_cells_proj)

  ### 4. Identify towers
  gdf_cells_proj = geo.fast_identify_clusters_within_distance(gdf_cells_proj, max_distance=antenna_same_tower_max_distance, n_jobs=n_jobs)
  gdf_cells_proj.rename(columns={'CLUSTER_ID':'tower_id'}, inplace=True)

  ### 4. Computation
  ### - # of cells and towers within cluster area
  ### - distance to closest cell
  ### https://shapely.readthedocs.io/en/stable/manual.html#object.buffer

  # @TODO: use paralellism 
  for index, row in tqdm(gdf_places_proj.iterrows(), total=gdf_places_proj.shape[0]):

    # nearest place
    nearest, distance_nearest = geo.find_nearest_place(all_cells, row.geometry)

    # number of cells and towers
    cells_in_area = {}
    towers_in_area = {}
    for m in meters:
      k = round(m/1000.,2)
      cells_area = geo.find_places_sqm(gdf_proj=gdf_cells_proj, point_proj=row.geometry, width=m)
      cells = cells_area.shape[0]
      towers = cells_area.tower_id.nunique()
      cells_in_area[k] = cells
      towers_in_area[k] = towers

    # update df
    df_places.loc[index,'distance_closest_cell'] = distance_nearest
    for k,v in cells_in_area.items():
      df_places.loc[index,'cells_in_{}km'.format(k)] = v
    for k,v in towers_in_area.items():
      df_places.loc[index,'towers_in_{}km'.format(k)] = v

  return df_places.drop(columns='geometry')