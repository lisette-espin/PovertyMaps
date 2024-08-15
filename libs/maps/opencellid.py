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

def update_opencellid_features_fast(fn_cells, fn_places, antenna_same_tower_max_distance=5.0, meters=METERS, n_jobs=1):
    ### 1. load cell ids
    print(fn_cells)
    gdf_cells =  geo.load_as_GeoDataFrame(fn=fn_cells, index_col=0, lat='lat', lon='lon', crs=geo.PROJ_DEG)

    ### 2. load survey data (DHS) or populated place
    df_places = ios.load_csv(fn_places, index_col=0)
    df_places.drop(columns=[c for c in df_places.columns if c not in ['gtID','OSMID','lon','lat']], inplace=True)
    lat,lon = ('lat','lon') #('LATNUM', 'LONGNUM') if 'LATNUM' in df_places.columns else ('lat','lon')
    gdf_places = geo.get_GeoDataFrame(df=df_places, lat=lat, lon=lon, crs=geo.PROJ_DEG)
    df_places = delete_nonprojected_variables(df_places, os.path.basename(__file__))

    ### 3. Projection to meters
    gdf_cells_proj = geo.get_projection(gdf_cells, geo.PROJ_MET)
    gdf_places_proj = geo.get_projection(gdf_places, geo.PROJ_MET)
    all_cells = geo.get_STRtree(gdf_cells_proj)

    
    ### 4. Identify towers
    gdf_cells_proj = geo.fast_identify_clusters_within_distance(gdf_cells_proj[['geometry']], max_distance=antenna_same_tower_max_distance, n_jobs=n_jobs)
    gdf_cells_proj.rename(columns={'CLUSTER_ID':'tower_id'}, inplace=True)
    index_cell = gdf_cells_proj.sindex
    
    
    ### 4. Computation
    ### - # of cells and towers within cluster area
    ### - distance to closest cell
    ### https://shapely.readthedocs.io/en/stable/manual.html#object.buffer
    gdf = gdf_places_proj.sjoin_nearest(gdf_cells_proj, how="left", distance_col="distance_closest_cell").drop_duplicates('geometry').copy()
    
    print('=======')
    print(gdf_places_proj.shape)
    print(gdf_cells_proj.shape)
    print(gdf.shape)
    print('=======')
    
    # within area
    for m in tqdm(meters, total=len(meters)):
        km = round(m/1000.,2)
        cells = [gdf_cells_proj.iloc[list(index_cell.intersection(poly.bounds))].shape[0] for poly in gdf.geometry.buffer(m/2., cap_style=3, join_style=1)]
        towers = [gdf_cells_proj.iloc[list(index_cell.intersection(poly.bounds))].tower_id.nunique() for poly in gdf.geometry.buffer(m/2., cap_style=3, join_style=1)]
        gdf.loc[:,f'cells_in_{km}km'] = cells
        gdf.loc[:,f'towers_in_{km}km'] = towers
        print(f'm:{m}, cells:{len(cells)}, towers:{len(towers)}, gdf:{gdf.shape}')
        
    return gdf.drop(columns=['geometry','lat','lon','index_right','tower_id'])


# ###########################################################################
# # NEW Functions (too slow anyway)
# ###########################################################################

# import pandas as pd
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import multiprocessing
# import functools

# def update_opencellid_features(fn_cells, fn_places, antenna_same_tower_max_distance=5.0, meters=METERS, n_jobs=1):
#     with tqdm(total=4) as pbar:
        
#         ### 1. load cell ids
#         print(fn_cells)
#         gdf_cells =  geo.load_as_GeoDataFrame(fn=fn_cells, index_col=0, lat='lat', lon='lon', crs=geo.PROJ_DEG)
#         pbar.update(1)

#         ### 2. load survey data (DHS) or populated place
#         df_places = ios.load_csv(fn_places, index_col=0)
#         lat,lon = ('lat','lon') #('LATNUM', 'LONGNUM') if 'LATNUM' in df_places.columns else ('lat','lon')
#         gdf_places = geo.get_GeoDataFrame(df=df_places, lat=lat, lon=lon, crs=geo.PROJ_DEG)
#         df_places = delete_nonprojected_variables(df_places, os.path.basename(__file__))
#         pbar.update(2)
        
#         ### 3. Projection to meters
#         gdf_cells_proj = geo.get_projection(gdf_cells, geo.PROJ_MET)
#         gdf_places_proj = geo.get_projection(gdf_places, geo.PROJ_MET)
#         all_cells = geo.get_STRtree(gdf_cells_proj)
#         pbar.update(3)
        
#         ### 4. Identify towers
#         num_cores = os.cpu_count()
#         gdf_cells_proj = geo.fast_identify_clusters_within_distance(gdf_cells_proj, max_distance=antenna_same_tower_max_distance, n_jobs=num_cores)
#         gdf_cells_proj.rename(columns={'CLUSTER_ID':'tower_id'}, inplace=True)
#         pbar.update(4)
        
#     ### 4. Computation
#     ### - # of cells and towers within cluster area
#     ### - distance to closest cell
#     ### https://shapely.readthedocs.io/en/stable/manual.html#object.buffer
#     df_results = parallel_process_dataframe(gdf_places_proj, gdf_cells_proj, all_cells, meters, max_workers=num_cores)
#     df_places = df_places.join(df_results)
#     return df_places.drop(columns='geometry')


# # Example function to process each row
# def process_row(index, row, gdf_cells_proj, all_cells, meters):
#     # nearest place
#     nearest, distance_nearest = geo.find_nearest_place(all_cells, row.geometry)

#     # number of cells and towers
#     cells_in_area = {}
#     towers_in_area = {}
#     for m in meters:
#         k = round(m/1000.,2)
#         cells_area = geo.find_places_sqm(gdf_proj=gdf_cells_proj, point_proj=row.geometry, width=m)
#         cells = cells_area.shape[0]
#         towers = cells_area.tower_id.nunique()
#         cells_in_area[k] = cells
#         towers_in_area[k] = towers

#     # update df
#     columns = ['distance_closest_cell'] + [ f'cells_in_{k}km' for k in cells_in_area.keys()] + [ f'towers_in_{k}km' for k in cells_in_area.keys()]
#     obj = {c:0.0 for c in columns}
    
#     obj['distance_closest_cell'] = distance_nearest
#     for k,v in cells_in_area.items():
#         obj[f'cells_in_{k}km'] = v
#     for k,v in towers_in_area.items():
#         obj[f'towers_in_{k}km'] = v

#     df = pd.DataFrame(obj, index=[index], columns=columns)
#     return df

# # Function to apply parallel processing

# def parallel_process_dataframe(gdf_places_proj, gdf_cells_proj, all_cells, meters, max_workers=1):
#     # Create a partial function with the additional arguments
#     partial_process_row = functools.partial(process_row, gdf_cells_proj=gdf_cells_proj, all_cells=all_cells, meters=meters)

#     # Create a list to store the results
#     results = []

#     # Use ProcessPoolExecutor to process rows in parallel
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         # Submit each row to the executor
#         future_to_row = {executor.submit(partial_process_row, index, row): index for index, row in gdf_places_proj.iterrows()}
        
#         # Collect the results as they complete
#         for future in as_completed(future_to_row):
#             index = future_to_row[future]
#             try:
#                 result = future.result()
#                 results.append(result)
#             except Exception as exc:
#                 print(f'Row {index} generated an exception: {exc}')
                      
#     # Combine results into a DataFrame
#     result_df = pd.DataFrame(pd.concat(results))
#     return result_df
