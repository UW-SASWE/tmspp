# %%
import ee
ee.Initialize(project='tmospp')

# # %%
# %load_ext autoreload
# %autoreload 2

from rat.core.run_sarea import run_sarea

# %%
from pathlib import Path
import geopandas as gpd
import hvplot.pandas
import holoviews as hv

hv.extension('bokeh')


RESULTS_DIR = Path(f'results/')
DATA_DIR = Path(f'data/')

# read the bounding box of the study area
### all 100 reservoirs
val_pts = gpd.read_file(Path('/tiger1/pdas47/tmsosPP/data/validation-locations/100-validation-reservoirs-grand-pts.geojson'))
val_polys = gpd.read_file(Path('/tiger1/pdas47/tmsosPP/data/validation-locations/100-validation-reservoirs-grand-polys.geojson'))

val_res_pt = val_pts
val_res_poly = val_polys

print(f"Number of reservoirs: {len(val_res_pt)}")
print(f"Reservoirs: {val_res_pt['tmsos_id'].values}")


# tmsos requires the start and end date, reservoir shapefile, a dictionary of columns names for unique_identifier and nominal area, and a datadirectory
import pandas as pd


start_date = '2019-01-01'
end_date = '2024-08-30'
reservoir_shpfile = val_res_poly
shpfile_column_dict = {
    'unique_identifier': 'tmsos_id',
    'area_column': 'AREA_SKM'
}
datadir = Path('data/tmsos/')
datadir.mkdir(exist_ok=True)

run_sarea(
    start_date = start_date, 
    end_date = end_date, 
    datadir = datadir,
    reservoirs_shpfile = reservoir_shpfile, 
    shpfile_column_dict = shpfile_column_dict, 
)