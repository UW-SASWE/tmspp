import ee
ee.Initialize(project='tmospp')

import sys
sys.path.append('/tiger1/pdas47/tmsosPP/models/RAT/src')

from rat.ee_utils.ee_aec_file_creator import aec_file_creator
import geopandas as gpd
from pathlib import Path
import hvplot.pandas
import pandas as pd
import holoviews as hv
import geoviews as gv
import numpy as np


# read the bounding box of the study area
val_pts = gpd.read_file(Path('/tiger1/pdas47/tmsosPP/data/validation-locations/100-validation-reservoirs-grand-pts.geojson'))
val_polys = gpd.read_file(Path('/tiger1/pdas47/tmsosPP/data/validation-locations/100-validation-reservoirs-grand-polys.geojson'))

selected_reservoirs = val_pts['tmsos_id'].tolist()  # select all 100 reservoirs

for reservoir in selected_reservoirs:
    print(f"PROCESSING {reservoir}")
    reservoir_shpfile = val_polys[val_polys['tmsos_id'] == reservoir]
    shpfile_column_dict = {
        'unique_identifier': 'tmsos_id',
    }
    aec_dir_path = Path(f'/tiger1/pdas47/tmsosPP/data/aec/srtm/')
    output_file = aec_dir_path / f"{reservoir}.csv"
    
    if output_file.exists():
        print(f"File {output_file} already exists. Skipping...")
        continue
    else:
        print(f"Creating {output_file}")
        aec_file_creator(
            reservoir_shpfile=reservoir_shpfile,
            shpfile_column_dict=shpfile_column_dict,
            aec_dir_path=aec_dir_path
        )