import papermill as pm
import xarray as xr
import hvplot.xarray
import hvplot.pandas
import numpy as np
import pandas as pd
import holoviews as hv
import geoviews as gv
import matplotlib as mpl

import geopandas as gpd
from pathlib import Path

from dask.distributed import LocalCluster, Client

import warnings

warnings.filterwarnings('ignore')

hv.extension('bokeh')


def process(PARAMS, notebook_path):
    notebook_path = Path(notebook_path)
    notebook_name = notebook_path.stem
    # dst_notebook_path = notebook_path.parent / 'papermill' / notebook_name / f"{reservoir}.ipynb"
    data_fraction = PARAMS.get('data_fraction')
    dst_notebook_path = notebook_path.parent / 'papermill' / notebook_name / f"{data_fraction}.ipynb"

    if not dst_notebook_path.parent.exists():
        print("Creating directory to save notebooks ran through papermill: ", str(dst_notebook_path.parent))
        dst_notebook_path.parent.mkdir()

    # print(f"processing {reservoir}")
    try:
        pm.execute_notebook(
            notebook_path,
            dst_notebook_path,
            parameters=PARAMS
        )
        return True
    except Exception as e:
        # print(f'Something went wrong, {reservoir}: {resname}')
        print(e)
        return False

def main(client=None):
    NOTEBOOK = "/tiger1/pdas47/tmsosPP/notebooks/03.10-ML-data-prep.ipynb"

    # val_pts = gpd.read_file(Path('/tiger1/pdas47/tmsosPP/data/validation-locations/locations-with-2023-24-insitu-pts-correct-db.geojson'))
    val_pts = gpd.read_file(Path('/tiger1/pdas47/tmsosPP/data/validation-locations/glws-tmsos-baseline-comparison-locations-pts.geojson'))
    val_polys = gpd.read_file(Path('/tiger1/pdas47/tmsosPP/data/validation-locations/glws-tmsos-baseline-comparison-locations-polys.geojson'))


    selected_reservoirs = val_polys['tmsos_id'].tolist()
    res_names_dict = val_polys[['tmsos_id', 'name']].set_index('tmsos_id').to_dict()['name']
    res_names = [res_names_dict[res] for res in selected_reservoirs]
    
    if client is None:
        for data_fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            PARAMS = dict(
                data_fraction=data_fraction
            )
            process(PARAMS=PARAMS, notebook_path=NOTEBOOK)
        # for reservoir, resname in zip(selected_reservoirs, res_names):
        #     PARAMS = dict(
        #         RESERVOIR=reservoir,
        #     )
        #     process(resname, PARAMS, NOTEBOOK)
    else:
        futures = client.map(process, [dict(data_fraction=param) for param in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]], [NOTEBOOK]*len(selected_reservoirs))
        # futures = client.map(process, res_names, [dict(RESERVOIR=reservoir) for reservoir in selected_reservoirs], [NOTEBOOK]*len(selected_reservoirs))
        results = client.gather(futures)
        print(results)
        print(f'Passed: {results.count(True)}/{len(results)}')

if __name__ == '__main__':
    clutser = LocalCluster(n_workers=25, threads_per_worker=1)
    client = Client(clutser)
    print(client.dashboard_link)

    main(client)