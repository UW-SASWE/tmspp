import xarray as xr

import easysnowdata
import geopandas as gpd
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
import dask

import geopandas as gpd
from pathlib import Path
import shutil


def get_hls_id(
        id, 
        start_date="2022-07-01", 
        end_date="2022-07-31", 
        val_res_poly_fn=Path('/tiger1/pdas47/tmsosPP/data/validation-locations/subset-validation-reservoirs-grand.geojson'),
        buffer=0.05
    ):
    val_res_poly = gpd.read_file(val_res_poly_fn)
    roi = val_res_poly.loc[val_res_poly['tmsos_id']==id]
    buffered_roi = roi.geometry.convex_hull.buffer(buffer)
    print(buffered_roi)
    hls = easysnowdata.remote_sensing.HLS(
        bbox_input=buffered_roi, start_date=start_date, end_date=end_date,
        bands=[
            'blue', 'green', 'red', 'nir narrow', 'swir 1', 'swir 2', 'Fmask'
        ]
    )
    data = None
    if hls.data is not None:
        hls.mask_data()

        raw_bands = hls.data
        hls.get_ndwi()
        ndwi = hls.ndwi.to_dataset(name='ndwi')

        data = xr.merge([raw_bands, ndwi])

        data = data.assign_coords(
            reservoir=((id))
        )

        data = data.drop_vars(['geometry', 'AssociatedBrowseImageUrls'])
    else:
        print(f'No data found between {start_date} and {end_date} for {id}')

    return data


def download_hls_reservoir(
        reservoir_id,
        start_date = "2019-12-01", end_date="2020-03-05",
        save_dir = Path('../data/hls'),
        dask_client=None,
        val_res_poly_fn=Path('/tiger1/pdas47/tmsosPP/data/validation-locations/subset-validation-reservoirs-grand.geojson'),
        buffer=0.05
    ):
    save_dir = Path(save_dir)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    print(f"Will download data for {reservoir_id} from {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")
    date_ranges = pd.date_range(start_date, end_date, freq='1MS', inclusive='both')
    
    download_date_range_args = list(
        zip(date_ranges[:-1], date_ranges[1:] - pd.Timedelta(1, 'days'))
    )

    download_date_range_args.append(
        (date_ranges[-1], pd.to_datetime(end_date))
    )

    dsses = []

    def _get_hls_for_subset(reservoir_id, subset_start, subset_end):
        start_date_subset = subset_start.tz_localize('utc')
        end_date_subset = subset_end.tz_localize('utc')
        save_fp = save_dir / f'{reservoir_id}' / f'{reservoir_id}_{subset_start:%Y-%m-%d}_{subset_end:%Y-%m-%d}.nc'
        save_fp.parent.mkdir(exist_ok=True)

        hls_data = get_hls_id(
            reservoir_id, 
            start_date_subset, 
            end_date_subset,
            val_res_poly_fn=val_res_poly_fn,
            buffer=buffer
        )
        if hls_data is not None:
            hls_data['platform'] = hls_data['platform'].astype(str)
            hls_data.to_netcdf(save_fp)

            return save_fp
        else:
            return False
    
    if dask_client is not None:
        for subset_start, subset_end in download_date_range_args:
            hls_data = dask_client.submit(_get_hls_for_subset, reservoir_id, subset_start, subset_end, retries=2)
            dsses.append(hls_data)
        
        return_statuses = dask_client.gather(dsses)
    else:
        return_statuses = []
        for subset_start, subset_end in download_date_range_args:
            hls_data = _get_hls_for_subset(reservoir_id, subset_start, subset_end)
            return_statuses.append(hls_data)
    
    return return_statuses


def main():
    data = Path('/tiger1/pdas47/tmsosPP/data')
    val_poly_fn = data / 'validation-locations/subset-validation-reservoirs-grand.geojson'
    selected_reservoirs = [
        '0505', # dumboor. India
        # '0810', # sirindhorn, Thailand.
        '0830', # Krasoew, Thailand.
        # '0502', # Bhakra dam, India.
        # '0518', # Bhadra, India.
        # '0349', # vaaldam, South Africa.
        # '0464', # Sterkspruit, South Africa.
        # '0214', # Cijara, Spain
        # '1498', # Toledo bend, US
        # '0936', # Arrow, Canada
    ]

    download_hls_reservoir(
        selected_reservoirs, 
        start_date="2023-07-01", end_date="2023-12-31",
        save_dir=Path("/tiger1/pdas47/tmsosPP/data/hls"),
        val_poly_fn=val_poly_fn,
        dask_client=None
        # dask_client=client
    )


if __name__ == '__main__':
    main()