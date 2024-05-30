import xarray as xr
import easysnowdata
import geopandas as gpd
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
import dask
import numpy as np
import geopandas as gpd
from pathlib import Path
import shutil
from rioxarray.crs import CRS
import subprocess
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from rioxarray.exceptions import NoDataInBounds


## Pekel dataset
def get_pekel_for_reservoir(roi, reservoir_id=None):
    pekel_ds = get_pekel_dataset(roi)

    pekel_ds = pekel_ds.assign_coords({
        'reservoir': [reservoir_id]
    })

    return pekel_ds

def get_pekel_dataset(roi, pekel_dir, scaling_factor=1.05):
    print(f"ROI CRS: ", roi.crs)
    roi_4326 = roi.to_crs('epsg:4326')
    
    fnx_l = int(roi_4326.total_bounds[0].round(-1)-10)
    fny_l = int(roi_4326.total_bounds[1].round(-1))
    fnx_u = int(roi_4326.total_bounds[2].round(-1))
    fny_u = int(roi_4326.total_bounds[3].round(-1)+10)
    tiles = [
        f"{abs(dx)}{'E' if dx >=0 else 'W'}_{abs(dy)}{'N' if dy >= 0 else 'S'}" for dx in range(fnx_l, fnx_u+1, 10) for dy in range(fny_l, fny_u+1, 10)]
    print("roi (epsg:4326) total bounds: ", roi_4326.total_bounds)
    print("tile limits: ", fnx_l, fny_l, fnx_u, fny_u)
    print("tiles: ", tiles)
    
    fns = [pekel_dir / f"occurrence_{tile}v1_4_2021.nc" for tile in tiles]
    ds = xr.open_mfdataset(fns, chunks={'x': 1024*5, 'y': 1024*5}, decode_coords="all").sel(band=1)
        
    ds['band_data'] = ds['band_data'].rio.write_nodata(-1)
    ds = ds.rename({'band_data': 'occurrence'})
    
    ds = ds.rio.set_crs('epsg:4326')
    ds = ds.rio.clip_box(*roi_4326.total_bounds)
    
    ds['occurrence'] = (ds['occurrence'] * scaling_factor).clip(0, 100)

    return ds

def get_occurrence_like(roi, reproject_match, pekel_dir, stretching_factor=1.05):
    """Get occurence over ROI. Always converts to estimated UTM projection.

    Inputs
    ------
    roi (geopandas.GeoDataFrame): ROI in WGS84.
    reproject_match (xarray.Dataset): Dataset to reproject to.
    stretching_factor (float): Factor to stretch the occurrence values by. Default is 1.05. 
        Multiples occurernce by this factor and clips to 0-100. This is to minimize the effect of 
        underestimation of occurrence which can happen in Pekel dataset since it is based on Landsat 
        optical data.
    """
    pekel_ds = get_pekel_dataset(roi, pekel_dir, stretching_factor)
    crs_wkt = reproject_match.spatial_ref.attrs['crs_wkt']
    # histogram to find threshold of occurrence
    occurrence = pekel_ds['occurrence'].rio.reproject_match(reproject_match).transpose('y', 'x')
    occurrence.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    occurrence.rio.write_crs(CRS.from_wkt(crs_wkt), inplace=True)
    occurrence.rio.write_nodata(np.nan, inplace=True)
    occurrence = occurrence.rio.clip(roi.to_crs(crs_wkt).geometry.values, drop=True)
    return occurrence

## SWOT
def download_swot_for_reservoir(roi, start_date, end_date, download_dir):
    roi = roi.to_crs('epsg:4326')
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    download_dir = Path(download_dir)
    bounds = ",".join([f"{x:.3f}" for x in roi.total_bounds])

    podaac_cmd = [
        'podaac-data-downloader',
        '-c', 'SWOT_L2_HR_Raster_100m_2.0',
        '-d', str(download_dir), 
        rf'-b="{bounds}"',
        '-gr="SWOT_L2_HR_Raster_100m_UTM*"',
        '--start-date', pd.to_datetime(start_date).strftime('%Y-%m-%dT%XZ'),
        '--end-date', pd.to_datetime(end_date).strftime('%Y-%m-%dT%XZ'),
    ]

    res = subprocess.run(" ".join(podaac_cmd), shell=True)
    res

def get_swot_id(
        id, 
        val_res_poly,
        start_date="2022-07-01", 
        end_date="2022-07-31", 
        buffer=2000, # m
        qual_mask_threshold=0.7,
        swot_dir = Path(f'../data/swot/raw'),
        gd_track_fn = Path('../data/swot_orbit/swot_orbit.geojson'),
        download = False
    ):
    roi = val_res_poly.loc[val_res_poly['tmsos_id']==id]
    buffered_roi = roi.to_crs(roi.estimate_utm_crs()).geometry.iloc[0].convex_hull.buffer(buffer)
    roi.crs = roi.estimate_utm_crs()
    roi.geometry = [buffered_roi]
    roi = roi.to_crs('epsg:4326')

    # download
    if download:
        download_swot_for_reservoir(roi, start_date, end_date, swot_dir)

    # determine swot pass number
    gd_track = gpd.read_file(gd_track_fn)

    matches = gd_track[gd_track.intersects(roi.iloc[0].geometry)]
    pass_ids = list(matches.ID_PASS)
    
    pass_id_l = []
    fns = []
    for pass_id in pass_ids:
        files = list(swot_dir.glob(f'*_{pass_id:03}_*'))
        pass_id_l.extend([pass_id]*len(files))
        fns.extend(files)

    fn_dates = [fn.name.split('_')[13] for fn in fns]

    datas = []

    pbar = tqdm(total=len(fns))
    for fn, fn_date, pass_id in zip(fns, fn_dates, pass_id_l):
        fn_date = pd.to_datetime(fn_date)
        pbar.set_description_str(f"Processing {fn_date}")
        if fn_date < pd.to_datetime(start_date) or fn_date > pd.to_datetime(end_date):
            continue
        data = xr.open_dataset(fn, decode_coords="all")
        date = pd.to_datetime(fn.name.split('_')[13])

        data = data.assign_coords(
            reservoir=((id)),
            time=((pd.to_datetime(date.date()))),
            pass_id = ((pass_id))
        )

        projection = roi.estimate_utm_crs()
        roi = roi.to_crs(projection)
        data = data.rio.write_crs(projection)
        try:
            data = data[[
                'water_area', 'water_area_qual', 
                'water_frac', 'water_frac_uncert', 
                'wse', 'wse_uncert', 'wse_qual'
            ]].rio.clip(roi.geometry.values, crs=projection, drop=True)
            datas.append(data)
        except NoDataInBounds as e:
            print(e)
            pass
        finally:
            pbar.update(1)

    try:
        data = xr.concat(datas, dim='time')
        
        data = data.groupby('time').mean(dim='time')

        data = data.chunk(chunks={
            'time': 30,
            'x': 4096,
            'y': 4096
        })

        return data
    except Exception as e:
        print(e)
        return None

## HLS
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
    RESERVOIR = '0505'
    buffer_amt = 800 # m

    swot_dir = Path('../data/swot/raw')
    data = Path('/tiger1/pdas47/tmsosPP/data')
    val_pt_fn = data / 'validation-locations/subset-validation-reservoirs-grand-pts.geojson'
    val_poly_fn = data / 'validation-locations/subset-validation-reservoirs-grand.geojson'
    
    selected_reservoirs = [
        '0505', # dumboor. India
        '0810', # sirindhorn, Thailand.
        '0830', # Krasoew, Thailand.
        '0502', # Bhakra dam, India.
        '0518', # Bhadra, India.
        '0349', # vaaldam, South Africa.
        '0464', # Sterkspruit, South Africa.
        '0214', # Cijara, Spain
        '1498', # Toledo bend, US
        '0936', # Arrow, Canada
    ]
    res_names = {
        '0505': 'Dumboor, In',
        '0810': 'Sirindhorn, Th',
        '0830': 'Krasoew, Th',
        '0502': 'Bhakra, In',
        '0518': 'Bhadra, In',
        '0349': 'Vaaldam, SA',
        '0464': 'Sterkspruit, SA',
        '0214': 'Cijara, Sp',
        '1498': 'Toledo Bend, US',
        '0936': 'Arrow, Ca'
    }
    RESERVOIR_NAME = res_names[RESERVOIR]

    val_pts = gpd.read_file(val_pt_fn)
    val_res_pt = val_pts.loc[val_pts['tmsos_id'].isin(selected_reservoirs)]
    val_polys = gpd.read_file(val_poly_fn)
    val_res_poly = val_polys.loc[val_polys['tmsos_id'].isin(selected_reservoirs)]

    roi = val_res_poly[val_res_poly['tmsos_id']==RESERVOIR]
    utm_crs = roi.estimate_utm_crs()

    buffered_roi_utm = roi.to_crs(utm_crs).geometry.buffer(buffer_amt)
    buffered_roi = roi.to_crs('epsg:4326')

    # download SWOT
    

    # # download Pekel data
    # pekel_dir = data / 'pekel/occurrence_nc'
    # occurrence = get_occurrence_like(
    #     buffered_roi, # WATER FRAC
    # )

    # download HLS reservoir data
    # download_hls_reservoir(
    #     selected_reservoirs, 
    #     start_date="2023-07-01", end_date="2023-12-31",
    #     save_dir=Path("/tiger1/pdas47/tmsosPP/data/hls"),
    #     val_poly_fn=val_poly_fn,
    #     dask_client=None
    #     # dask_client=client
    # )


if __name__ == '__main__':
    main()