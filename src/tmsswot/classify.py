import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from rioxarray.crs import CRS


## SWOT
def get_swot_missing_data(swot_ds, roi, area_qual_threshold, wse_qual_threshold):
    """return a mask of missing data in the SWOT dataset, and a rate of missing data within the ROI."""
    missing_mask = swot_ds['water_area_qual'] > area_qual_threshold
    missing_mask = missing_mask | (swot_ds['wse_qual'] > wse_qual_threshold)
    missing_mask = missing_mask | (np.isnan(swot_ds['water_area_qual']))
    missing_mask = missing_mask * 1
    missing_mask = missing_mask.rename('missing_mask')
    missing_mask = missing_mask.rio.set_spatial_dims('y', 'x')
    utm_crs = roi.estimate_utm_crs()
    missing_mask = missing_mask.rio.write_crs(utm_crs)

    missing_mask_rate = missing_mask.mean(dim=('x', 'y'))
    missing_mask_rate = missing_mask_rate.rename('missing_mask_rate')

    return xr.merge([missing_mask, missing_mask_rate])

def classify_swot(
        swot_ds, 
        roi, 
        occurrence, 
        water_frac_threshold=0.7, 
        area_qual_threshold=1, 
        wse_qual_threshold=1, 
        zg_omega=0.3
    ):
    swot_ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    swot_ds.rio.write_crs(CRS.from_wkt(swot_ds.spatial_ref.attrs['crs_wkt']), inplace=True)

    water_map = (swot_ds['water_frac'].where(swot_ds['water_area_qual'] <= area_qual_threshold) > water_frac_threshold) * 1
    water_map = water_map.rio.set_crs(swot_ds.rio.crs)
    water_map.attrs['standard_name'] = "surface_water"
    water_map.attrs['long_name'] = "Surface water of lake/reservoir observed by SWOT"
    water_map = water_map.rename('surface_water')

    missing_mask = get_swot_missing_data(swot_ds, roi, area_qual_threshold, wse_qual_threshold)
    # enhance using historical occurrence data
    def zg(water_map_block):
        times = water_map_block.time
        res = xr.zeros_like(water_map_block)

        for i, time in enumerate(times):
            water_map_slice = water_map_block.sel(time=time)
            if missing_mask['missing_mask_rate'].sel(time=time) <= 0.95:
                mask_slice = missing_mask['missing_mask'].sel(time=time)
                
                occurrence_slice = xr.where(~mask_slice, occurrence, np.nan)
                occurrence_slice = xr.where(water_map_slice==1, occurrence_slice, 0)
                occurrence_counts, occurrence_values = np.histogram(occurrence_slice.values, bins=99, range=(1, 100))
                count_threshold = np.nanmean(occurrence_counts) * zg_omega
                occurrence_idx = np.nanargmax(np.where(occurrence_counts > count_threshold) if (occurrence_counts > count_threshold).sum() > 0 else [0])
                occurrence_idx = np.nan if occurrence_idx == 0 else occurrence_idx
                
                pekel_estimated_map = (occurrence >= occurrence_idx)
        
                corrected_map = xr.where(
                    missing_mask['missing_mask'].sel(time=time) == 1, 
                    pekel_estimated_map,
                    water_map_slice
                )
                res.data[i] = corrected_map
            else:
                res.data[i] = np.full_like(water_map_slice, 2)
        
        return res
    
    water_map = water_map.chunk({'time': 50, 'x': -1, 'y': -1})
    water_map = water_map.transpose('time', 'y', 'x')
    water_map_uncorrected = water_map.copy()
    water_map = water_map.map_blocks(
        zg, template=water_map
    ).rename("surface_water")
    water_map.rio.set_nodata(2, inplace=True)
    
    # water_map_area = swot_ds['water_area'].where(water_map == 1).sum(dim=('x', 'y')) * 1e-6
    area_pixels = xr.where(water_map==1, swot_ds['water_area'].fillna(1e4), np.nan)
    water_map_area = area_pixels.sum(dim=('x', 'y')) * 1e-6
    water_map_area.attrs['standard_name'] = "surface_water_area"
    water_map_area.attrs['long_name'] = "Surface water area of lake/reservoir observed by SWOT"
    water_map_area.attrs['unit'] = "km^2"
    water_map_area = water_map_area.rename('surface_water_area')

    water_surface_elevation_raster = xr.where(missing_mask['missing_mask'], np.nan, swot_ds['wse'])
    water_surface_elevation_raster = water_surface_elevation_raster.where(swot_ds['wse_qual'] <= wse_qual_threshold)
    water_surface_elevation_raster = xr.where(water_map_uncorrected == 1, water_surface_elevation_raster, np.nan).rename('surface_water_elevation_raster')
    water_surface_elevation_raster.attrs['standard_name'] = "surface_water_elevation_raster"
    water_surface_elevation_raster.attrs['long_name'] = "Surface water elevation of lake/reservoir observed by SWOT"
    water_surface_elevation_raster.attrs['unit'] = "m"

    water_surface_elevation = water_surface_elevation_raster.mean(dim=['x', 'y'])
    water_surface_elevation.attrs['standard_name'] = "surface_water_elevation"
    water_surface_elevation.attrs['long_name'] = "Water Surface Elevation observed by SWOT"
    water_surface_elevation.attrs['unit'] = "m"
    water_surface_elevation = water_surface_elevation.rename('surface_water_elevation')

    storage_change = (water_map_area[1:] - water_map_area[:-1]) * (water_surface_elevation[1:] + water_surface_elevation[:-1]) * 0.5 * 1e6
    storage_change.attrs['standard_name'] = "storage_change"
    storage_change.attrs['long_name'] = "Change in storage of lake/reservoir observed by SWOT"
    storage_change.attrs['unit'] = "m^3"
    storage_change = storage_change.rename('storage_change')

    time_since_last_obs = (water_map_area.time[1:] - water_map_area.time[:-1]).astype('timedelta64[D]')
    time_since_last_obs.attrs['standard_name'] = "time_since_last_obs"
    time_since_last_obs.attrs['long_name'] = "Time since last observation of lake/reservoir observed by SWOT"
    time_since_last_obs.attrs['unit'] = "days"
    time_since_last_obs = time_since_last_obs.rename('time_since_last_obs')

    swot_ds = xr.merge([
        water_map, water_map_area, water_surface_elevation_raster, 
        water_surface_elevation, missing_mask, storage_change, time_since_last_obs
    ])

    return swot_ds


## HLS
# Get value of QC bit based on location
def get_qc_bit(ar, bit):
    # taken from Helen's fantastic repo https://github.com/UW-GDA/mekong-water-quality/blob/main/02_pull_hls.ipynb
    return (ar // (2**bit)) - ((ar // (2**bit)) // 2 * 2)


def get_hls_missing_data(hls_ds, roi, clip_to_roi=True):
    # transform
    utm_crs = roi.estimate_utm_crs()
    # fmask = fmask.rio.reproject(utm_crs)
    roi = roi.to_crs(utm_crs)
    print("utm CRS of ROI: ", utm_crs)
    print("hls_ds CRS: ", hls_ds.rio.crs)
    print("hls_ds bounds: ", hls_ds.rio.bounds())
    
    print("roi CRS: ", roi.crs)
    print("roi bounds: ", roi.total_bounds)

    mask = xr.full_like(hls_ds['Fmask'], 9)
    mask.data = np.isnan(hls_ds['Fmask'])
    mask.rio.write_nodata(9)
    mask.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
    mask.rio.write_crs(utm_crs, inplace=True)
    mask = mask.rename("cloud_mask")

    if clip_to_roi:
        mask = mask.rio.clip_box(*roi.total_bounds)
        mask = mask.rio.clip(roi.geometry.values, from_disk=True)

    cloudy_pixels = (mask.sum(dim=['x', 'y'])/(~np.isnan(mask)).sum(dim=['x', 'y'])).rename('cloudy_pixels')
    cloudy_pixels.attrs.update({
        'units': 'fraction', 'long_name': 'fraction of cloudy pixels'
    })

    return xr.merge([mask, cloudy_pixels])


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



def zhao_and_gao(water_map, occurrence, cloud_mask, omega=0.2):
    occurrence_orig = occurrence.rio.reproject_match(water_map.fillna(0))
    # histogram to find threshold of occurrence
    
    # zg
    def zg(water_map_block):
        times = water_map_block.time
        res = xr.zeros_like(water_map_block)

        for i, time in enumerate(times):
            water_map_slice = water_map_block.sel(time=time)
            if cloud_mask['cloudy_pixels'].sel(time=time) <= 0.95:
                mask_slice = cloud_mask['cloud_mask'].sel(time=time)
                
                occurrence_slice = xr.where(~mask_slice, occurrence_orig, np.nan)
                occurrence_slice = xr.where(water_map_slice==1, occurrence_slice, 0)
                occurrence_counts, occurrence_values = np.histogram(occurrence_slice.values, bins=99, range=(1, 100))
                count_threshold = np.nanmean(occurrence_counts) * omega
                occurrence_idx = np.nanargmax(np.where(occurrence_counts > count_threshold) if (occurrence_counts > count_threshold).sum() > 0 else [0])
                occurrence_idx = np.nan if occurrence_idx == 0 else occurrence_idx
                
                pekel_estimated_map = (occurrence_orig >= occurrence_idx)
        
                corrected_map = xr.where(
                    cloud_mask['cloud_mask'].sel(time=time) == 1, 
                    pekel_estimated_map,
                    # xr.where(
                    #     water_map_slice == 0,
                    #     pekel_estimated_map,
                    #     water_map_slice
                    # )
                    water_map_slice
                )
                res.data[i] = corrected_map
            else:
                res.data[i] = np.full_like(water_map_slice, 2)
        
        return res
    
    water_map = water_map.chunk({'time': 50, 'x': -1, 'y': -1})
    water_map = water_map.transpose('time', 'y', 'x')
    # res = water_map.groupby('time').apply(zg).rename("surface_water")
    res = water_map.map_blocks(
        zg, template=water_map
    ).rename("surface_water")
    res.rio.set_nodata(2)
    
    return res


def classify_hls(hls_ds, roi, occurrence):
    hls_ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    hls_ds.rio.write_crs(CRS.from_wkt(hls_ds.spatial_ref.attrs['crs_wkt']), inplace=True)
    
    # add surface water map
    water_map = (hls_ds['ndwi'] > 0).astype(np.uint8)
    CLOUD_NODATA = 2
    water_map = xr.where(~np.isnan(hls_ds['ndwi']), water_map, CLOUD_NODATA)
    water_map.rio.write_nodata(CLOUD_NODATA, inplace=True)
    water_map.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    water_map.rio.write_crs(CRS.from_wkt(hls_ds.spatial_ref.attrs['crs_wkt']), inplace=True)
    water_map.attrs['standard_name'] = "surface_water_uncorrected"
    water_map.attrs['long_name'] = "Surface water of lake/reservoir uncorrected for clouds"
    water_map = water_map.rename('surface_water_uncorrected')
    
    # add area time series as a variable
    water_map_area = (water_map.where(water_map==1).sum(dim=['x', 'y']) * 30*30 * 1e-6) # km2
    water_map_area.attrs['standard_name'] = 'surface_water_area_uncorrected'
    water_map_area.attrs['long_name'] = "Surface water area of lake/reservoir uncorrected for clouds"
    water_map_area.attrs['unit'] = 'km^2'
    water_map_area = water_map_area.rename('surface_water_area_uncorrected')

    # get cloud mask
    cloud_mask = get_hls_missing_data(hls_ds, roi).load()

    # apply cloud cover correction
    enhanced_water_map = zhao_and_gao(water_map, occurrence, cloud_mask)
    enhanced_water_map = enhanced_water_map.to_dataset()

    # add area time series after correction
    en_water_map_area = enhanced_water_map['surface_water'].where(enhanced_water_map['surface_water']==1).sum(dim=['x', 'y']) * 30*30 * 1e-6 # km2
    en_water_map_area = en_water_map_area.where(en_water_map_area > 0)
    en_water_map_area.attrs['standard_name'] = 'surface_water_area'
    en_water_map_area.attrs['long_name'] = "Surface water area of lake/reservoir corrected for clouds"
    en_water_map_area.attrs['unit'] = 'km^2'
    en_water_map_area = en_water_map_area.rename('surface_water_area')

    return xr.merge([
        water_map, 
        enhanced_water_map, 
        water_map_area, 
        en_water_map_area, 
        hls_ds['ndwi'],
        cloud_mask
    ])
