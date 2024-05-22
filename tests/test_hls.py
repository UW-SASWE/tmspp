import pytest
from tmsswot.data import download_hls_reservoir
from pathlib import Path
import xarray as xr
from dask.distributed import LocalCluster, Client

def test_hls_download_cluster():
    data = Path('tests/data/')
    val_poly_fn = data / 'validation-locations/subset-validation-reservoirs-grand.geojson'
    print(val_poly_fn, val_poly_fn.exists())

    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(cluster)
    reservoir_id = '0830'
    save_fn = Path(f"tests/data/hls/{reservoir_id}/")
    if save_fn.exists():
        import shutil
        shutil.rmtree(save_fn)
        
    data = download_hls_reservoir(
        reservoir_id, start_date = '2019-01-01', end_date = '2019-01-30',
        save_dir='tests/data/hls/', dask_client=client,
        val_res_poly_fn=val_poly_fn
    )

    assert Path(f"tests/data/hls/{reservoir_id}").exists()
    ds = xr.open_mfdataset(Path(f"tests/data/hls/{reservoir_id}/").glob("*.nc"), engine='netcdf4')
    print(ds)
    assert ds is not None


def test_hls_download_no_cluster():
    data = Path('tests/data/')
    val_poly_fn = data / 'validation-locations/subset-validation-reservoirs-grand.geojson'
    print(val_poly_fn, val_poly_fn.exists())

    reservoir_id = '0505'
    save_fn = Path(f"tests/data/hls/{reservoir_id}/")
    if save_fn.exists():
        import shutil
        shutil.rmtree(save_fn)
        
    data = download_hls_reservoir(
        reservoir_id, start_date = '2019-01-01', end_date = '2019-01-30',
        save_dir='tests/data/hls/', dask_client=None,
        val_res_poly_fn=val_poly_fn
    )

    assert Path(f"tests/data/hls/{reservoir_id}").exists()
    ds = xr.open_mfdataset(Path(f"tests/data/hls/{reservoir_id}/").glob("*.nc"), engine='netcdf4')
    print(ds)
    assert ds is not None