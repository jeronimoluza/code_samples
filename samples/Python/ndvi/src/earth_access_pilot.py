import geopandas as gpd
import earthaccess
import os
from shapely import wkt

auth = earthaccess.login()


# Define search parameters
short_name = "MOD13Q1"
version = "061"  # Latest version of MOD13Q1
start_date = "2023-01-01"
end_date = "2023-12-31"
count = 10

region = gpd.GeoDataFrame(
    [
        wkt.loads(
            "POLYGON((-48.837 -1.83, -47.6921 -1.83, -47.6921 -3.3766, -48.837 -3.3766, -48.837 -1.83))"
        )
    ],
    columns=["geometry"],
)

bounding_box = tuple(region.total_bounds)


# Define download directory
download_dir = "../data/MODIS_MOD13Q1"
os.makedirs(download_dir, exist_ok=True)


# Search for MOD13Q1 granules
granules = earthaccess.search_data(
    short_name=short_name,
    version=version,
    temporal=(start_date, end_date),
    bounding_box=bounding_box,
    count=10,
)

print(f"Found {len(granules)} granules.")

# Download granules
files = earthaccess.download(granules, local_path=download_dir)

print(f"Downloaded {len(files)} files to {download_dir}.")
