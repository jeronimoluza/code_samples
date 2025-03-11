# README: Earth Observation Data Processing Script
#
# This script provides functions for downloading, processing, and extracting NDVI data from NASA MODIS.
# It is designed to work with MODIS data retrieved via Earth Access and stored in HDF format.
#
# Features:
# - Retrieve bounding boxes for geospatial regions.
# - Download satellite data using the `earthaccess` API.
# - Construct polygons from grid coordinates.
# - Extract metadata from HDF files.
# - Process and clean numerical data arrays.
#
# Dependencies:
# - `earthaccess` (for data retrieval)
# - `geopandas`, `shapely`, `numpy`, `pandas` (for geospatial processing)
# - `joblib` (for parallel processing)
# - `pyhdf` (for handling HDF files)
# - `pyproj` (for coordinate transformations)
# - `tqdm` (for progress tracking)
#
# Functions:
#
# 1. get_bounding_box(region)
#    - Extracts bounding box coordinates from different geospatial data formats (Polygon, MultiPolygon, GeoDataFrame, or WKT string).
#
# 2. download_earthaccess(download_dir, short_name, version, start_date, end_date, region, count=10)
#    - Searches for and downloads MODIS data files from Earth Access within a specified date range and region.
#
# 3. create_polygon(i, j, x_min, x_max, y_min, y_max)
#    - Constructs a bounding box polygon for a pixel given its grid coordinates.
#
# 4. extract_metadata(hdf_object, data_field)
#    - Extracts key metadata attributes (e.g., scale factor, valid range, spatial coordinates) from an HDF file.
#
# 5. treat_data_array(data_array, _FillValue, scale_factor, valid_range, add_offset)
#    - Cleans and processes numerical data arrays by applying scale factors, offsets, and handling missing values.
#
# Usage Example:
# ```python
# region = "POLYGON ((-100 40, -100 45, -95 45, -95 40, -100 40))"
# download_dir = "./data"
# short_name = "MOD13Q1"
# version = "061"
# start_date = "2023-01-01"
# end_date = "2023-12-31"
# files = download_earthaccess(download_dir, short_name, version, start_date, end_date, region)
# ```

import os
import re

import earthaccess
import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyhdf.SD import SD, SDC
from pyproj import Proj, Transformer
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon
from tqdm import tqdm


def get_bounding_box(region):
    """
    Returns the bounding box of the given region in the format
    (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).

    Parameters:
    - region (Polygon, MultiPolygon, GeoDataFrame, or WKT string): The input region.

    Returns:
    - tuple: Bounding box (min_lon, min_lat, max_lon, max_lat).
    """
    if isinstance(region, (Polygon, MultiPolygon)):
        (
            minx,
            miny,
            maxx,
            maxy,
        ) = region.bounds  # Directly get bounds for shapely geometries

    elif isinstance(region, gpd.GeoDataFrame):
        (
            minx,
            miny,
            maxx,
            maxy,
        ) = region.total_bounds  # Get total bounds for GeoDataFrame

    elif isinstance(region, str):
        try:
            geom = wkt.loads(region)  # Load WKT string
            if isinstance(geom, (Polygon, MultiPolygon)):
                minx, miny, maxx, maxy = geom.bounds
            else:
                raise ValueError(
                    "WKT does not represent a valid Polygon or MultiPolygon."
                )
        except Exception as e:
            raise ValueError(f"Invalid WKT string: {e}")

    else:
        raise TypeError(
            "Unsupported region type. Must be a Polygon, MultiPolygon, GeoDataFrame, or WKT string."
        )

    return (
        minx,
        miny,
        maxx,
        maxy,
    )  # Return in (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat) format


def download_earthaccess(
    download_dir: str,
    short_name: str,
    version: str,
    start_date: str,
    end_date: str,
    region,
    count: int = 10,
) -> list:
    """
    Downloads Earth observation data from the Earth Access platform.

    This function searches for and downloads MODIS granules within a specified region and date range.
    It uses the Earth Access Python library to perform the search and download operations.

    Parameters:
    - download_dir (str): The directory where the downloaded files will be saved.
    - short_name (str): The short name of the dataset (e.g., 'MOD13Q1').
    - version (str): The version of the dataset (e.g., '061').
    - start_date (str): The start date of the search period in 'YYYY-MM-DD' format.
    - end_date (str): The end date of the search period in 'YYYY-MM-DD' format.
    - region (Polygon, MultiPolygon, GeoDataFrame, or WKT string): The region of interest.
    - count (int, optional): The maximum number of granules to download. Default is 10.

    Returns:
    - list: A list of downloaded file paths.
    """
    bounding_box = get_bounding_box(region=region)

    download_dir = download_dir + f"/{short_name}_{version}"
    # Define download directory
    os.makedirs(download_dir, exist_ok=True)

    # Search for MOD13Q1 granules
    granules = earthaccess.search_data(
        short_name=short_name,
        version=version,
        temporal=(start_date, end_date),
        bounding_box=bounding_box,
        count=count,
    )

    print(f"Found {len(granules)} granules.")
    print(f"Downloading {count} granules.")

    # Download granules
    files = earthaccess.download(granules, local_path=download_dir)

    print(f"Downloaded {len(files)} files to {download_dir}.")
    return files


def create_polygon(
    i: int,
    j: int,
    x_min: np.ndarray,
    x_max: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> Polygon:
    """
    Create a bounding box polygon for a given pixel.

    This function constructs a bounding box polygon for a specific pixel in a grid using the provided
    minimum and maximum x and y coordinates. The polygon is defined by the lower-left, lower-right,
    upper-right, and upper-left corners, and is closed by connecting the last point to the first.

    Parameters:
    - i (int): The row index of the pixel in the grid.
    - j (int): The column index of the pixel in the grid.
    - x_min (numpy.ndarray): An array containing the minimum x-coordinate for each column in the grid.
    - x_max (numpy.ndarray): An array containing the maximum x-coordinate for each column in the grid.
    - y_min (numpy.ndarray): An array containing the minimum y-coordinate for each row in the grid.
    - y_max (numpy.ndarray): An array containing the maximum y-coordinate for each row in the grid.

    Returns:
    - shapely.geometry.Polygon: A bounding box polygon for the given pixel.
    """
    return Polygon(
        [
            (x_min[j], y_min[i]),  # Lower-left
            (x_max[j], y_min[i]),  # Lower-right
            (x_max[j], y_max[i]),  # Upper-right
            (x_min[j], y_max[i]),  # Upper-left
            (x_min[j], y_min[i]),  # Close polygon
        ]
    )


def extract_metadata(hdf_object: SD, data_field: str) -> tuple:
    """
    Extract metadata from an HDF object for a specific data field.

    Parameters:
    - hdf_object (SD): An HDF object opened with the SD library.
    - data_field (str): The name of the data field for which metadata is to be extracted.

    Returns:
    - tuple: A tuple containing the following metadata values:
        - _FillValue (int or float): The fill value for the data field.
        - scale_factor (int or float): The scale factor for the data field.
        - valid_range (tuple): A tuple containing the valid range for the data field.
        - add_offset (int or float): The add offset for the data field.
        - upper_left_x (float): The x-coordinate of the upper-left corner of the grid.
        - upper_left_y (float): The y-coordinate of the upper-left corner of the grid.
        - lower_right_x (float): The x-coordinate of the lower-right corner of the grid.
        - lower_right_y (float): The y-coordinate of the lower-right corner of the grid.
    """
    struct_metadata = hdf_object.attributes(full=1)["StructMetadata.0"][0]
    data_field_metadata = hdf_object.select(data_field).attributes(full=1)
    lna = data_field_metadata["long_name"][0]
    valid_range = data_field_metadata["valid_range"][0]
    _FillValue = data_field_metadata["_FillValue"][0]
    scale_factor = data_field_metadata["scale_factor"][0]
    units = data_field_metadata["units"][0]
    add_offset = data_field_metadata["add_offset"][0]

    # Construct the grid.  The needed information is in a global attribute
    # called 'StructMetadata.0'.  Use regular expressions to tease out the
    # extents of the grid.
    ul_regex = re.compile(
        r"""UpperLeftPointMtrs=\(
                              (?P<upper_left_x>[+-]?\d+\.\d+)
                              ,
                              (?P<upper_left_y>[+-]?\d+\.\d+)
                              \)""",
        re.VERBOSE,
    )

    match = ul_regex.search(struct_metadata)
    upper_left_x = np.float64(match.group("upper_left_x"))
    upper_left_y = np.float64(match.group("upper_left_y"))

    lr_regex = re.compile(
        r"""LowerRightMtrs=\(
                              (?P<lower_right_x>[+-]?\d+\.\d+)
                              ,
                              (?P<lower_right_y>[+-]?\d+\.\d+)
                              \)""",
        re.VERBOSE,
    )
    match = lr_regex.search(struct_metadata)
    lower_right_x = np.float64(match.group("lower_right_x"))
    lower_right_y = np.float64(match.group("lower_right_y"))

    return (
        _FillValue,
        scale_factor,
        valid_range,
        add_offset,
        upper_left_x,
        upper_left_y,
        lower_right_x,
        lower_right_y,
    )


def treat_data_array(
    data_array: np.ndarray,
    _FillValue: int,
    scale_factor: float,
    valid_range: list,
    add_offset: float,
) -> np.ndarray:
    """
    Processes and cleans the input data array by applying the given attributes.

    Parameters:
    - data_array (numpy.ndarray): The input data array to be processed.
    - _FillValue (int or float): The fill value for the data array.
    - scale_factor (int or float): The scale factor for the data array.
    - valid_range (tuple): A tuple containing the valid range for the data array.
    - add_offset (int or float): The add offset for the data array.

    Returns:
    - numpy.ndarray: The processed and cleaned data array.
    """
    data_array = data_array.get().astype(np.float64)
    # Apply the attributes to the data.
    invalid = np.logical_or(data_array < valid_range[0], data_array > valid_range[1])
    invalid = np.logical_or(invalid, data_array == _FillValue)
    data_array[invalid] = np.nan
    data_array = (data_array - add_offset) / scale_factor
    data_array = np.ma.masked_array(data_array, np.isnan(data_array))
    return data_array

def create_polygon(
    i: int,
    j: int,
    x_min: np.ndarray,
    x_max: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> Polygon:
    """
    Create a bounding box polygon for a given pixel.

    This function constructs a bounding box polygon for a specific pixel in a grid using the provided
    minimum and maximum x and y coordinates. The polygon is defined by the lower-left, lower-right,
    upper-right, and upper-left corners, and is closed by connecting the last point to the first.

    Parameters:
    - i (int): The row index of the pixel in the grid.
    - j (int): The column index of the pixel in the grid.
    - x_min (numpy.ndarray): An array containing the minimum x-coordinate for each column in the grid.
    - x_max (numpy.ndarray): An array containing the maximum x-coordinate for each column in the grid.
    - y_min (numpy.ndarray): An array containing the minimum y-coordinate for each row in the grid.
    - y_max (numpy.ndarray): An array containing the maximum y-coordinate for each row in the grid.

    Returns:
    - shapely.geometry.Polygon: A bounding box polygon for the given pixel.
    """
    return Polygon(
        [
            (x_min[j], y_min[i]),  # Lower-left
            (x_max[j], y_min[i]),  # Lower-right
            (x_max[j], y_max[i]),  # Upper-right
            (x_min[j], y_max[i]),  # Upper-left
            (x_min[j], y_min[i]),  # Close polygon
        ]
    )

def generate_polygon_data(
    data: np.ndarray,
    upper_left_x: float,
    upper_left_y: float,
    lower_right_x: float,
    lower_right_y: float,
    n_jobs: int = -1,
) -> tuple:
    """
    Generate polygons from a 2D data array representing a gridded region.

    Parameters:
    - data (numpy.ndarray): 2D array representing the gridded data.
    - upper_left_x (float): X-coordinate of the upper-left corner of the grid.
    - upper_left_y (float): Y-coordinate of the upper-left corner of the grid.
    - lower_right_x (float): X-coordinate of the lower-right corner of the grid.
    - lower_right_y (float): Y-coordinate of the lower-right corner of the grid.
    - n_jobs (int, optional): Number of jobs to run in parallel. Default is -1, which means using all available processors.

    Returns:
    - polygons (list): List of shapely Polygon objects representing the gridded data.
    - values (numpy.ndarray): Flattened 1D array of data values corresponding to the polygons.
    """
    print("Polygonizing data...")

    ny, nx = data.shape

    # Generate vectorized grid coordinates
    x = np.linspace(upper_left_x, lower_right_x, nx)
    y = np.linspace(upper_left_y, lower_right_y, ny)

    # Compute pixel size
    pixel_width = (lower_right_x - upper_left_x) / nx
    pixel_height = (lower_right_y - upper_left_y) / ny

    # Compute lower-left and upper-right corners
    x_min = x[:-1]  # Shape (nx-1,)
    y_min = y[:-1]  # Shape (ny-1,)
    x_max = x_min + pixel_width
    y_max = y_min + pixel_height

    # Generate all (i, j) index pairs
    index_pairs = [(i, j) for i in range(nx - 1) for j in range(ny - 1)]

    # Parallelized polygon creation
    polygons = Parallel(n_jobs=n_jobs)(
        delayed(create_polygon)(i, j, x_min, x_max, y_min, y_max)
        for i, j in tqdm(index_pairs, total=len(index_pairs))
    )

    # Flatten the data array to match polygons
    values = data[:-1, :-1].ravel()
    print("Done!")

    return polygons, values


def create_geodataframe_and_reproject(
    polygons: list, values: list, data_field: str, polygon_to_check: bool = None
) -> pd.DataFrame:
    """
    Creates a GeoDataFrame from the given polygons and values, reprojects the data to WGS84,
    and optionally filters the data based on a given polygon.

    Parameters:
    - polygons (list): A list of shapely Polygon objects representing the gridded data.
    - values (numpy.ndarray): A flattened 1D array of data values corresponding to the polygons.
    - polygon_to_check (shapely.geometry.Polygon, optional): A polygon to check for intersection.
        If provided, only the polygons that intersect with this polygon will be included in the
        resulting GeoDataFrame. Default is None.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame containing the reprojected polygons and their corresponding
        values. The GeoDataFrame will have two columns: 'value' and 'wkt', representing the data
        values and the polygons in Well-Known Text (WKT) format, respectively.
    """
    print("Creating GeoDataFrame and reprojecting...")
    sinu_proj = "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext"
    gdf = gpd.GeoDataFrame({data_field: values, "geometry": polygons}, crs=sinu_proj)
    # Reproject from Sinusoidal to WGS84
    gdf = gdf.to_crs("EPSG:4326")
    if polygon_to_check:
        gdf = gdf[gdf.geometry.intersects(polygon_to_check)]

    # Convert to WKT
    gdf["wkt"] = gdf["geometry"].apply(lambda geom: geom.wkt)
    print("Done!")

    return gdf[[data_field, "wkt"]]


def process_file(
    filepath: str, data_field: str, geometry: Polygon = None
) -> gpd.GeoDataFrame:
    """
    Processes a single HDF file and generates a GeoDataFrame containing polygons and their corresponding values.

    Parameters:
    - filepath (str): The path to the HDF file to be processed.
    - data_field (str): The name of the data field within the HDF file to be processed.
    - geometry (shapely.geometry.Polygon, optional): A polygon to check for intersection.
        If provided, only the polygons that intersect with this polygon will be included in the resulting GeoDataFrame.
        Default is None.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame containing the reprojected polygons and their corresponding values.
        The GeoDataFrame will have two columns: 'value' and 'wkt', representing the data values and the polygons in
        Well-Known Text (WKT) format, respectively.
    """
    print(f"Processing file: {filepath}")
    hdf_file = SD(filepath, SDC.READ)
    print("Extracting from HDF...")
    (
        _FillValue,
        scale_factor,
        valid_range,
        add_offset,
        upper_left_x,
        upper_left_y,
        lower_right_x,
        lower_right_y,
    ) = extract_metadata(hdf_file, data_field)
    data = hdf_file.select(data_field)
    data = treat_data_array(data, _FillValue, scale_factor, valid_range, add_offset)
    hdf_file.end()
    print("Done!")

    polygons, values = generate_polygon_data(
        data, upper_left_x, upper_left_y, lower_right_x, lower_right_y, n_jobs=-1
    )
    return create_geodataframe_and_reproject(polygons, values, data_field, geometry)

